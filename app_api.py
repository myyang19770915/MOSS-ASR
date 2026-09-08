"""FastAPI service for MOSS transcription, diarization, and LLM proofreading."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from moss_asr import (
    ASRConfig,
    AudioChunkConfig,
    MossASRPipeline,
    PipelineConfig,
    ProofreadConfig,
    TranscriptExporter,
)
from moss_asr.config import default_litellm_user_agent
from moss_asr.benchmarking import (
    BenchmarkManifestError,
    find_manifest_sample,
    list_manifests,
    load_manifest,
    manifest_stats,
    resolve_manifest_path,
    score_transcript,
    select_metric,
    summarize_scores,
)
from moss_asr.vllm_client import VLLMClient


APP_VERSION = "1.5.0"
MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
TEMP_UPLOAD_DIR = Path(tempfile.gettempdir()) / "moss_asr_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_DATA_ROOT = Path(
    os.getenv("MOSS_BENCHMARK_DATA_ROOT", str(BASE_DIR / "benchmarks"))
).resolve()

DEFAULT_VLLM_URL = os.getenv("MOSS_VLLM_URL", "http://127.0.0.1:18000")
DEFAULT_LITELLM_URL = os.getenv("LITELLM_API_BASE", "https://litellm.my-yang.online/v1")
DEFAULT_LITELLM_MODEL = os.getenv("LITELLM_MODEL", "deepseek-v4-flash")
ASR_ENGINE = os.getenv("MOSS_ASR_ENGINE", "vllm").lower()
MAX_UPLOAD_BYTES = int(os.getenv("MOSS_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
MAX_CONCURRENT_JOBS = max(1, int(os.getenv("MOSS_MAX_CONCURRENT_JOBS", "1")))
UPLOAD_CHUNK_BYTES = 1024 * 1024
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high"}
SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv",
}
INFERENCE_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)
# A complete locally mounted Common Voice zh-TW subset contains 500 rows. Keep
# the ceiling bounded, while allowing one full split-sized run when the host
# has time available; the UI still defaults to a short 20-row smoke test.
MAX_BENCHMARK_SAMPLES = max(1, min(500, int(os.getenv("MOSS_MAX_BENCHMARK_SAMPLES", "500"))))

BENCHMARK_DATASETS = [
    {
        "name": "Common Voice 25 zh-TW（OpenFormosa）",
        "languages": "繁體中文／台灣華語（zh-TW）",
        "license": "CC0-1.0",
        "best_for": "本專案內建可重現 500 筆 test 子集；一般繁中 ASR 基準",
        "url": "https://huggingface.co/datasets/OpenFormosa/common_voice_25_zh-TW",
    },
    {
        "name": "AISHELL-1",
        "languages": "普通話（Mandarin Chinese）",
        "license": "Apache-2.0",
        "best_for": "公開、標準化的中文朗讀語音；適合正式中文 ASR 對照",
        "url": "https://huggingface.co/datasets/shenyunhang/AISHELL-1",
    },
    {
        "name": "Primewords Chinese Corpus Set 1",
        "languages": "普通話（行動裝置錄音）",
        "license": "CC BY-NC-ND 4.0",
        "best_for": "行動裝置／口語中文；使用前請確認非商業與不可改作條款",
        "url": "https://us.openslr.org/47/",
    },
    {
        "name": "MInDS-14",
        "languages": "14 種語言（客服／銀行語境）",
        "license": "CC BY 4.0",
        "best_for": "短句、多語客服指令與跨語言快速驗證",
        "url": "https://huggingface.co/datasets/PolyAI/minds14",
    },
    {
        "name": "LibriSpeech ASR",
        "languages": "English（朗讀書籍）",
        "license": "CC BY 4.0",
        "best_for": "英文朗讀語音的標準 ASR 基準",
        "url": "https://huggingface.co/datasets/openslr/librispeech_asr",
    },
    {
        "name": "Mozilla Common Voice",
        "languages": "100+（含中文、台語、粵語、日韓與歐洲語言）",
        "license": "CC0-1.0",
        "best_for": "多口音、群眾錄音與自訂語言小樣本",
        "url": "https://commonvoice.mozilla.org/datasets",
    },
    {
        "name": "Google FLEURS",
        "languages": "102 種語言／10+ 語系",
        "license": "CC BY 4.0",
        "best_for": "固定跨語言基準與可重現比較",
        "url": "https://huggingface.co/datasets/google/fleurs",
    },
    {
        "name": "VoxPopuli",
        "languages": "18 種歐洲語言 + 15 種英語口音",
        "license": "CC0-1.0（請同時確認原始資料條款）",
        "best_for": "議會／演講型長句與非母語英語口音",
        "url": "https://huggingface.co/datasets/facebook/voxpopuli",
    },
    {
        "name": "ML-SUPERB 2.0",
        "languages": "141 種語言開發集",
        "license": "依來源資料集而定",
        "best_for": "低資源語言與跨語言涵蓋度研究",
        "url": "https://multilingual.superbbenchmark.org/challenge-interspeech2025/data_description",
    },
]

# MOSS/vLLM accepts ISO-like primary language codes, not dataset locales such
# as ``en-US`` or ``cmn_Hans_CN``. Keep the locale for score reporting, but
# only pass a supported primary hint to the model. An unknown code becomes
# auto-detect rather than failing an entire benchmark run.
MOSS_LANGUAGE_HINTS = {
    "af", "ar", "hy", "az", "be", "bs", "bg", "ca", "zh", "hr", "cs", "da",
    "nl", "en", "et", "fi", "fr", "gl", "de", "el", "he", "hi", "hu", "is",
    "id", "it", "ja", "kn", "kk", "ko", "lv", "lt", "mk", "ms", "mr", "mi",
    "ne", "no", "fa", "pl", "pt", "ro", "ru", "sr", "sk", "sl", "es", "sw",
    "sv", "tl", "ta", "th", "tr", "uk", "ur", "vi", "cy",
}
MOSS_LANGUAGE_ALIASES = {
    "cmn": "zh", "jpn": "ja", "kor": "ko", "tha": "th", "lao": "lo",
    "deu": "de", "ger": "de", "fra": "fr", "fre": "fr", "spa": "es",
    "por": "pt", "ita": "it", "nld": "nl", "dut": "nl", "pol": "pl",
    "ron": "ro", "rum": "ro", "rus": "ru", "ukr": "uk", "vie": "vi",
    "ces": "cs", "cze": "cs", "ell": "el", "gre": "el", "heb": "he",
    "fas": "fa", "pes": "fa", "ind": "id", "tgl": "tl", "fil": "tl",
}


def _cors_origins() -> list[str]:
    raw = os.getenv("MOSS_CORS_ORIGINS", "*")
    values = [value.strip() for value in raw.split(",") if value.strip()]
    return values or ["*"]


app = FastAPI(
    title="MOSS-Transcribe-Diarize ASR & Proofreading Server",
    description="Speech-to-text with diarization, timestamps, and streaming LLM proofreading.",
    version=APP_VERSION,
)
cors_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    """Serve the standalone web application."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/benchmark", response_class=FileResponse)
async def benchmark_page() -> FileResponse:
    """Serve the local multi-language ASR benchmark UI."""
    return FileResponse(WEB_DIR / "benchmark.html")


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    """Cheap liveness check used by the container health probe."""
    return {"status": "ok", "version": APP_VERSION, "vllm_url": DEFAULT_VLLM_URL}


@app.get("/readyz")
async def readyz() -> Dict[str, Any]:
    """Readiness check that also verifies the configured model server."""
    client = VLLMClient(ASRConfig(vllm_base_url=DEFAULT_VLLM_URL))
    ready = await asyncio.to_thread(client.check_health)
    if not ready:
        raise HTTPException(status_code=503, detail="vLLM 模型服務尚未就緒")
    return {"status": "ready", "version": APP_VERSION, "vllm_url": DEFAULT_VLLM_URL}


@app.get("/api/config")
async def api_config() -> Dict[str, Any]:
    """Return non-secret defaults required by the web UI."""
    return {
        "version": APP_VERSION,
        "vllm_url": DEFAULT_VLLM_URL,
        "model_id": MODEL_ID,
        "litellm_url": DEFAULT_LITELLM_URL,
        "litellm_model": DEFAULT_LITELLM_MODEL,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_upload_mb": round(MAX_UPLOAD_BYTES / 1024 / 1024),
        "max_audio_minutes": 90,
        "reasoning_efforts": sorted(REASONING_EFFORTS),
        "benchmark_max_samples": MAX_BENCHMARK_SAMPLES,
    }


def _validate_http_url(value: str, field_name: str) -> str:
    normalized = (value or "").strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} 必須是有效的 http 或 https URL")
    return normalized


def _validate_request_options(
    *,
    vllm_url: str,
    model_id: str,
    proofread: bool,
    llm_url: str,
    llm_model: str,
    reasoning_effort: Optional[str],
    max_chunk_sec: float,
) -> tuple[str, str, str, str, Optional[str], float]:
    clean_vllm_url = _validate_http_url(vllm_url, "vLLM URL")
    clean_model_id = (model_id or "").strip()
    clean_llm_model = (llm_model or "").strip()
    clean_effort = (reasoning_effort or "none").strip().lower()

    if not clean_model_id or len(clean_model_id) > 256:
        raise ValueError("ASR model_id 不可為空且最多 256 個字元")
    if not 30.0 <= max_chunk_sec <= 5400.0:
        raise ValueError("max_chunk_sec 必須介於 30 與 5400 秒")
    if clean_effort not in REASONING_EFFORTS:
        raise ValueError("reasoning_effort 必須是 none、minimal、low、medium 或 high")

    clean_llm_url = _validate_http_url(llm_url, "LiteLLM URL") if proofread else (llm_url or "").strip()
    if proofread and (not clean_llm_model or len(clean_llm_model) > 256):
        raise ValueError("啟用智慧校對時，LLM 模型名稱不可為空")
    return clean_vllm_url, clean_model_id, clean_llm_url, clean_llm_model, clean_effort, max_chunk_sec


def _parse_hotwords(raw: Optional[str]) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for item in (raw or "").split(","):
        word = item.strip()
        if not word or word in seen:
            continue
        if len(word) > 100:
            raise ValueError("每個熱詞最多 100 個字元")
        seen.add(word)
        words.append(word)
        if len(words) >= 100:
            break
    return words


async def _save_upload(file: UploadFile) -> tuple[Path, int]:
    """Persist an upload with extension, empty-file, and 100 MiB enforcement."""
    original_name = Path(file.filename or "audio.wav").name
    suffix = Path(original_name).suffix.lower() or ".wav"
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(status_code=415, detail=f"不支援 {suffix} 檔案；支援格式：{supported}")

    destination = TEMP_UPLOAD_DIR / f"upload_{os.urandom(8).hex()}{suffix}"
    total = 0
    try:
        with destination.open("xb") as output:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"檔案超過 {round(MAX_UPLOAD_BYTES / 1024 / 1024)} MB 上限",
                    )
                output.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="上傳檔案是空的")
        return destination, total
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


def _remove_temp_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _build_pipeline(
    *,
    vllm_url: str,
    model_id: str,
    include_timestamps: bool,
    include_speakers: bool,
    hotwords: Optional[str],
    proofread: bool,
    llm_url: str,
    llm_key: Optional[str],
    llm_model: str,
    reasoning_effort: Optional[str],
    adaptive_proofread: bool,
    max_chunk_sec: float,
    language: Optional[str] = None,
) -> MossASRPipeline:
    if ASR_ENGINE not in {"vllm", "transformers"}:
        raise RuntimeError("MOSS_ASR_ENGINE must be 'vllm' or 'transformers'.")

    (
        clean_vllm_url,
        clean_model_id,
        clean_llm_url,
        clean_llm_model,
        clean_effort,
        clean_max_chunk,
    ) = _validate_request_options(
        vllm_url=vllm_url,
        model_id=model_id,
        proofread=proofread,
        llm_url=llm_url,
        llm_model=llm_model,
        reasoning_effort=reasoning_effort,
        max_chunk_sec=max_chunk_sec,
    )

    effective_llm_key = (llm_key or os.getenv("LITELLM_API_KEY", "")).strip()
    hotwords_list = _parse_hotwords(hotwords)
    clean_language = (language or "").strip()
    if clean_language and (
        len(clean_language) > 32
        or not all(char.isalnum() or char in {"-", "_"} for char in clean_language)
    ):
        raise ValueError("language 必須是有效的語言代碼")
    config = PipelineConfig(
        audio=AudioChunkConfig(max_duration_seconds=clean_max_chunk),
        asr=ASRConfig(
            model_id=clean_model_id,
            vllm_base_url=clean_vllm_url,
            include_timestamps=include_timestamps,
            include_speakers=include_speakers,
            hotwords=hotwords_list,
            language=clean_language or None,
        ),
        proofread=ProofreadConfig(
            enabled=proofread,
            llm_api_base=clean_llm_url,
            llm_api_key=effective_llm_key,
            llm_model=clean_llm_model,
            reasoning_effort=clean_effort,
            adaptive_reasoning=adaptive_proofread,
            remove_stutter=proofread,
            normalize_punctuation=proofread,
            glossary_terms=hotwords_list,
        ),
    )
    return MossASRPipeline(config=config, use_local_transformers=(ASR_ENGINE == "transformers"))


def _clean_benchmark_language(value: str) -> Optional[str]:
    clean_value = (value or "auto").strip()
    if clean_value.lower() in {"", "auto", "und"}:
        return None
    if len(clean_value) > 32 or not all(char.isalnum() or char in {"-", "_"} for char in clean_value):
        raise ValueError("語言代碼格式不正確")
    return clean_value


def _benchmark_language_hint(value: str) -> Optional[str]:
    """Map dataset locales to a MOSS-supported language hint, or auto mode."""
    clean_value = _clean_benchmark_language(value)
    if not clean_value:
        return None
    primary = clean_value.lower().replace("_", "-").split("-", 1)[0]
    hint = MOSS_LANGUAGE_ALIASES.get(primary, primary)
    return hint if hint in MOSS_LANGUAGE_HINTS else None


def _result_payload(
    result: Any,
    *,
    include_timestamps: bool,
    include_speakers: bool,
) -> Dict[str, Any]:
    return {
        "status": "success",
        "audio_duration": result.audio_duration,
        "elapsed_time": result.elapsed_time,
        "model": result.model,
        "is_proofread": result.is_proofread,
        "proofread_notes": result.proofread_notes,
        "full_text": result.full_text,
        "formatted_text": result.get_formatted_text(
            include_timestamps=include_timestamps,
            include_speakers=include_speakers,
        ),
        "srt_text": TranscriptExporter.to_srt(result, include_speakers=include_speakers),
        "segments": [segment.to_dict() for segment in result.segments],
    }


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


def _check_litellm_connection(llm_url: str, api_key: str, model: str) -> Dict[str, Any]:
    """Validate a LiteLLM key and report whether the requested model is visible."""
    models_url = f"{llm_url.rstrip('/')}/models"
    request = urllib.request.Request(
        models_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": default_litellm_user_agent(),
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = body.strip()
        try:
            parsed = json.loads(detail)
            error = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
            detail = str(error.get("message") or error.get("detail") or error) if isinstance(error, dict) else str(error)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        detail = detail.replace(api_key, "***")[:800]
        hint = "API Key 無效或已過期" if exc.code == 401 else "API Key 被拒絕或沒有團隊／模型權限"
        raise HTTPException(status_code=exc.code, detail=f"{hint}：{detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"無法連線 LiteLLM：{exc.reason}") from exc

    model_ids = {
        str(item.get("id"))
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("id")
    }
    available = model in model_ids
    return {
        "status": "ok",
        "authenticated": True,
        "model": model,
        "model_available": available,
        "models_count": len(model_ids),
        "message": "Key 驗證成功，模型可使用" if available else "Key 驗證成功，但可用模型清單中找不到此模型",
    }


@app.post("/api/proofread/check")
async def api_proofread_check(
    llm_url: str = Form(DEFAULT_LITELLM_URL),
    llm_key: Optional[str] = Form(""),
    llm_model: str = Form(DEFAULT_LITELLM_MODEL),
) -> Dict[str, Any]:
    """Check LiteLLM authentication and model visibility without running inference."""
    clean_url = _validate_http_url(llm_url, "LiteLLM URL")
    clean_model = (llm_model or "").strip()
    effective_key = (llm_key or os.getenv("LITELLM_API_KEY", "")).strip()
    if not effective_key:
        raise HTTPException(status_code=400, detail="請輸入 API Key，或在伺服器設定 LITELLM_API_KEY")
    if not clean_model:
        raise HTTPException(status_code=400, detail="請輸入校對模型名稱")
    return await asyncio.to_thread(_check_litellm_connection, clean_url, effective_key, clean_model)


@app.post("/api/transcribe")
async def api_transcribe(
    file: UploadFile = File(...),
    vllm_url: str = Form(DEFAULT_VLLM_URL),
    model_id: str = Form(MODEL_ID),
    include_timestamps: bool = Form(True),
    include_speakers: bool = Form(True),
    hotwords: Optional[str] = Form(""),
    proofread: bool = Form(False),
    llm_url: str = Form(DEFAULT_LITELLM_URL),
    llm_key: Optional[str] = Form(""),
    llm_model: str = Form(DEFAULT_LITELLM_MODEL),
    reasoning_effort: Optional[str] = Form("medium"),
    adaptive_proofread: bool = Form(True),
    max_chunk_sec: float = Form(1800.0),
) -> Dict[str, Any]:
    """Transcribe one uploaded media file and return a final JSON response."""
    temp_file, _ = await _save_upload(file)
    try:
        pipeline = _build_pipeline(
            vllm_url=vllm_url,
            model_id=model_id,
            include_timestamps=include_timestamps,
            include_speakers=include_speakers,
            hotwords=hotwords,
            proofread=proofread,
            llm_url=llm_url,
            llm_key=llm_key,
            llm_model=llm_model,
            reasoning_effort=reasoning_effort,
            adaptive_proofread=adaptive_proofread,
            max_chunk_sec=max_chunk_sec,
        )

        def run() -> Any:
            with INFERENCE_SLOTS:
                return pipeline.process(temp_file)

        result = await asyncio.to_thread(run)
        return _result_payload(result, include_timestamps=include_timestamps, include_speakers=include_speakers)
    except Exception as exc:
        raise _http_error(exc) from exc
    finally:
        _remove_temp_file(temp_file)


@app.get("/api/benchmark/catalog")
async def api_benchmark_catalog() -> Dict[str, Any]:
    """Expose only relative benchmark manifests and public dataset references."""
    return {
        "manifests": list_manifests(BENCHMARK_DATA_ROOT),
        "max_samples": MAX_BENCHMARK_SAMPLES,
        "manifest_formats": ["jsonl", "csv"],
        "datasets": BENCHMARK_DATASETS,
        "mount_hint": "將資料集放在主機 benchmarks/，Docker 內會以唯讀 /benchmarks 掛載。",
    }


@app.get("/api/benchmark/preview")
async def api_benchmark_preview(
    manifest: str, limit: int = 8, language: str = "auto"
) -> Dict[str, Any]:
    """Return a safe, small preview of one locally mounted benchmark dataset."""
    try:
        preview_limit = max(1, min(30, int(limit)))
        clean_manifest = resolve_manifest_path(manifest, BENCHMARK_DATA_ROOT)
        selected_language = _clean_benchmark_language(language)
        all_stats = manifest_stats(clean_manifest, BENCHMARK_DATA_ROOT)
        stats = manifest_stats(
            clean_manifest, BENCHMARK_DATA_ROOT, language=selected_language
        )
        samples = load_manifest(
            clean_manifest,
            BENCHMARK_DATA_ROOT,
            max_samples=preview_limit,
            language=selected_language,
        )
    except (BenchmarkManifestError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "manifest": manifest,
        "dataset": clean_manifest.parent.name,
        "samples": stats["samples"],
        "total_samples": all_stats["samples"],
        "languages": all_stats["languages"],
        "selected_language": selected_language or "auto",
        "preview": [
            {
                "id": sample.sample_id,
                "language": sample.language,
                "reference": sample.reference,
                "audio_url": f"/api/benchmark/audio?manifest={urllib.parse.quote(manifest, safe='')}&sample_id={urllib.parse.quote(sample.sample_id, safe='')}",
            }
            for sample in samples
        ],
    }


@app.get("/api/benchmark/audio")
async def api_benchmark_audio(manifest: str, sample_id: str) -> FileResponse:
    """Stream one manifest-declared audio file; arbitrary filesystem paths are never accepted."""
    try:
        clean_manifest = resolve_manifest_path(manifest, BENCHMARK_DATA_ROOT)
        sample = find_manifest_sample(clean_manifest, BENCHMARK_DATA_ROOT, sample_id)
    except BenchmarkManifestError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        sample.audio_path,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


@app.post("/api/benchmark/run")
async def api_benchmark_run(
    manifest: str = Form(...),
    vllm_url: str = Form(DEFAULT_VLLM_URL),
    model_id: str = Form(MODEL_ID),
    dataset_language: str = Form("auto"),
    language_override: str = Form("auto"),
    metric: str = Form("auto"),
    max_samples: int = Form(20),
    max_chunk_sec: float = Form(1800.0),
) -> StreamingResponse:
    """Run a local audio/reference manifest and stream per-sample ASR scores."""
    try:
        if not 1 <= max_samples <= MAX_BENCHMARK_SAMPLES:
            raise ValueError(f"單次測試筆數必須介於 1 與 {MAX_BENCHMARK_SAMPLES}")
        clean_manifest = resolve_manifest_path(manifest, BENCHMARK_DATA_ROOT)
        dataset_language_filter = _clean_benchmark_language(dataset_language)
        samples = load_manifest(
            clean_manifest,
            BENCHMARK_DATA_ROOT,
            max_samples=max_samples,
            language=dataset_language_filter,
        )
        selected_metric = select_metric("und", metric)
        selected_language = _clean_benchmark_language(language_override)
        clean_vllm_url, clean_model_id, _, _, _, clean_max_chunk = _validate_request_options(
            vllm_url=vllm_url,
            model_id=model_id,
            proofread=False,
            llm_url="",
            llm_model="",
            reasoning_effort="none",
            max_chunk_sec=max_chunk_sec,
        )
    except (BenchmarkManifestError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[Optional[tuple[str, Dict[str, Any]]]] = asyncio.Queue()
    accepting_events = threading.Event()
    accepting_events.set()

    def publish(event: str, payload: Dict[str, Any]) -> None:
        if not accepting_events.is_set():
            return
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, (event, payload))
        except RuntimeError:
            accepting_events.clear()

    def worker() -> None:
        started_at = time.monotonic()
        completed_rows: list[dict[str, Any]] = []
        pipelines: dict[str, MossASRPipeline] = {}
        try:
            publish(
                "queued",
                {
                    "manifest": manifest,
                    "dataset_language": dataset_language_filter or "auto",
                    "samples_total": len(samples),
                    "message": "Benchmark 已排入 GPU 佇列；將依序執行，避免影響長音訊轉錄。",
                },
            )
            with INFERENCE_SLOTS:
                for position, sample in enumerate(samples, start=1):
                    effective_language = selected_language or sample.language
                    language_hint = _benchmark_language_hint(effective_language)
                    cache_key = language_hint or "auto"
                    pipeline = pipelines.get(cache_key)
                    if pipeline is None:
                        pipeline = _build_pipeline(
                            vllm_url=clean_vllm_url,
                            model_id=clean_model_id,
                            include_timestamps=False,
                            include_speakers=False,
                            hotwords="",
                            proofread=False,
                            llm_url="",
                            llm_key="",
                            llm_model="",
                            reasoning_effort="none",
                            adaptive_proofread=False,
                            max_chunk_sec=clean_max_chunk,
                            language=language_hint,
                        )
                        pipelines[cache_key] = pipeline
                    publish(
                        "sample_started",
                        {
                            "position": position,
                            "samples_total": len(samples),
                            "id": sample.sample_id,
                            "language": effective_language,
                            "message": f"正在轉錄 {position}/{len(samples)}：{sample.sample_id}",
                        },
                    )
                    result = pipeline.process(sample.audio_path)
                    hypothesis = result.full_text
                    score = score_transcript(
                        sample.reference,
                        hypothesis,
                        language=effective_language,
                        requested_metric=metric,
                    )
                    row = {
                        "id": sample.sample_id,
                        "language": effective_language,
                        "reference": sample.reference,
                        "hypothesis": hypothesis,
                        "audio_duration": result.audio_duration,
                        "elapsed_time": result.elapsed_time,
                        "score": score,
                    }
                    completed_rows.append(row)
                    publish(
                        "sample_complete",
                        {
                            "position": position,
                            "samples_total": len(samples),
                            "row": row,
                            "summary": summarize_scores(completed_rows),
                        },
                    )
            summary = summarize_scores(completed_rows)
            publish(
                "complete",
                {
                    "manifest": manifest,
                    "metric_requested": metric,
                    "metric_default": selected_metric,
                    "language_override": selected_language or "auto",
                    "dataset_language": dataset_language_filter or "auto",
                    "elapsed_time": time.monotonic() - started_at,
                    "summary": summary,
                    "rows": completed_rows,
                },
            )
        except Exception as exc:
            publish("error", {"detail": str(exc), "completed": len(completed_rows)})
        finally:
            if accepting_events.is_set():
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

    threading.Thread(target=worker, name="moss-asr-benchmark", daemon=True).start()

    async def event_stream() -> AsyncIterator[str]:
        try:
            while True:
                item = await event_queue.get()
                if item is None:
                    break
                event, payload = item
                yield _sse_event(event, payload)
        finally:
            accepting_events.clear()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/transcribe/stream")
async def api_transcribe_stream(
    file: UploadFile = File(...),
    vllm_url: str = Form(DEFAULT_VLLM_URL),
    model_id: str = Form(MODEL_ID),
    include_timestamps: bool = Form(True),
    include_speakers: bool = Form(True),
    hotwords: Optional[str] = Form(""),
    proofread: bool = Form(False),
    llm_url: str = Form(DEFAULT_LITELLM_URL),
    llm_key: Optional[str] = Form(""),
    llm_model: str = Form(DEFAULT_LITELLM_MODEL),
    reasoning_effort: Optional[str] = Form("medium"),
    adaptive_proofread: bool = Form(True),
    max_chunk_sec: float = Form(1800.0),
) -> StreamingResponse:
    """Stream ASR deltas, stable transcription, proofreading deltas, and final output."""
    temp_file, upload_size = await _save_upload(file)
    try:
        pipeline = _build_pipeline(
            vllm_url=vllm_url,
            model_id=model_id,
            include_timestamps=include_timestamps,
            include_speakers=include_speakers,
            hotwords=hotwords,
            proofread=proofread,
            llm_url=llm_url,
            llm_key=llm_key,
            llm_model=llm_model,
            reasoning_effort=reasoning_effort,
            adaptive_proofread=adaptive_proofread,
            max_chunk_sec=max_chunk_sec,
        )
    except Exception as exc:
        _remove_temp_file(temp_file)
        raise _http_error(exc) from exc

    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[Optional[tuple[str, Dict[str, Any]]]] = asyncio.Queue()
    accepting_events = threading.Event()
    accepting_events.set()

    def publish(event: str, payload: Dict[str, Any]) -> None:
        if not accepting_events.is_set():
            return
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, (event, payload))
        except RuntimeError:
            accepting_events.clear()

    def worker() -> None:
        try:
            publish("queued", {"message": "工作已排入 GPU 佇列", "upload_bytes": upload_size})
            with INFERENCE_SLOTS:
                publish("status", {"message": "GPU 已就緒，開始處理", "progress": 0.02})
                result = pipeline.process_stream(temp_file, event_callback=publish)
            publish(
                "complete",
                _result_payload(
                    result,
                    include_timestamps=include_timestamps,
                    include_speakers=include_speakers,
                ),
            )
        except Exception as exc:
            publish("error", {"detail": str(exc)})
        finally:
            _remove_temp_file(temp_file)
            if accepting_events.is_set():
                loop.call_soon_threadsafe(event_queue.put_nowait, None)

    threading.Thread(target=worker, name="moss-asr-stream", daemon=True).start()

    async def event_stream() -> AsyncIterator[str]:
        try:
            while True:
                item = await event_queue.get()
                if item is None:
                    break
                event, payload = item
                yield _sse_event(event, payload)
        finally:
            accepting_events.clear()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("MOSS_API_PORT", "7860")))
