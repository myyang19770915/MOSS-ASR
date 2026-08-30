"""FastAPI service for MOSS transcription, diarization, and LLM proofreading."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import urllib.error
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
from moss_asr.vllm_client import VLLMClient


APP_VERSION = "1.2.5"
MODEL_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
TEMP_UPLOAD_DIR = Path(tempfile.gettempdir()) / "moss_asr_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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
    config = PipelineConfig(
        audio=AudioChunkConfig(max_duration_seconds=clean_max_chunk),
        asr=ASRConfig(
            model_id=clean_model_id,
            vllm_base_url=clean_vllm_url,
            include_timestamps=include_timestamps,
            include_speakers=include_speakers,
            hotwords=hotwords_list,
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
