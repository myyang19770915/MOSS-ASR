"""Validation and upload-boundary tests for the FastAPI service."""

import asyncio
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

import app_api


def test_validate_http_url_and_reasoning_options():
    assert app_api._validate_http_url("http://vllm:8000/", "vLLM URL") == "http://vllm:8000"
    with pytest.raises(ValueError, match="有效"):
        app_api._validate_http_url("file:///etc/passwd", "vLLM URL")
    with pytest.raises(ValueError, match="reasoning_effort"):
        app_api._validate_request_options(
            vllm_url="http://vllm:8000",
            model_id=app_api.MODEL_ID,
            proofread=True,
            llm_url="https://example.com/v1",
            llm_model="test-model",
            reasoning_effort="extreme",
            max_chunk_sec=1800,
        )


def test_parse_hotwords_deduplicates_and_preserves_order():
    assert app_api._parse_hotwords("OpenMOSS, vLLM, OpenMOSS, 人名") == ["OpenMOSS", "vLLM", "人名"]


def test_build_pipeline_disables_rule_rewrites_when_proofreading_is_off():
    pipeline = app_api._build_pipeline(
        vllm_url="http://vllm:8000",
        model_id=app_api.MODEL_ID,
        include_timestamps=True,
        include_speakers=True,
        hotwords="",
        proofread=False,
        llm_url="",
        llm_key="",
        llm_model="",
        reasoning_effort="medium",
        adaptive_proofread=True,
        max_chunk_sec=1800,
    )
    assert pipeline.config.proofread.enabled is False
    assert pipeline.config.proofread.remove_stutter is False
    assert pipeline.config.proofread.normalize_punctuation is False


def test_build_pipeline_passes_an_optional_language_hint():
    pipeline = app_api._build_pipeline(
        vllm_url="http://vllm:8000",
        model_id=app_api.MODEL_ID,
        include_timestamps=False,
        include_speakers=False,
        hotwords="",
        proofread=False,
        llm_url="",
        llm_key="",
        llm_model="",
        reasoning_effort="none",
        adaptive_proofread=False,
        max_chunk_sec=1800,
        language="zh-TW",
    )
    assert pipeline.config.asr.language == "zh-TW"


def test_save_upload_rejects_unsupported_extension():
    upload = UploadFile(filename="notes.txt", file=BytesIO(b"not audio"))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(app_api._save_upload(upload))
    assert caught.value.status_code == 415


def test_save_upload_enforces_api_size_limit(monkeypatch):
    monkeypatch.setattr(app_api, "MAX_UPLOAD_BYTES", 3)
    upload = UploadFile(filename="sample.wav", file=BytesIO(b"1234"))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(app_api._save_upload(upload))
    assert caught.value.status_code == 413


def test_litellm_connection_reports_model_visibility(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "deepseek-v4-flash"}]}).encode()

    def fake_urlopen(request, **_kwargs):
        captured["request"] = request
        return Response()

    monkeypatch.setattr(app_api.urllib.request, "urlopen", fake_urlopen)
    result = app_api._check_litellm_connection(
        "https://example.invalid/v1", "secret-key", "deepseek-v4-flash"
    )
    assert result["authenticated"] is True
    assert result["model_available"] is True
    assert captured["request"].get_header("User-agent").startswith("Mozilla/5.0")
    assert captured["request"].get_header("Accept") == "application/json"


def test_empty_litellm_user_agent_env_keeps_safe_default(monkeypatch):
    monkeypatch.setenv("LITELLM_USER_AGENT", "")
    assert app_api.default_litellm_user_agent().startswith("Mozilla/5.0")


def test_benchmark_run_streams_a_scored_sample(monkeypatch, tmp_path):
    data_root = tmp_path / "benchmarks"
    dataset = data_root / "smoke"
    audio_dir = dataset / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "one.wav").write_bytes(b"placeholder")
    (dataset / "test.jsonl").write_text(
        '{"id":"one","audio":"audio/one.wav","text":"hello world","language":"en"}\n',
        encoding="utf-8",
    )

    class FakePipeline:
        def process(self, _audio_path):
            return SimpleNamespace(full_text="hello world", audio_duration=1.0, elapsed_time=0.01)

    monkeypatch.setattr(app_api, "BENCHMARK_DATA_ROOT", data_root)
    monkeypatch.setattr(app_api, "_build_pipeline", lambda **_kwargs: FakePipeline())

    async def collect_events():
        response = await app_api.api_benchmark_run(
            manifest="smoke/test.jsonl",
            vllm_url="http://vllm:8000",
            model_id=app_api.MODEL_ID,
            language_override="auto",
            metric="auto",
            max_samples=1,
            max_chunk_sec=1800,
        )
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    events = asyncio.run(collect_events())
    assert "event: sample_complete" in events
    assert "event: complete" in events
    assert '"accuracy_percent": 100.0' in events


def test_benchmark_language_hints_normalize_dataset_locales():
    assert app_api._benchmark_language_hint("en-US") == "en"
    assert app_api._benchmark_language_hint("zh-CN") == "zh"
    assert app_api._benchmark_language_hint("cmn_Hans_CN") == "zh"
    assert app_api._benchmark_language_hint("yue-Hant-HK") is None


def test_benchmark_preview_and_audio_only_resolve_manifest_samples(monkeypatch, tmp_path):
    root = tmp_path / "benchmarks"
    dataset = root / "preview"
    audio_dir = dataset / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "one.wav").write_bytes(b"placeholder")
    (dataset / "test.jsonl").write_text(
        '{"id":"one","audio":"audio/one.wav","text":"hello world","language":"en-US"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(app_api, "BENCHMARK_DATA_ROOT", root)
    preview = asyncio.run(app_api.api_benchmark_preview("preview/test.jsonl", 8))
    assert preview["samples"] == 1
    assert preview["preview"][0]["audio_url"].endswith("sample_id=one")
    response = asyncio.run(app_api.api_benchmark_audio("preview/test.jsonl", "one"))
    assert str(response.path).endswith("one.wav")
