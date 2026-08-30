"""Validation and upload-boundary tests for the FastAPI service."""

import asyncio
import json
from io import BytesIO

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
