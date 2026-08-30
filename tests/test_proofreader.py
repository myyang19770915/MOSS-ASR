"""Unit tests for RuleBasedProofreader and ProofreadingPipeline."""

import json
import urllib.error
from io import BytesIO

import pytest
from moss_asr.proofreader import LLMProofreader, RuleBasedProofreader, TranscriptProofreadingPipeline
from moss_asr.config import ProofreadConfig
from moss_asr.transcript_parser import TranscriptSegment, TranscriptResult


def test_rule_based_homophone_correction():
    reader = RuleBasedProofreader(
        custom_replacements={"快記": "會計", "計器": "機器"}
    )
    raw = "我們今天要在伺服器上佈署新的計器學習模型，請快記部門核算預算。"
    cleaned = reader.clean_text(raw)
    
    assert "部署" in cleaned
    assert "機器學習模型" in cleaned
    assert "會計部門" in cleaned


def test_rule_based_stutter_removal():
    reader = RuleBasedProofreader(remove_stutter=True)
    raw = "我、我、我們那個那個就是就是要做這個。"
    cleaned = reader.clean_text(raw)
    
    assert "我、我" not in cleaned
    assert "那個那個" not in cleaned
    assert "就是就是" not in cleaned


def test_rule_based_stutter_removal_preserves_english_double_letters():
    reader = RuleBasedProofreader(remove_stutter=True)
    cleaned = reader.clean_text("Hello hello, this speech is still clear.")

    assert cleaned == "Hello, this speech is still clear."


def test_rule_based_punctuation_normalization():
    reader = RuleBasedProofreader(normalize_punctuation=True)
    raw = "你好，，今天天氣真好！！我們一起去吃飯吧。。"
    cleaned = reader.clean_text(raw)
    
    assert "，，" not in cleaned
    assert "！！" not in cleaned
    assert "。。" not in cleaned


def test_pipeline_proofreading():
    config = ProofreadConfig(
        enabled=False,  # Test rule-based path without remote API call
        remove_stutter=True,
        custom_replacements={"佈署": "部署"},
    )
    pipeline = TranscriptProofreadingPipeline(config)
    
    segments = [
        TranscriptSegment(start=0.0, end=2.0, speaker="S01", text="我..我們準備佈署服務。。"),
    ]
    res = TranscriptResult(raw_text="raw", segments=segments, audio_duration=2.0)
    proofread_res = pipeline.process(res)
    
    assert proofread_res.is_proofread is True
    assert "部署" in proofread_res.segments[0].text
    assert "。。" not in proofread_res.segments[0].text


def test_requested_llm_without_key_is_not_reported_as_successful():
    pipeline = TranscriptProofreadingPipeline(
        ProofreadConfig(enabled=True, llm_api_key="", remove_stutter=True)
    )
    result = pipeline.process(
        TranscriptResult(
            raw_text="raw",
            segments=[TranscriptSegment(start=0, end=1, speaker="S01", text="準備佈署。")],
        )
    )
    assert result.is_proofread is False
    assert "未提供 API Key" in (result.proofread_notes or "")


def test_llm_proofreader_streams_json_and_keeps_segment_structure(monkeypatch):
    """Streaming deltas are forwarded while timestamps/speakers remain local."""
    sent_body = {}
    response_json = {
        "corrected_segments": [
            {"index": 0, "corrected_text": "我們準備部署服務。", "changes_made": "修正錯別字"},
            {"index": 1, "corrected_text": "第二段保持原文。", "changes_made": "無"},
        ],
        "summary": "已完成校對。",
    }
    streamed = json.dumps(response_json, ensure_ascii=False)

    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            midpoint = len(streamed) // 2
            for piece in (streamed[:midpoint], streamed[midpoint:]):
                yield f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]}, ensure_ascii=False)}\n".encode()
            yield b"data: [DONE]\n"

    def fake_urlopen(request, timeout):
        sent_body.update(json.loads(request.data.decode("utf-8")))
        sent_body["_user_agent"] = request.get_header("User-agent")
        sent_body["_accept"] = request.get_header("Accept")
        assert timeout == 600.0
        return FakeResponse()

    monkeypatch.setattr("moss_asr.proofreader.urllib.request.urlopen", fake_urlopen)
    reader = LLMProofreader(
        ProofreadConfig(
            enabled=True,
            llm_api_base="https://example.invalid/v1",
            llm_api_key="test-key",
            llm_model="deepseek-v4-flash",
            reasoning_effort="high",
            glossary_terms=["OpenMOSS", "vLLM"],
        )
    )
    deltas = []
    original = [
        TranscriptSegment(start=1.0, end=2.0, speaker="S01", text="我們準備佈署服務。"),
        TranscriptSegment(start=2.0, end=3.0, speaker="S02", text="第二段保持原文。"),
    ]

    corrected, summary = reader.proofread(original, stream_callback=deltas.append)

    assert sent_body["stream"] is True
    assert sent_body["max_completion_tokens"] == 16384
    assert sent_body["_user_agent"].startswith("Mozilla/5.0")
    assert sent_body["_accept"] == "application/json, text/event-stream"
    assert sent_body["reasoning_effort"] == "high"
    assert "不要只處理英文空格" in sent_body["messages"][1]["content"]
    assert "同音字、近音字" in sent_body["messages"][0]["content"]
    assert "OpenMOSS" in sent_body["messages"][1]["content"]
    assert "".join(deltas) == streamed
    assert "LLM 校對呼叫 1 次" in summary
    assert summary.endswith("已完成校對。")
    assert [(seg.start, seg.end, seg.speaker) for seg in corrected] == [
        (1.0, 2.0, "S01"),
        (2.0, 3.0, "S02"),
    ]
    assert [seg.text for seg in corrected] == ["我們準備部署服務。", "第二段保持原文。"]


def test_reasoning_only_stream_retries_with_lower_effort(monkeypatch):
    response_json = {
        "corrected_segments": [
            {"index": 0, "corrected_text": "我們應該再次部署。", "changes_made": "同音字"}
        ],
        "summary": "已完成校對。",
    }
    streamed = json.dumps(response_json, ensure_ascii=False)
    sent_bodies = []

    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(self.lines)

    def fake_urlopen(request, timeout):
        sent_bodies.append(json.loads(request.data.decode("utf-8")))
        assert timeout == 600.0
        if len(sent_bodies) == 1:
            return FakeResponse([
                b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"completion_tokens_details":{"reasoning_tokens":25}}}\n',
                b"data: [DONE]\n",
            ])
        return FakeResponse([
            f"data: {json.dumps({'choices': [{'delta': {'content': streamed}}]}, ensure_ascii=False)}\n".encode(),
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b"data: [DONE]\n",
        ])

    monkeypatch.setattr("moss_asr.proofreader.urllib.request.urlopen", fake_urlopen)
    reader = LLMProofreader(
        ProofreadConfig(
            enabled=True,
            llm_api_base="https://example.invalid/v1",
            llm_api_key="test-key",
            llm_model="deepseek-v4-flash",
            reasoning_effort="medium",
        )
    )
    deltas = []
    corrected, summary = reader.proofread(
        [TranscriptSegment(start=0, end=1, text="我們因該在次佈署。")],
        stream_callback=deltas.append,
    )

    assert [body["reasoning_effort"] for body in sent_bodies] == ["medium", "low"]
    assert "".join(deltas) == streamed
    assert corrected[0].text == "我們應該再次部署。"
    assert "自動改用 low" in summary


def test_long_document_adaptive_mode_starts_with_low_and_reports_progress(monkeypatch):
    response_json = {
        "corrected_segments": [
            {"index": i, "corrected_text": f"第 {i} 段", "changes_made": "無"}
            for i in range(100)
        ],
        "summary": "長文完成。",
    }
    streamed = json.dumps(response_json, ensure_ascii=False)
    sent_bodies = []

    class FakeResponse:
        headers = {"Content-Type": "text/event-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            yield f"data: {json.dumps({'choices': [{'delta': {'content': streamed}}]}, ensure_ascii=False)}\n".encode()
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n'
            yield b"data: [DONE]\n"

    def fake_urlopen(request, timeout):
        sent_bodies.append(json.loads(request.data.decode("utf-8")))
        assert timeout == 600.0
        return FakeResponse()

    monkeypatch.setattr("moss_asr.proofreader.urllib.request.urlopen", fake_urlopen)
    reader = LLMProofreader(
        ProofreadConfig(
            enabled=True,
            llm_api_base="https://example.invalid/v1",
            llm_api_key="test-key",
            llm_model="deepseek-v4-flash",
            reasoning_effort="medium",
            adaptive_reasoning=True,
        )
    )
    progress = []
    corrected, summary = reader.proofread(
        [TranscriptSegment(start=i, end=i + 1, text=f"第 {i} 段") for i in range(100)],
        stream_callback=lambda _delta: None,
        progress_callback=progress.append,
    )

    assert len(sent_bodies) == 1
    assert sent_bodies[0]["reasoning_effort"] == "low"
    assert len(corrected) == 100
    assert any(item["phase"] == "optimized" for item in progress)
    assert any(item["phase"] == "writing" for item in progress)
    assert "長文加速" in summary


def test_rule_preview_is_reported_before_llm_processing():
    pipeline = TranscriptProofreadingPipeline(
        ProofreadConfig(enabled=False, remove_stutter=True, custom_replacements={"佈署": "部署"})
    )
    progress = []
    pipeline.process(
        TranscriptResult(
            raw_text="raw",
            segments=[TranscriptSegment(start=0, end=1, text="準備佈署。")],
        ),
        progress_callback=progress.append,
    )
    assert progress[0]["phase"] == "rules_complete"
    assert progress[0]["segments"][0]["text"] == "準備部署。"


def test_llm_http_error_includes_provider_detail_without_key(monkeypatch):
    def denied(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://example.invalid/v1/chat/completions",
            403,
            "Forbidden",
            {},
            BytesIO(b'{"error":{"message":"model access denied for test-key"}}'),
        )

    monkeypatch.setattr("moss_asr.proofreader.urllib.request.urlopen", denied)
    reader = LLMProofreader(
        ProofreadConfig(
            enabled=True,
            llm_api_base="https://example.invalid/v1",
            llm_api_key="test-key",
            llm_model="restricted-model",
        )
    )
    with pytest.raises(RuntimeError) as caught:
        reader.proofread([TranscriptSegment(start=0, end=1, text="測試")])
    message = str(caught.value)
    assert "model access denied" in message
    assert "test-key" not in message
    assert "模型／團隊" in message
