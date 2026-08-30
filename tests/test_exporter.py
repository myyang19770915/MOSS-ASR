"""Unit tests for TranscriptExporter."""

import json
from pathlib import Path
import tempfile
import pytest
from moss_asr.exporter import TranscriptExporter
from moss_asr.transcript_parser import TranscriptParser, TranscriptResult, TranscriptSegment


@pytest.fixture
def sample_result():
    segments = [
        TranscriptSegment(start=1.25, end=4.50, speaker="S01", text="歡迎來到會議。"),
        TranscriptSegment(start=5.00, end=8.20, speaker="S02", text="謝謝主持人。"),
    ]
    return TranscriptResult(
        raw_text="raw",
        segments=segments,
        audio_duration=10.0,
        elapsed_time=1.5,
        model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
    )


def test_export_txt(sample_result):
    txt = TranscriptExporter.to_txt(sample_result, include_timestamps=True, include_speakers=True)
    assert "[00:01.25 -> 00:04.50]" in txt
    assert "[S01]" in txt
    assert "歡迎來到會議。" in txt


def test_export_srt(sample_result):
    srt = TranscriptExporter.to_srt(sample_result, include_speakers=True)
    assert "00:00:01,250 --> 00:00:04,500" in srt
    assert "[S01] 歡迎來到會議。" in srt
    assert "00:00:05,000 --> 00:00:08,200" in srt


def test_timestamp_only_model_output_exports_as_srt():
    raw = "[21.96]AI 是時下最夯的詞彙[24.26][24.26]很多人已經用 ChatGPT 寫文章[26.86]"
    segments = TranscriptParser.parse(raw)
    srt = TranscriptExporter.to_srt(
        TranscriptResult(raw_text=raw, segments=segments),
        include_speakers=False,
    )

    assert srt == (
        "1\n00:00:21,960 --> 00:00:24,260\nAI 是時下最夯的詞彙\n\n"
        "2\n00:00:24,260 --> 00:00:26,860\n很多人已經用 ChatGPT 寫文章\n"
    )


def test_export_vtt(sample_result):
    vtt = TranscriptExporter.to_vtt(sample_result, include_speakers=True)
    assert "WEBVTT" in vtt
    assert "00:00:01.250 --> 00:00:04.500" in vtt
    assert "<v S01>歡迎來到會議。" in vtt


def test_export_json(sample_result):
    json_str = TranscriptExporter.to_json(sample_result)
    data = json.loads(json_str)
    assert data["audio_duration"] == 10.0
    assert len(data["segments"]) == 2
    assert data["segments"][0]["speaker"] == "S01"


def test_export_csv(sample_result):
    csv_str = TranscriptExporter.to_csv(sample_result)
    assert "Start (s),End (s),Duration (s),Speaker,Text" in csv_str
    assert "1.250,4.500,3.250,S01,歡迎來到會議。" in csv_str


def test_save_all(sample_result):
    with tempfile.TemporaryDirectory() as tmpdir:
        saved = TranscriptExporter.save_all(sample_result, tmpdir, base_name="test_out")
        assert Path(saved["txt"]).exists()
        assert Path(saved["md"]).exists()
        assert Path(saved["srt"]).exists()
        assert Path(saved["vtt"]).exists()
        assert Path(saved["json"]).exists()
        assert Path(saved["csv"]).exists()
