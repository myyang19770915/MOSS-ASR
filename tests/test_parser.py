"""Unit tests for TranscriptParser and TranscriptSegment."""

import pytest
from moss_asr.transcript_parser import TranscriptParser, TranscriptSegment, TranscriptResult


def test_parse_canonical_format():
    raw = "[0.48][S01]歡迎大家參加今天的會議。[2.35][3.10][S02]好的，我已經準備好簡報了。[6.50]"
    segments = TranscriptParser.parse(raw)
    
    assert len(segments) == 2
    assert segments[0].start == 0.48
    assert segments[0].end == 2.35
    assert segments[0].speaker == "S01"
    assert segments[0].text == "歡迎大家參加今天的會議。"

    assert segments[1].start == 3.10
    assert segments[1].end == 6.50
    assert segments[1].speaker == "S02"
    assert segments[1].text == "好的，我已經準備好簡報了。"


def test_parse_with_time_offset():
    raw = "[1.00][S01]第一分段測試。[3.50]"
    segments = TranscriptParser.parse(raw, time_offset=100.0)
    
    assert len(segments) == 1
    assert segments[0].start == 101.0
    assert segments[0].end == 103.5
    assert segments[0].speaker == "S01"
    assert segments[0].text == "第一分段測試。"


def test_parse_speaker_only_fallback():
    raw = "[S01] 這是純說話人標籤的文字 [S02] 這是第二個人的回覆"
    segments = TranscriptParser.parse(raw)
    
    assert len(segments) == 2
    assert segments[0].speaker == "S01"
    assert "這是純說話人標籤的文字" in segments[0].text
    assert segments[1].speaker == "S02"
    assert "這是第二個人的回覆" in segments[1].text


def test_parse_timestamp_only_stream_with_duplicate_boundaries():
    raw = (
        "[21.96]AI 是時下最夯的詞彙[24.26]"
        "[24.26]很多人已經用 ChatGPT 寫文章[26.86]"
        "[26.86]甚至設計動態人像[28.86]"
    )
    segments = TranscriptParser.parse(raw)

    assert len(segments) == 3
    assert (segments[0].start, segments[0].end) == (21.96, 24.26)
    assert segments[0].text == "AI 是時下最夯的詞彙"
    assert (segments[1].start, segments[1].end) == (24.26, 26.86)
    assert segments[1].text == "很多人已經用 ChatGPT 寫文章"
    assert (segments[2].start, segments[2].end) == (26.86, 28.86)


def test_segment_to_display_text():
    seg = TranscriptSegment(start=1.5, end=4.2, speaker="S01", text="測試內容")
    
    # Both timestamps and speakers
    disp1 = seg.to_display_text(include_timestamps=True, include_speakers=True)
    assert "[00:01.50 -> 00:04.20]" in disp1
    assert "[S01]" in disp1
    assert "測試內容" in disp1
    
    # Speaker name mapping
    disp2 = seg.to_display_text(include_timestamps=True, include_speakers=True, speaker_names={"S01": "王大明"})
    assert "[王大明]" in disp2
    
    # Without timestamps
    disp3 = seg.to_display_text(include_timestamps=False, include_speakers=True)
    assert "->" not in disp3
    assert "[S01] 測試內容" == disp3
    
    # Without speakers
    disp4 = seg.to_display_text(include_timestamps=False, include_speakers=False)
    assert disp4 == "測試內容"


def test_transcript_result():
    segments = [
        TranscriptSegment(start=0.0, end=2.0, speaker="S01", text="語音轉錄測試。"),
        TranscriptSegment(start=2.5, end=5.0, speaker="S02", text="系統運作正常。"),
    ]
    res = TranscriptResult(raw_text="raw", segments=segments, audio_duration=5.0)
    
    assert res.full_text == "語音轉錄測試。 系統運作正常。"
    assert len(res.to_dict()["segments"]) == 2
