"""Unit tests for PromptBuilder."""

import pytest
from moss_asr.prompt_builder import PromptBuilder
from moss_asr.config import ASRConfig


def test_prompt_builder_default():
    prompt = PromptBuilder.build(include_timestamps=True, include_speakers=True, language="zh")
    assert "起始时间戳和说话人编号" in prompt
    assert "[S01]" in prompt


def test_prompt_builder_speaker_only():
    prompt = PromptBuilder.build(include_timestamps=False, include_speakers=True, language="zh")
    assert "转录为文本" in prompt
    assert "[S01]" in prompt
    assert "起始时间戳" not in prompt


def test_prompt_builder_timestamp_only():
    prompt = PromptBuilder.build(include_timestamps=True, include_speakers=False, language="zh")
    assert "起始时间戳" in prompt
    assert "说话人编号" not in prompt


def test_prompt_builder_plain():
    prompt = PromptBuilder.build(include_timestamps=False, include_speakers=False, language="zh")
    assert "純文字內容" in prompt


def test_prompt_builder_with_hotwords():
    prompt = PromptBuilder.build(
        include_timestamps=True,
        include_speakers=True,
        hotwords=["OpenMOSS", "vLLM", "Diarization"],
        language="zh",
    )
    assert "热词提示：OpenMOSS, vLLM, Diarization" in prompt


def test_prompt_builder_from_config():
    config = ASRConfig(
        include_timestamps=True,
        include_speakers=False,
        hotwords=["PyTorch"],
        language="zh",
    )
    prompt = PromptBuilder.from_config(config)
    assert "起始时间戳" in prompt
    assert "热词提示：PyTorch" in prompt
