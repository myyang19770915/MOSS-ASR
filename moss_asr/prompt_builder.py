"""Prompt builder for MOSS-Transcribe-Diarize based on feature flags."""

from __future__ import annotations

from typing import List, Optional
from .config import ASRConfig


class PromptBuilder:
    """Constructs prompt strings for MOSS-Transcribe-Diarize."""

    # Default canonical prompt: Timestamps + Speaker Diarization
    PROMPT_ZH_FULL = (
        "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
        "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
        "并在段末标注结束时间戳，以清晰标明该段语音范围。"
    )
    PROMPT_EN_FULL = (
        "Transcribe the audio. For each segment, start with the timestamp and speaker ID "
        "([S01], [S02], [S03], ...), then the spoken text, and end with the segment timestamp."
    )

    # Speaker-only (No timestamps)
    PROMPT_ZH_SPEAKER_ONLY = "转录为文本，使用 [S01] [S02] [S03]等说话人标签。"
    PROMPT_EN_SPEAKER_ONLY = (
        "Transcribe the audio as text using speaker labels such as [S01], [S02], and [S03]."
    )

    # Timestamp-only (Single speaker / no diarization)
    PROMPT_ZH_TIMESTAMP_ONLY = (
        "请将音频转写为文本，每一段需以起始时间戳开头，正文为对应的语音内容，并在段末标注结束时间戳。"
    )
    PROMPT_EN_TIMESTAMP_ONLY = (
        "Transcribe the audio with timestamps for each segment ([start_time]text[end_time])."
    )

    # Plain text (No timestamps, no diarization)
    PROMPT_ZH_PLAIN = "请将音频完整转写为純文字內容，保持語意流暢與正確標點符號。"
    PROMPT_EN_PLAIN = "Transcribe the spoken audio into continuous text accurately."

    @classmethod
    def build(
        cls,
        include_timestamps: bool = True,
        include_speakers: bool = True,
        hotwords: Optional[List[str]] = None,
        custom_prompt: Optional[str] = None,
        language: str = "zh",
    ) -> str:
        """Build prompt string dynamically based on configuration options.
        
        Args:
            include_timestamps: Whether to request timestamps in output.
            include_speakers: Whether to request [S01], [S02] speaker labels.
            hotwords: Optional list of domain hotwords / terminology.
            custom_prompt: Optional custom base prompt string override.
            language: Language preference for prompt ('zh' or 'en').
        """
        if custom_prompt and custom_prompt.strip():
            base_prompt = custom_prompt.strip()
        else:
            is_en = language.lower().startswith("en")
            if include_timestamps and include_speakers:
                base_prompt = cls.PROMPT_EN_FULL if is_en else cls.PROMPT_ZH_FULL
            elif include_speakers and not include_timestamps:
                base_prompt = cls.PROMPT_EN_SPEAKER_ONLY if is_en else cls.PROMPT_ZH_SPEAKER_ONLY
            elif include_timestamps and not include_speakers:
                base_prompt = cls.PROMPT_EN_TIMESTAMP_ONLY if is_en else cls.PROMPT_ZH_TIMESTAMP_ONLY
            else:
                base_prompt = cls.PROMPT_EN_PLAIN if is_en else cls.PROMPT_ZH_PLAIN

        # Attach hotwords if present
        if hotwords:
            clean_hotwords = [hw.strip() for hw in hotwords if hw.strip()]
            if clean_hotwords:
                hw_str = ", ".join(clean_hotwords)
                if language.lower().startswith("en"):
                    base_prompt += f" Hotwords: {hw_str}"
                else:
                    base_prompt += f" 热词提示：{hw_str}"

        return base_prompt

    @classmethod
    def from_config(cls, config: ASRConfig) -> str:
        """Build prompt from ASRConfig object."""
        return cls.build(
            include_timestamps=config.include_timestamps,
            include_speakers=config.include_speakers,
            hotwords=config.hotwords,
            custom_prompt=config.custom_prompt,
            language=config.language or "zh",
        )
