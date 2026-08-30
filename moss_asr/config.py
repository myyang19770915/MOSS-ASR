"""Configuration dataclasses for MOSS-Transcribe-Diarize pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict


# Cloudflare Browser Integrity Check rejects Python's default
# ``Python-urllib/x.y`` signature on some API routes (Error 1010).  Keep the
# value configurable for operators while using the browser-compatible header
# that was verified against the project's LiteLLM gateway.
DEFAULT_LITELLM_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36 MOSS-Transcribe-Diarize"
)


def default_litellm_user_agent() -> str:
    configured = os.getenv("LITELLM_USER_AGENT", "").strip()
    return configured or DEFAULT_LITELLM_USER_AGENT


@dataclass
class AudioChunkConfig:
    """Configuration for audio loading and chunking."""
    sampling_rate: int = 16000
    max_duration_seconds: float = 1800.0  # 30 minutes single-pass limit
    max_total_duration_seconds: float = 5400.0  # 90 minutes per uploaded job
    chunk_overlap_seconds: float = 2.0     # Overlap for sliding window if hard split
    silence_threshold_db: float = -30.0   # Silence detection threshold in dB
    min_silence_duration: float = 0.5     # Minimum silence duration to cut (seconds)
    supported_extensions: List[str] = field(
        default_factory=lambda: [
            ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
            ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv"
        ]
    )


@dataclass
class ASRConfig:
    """Configuration for ASR transcription and diarization."""
    model_id: str = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    vllm_base_url: str = field(
        default_factory=lambda: os.getenv("MOSS_VLLM_URL", "http://127.0.0.1:18000")
    )
    api_key: Optional[str] = None
    timeout: float = 600.0
    
    # Feature toggles
    include_timestamps: bool = True
    include_speakers: bool = True
    
    # Decoding parameters
    temperature: float = 0.0
    max_new_tokens: int = 4096
    language: Optional[str] = None  # e.g., 'zh', 'en'
    
    # Custom hotwords
    hotwords: List[str] = field(default_factory=list)
    custom_prompt: Optional[str] = None
    
    # Speaker name mapping (e.g., {'S01': 'Alice', 'S02': 'Bob'})
    speaker_names: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProofreadConfig:
    """Configuration for LLM and rule-based transcript proofreading."""
    enabled: bool = False
    llm_api_base: Optional[str] = None       # e.g. https://api.openai.com/v1 or http://localhost:8000/v1
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"           # or qwen2.5-7b, deepseek-chat, etc.
    temperature: float = 0.1
    request_timeout_seconds: float = 600.0
    http_user_agent: str = field(default_factory=default_litellm_user_agent)
    # Long transcripts need room for both reasoning tokens and the final JSON.
    max_completion_tokens: int = 16384
    # Avoid a reasoning-only first call for long documents unless the user
    # explicitly opts out and wants the selected effort preserved exactly.
    adaptive_reasoning: bool = True
    # Passed to OpenAI-compatible providers that support reasoning controls.
    # Common values: none, minimal, low, medium, high.
    reasoning_effort: Optional[str] = "medium"
    
    # Rule-based cleanups
    remove_stutter: bool = True              # Remove repeated filler words ("這..這個")
    normalize_punctuation: bool = True      # Fix trailing punctuation
    custom_replacements: Dict[str, str] = field(default_factory=dict)  # Custom glossary replacements
    glossary_terms: List[str] = field(default_factory=list)  # Trusted names/terms shared with the LLM
    
    # LLM proofreading settings
    preserve_timestamps: bool = True
    preserve_speakers: bool = True
    language_target: str = "auto"           # 'zh-TW', 'zh-CN', 'en', 'auto'


@dataclass
class PipelineConfig:
    """Master pipeline configuration."""
    audio: AudioChunkConfig = field(default_factory=AudioChunkConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    proofread: ProofreadConfig = field(default_factory=ProofreadConfig)
