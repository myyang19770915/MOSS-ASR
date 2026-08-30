"""Robust parser for MOSS-Transcribe-Diarize transcript outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Iterator, Iterable, Dict, Any


@dataclass
class TranscriptSegment:
    """A structured transcript segment with time range, speaker, and text."""
    start: float = 0.0
    end: float = 0.0
    speaker: str = "S01"
    text: str = ""
    confidence: Optional[float] = None
    
    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)
    
    def format_time(self, include_hours: bool = False) -> str:
        """Format timestamp as MM:SS.cc or HH:MM:SS.cc"""
        def _sec_to_str(s: float) -> str:
            m, s = divmod(s, 60)
            h, m = divmod(m, 60)
            if include_hours or h > 0:
                return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"
            return f"{int(m):02d}:{s:05.2f}"
        return f"[{_sec_to_str(self.start)} -> {_sec_to_str(self.end)}]"

    def to_display_text(
        self,
        include_timestamps: bool = True,
        include_speakers: bool = True,
        speaker_names: Optional[Dict[str, str]] = None,
    ) -> str:
        """Render segment text based on display options."""
        parts = []
        if include_timestamps:
            parts.append(self.format_time())
        if include_speakers and self.speaker:
            name = (speaker_names or {}).get(self.speaker, self.speaker)
            parts.append(f"[{name}]")
        parts.append(self.text)
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "speaker": self.speaker,
            "text": self.text,
        }


@dataclass
class TranscriptResult:
    """Container for complete transcription result."""
    raw_text: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    audio_duration: float = 0.0
    elapsed_time: float = 0.0
    model: str = ""
    is_proofread: bool = False
    proofread_notes: Optional[str] = None
    
    @property
    def full_text(self) -> str:
        """Combined clean spoken text across all segments."""
        return " ".join(seg.text for seg in self.segments if seg.text.strip())

    def get_formatted_text(
        self,
        include_timestamps: bool = True,
        include_speakers: bool = True,
        speaker_names: Optional[Dict[str, str]] = None,
    ) -> str:
        """Render multi-line formatted transcript."""
        lines = []
        for seg in self.segments:
            if not seg.text.strip():
                continue
            lines.append(
                seg.to_display_text(
                    include_timestamps=include_timestamps,
                    include_speakers=include_speakers,
                    speaker_names=speaker_names,
                )
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "full_text": self.full_text,
            "audio_duration": round(self.audio_duration, 2),
            "elapsed_time": round(self.elapsed_time, 2),
            "model": self.model,
            "is_proofread": self.is_proofread,
            "proofread_notes": self.proofread_notes,
            "segments": [seg.to_dict() for seg in self.segments],
        }


class TranscriptParser:
    """Parser for MOSS canonical output format: [start][Sxx]text[end]"""

    # Regex for fast parsing of standard format
    # Example: [0.48][S01]Hello world[1.66]
    CANONICAL_PATTERN = re.compile(
        r"\[(\d+(?:\.\d+)?)\]\s*\[(S\d+)\]\s*(.*?)\s*\[(\d+(?:\.\d+)?)\]",
        re.DOTALL,
    )
    
    # Fallback pattern for speaker-only format: [S01] text
    SPEAKER_ONLY_PATTERN = re.compile(r"\[(S\d+)\]\s*([^\[]+)")

    # Timestamp-only output emitted when speaker diarization is not requested.
    # Example: [21.96]第一句字幕[24.26][24.26]第二句字幕[26.86]
    TIMESTAMP_TOKEN_PATTERN = re.compile(r"\[(\d+(?:\.\d+)?)\]")

    @classmethod
    def parse(cls, raw_text: str, time_offset: float = 0.0) -> List[TranscriptSegment]:
        """Parse raw MOSS output string into TranscriptSegment list.
        
        Args:
            raw_text: Raw output string from model.
            time_offset: Timestamp offset (used when stitching chunks).
        """
        if not raw_text or not raw_text.strip():
            return []

        cleaned_text = raw_text.strip()
        segments: List[TranscriptSegment] = []

        # 1. Try canonical parsing [start][Sxx]text[end]
        matches = list(cls.CANONICAL_PATTERN.finditer(cleaned_text))
        if matches:
            for match in matches:
                start_sec = float(match.group(1)) + time_offset
                speaker = match.group(2)
                text = match.group(3).strip()
                end_sec = float(match.group(4)) + time_offset
                
                # Sanity check for timestamps
                if end_sec < start_sec:
                    end_sec = start_sec + 0.1
                    
                if text:
                    segments.append(
                        TranscriptSegment(
                            start=round(start_sec, 3),
                            end=round(end_sec, 3),
                            speaker=speaker,
                            text=text,
                        )
                    )
            if segments:
                return segments

        # 2. Timestamp-only stream: [start]text[end][next_start]text[next_end].
        # The model often repeats a boundary timestamp, so empty spans are
        # deliberately ignored.  Each non-empty span becomes one subtitle cue.
        timestamp_only_segments = cls._parse_timestamp_only_stream(cleaned_text, time_offset)
        if timestamp_only_segments:
            return timestamp_only_segments

        # 3. Fallback: Character state machine parser (handles non-standard tokens)
        state_segments = cls._parse_state_machine(cleaned_text, time_offset)
        if state_segments:
            return state_segments

        # 4. Fallback: Speaker only pattern [S01] text
        speaker_matches = list(cls.SPEAKER_ONLY_PATTERN.finditer(cleaned_text))
        if speaker_matches:
            cur_time = time_offset
            for match in speaker_matches:
                speaker = match.group(1)
                text = match.group(2).strip()
                if text:
                    # Estimate ~0.2s per character as dummy duration if timestamps absent
                    dur = max(1.0, len(text) * 0.2)
                    segments.append(
                        TranscriptSegment(
                            start=round(cur_time, 2),
                            end=round(cur_time + dur, 2),
                            speaker=speaker,
                            text=text,
                        )
                    )
                    cur_time += dur
            if segments:
                return segments

        # 5. Final Fallback: Plain raw text
        return [
            TranscriptSegment(
                start=round(time_offset, 2),
                end=round(time_offset + max(1.0, len(cleaned_text) * 0.2), 2),
                speaker="S01",
                text=cleaned_text,
            )
        ]

    @classmethod
    def _parse_timestamp_only_stream(
        cls,
        text: str,
        time_offset: float = 0.0,
    ) -> List[TranscriptSegment]:
        """Parse timestamp-only MOSS output into subtitle-ready segments.

        A timestamp is the start of the following text and the next timestamp
        closes that cue.  Consecutive duplicate boundary tokens are normal in
        MOSS output and produce an empty span, which is skipped.
        """
        matches = list(cls.TIMESTAMP_TOKEN_PATTERN.finditer(text))
        if len(matches) < 2:
            return []

        segments: List[TranscriptSegment] = []
        for current, following in zip(matches, matches[1:]):
            cue_text = text[current.end():following.start()].strip()
            if not cue_text:
                continue

            start = float(current.group(1))
            end = float(following.group(1))
            segments.append(
                TranscriptSegment(
                    start=round(start + time_offset, 3),
                    end=round(max(end, start) + time_offset, 3),
                    speaker="S01",
                    text=cue_text,
                )
            )

        # Preserve a final cue if the model omitted its closing timestamp.
        final_marker = matches[-1]
        trailing_text = text[final_marker.end():].strip()
        if trailing_text:
            start = float(final_marker.group(1))
            estimated_end = start + max(0.1, len(trailing_text) * 0.2)
            segments.append(
                TranscriptSegment(
                    start=round(start + time_offset, 3),
                    end=round(estimated_end + time_offset, 3),
                    speaker="S01",
                    text=trailing_text,
                )
            )

        return segments

    @staticmethod
    def _parse_state_machine(text: str, time_offset: float = 0.0) -> List[TranscriptSegment]:
        """Robust state machine scanner to parse token streams."""
        segments: List[TranscriptSegment] = []
        tokens = re.split(r"(\[[^\]]+\])", text)
        
        start: Optional[float] = None
        speaker: Optional[str] = None
        text_buf: List[str] = []
        
        for token in tokens:
            if not token:
                continue
            if token.startswith("[") and token.endswith("]"):
                inner = token[1:-1].strip()
                # Check if timestamp (e.g. 0.48 or 12.3)
                try:
                    ts = float(inner)
                    if start is None:
                        start = ts
                    else:
                        # End timestamp found
                        end = ts
                        t = "".join(text_buf).strip()
                        if t and speaker:
                            segments.append(
                                TranscriptSegment(
                                    start=round(start + time_offset, 3),
                                    end=round(max(end, start) + time_offset, 3),
                                    speaker=speaker,
                                    text=t,
                                )
                            )
                        # Reset for next segment
                        start = end
                        speaker = None
                        text_buf.clear()
                    continue
                except ValueError:
                    pass
                
                # Check if speaker (e.g. S01, S02)
                if inner.startswith("S") and inner[1:].isdigit():
                    speaker = inner
                    continue
            
            # Regular text content
            text_buf.append(token)

        # Flush any trailing segment
        if start is not None and speaker is not None and text_buf:
            t = "".join(text_buf).strip()
            if t:
                segments.append(
                    TranscriptSegment(
                        start=round(start + time_offset, 3),
                        end=round(start + time_offset + max(1.0, len(t) * 0.2), 3),
                        speaker=speaker,
                        text=t,
                    )
                )

        return segments
