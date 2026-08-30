"""Export transcript results to multiple formats (TXT, MD, SRT, VTT, JSON, CSV)."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Optional, Dict, List

from .transcript_parser import TranscriptResult, TranscriptSegment


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm"""
    millis = max(0, round(float(seconds) * 1000))
    hours, rem = divmod(millis, 3_600_000)
    mins, rem = divmod(rem, 60_000)
    secs, msec = divmod(rem, 1000)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{msec:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds into WebVTT timestamp HH:MM:SS.mmm"""
    millis = max(0, round(float(seconds) * 1000))
    hours, rem = divmod(millis, 3_600_000)
    mins, rem = divmod(rem, 60_000)
    secs, msec = divmod(rem, 1000)
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{msec:03d}"


class TranscriptExporter:
    """Exports TranscriptResult into various standard formats."""

    @staticmethod
    def to_txt(
        result: TranscriptResult,
        include_timestamps: bool = True,
        include_speakers: bool = True,
        speaker_names: Optional[Dict[str, str]] = None,
    ) -> str:
        """Export as plain text lines."""
        return result.get_formatted_text(
            include_timestamps=include_timestamps,
            include_speakers=include_speakers,
            speaker_names=speaker_names,
        )

    @staticmethod
    def to_markdown(
        result: TranscriptResult,
        title: str = "ASR 語音轉錄結果",
        include_timestamps: bool = True,
        include_speakers: bool = True,
        speaker_names: Optional[Dict[str, str]] = None,
    ) -> str:
        """Export as structured Markdown document."""
        lines = [
            f"# {title}",
            "",
            f"- **音訊總時長**: `{result.audio_duration:.2f}` 秒",
            f"- **處理耗時**: `{result.elapsed_time:.2f}` 秒",
            f"- **使用模型**: `{result.model or 'MOSS-Transcribe-Diarize'}`",
            f"- **校對狀態**: `{'已完成校對' if result.is_proofread else '未校對'}`",
        ]
        if result.proofread_notes:
            lines.append(f"- **校對註記**: {result.proofread_notes}")

        lines.extend([
            "",
            "## 完整轉錄文字",
            "",
            f"> {result.full_text}",
            "",
            "## 時間序逐字稿",
            "",
        ])

        for seg in result.segments:
            if not seg.text.strip():
                continue
            t_str = f"`{seg.format_time()}` " if include_timestamps else ""
            spk_name = (speaker_names or {}).get(seg.speaker, seg.speaker) if include_speakers else ""
            spk_str = f"**[{spk_name}]**: " if spk_name else ""
            lines.append(f"- {t_str}{spk_str}{seg.text}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def to_srt(
        result: TranscriptResult,
        include_speakers: bool = True,
        speaker_names: Optional[Dict[str, str]] = None,
    ) -> str:
        """Export as standard SRT subtitle format."""
        blocks = []
        for idx, seg in enumerate(result.segments, start=1):
            if not seg.text.strip():
                continue
            start_str = format_srt_timestamp(seg.start)
            end_str = format_srt_timestamp(seg.end)
            spk_name = (speaker_names or {}).get(seg.speaker, seg.speaker) if include_speakers and seg.speaker else ""
            text_str = f"[{spk_name}] {seg.text}" if spk_name else seg.text

            blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text_str}")

        return "\n\n".join(blocks) + "\n"

    @staticmethod
    def to_vtt(
        result: TranscriptResult,
        include_speakers: bool = True,
        speaker_names: Optional[Dict[str, str]] = None,
    ) -> str:
        """Export as WebVTT subtitle format."""
        blocks = ["WEBVTT\n"]
        for idx, seg in enumerate(result.segments, start=1):
            if not seg.text.strip():
                continue
            start_str = format_vtt_timestamp(seg.start)
            end_str = format_vtt_timestamp(seg.end)
            spk_name = (speaker_names or {}).get(seg.speaker, seg.speaker) if include_speakers and seg.speaker else ""
            text_str = f"<v {spk_name}>{seg.text}" if spk_name else seg.text

            blocks.append(f"{idx}\n{start_str} --> {end_str}\n{text_str}")

        return "\n\n".join(blocks) + "\n"

    @staticmethod
    def to_json(result: TranscriptResult, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=indent)

    @staticmethod
    def to_csv(result: TranscriptResult) -> str:
        """Export as CSV spreadsheet format."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Index", "Start (s)", "End (s)", "Duration (s)", "Speaker", "Text"])
        for idx, seg in enumerate(result.segments, start=1):
            writer.writerow([
                idx,
                f"{seg.start:.3f}",
                f"{seg.end:.3f}",
                f"{seg.duration:.3f}",
                seg.speaker,
                seg.text,
            ])
        return output.getvalue()

    @classmethod
    def save_all(
        cls,
        result: TranscriptResult,
        output_dir: str | Path,
        base_name: str = "transcript",
        include_timestamps: bool = True,
        include_speakers: bool = True,
        speaker_names: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Path]:
        """Save all format files to destination directory."""
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: Dict[str, Path] = {}

        # 1. Plain text
        txt_path = out_dir / f"{base_name}.txt"
        txt_path.write_text(
            cls.to_txt(result, include_timestamps, include_speakers, speaker_names),
            encoding="utf-8",
        )
        saved["txt"] = txt_path

        # 2. Markdown
        md_path = out_dir / f"{base_name}.md"
        md_path.write_text(
            cls.to_markdown(result, title=f"轉錄逐字稿 - {base_name}", include_timestamps=include_timestamps, include_speakers=include_speakers, speaker_names=speaker_names),
            encoding="utf-8",
        )
        saved["md"] = md_path

        # 3. SRT
        srt_path = out_dir / f"{base_name}.srt"
        srt_path.write_text(
            cls.to_srt(result, include_speakers, speaker_names),
            encoding="utf-8",
        )
        saved["srt"] = srt_path

        # 4. WebVTT
        vtt_path = out_dir / f"{base_name}.vtt"
        vtt_path.write_text(
            cls.to_vtt(result, include_speakers, speaker_names),
            encoding="utf-8",
        )
        saved["vtt"] = vtt_path

        # 5. JSON
        json_path = out_dir / f"{base_name}.json"
        json_path.write_text(
            cls.to_json(result),
            encoding="utf-8",
        )
        saved["json"] = json_path

        # 6. CSV
        csv_path = out_dir / f"{base_name}.csv"
        csv_path.write_text(
            cls.to_csv(result),
            encoding="utf-8",
        )
        saved["csv"] = csv_path

        return saved
