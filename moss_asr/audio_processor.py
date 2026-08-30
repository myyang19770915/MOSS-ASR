"""Audio processing, format conversion, and silence-aware auto-chunking using FFmpeg."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from .config import AudioChunkConfig


@dataclass
class AudioMetadata:
    """Audio metadata extracted via ffprobe."""
    path: Path
    duration: float
    sampling_rate: int
    channels: int
    format_name: str
    is_video: bool


@dataclass
class AudioChunk:
    """An audio segment ready for model inference."""
    index: int
    path: Path
    start_offset: float
    duration: float
    is_temp: bool = True
    cleanup_dir: Optional[Path] = None

    def cleanup(self) -> None:
        """Remove temporary file if applicable."""
        if self.is_temp and self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass
        if self.cleanup_dir:
            try:
                self.cleanup_dir.rmdir()
            except OSError:
                pass


class AudioProcessor:
    """Handles audio inspection, format normalization to 16kHz mono WAV, and auto-chunking."""

    def __init__(self, config: Optional[AudioChunkConfig] = None):
        self.config = config or AudioChunkConfig()
        self._check_ffmpeg()

    @staticmethod
    def _check_ffmpeg() -> None:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("`ffmpeg` is not installed or not found on PATH.")
        if not shutil.which("ffprobe"):
            raise RuntimeError("`ffprobe` is not installed or not found on PATH.")

    def probe(self, media_path: str | Path) -> AudioMetadata:
        """Inspect media file properties using ffprobe."""
        path = Path(media_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        cmd = [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "ffprobe 無法解析媒體檔案").strip()
            raise ValueError(f"無法讀取媒體資訊：{detail[-800:]}") from exc
        data = json.loads(res.stdout or "{}")

        streams = data.get("streams", [])
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)

        if not audio_stream:
            raise ValueError(f"No audio stream found in file: {path}")

        format_info = data.get("format", {})
        duration = float(format_info.get("duration") or audio_stream.get("duration") or 0.0)
        if duration <= 0:
            raise ValueError("媒體檔案沒有有效的音訊時長")
        sampling_rate = int(audio_stream.get("sample_rate") or 16000)
        channels = int(audio_stream.get("channels") or 1)
        format_name = format_info.get("format_name", "")

        return AudioMetadata(
            path=path,
            duration=duration,
            sampling_rate=sampling_rate,
            channels=channels,
            format_name=format_name,
            is_video=video_stream is not None,
        )

    def convert_to_wav(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Path:
        """Convert any audio/video container (mp3, m4a, mp4, etc.) to 16kHz mono WAV."""
        input_p = Path(input_path).expanduser().resolve()
        if output_path is None:
            fd, temp_wav = tempfile.mkstemp(suffix=".wav", prefix="moss_audio_")
            os.close(fd)
            out_p = Path(temp_wav)
        else:
            out_p = Path(output_path).expanduser().resolve()
            out_p.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["ffmpeg", "-y", "-v", "error"]
        if start_time is not None and start_time > 0:
            cmd.extend(["-ss", f"{start_time:.3f}"])
        cmd.extend(["-i", str(input_p)])
        if duration is not None and duration > 0:
            cmd.extend(["-t", f"{duration:.3f}"])

        cmd.extend([
            "-vn",                      # Disable video recording
            "-acodec", "pcm_s16le",      # Standard 16-bit PCM
            "-ar", str(self.config.sampling_rate),  # 16000 Hz
            "-ac", "1",                  # Mono channel
            str(out_p),
        ])

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            if out_p.exists():
                out_p.unlink(missing_ok=True)
            detail = (exc.stderr or "ffmpeg 轉檔失敗").strip()
            raise ValueError(f"音訊轉檔失敗：{detail[-800:]}") from exc
        return out_p

    def detect_silence_points(
        self,
        audio_path: str | Path,
        silence_threshold_db: Optional[float] = None,
        min_silence_duration: Optional[float] = None,
    ) -> List[float]:
        """Detect silence midpoints in audio using ffmpeg silencedetect."""
        thresh = self.config.silence_threshold_db if silence_threshold_db is None else silence_threshold_db
        min_dur = self.config.min_silence_duration if min_silence_duration is None else min_silence_duration

        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-af", f"silencedetect=noise={thresh}dB:d={min_dur}",
            "-f", "null",
            "-",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        stderr = res.stderr

        silence_starts = [float(m) for m in re.findall(r"silence_start:\s*([\d\.]+)", stderr)]
        silence_ends = [float(m) for m in re.findall(r"silence_end:\s*([\d\.]+)", stderr)]

        # Find midpoints of silence intervals
        split_points: List[float] = []
        for s_start, s_end in zip(silence_starts, silence_ends):
            split_points.append((s_start + s_end) / 2.0)

        return sorted(split_points)

    def prepare_chunks(
        self,
        media_path: str | Path,
        work_dir: Optional[str | Path] = None,
    ) -> Tuple[List[AudioChunk], AudioMetadata]:
        """Inspect media and split into chunks if duration exceeds max_duration_seconds.
        
        Returns:
            Tuple of (list of AudioChunk objects, AudioMetadata).
        """
        meta = self.probe(media_path)
        max_dur = self.config.max_duration_seconds
        if max_dur <= 0:
            raise ValueError("max_duration_seconds 必須大於 0")
        if meta.duration > self.config.max_total_duration_seconds:
            limit_seconds = self.config.max_total_duration_seconds
            limit_label = (
                f"{limit_seconds / 60:.0f} 分鐘"
                if limit_seconds >= 60 and limit_seconds % 60 == 0
                else f"{limit_seconds:g} 秒"
            )
            raise ValueError(f"音訊長度超過 {limit_label}上限")

        # Case 1: Audio duration within limit -> single 16kHz mono WAV conversion
        if meta.duration <= max_dur:
            wav_path = self.convert_to_wav(meta.path)
            chunk = AudioChunk(
                index=0,
                path=wav_path,
                start_offset=0.0,
                duration=meta.duration,
                is_temp=True,
            )
            return [chunk], meta

        # Case 2: Audio exceeds limit -> intelligent silence-aware slicing
        owns_temp_dir = work_dir is None
        temp_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="moss_chunks_"))
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Convert full audio to standard WAV first to accurately detect silence
        full_wav = self.convert_to_wav(meta.path, temp_dir / "full_normalized.wav")
        silence_points = self.detect_silence_points(full_wav)

        chunks: List[AudioChunk] = []
        cur_start = 0.0
        total_duration = meta.duration
        chunk_idx = 0

        while cur_start < total_duration:
            target_end = cur_start + max_dur
            if target_end >= total_duration:
                # Last remaining chunk
                actual_end = total_duration
            else:
                # Find a silence point near target_end (within +/- 30 seconds)
                window_min = max(cur_start + 60.0, target_end - 45.0)
                window_max = target_end + 15.0
                candidates = [p for p in silence_points if window_min <= p <= window_max]

                if candidates:
                    # Choose silence point closest to target_end
                    actual_end = min(candidates, key=lambda p: abs(p - target_end))
                else:
                    # Fallback to fixed cut if no suitable silence found
                    actual_end = target_end

            chunk_dur = actual_end - cur_start
            chunk_file = temp_dir / f"chunk_{chunk_idx:03d}_{cur_start:.1f}_{actual_end:.1f}.wav"
            
            self.convert_to_wav(
                full_wav,
                output_path=chunk_file,
                start_time=cur_start,
                duration=chunk_dur,
            )

            chunks.append(
                AudioChunk(
                    index=chunk_idx,
                    path=chunk_file,
                    start_offset=cur_start,
                    duration=chunk_dur,
                    is_temp=True,
                    cleanup_dir=temp_dir if owns_temp_dir else None,
                )
            )

            chunk_idx += 1
            cur_start = actual_end

        # Cleanup intermediate full WAV
        if full_wav.exists():
            full_wav.unlink()

        return chunks, meta
