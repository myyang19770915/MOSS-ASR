"""Unit tests for AudioProcessor with synthetic media."""

import subprocess
import tempfile
from pathlib import Path
import pytest

from moss_asr.audio_processor import AudioProcessor
from moss_asr.config import AudioChunkConfig


@pytest.fixture(scope="module")
def sample_media_files():
    """Generate temporary synthetic audio and video files using ffmpeg."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="moss_test_media_"))
    
    wav_path = tmp_dir / "sample.wav"
    mp3_path = tmp_dir / "sample.mp3"
    mp4_path = tmp_dir / "sample.mp4"
    long_wav_path = tmp_dir / "long_sample.wav"

    # 1. 2-second sine wave WAV (440Hz, stereo 44100Hz)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-ar", "44100", "-ac", "2",
        str(wav_path),
    ], check=True)

    # 2. 2-second MP3
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        str(mp3_path),
    ], check=True)

    # 3. 2-second MP4 video with audio
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest",
        str(mp4_path),
    ], check=True)

    # 4. 6-second WAV with silence in the middle (0-2s tone, 2-4s silence, 4-6s tone)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=2",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
        "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[outa]",
        "-map", "[outa]",
        str(long_wav_path),
    ], check=True)

    yield {
        "wav": wav_path,
        "mp3": mp3_path,
        "mp4": mp4_path,
        "long_wav": long_wav_path,
    }


def test_probe_media(sample_media_files):
    processor = AudioProcessor()
    
    # Test WAV
    meta_wav = processor.probe(sample_media_files["wav"])
    assert 1.9 <= meta_wav.duration <= 2.1
    assert meta_wav.is_video is False

    # Test MP4
    meta_mp4 = processor.probe(sample_media_files["mp4"])
    assert 1.9 <= meta_mp4.duration <= 2.1
    assert meta_mp4.is_video is True


def test_convert_to_wav(sample_media_files):
    processor = AudioProcessor()
    
    # Convert MP4 to 16kHz mono WAV
    out_wav = processor.convert_to_wav(sample_media_files["mp4"])
    assert out_wav.exists()
    
    meta = processor.probe(out_wav)
    assert meta.sampling_rate == 16000
    assert meta.channels == 1
    out_wav.unlink()


def test_chunking_short_audio(sample_media_files):
    processor = AudioProcessor(AudioChunkConfig(max_duration_seconds=10.0))
    chunks, meta = processor.prepare_chunks(sample_media_files["mp3"])
    
    assert len(chunks) == 1
    assert chunks[0].start_offset == 0.0
    for c in chunks:
        c.cleanup()


def test_chunking_long_audio(sample_media_files):
    # Set max duration to 3 seconds so 6s audio triggers chunking
    config = AudioChunkConfig(
        max_duration_seconds=3.0,
        silence_threshold_db=-20.0,
        min_silence_duration=0.5,
    )
    processor = AudioProcessor(config)
    chunks, meta = processor.prepare_chunks(sample_media_files["long_wav"])
    
    assert len(chunks) >= 2
    assert chunks[0].start_offset == 0.0
    assert chunks[1].start_offset > 0.0

    # Ensure all chunks exist and are valid WAV files
    for chunk in chunks:
        assert chunk.path.exists()
        chunk_meta = processor.probe(chunk.path)
        assert chunk_meta.sampling_rate == 16000
        chunk.cleanup()


def test_total_duration_limit_is_enforced(sample_media_files):
    processor = AudioProcessor(
        AudioChunkConfig(max_duration_seconds=3.0, max_total_duration_seconds=1.0)
    )
    with pytest.raises(ValueError, match="超過 1 秒上限"):
        processor.prepare_chunks(sample_media_files["wav"])
