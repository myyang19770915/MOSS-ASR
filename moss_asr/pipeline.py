"""End-to-End Pipeline for MOSS-Transcribe-Diarize ASR with Proofreading."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from .audio_processor import AudioProcessor
from .config import PipelineConfig, ASRConfig, AudioChunkConfig, ProofreadConfig
from .exporter import TranscriptExporter
from .prompt_builder import PromptBuilder
from .proofreader import TranscriptProofreadingPipeline
from .transcript_parser import TranscriptParser, TranscriptResult, TranscriptSegment
from .transformers_engine import TransformersEngine
from .vllm_client import VLLMClient


class MossASRPipeline:
    """Master pipeline orchestrating audio preprocessing, chunking, transcription,

    time alignment, proofreading, and export.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        use_local_transformers: bool = False,
    ):
        self.config = config or PipelineConfig()
        self.audio_processor = AudioProcessor(self.config.audio)
        self.use_local_transformers = use_local_transformers

        if use_local_transformers:
            self.engine = TransformersEngine(self.config.asr)
        else:
            self.engine = VLLMClient(self.config.asr)

        self.proofreader = TranscriptProofreadingPipeline(self.config.proofread)

    def process(
        self,
        media_path: str | Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> TranscriptResult:
        return self._process(media_path, progress_callback=progress_callback)

    def process_stream(
        self,
        media_path: str | Path,
        event_callback: Callable[[str, Dict[str, Any]], None],
    ) -> TranscriptResult:
        """Process media while forwarding ASR and LLM generation deltas."""
        return self._process(media_path, event_callback=event_callback)

    def _process(
        self,
        media_path: str | Path,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> TranscriptResult:
        """Run full transcription and proofreading pipeline on an audio or video file.
        
        Args:
            media_path: Path to mp3, wav, m4a, mp4, etc.
            progress_callback: Optional callback receiving (status_message, progress_fraction).
            
        Returns:
            TranscriptResult with full metadata, segments, and proofreading information.
        """
        path = Path(media_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        start_time = time.time()

        def report(message: str, progress: float) -> None:
            if progress_callback:
                progress_callback(message, progress)
            if event_callback:
                event_callback("status", {"message": message, "progress": progress})

        report("正在檢查與準備音訊...", 0.05)

        # 1. Inspect and slice audio into chunks if length exceeds limit
        chunks, meta = self.audio_processor.prepare_chunks(path)
        total_chunks = len(chunks)

        report(
            f"音訊時長 {meta.duration:.1f}s，已準備 {total_chunks} 個音訊片段...",
            0.15,
        )

        all_segments: List[TranscriptSegment] = []
        raw_outputs: List[str] = []

        try:
            # 2. Iterate through each chunk and run inference
            prompt = PromptBuilder.from_config(self.config.asr)

            for idx, chunk in enumerate(chunks, start=1):
                chunk_progress = 0.15 + (0.65 * (idx / total_chunks))
                report(
                    f"正在轉錄分段 [{idx}/{total_chunks}] (偏移: {chunk.start_offset:.1f}s)...",
                    chunk_progress,
                )

                def on_transcription_delta(delta: str, chunk_index: int = idx) -> None:
                    if event_callback:
                        event_callback(
                            "transcription_delta",
                            {
                                "chunk_index": chunk_index,
                                "chunks_total": total_chunks,
                                "delta": delta,
                            },
                        )

                transcribe_kwargs: Dict[str, Any] = {
                    "prompt": prompt,
                    "max_new_tokens": self.config.asr.max_new_tokens,
                    "temperature": self.config.asr.temperature,
                }
                if not self.use_local_transformers:
                    transcribe_kwargs["progress_callback"] = on_transcription_delta
                result_dict = self.engine.transcribe(chunk.path, **transcribe_kwargs)

                chunk_text = result_dict.get("text", "")
                raw_outputs.append(chunk_text)

                # Parse chunk tokens and apply time offset
                chunk_segments = TranscriptParser.parse(
                    chunk_text,
                    time_offset=chunk.start_offset,
                )
                all_segments.extend(chunk_segments)
                if event_callback:
                    event_callback(
                        "transcription_chunk_complete",
                        {
                            "chunk_index": idx,
                            "chunks_total": total_chunks,
                            "segments": [segment.to_dict() for segment in chunk_segments],
                        },
                    )

        finally:
            # Clean up temporary chunk WAV files
            for chunk in chunks:
                chunk.cleanup()

        # 3. Post-process segments (sort by start timestamp)
        all_segments.sort(key=lambda s: (s.start, s.end))

        raw_combined = "\n".join(raw_outputs)
        elapsed_total = time.time() - start_time

        initial_result = TranscriptResult(
            raw_text=raw_combined,
            segments=all_segments,
            audio_duration=meta.duration,
            elapsed_time=elapsed_total,
            model=self.config.asr.model_id,
            is_proofread=False,
        )

        # Publish a stable, fully parsed transcript before any correction
        # starts.  This lets clients render usable subtitles while the LLM
        # performs the slower second-stage review.
        if event_callback:
            event_callback(
                "transcription_complete",
                {
                    "audio_duration": initial_result.audio_duration,
                    "elapsed_time": initial_result.elapsed_time,
                    "full_text": initial_result.full_text,
                    "formatted_text": initial_result.get_formatted_text(
                        include_timestamps=self.config.asr.include_timestamps,
                        include_speakers=self.config.asr.include_speakers,
                    ),
                    "srt_text": TranscriptExporter.to_srt(
                        initial_result,
                        include_speakers=self.config.asr.include_speakers,
                    ),
                    "segments": [segment.to_dict() for segment in initial_result.segments],
                },
            )

        # 4. Proofreading & Error Correction
        if (
            self.config.proofread.enabled
            or self.config.proofread.remove_stutter
            or self.config.proofread.normalize_punctuation
            or self.config.proofread.custom_replacements
        ):
            report("正在進行文稿校對與潤飾...", 0.90)
            if event_callback and self.config.proofread.enabled and self.proofreader.llm_engine:
                event_callback("proofread_started", {"model": self.config.proofread.llm_model})
            elif event_callback and self.config.proofread.enabled:
                event_callback("proofread_unavailable", {"detail": "未提供 LiteLLM API Key，無法執行智慧校對。"})

            def on_proofread_delta(delta: str) -> None:
                if event_callback:
                    event_callback("proofread_delta", {"delta": delta})

            def on_proofread_progress(payload: Dict[str, Any]) -> None:
                if event_callback:
                    event_callback("proofread_progress", payload)

            final_result = self.proofreader.process(
                initial_result,
                stream_callback=(
                    on_proofread_delta
                    if event_callback and self.config.proofread.enabled and self.proofreader.llm_engine
                    else None
                ),
                progress_callback=(
                    on_proofread_progress
                    if event_callback and self.config.proofread.enabled
                    else None
                ),
            )
        else:
            final_result = initial_result

        final_result.elapsed_time = time.time() - start_time

        report("轉錄與校對完成！", 1.0)

        return final_result

    def process_and_export(
        self,
        media_path: str | Path,
        output_dir: str | Path,
        base_name: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Path]:
        """Convenience method to process media and save all standard formats."""
        path = Path(media_path)
        out_base = base_name or path.stem

        result = self.process(path, progress_callback=progress_callback)

        saved = TranscriptExporter.save_all(
            result=result,
            output_dir=output_dir,
            base_name=out_base,
            include_timestamps=self.config.asr.include_timestamps,
            include_speakers=self.config.asr.include_speakers,
            speaker_names=self.config.asr.speaker_names,
        )
        return saved
