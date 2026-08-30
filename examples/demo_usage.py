"""Example demonstration of MOSS-Transcribe-Diarize Python Pipeline."""

from pathlib import Path
from moss_asr import (
    PipelineConfig,
    ASRConfig,
    AudioChunkConfig,
    ProofreadConfig,
    MossASRPipeline,
    TranscriptExporter,
)


def main():
    # 1. Configure the pipeline
    config = PipelineConfig(
        audio=AudioChunkConfig(
            max_duration_seconds=1800.0,  # Auto-chunk if audio > 30 minutes
        ),
        asr=ASRConfig(
            vllm_base_url="http://localhost:8000",
            model_id="OpenMOSS-Team/MOSS-Transcribe-Diarize",
            include_timestamps=True,      # Toggle timestamps
            include_speakers=True,        # Toggle speaker diarization ([S01], [S02])
            hotwords=["OpenMOSS", "vLLM", "人工智慧", "語音轉錄"],  # Domain hotwords
            speaker_names={"S01": "主持人", "S02": "講者"},      # Custom speaker naming
        ),
        proofread=ProofreadConfig(
            enabled=True,                 # Enable transcript proofreading
            llm_api_base="https://api.openai.com/v1",
            llm_api_key="sk-...",         # Set your LLM API Key here
            llm_model="gpt-4o-mini",
            remove_stutter=True,          # Remove spoken stutter & filler words
            custom_replacements={
                "佈署": "部署",
                "計器": "機器",
            },
        ),
    )

    pipeline = MossASRPipeline(config=config, use_local_transformers=False)

    # 2. Input audio or video file (supports .mp3, .wav, .m4a, .mp4)
    media_file = "sample_meeting.mp4"
    print(f"開始處理音訊檔案: {media_file}")

    # 3. Progress callback
    def on_progress(message: str, fraction: float):
        print(f"[{int(fraction * 100):3d}%] {message}")

    # Note: In real usage with running vLLM service, call:
    # result = pipeline.process(media_file, progress_callback=on_progress)
    #
    # 4. Exporting results:
    # saved_files = TranscriptExporter.save_all(
    #     result=result,
    #     output_dir="./transcripts",
    #     base_name="meeting_transcript",
    #     include_timestamps=True,
    #     include_speakers=True,
    #     speaker_names={"S01": "主持人", "S02": "講者"},
    # )
    # print("已輸出格式:", list(saved_files.keys()))


if __name__ == "__main__":
    main()
