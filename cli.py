#!/usr/bin/env python3
"""Command Line Interface (CLI) for MOSS-Transcribe-Diarize ASR & Proofreading Pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from moss_asr import (
    PipelineConfig,
    ASRConfig,
    AudioChunkConfig,
    ProofreadConfig,
    MossASRPipeline,
    TranscriptExporter,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MOSS-Transcribe-Diarize ASR & Proofreading CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input & Output
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input audio or video file (supports .mp3, .wav, .m4a, .mp4, .mkv, .mov, etc.)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./output",
        help="Directory to save exported transcript files (.txt, .md, .srt, .vtt, .json, .csv)",
    )
    parser.add_argument(
        "--print-format",
        choices=["txt", "md", "srt", "vtt", "json", "none"],
        default="txt",
        help="Format to print to standard output",
    )

    # Serving & Backend
    parser.add_argument(
        "--vllm-url",
        type=str,
        default=os.getenv("MOSS_VLLM_URL", "http://127.0.0.1:18000"),
        help="URL of vLLM / SGLang OpenAI-compatible server",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        help="Model identifier on server or HuggingFace Hub",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use direct local HuggingFace Transformers inference instead of remote vLLM server",
    )

    # Features (Timestamps, Speaker Diarization, Hotwords)
    parser.add_argument(
        "--timestamps",
        dest="include_timestamps",
        action="store_true",
        default=True,
        help="Enable timestamp generation and output",
    )
    parser.add_argument(
        "--no-timestamps",
        dest="include_timestamps",
        action="store_false",
        help="Disable timestamp generation and output",
    )
    parser.add_argument(
        "--diarize",
        dest="include_speakers",
        action="store_true",
        default=True,
        help="Enable speaker diarization ([S01], [S02])",
    )
    parser.add_argument(
        "--no-diarize",
        dest="include_speakers",
        action="store_false",
        help="Disable speaker diarization",
    )
    parser.add_argument(
        "--hotwords",
        type=str,
        default="",
        help="Comma-separated hotwords/terms for domain prompt hint (e.g. 'OpenMOSS,vLLM,Transformer')",
    )
    parser.add_argument(
        "--speaker-map",
        type=str,
        default="",
        help="JSON mapping of speaker labels to real names, e.g. '{\"S01\": \"張經理\", \"S02\": \"李工程師\"}'",
    )

    # Audio chunking
    parser.add_argument(
        "--max-chunk-sec",
        type=float,
        default=1800.0,
        help="Max audio chunk duration in seconds before auto silence-slicing",
    )

    # Proofreading options
    parser.add_argument(
        "--proofread",
        action="store_true",
        default=False,
        help="Enable LLM intelligent transcript proofreading and error correction",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI-compatible LLM endpoint base URL for proofreading",
    )
    parser.add_argument(
        "--llm-key",
        type=str,
        default="",
        help="API Key for LLM proofreading",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name for proofreading (e.g. gpt-4o-mini, qwen2.5-7b-instruct, deepseek-chat)",
    )
    parser.add_argument(
        "--no-stutter-clean",
        dest="remove_stutter",
        action="store_false",
        default=True,
        help="Disable automatic rule-based speech stutter and repetition removal",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"錯誤: 輸入檔案不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Parse hotwords
    hotwords_list = [w.strip() for w in args.hotwords.split(",") if w.strip()] if args.hotwords else []

    # Parse speaker mapping
    speaker_names = {}
    if args.speaker_map:
        try:
            if Path(args.speaker_map).exists():
                speaker_names = json.loads(Path(args.speaker_map).read_text(encoding="utf-8"))
            else:
                speaker_names = json.loads(args.speaker_map)
        except Exception as e:
            print(f"警告: 解析 speaker-map 失敗 ({e})，使用原始 S01/S02 標籤", file=sys.stderr)

    # Setup configuration
    config = PipelineConfig(
        audio=AudioChunkConfig(
            max_duration_seconds=args.max_chunk_sec,
        ),
        asr=ASRConfig(
            model_id=args.model_id,
            vllm_base_url=args.vllm_url,
            include_timestamps=args.include_timestamps,
            include_speakers=args.include_speakers,
            hotwords=hotwords_list,
            speaker_names=speaker_names,
        ),
        proofread=ProofreadConfig(
            enabled=args.proofread,
            llm_api_base=args.llm_url,
            llm_api_key=args.llm_key,
            llm_model=args.llm_model,
            remove_stutter=args.remove_stutter,
        ),
    )

    print("==================================================")
    print(" 🎙️  MOSS-Transcribe-Diarize 語音轉錄與校對系統")
    print("==================================================")
    print(f"輸入檔案: {input_path.name}")
    print(f"後端引擎: {'本地 Transformers' if args.local else f'vLLM 服務 ({args.vllm_url})'}")
    print(f"時間戳記: {'開啟' if args.include_timestamps else '關閉'}")
    print(f"說話人分離: {'開啟' if args.include_speakers else '關閉'}")
    print(f"熱詞提示: {', '.join(hotwords_list) if hotwords_list else '無'}")
    print(f"文稿校對: {'開啟 (LLM + 規則校對)' if args.proofread else '基礎規則/無 LLM 校對'}")
    print("==================================================\n")

    def on_progress(msg: str, frac: float):
        percent = int(frac * 100)
        bar = "█" * (percent // 4) + "░" * (25 - (percent // 4))
        print(f"\r[{bar}] {percent:3d}% - {msg}", end="", flush=True)

    pipeline = MossASRPipeline(config=config, use_local_transformers=args.local)
    
    try:
        result = pipeline.process(input_path, progress_callback=on_progress)
        print("\n\n✅ 轉錄與校對處理完成！\n")
    except Exception as exc:
        print(f"\n\n❌ 處理過程中發生錯誤: {exc}", file=sys.stderr)
        sys.exit(1)

    # Save outputs
    output_dir = Path(args.output_dir).expanduser().resolve()
    saved_files = TranscriptExporter.save_all(
        result=result,
        output_dir=output_dir,
        base_name=input_path.stem,
        include_timestamps=args.include_timestamps,
        include_speakers=args.include_speakers,
        speaker_names=speaker_names,
    )

    print(f"📁 檔案已成功輸出至: {output_dir}")
    for fmt, fpath in saved_files.items():
        print(f"  - [{fmt.upper()}] {fpath.name}")

    # Print to stdout if requested
    if args.print_format != "none":
        print("\n" + "=" * 50)
        print(f"📋 轉錄結果預覽 ({args.print_format.upper()}):")
        print("=" * 50)
        if args.print_format == "txt":
            print(TranscriptExporter.to_txt(result, args.include_timestamps, args.include_speakers, speaker_names))
        elif args.print_format == "md":
            print(TranscriptExporter.to_markdown(result, include_timestamps=args.include_timestamps, include_speakers=args.include_speakers, speaker_names=speaker_names))
        elif args.print_format == "srt":
            print(TranscriptExporter.to_srt(result, include_speakers=args.include_speakers, speaker_names=speaker_names))
        elif args.print_format == "vtt":
            print(TranscriptExporter.to_vtt(result, include_speakers=args.include_speakers, speaker_names=speaker_names))
        elif args.print_format == "json":
            print(TranscriptExporter.to_json(result))
        print("=" * 50)


if __name__ == "__main__":
    main()
