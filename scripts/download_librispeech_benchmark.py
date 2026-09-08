"""Download a bounded LibriSpeech validation subset into a local benchmark mount.

The downloaded audio and generated manifest are intentionally ignored by Git.
This script uses Hugging Face's public dataset-server API and does not require a
token. It is designed for repeatable small-scale, local ASR evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlencode


ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "openslr/librispeech_asr"


def request_rows(offset: int, count: int) -> list[dict]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": "clean",
            "split": "validation",
            "offset": offset,
            "length": count,
        }
    )
    request = urllib.request.Request(f"{ROWS_URL}?{query}", headers={"User-Agent": "MOSS-ASR-Benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("資料來源沒有回傳可下載的 LibriSpeech rows")
    return rows


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "MOSS-ASR-Benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True, help="Mounted benchmark dataset directory")
    parser.add_argument("--offset", type=int, default=0, help="Validation row offset")
    parser.add_argument("--count", type=int, default=100, help="Number of samples to download (1-100)")
    args = parser.parse_args()
    if args.offset < 0 or not 1 <= args.count <= 100:
        parser.error("--offset must be non-negative and --count must be 1-100")

    destination = args.destination.resolve()
    audio_dir = destination / "audio"
    manifest = destination / "validation.jsonl"
    audio_dir.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                existing_ids.add(str(json.loads(line).get("id", "")))

    rows = request_rows(args.offset, args.count)
    if len(rows) < args.count:
        raise RuntimeError(f"資料來源只回傳 {len(rows)} 筆，少於要求的 {args.count} 筆")

    entries: list[dict[str, str]] = []
    for position, item in enumerate(rows[: args.count], start=1):
        row = item.get("row", {})
        sample_id = str(row.get("id", "")).strip()
        text = str(row.get("text", "")).strip()
        audio = row.get("audio") or []
        audio_url = audio[0].get("src") if audio and isinstance(audio[0], dict) else None
        if not sample_id or not text or not audio_url:
            raise RuntimeError(f"第 {position} 筆缺少 id、text 或音檔 URL")
        audio_name = f"{sample_id}.flac"
        audio_path = audio_dir / audio_name
        if not audio_path.is_file() or not audio_path.stat().st_size:
            print(f"[{position}/{args.count}] downloading {sample_id}", flush=True)
            download(str(audio_url), audio_path)
        if sample_id not in existing_ids:
            entries.append(
                {"id": sample_id, "audio": f"audio/{audio_name}", "text": text, "language": "en-US"}
            )
            existing_ids.add(sample_id)

    if entries:
        with manifest.open("a", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Ready: {manifest} ({len(existing_ids)} samples total; {len(entries)} new manifest rows)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
