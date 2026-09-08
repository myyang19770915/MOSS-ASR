"""Download a bounded Common Voice 25 zh-TW test subset for local ASR scoring.

The source is OpenFormosa/common_voice_25_zh-TW on Hugging Face. It is a
public CC0-1.0 dataset mirror; downloaded audio and its manifest are ignored
by this repository, so they remain local to the benchmark mount.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlencode


ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "OpenFormosa/common_voice_25_zh-TW"
CONFIG = "default"
SPLIT = "test"
USER_AGENT = "MOSS-ASR-Benchmark/1.0"


def fetch_rows(offset: int, length: int) -> list[dict]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(f"{ROWS_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("資料來源沒有回傳可下載的 Common Voice rows")
    return rows


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if not temporary.stat().st_size:
            raise RuntimeError("下載到空白音檔")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def existing_ids(manifest: Path) -> set[str]:
    if not manifest.is_file():
        return set()
    ids: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            sample_id = str(json.loads(line).get("id", "")).strip()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"既有 manifest 第 {line_number} 行不是有效 JSON") from exc
        if sample_id:
            ids.add(sample_id)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True, help="Mounted benchmark dataset directory")
    parser.add_argument("--offset", type=int, default=0, help="Official test split row offset")
    parser.add_argument("--count", type=int, default=500, help="Number of samples to retain (1-1000)")
    args = parser.parse_args()
    if args.offset < 0 or not 1 <= args.count <= 1000:
        parser.error("--offset must be non-negative and --count must be 1-1000")

    destination = args.destination.resolve()
    audio_dir = destination / "audio"
    manifest = destination / "test.jsonl"
    audio_dir.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".download.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("已有 Common Voice zh-TW 下載工作進行中，請等待它完成") from exc

    try:
        ids = existing_ids(manifest)
        if len(ids) >= args.count:
            print(f"Ready: {manifest} ({len(ids)} samples already present)")
            return 0

        offset = args.offset
        while len(ids) < args.count:
            rows = fetch_rows(offset, min(100, args.count - len(ids)))
            if not rows:
                raise RuntimeError("官方 test split 已無更多資料，無法達到指定筆數")
            entries: list[dict[str, str]] = []
            for item in rows:
                row = item.get("row", {})
                row_index = item.get("row_idx")
                text = str(row.get("sentence", "")).strip()
                audio = row.get("audio") or []
                audio_url = audio[0].get("src") if audio and isinstance(audio[0], dict) else None
                if row_index is None or not text or not audio_url:
                    raise RuntimeError("Common Voice row 缺少 row_idx、sentence 或音檔 URL")
                sample_id = f"commonvoice-zh-TW-{row_index}"
                if sample_id in ids:
                    continue
                audio_name = f"{sample_id}.mp3"
                audio_path = audio_dir / audio_name
                if not audio_path.is_file() or not audio_path.stat().st_size:
                    print(f"[{len(ids) + 1}/{args.count}] downloading {sample_id}", flush=True)
                    download(str(audio_url), audio_path)
                entries.append(
                    {
                        "id": sample_id,
                        "audio": f"audio/{audio_name}",
                        "text": text,
                        "language": "zh-TW",
                    }
                )
                ids.add(sample_id)
                if len(ids) >= args.count:
                    break
            if entries:
                with manifest.open("a", encoding="utf-8", newline="\n") as handle:
                    for entry in entries:
                        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            offset += len(rows)

        print(f"Ready: {manifest} ({len(ids)} samples total; source {DATASET}/{SPLIT})")
        return 0
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
