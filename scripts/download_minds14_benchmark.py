"""Build a reproducible MInDS-14 benchmark in a local mount.

Audio and the generated JSONL manifest are intentionally ignored by Git. The
script fetches public, CC BY 4.0 dataset rows from Hugging Face's dataset-server
API. It can balance samples across all 14 language configurations, or retrieve a
fixed number of samples for one locale such as ``zh-CN``.
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
DATASET = "PolyAI/minds14"
CONFIGS = [
    "cs-CZ", "de-DE", "en-AU", "en-GB", "en-US", "es-ES", "fr-FR",
    "it-IT", "ko-KR", "nl-NL", "pl-PL", "pt-PT", "ru-RU", "zh-CN",
]
EXISTING_CONFIGS = {"en-US", "de-DE", "es-ES", "fr-FR", "ko-KR", "zh-CN"}


def fetch_rows(config: str, offset: int, count: int) -> list[dict]:
    query = urlencode(
        {"dataset": DATASET, "config": config, "split": "train", "offset": offset, "length": count}
    )
    request = urllib.request.Request(f"{ROWS_URL}?{query}", headers={"User-Agent": "MOSS-ASR-Benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) < count:
        raise RuntimeError(f"{config} 只回傳 {len(rows) if isinstance(rows, list) else 0} 筆資料")
    return rows[:count]


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "MOSS-ASR-Benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--config",
        action="append",
        choices=CONFIGS,
        help="要下載的 MInDS-14 locale；可重複指定。未指定時涵蓋全部 14 種語言。",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="此 manifest 的目標樣本數（預設：100）。",
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count 必須至少為 1")
    destination = args.destination.resolve()
    configs = args.config or CONFIGS
    audio_dir = destination / "audio"
    manifest = destination / "test.jsonl"
    audio_dir.mkdir(parents=True, exist_ok=True)
    lock_path = destination / ".download.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("已有 MInDS-14 下載工作進行中，請等待它完成") from exc

    try:
        existing_ids: set[str] = set()
        existing_by_config = {config: 0 for config in configs}
        if manifest.is_file():
            for line in manifest.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    existing_ids.add(str(entry.get("id", "")))
                    language = str(entry.get("language", "")).strip()
                    if language in existing_by_config:
                        existing_by_config[language] += 1

        if len(existing_ids) > args.count:
            raise RuntimeError(
                f"既有 manifest 已有 {len(existing_ids)} 筆，超過指定目標 {args.count} 筆；"
                "請使用新的 destination 或提高 --count"
            )

        # Assign the requested total as evenly as possible. A single --config
        # therefore retrieves exactly that many rows for the requested locale.
        base_count, remainder = divmod(args.count, len(configs))
        planned_counts = {
            config: base_count + (1 if index < remainder else 0)
            for index, config in enumerate(configs)
        }
        new_entries: list[dict[str, str]] = []
        target_total = args.count
        for config in configs:
            if len(existing_ids) >= target_total:
                break
            needed = max(0, planned_counts[config] - existing_by_config[config])
            if not needed:
                continue
            # The legacy multilingual smoke fixture contains one hand-curated
            # row for these locales. Its ID has no row index, so retain the
            # historical offset only when extending that original fixture.
            offset = 1 if destination.name == "minds14-smoke" and config in EXISTING_CONFIGS else 0
            rows = fetch_rows(config, offset, needed)
            for item in rows:
                row = item.get("row", {})
                source_id = str(row.get("path") or row.get("id") or "").strip()
                text = str(row.get("transcription", "")).strip()
                audio = row.get("audio") or []
                audio_url = audio[0].get("src") if audio and isinstance(audio[0], dict) else None
                if not source_id or not text or not audio_url:
                    raise RuntimeError(f"{config} 出現缺少 path、transcription 或音檔 URL 的資料")
                sample_id = f"minds14-{config}-{item.get('row_idx', len(existing_ids))}"
                if sample_id in existing_ids:
                    continue
                audio_name = f"{sample_id}.wav"
                audio_path = audio_dir / audio_name
                if not audio_path.is_file() or not audio_path.stat().st_size:
                    print(f"[{len(existing_ids) + 1}/{target_total}] downloading {sample_id}", flush=True)
                    download(str(audio_url), audio_path)
                new_entries.append(
                    {"id": sample_id, "audio": f"audio/{audio_name}", "text": text, "language": config}
                )
                existing_ids.add(sample_id)
                if len(existing_ids) >= target_total:
                    break

        if len(existing_ids) != target_total:
            raise RuntimeError(f"預期 {target_total} 筆，實際只有 {len(existing_ids)} 筆")
        if new_entries:
            with manifest.open("a", encoding="utf-8", newline="\n") as handle:
                for entry in new_entries:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Ready: {manifest} ({len(existing_ids)} samples total; {len(new_entries)} new manifest rows)")
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
