"""Deduplicate a local JSONL benchmark manifest and retain an exact sample count."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    if args.target < 1:
        parser.error("--target must be positive")
    manifest = args.manifest.resolve()
    if manifest.suffix.lower() != ".jsonl" or not manifest.is_file():
        parser.error("--manifest must be an existing .jsonl file")

    unique_rows: list[dict] = []
    seen_ids: set[str] = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise RuntimeError(f"Manifest line {number} has no id")
        if sample_id not in seen_ids:
            unique_rows.append(row)
            seen_ids.add(sample_id)
    if len(unique_rows) < args.target:
        raise RuntimeError(f"Only {len(unique_rows)} unique samples; target is {args.target}")

    retained = unique_rows[: args.target]
    temp_path = manifest.with_suffix(".jsonl.tmp")
    temp_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained), encoding="utf-8", newline="\n"
    )
    temp_path.replace(manifest)
    print(f"Normalized {manifest}: {len(retained)} unique samples retained")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Normalization failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
