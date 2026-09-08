"""Local manifest loading and language-aware ASR scoring utilities."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv",
}
MANIFEST_EXTENSIONS = {".jsonl", ".csv"}
CHARACTER_METRIC_LANGUAGES = {
    "zh", "cmn", "yue", "ja", "jpn", "ko", "kor", "th", "tha",
    "lo", "lao", "km", "khm", "my", "mya",
}
_PUNCTUATION_RE = re.compile(r"[\W_]+", re.UNICODE)
_LANGUAGE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")


class BenchmarkManifestError(ValueError):
    """Raised when a benchmark manifest is unsafe or structurally invalid."""


@dataclass(frozen=True)
class BenchmarkSample:
    """One audio/reference pair from a local benchmark manifest."""

    sample_id: str
    audio_path: Path
    reference: str
    language: str = "und"


def _resolve_under_root(path: Path, root: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise BenchmarkManifestError(f"{label} 必須位於 benchmark 資料根目錄內") from exc
    return resolved_path


def resolve_manifest_path(manifest_name: str, data_root: Path) -> Path:
    """Resolve a UI-selected manifest relative to the approved data root."""
    name = (manifest_name or "").strip().replace("\\", "/")
    if not name or name.startswith("/") or ".." in Path(name).parts:
        raise BenchmarkManifestError("請從已掛載資料集清單選擇 manifest")
    candidate = _resolve_under_root(data_root / name, data_root, label="Manifest")
    if candidate.suffix.lower() not in MANIFEST_EXTENSIONS:
        raise BenchmarkManifestError("Manifest 僅支援 .jsonl 或 .csv")
    if not candidate.is_file():
        raise BenchmarkManifestError("找不到指定的 benchmark manifest")
    return candidate


def list_manifests(data_root: Path) -> list[dict[str, Any]]:
    """Return safe, relative manifest paths discoverable below the mounted root."""
    if not data_root.is_dir():
        return []
    manifests: list[dict[str, Any]] = []
    for candidate in sorted(data_root.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in MANIFEST_EXTENSIONS:
            continue
        try:
            relative = candidate.resolve().relative_to(data_root.resolve())
        except ValueError:
            continue
        if ".example." in candidate.name:
            continue
        if len(relative.parts) > 5:
            continue
        item: dict[str, Any] = {
            "name": relative.as_posix(),
            "format": candidate.suffix.lower().removeprefix("."),
            "bytes": candidate.stat().st_size,
        }
        try:
            item.update(manifest_stats(candidate, data_root))
        except BenchmarkManifestError as exc:
            item["invalid"] = str(exc)
        manifests.append(item)
    return manifests[:100]


def _read_manifest_rows(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise BenchmarkManifestError(f"JSONL 第 {line_number} 行不是有效 JSON") from exc
                if not isinstance(row, dict):
                    raise BenchmarkManifestError(f"JSONL 第 {line_number} 行必須是物件")
                yield row
        return

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise BenchmarkManifestError("CSV 缺少欄位標題")
        for row in reader:
            yield dict(row)


def _value(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _iter_manifest_samples(manifest_path: Path, data_root: Path) -> Iterable[BenchmarkSample]:
    """Yield validated samples without exposing paths outside the mounted root."""
    safe_manifest = _resolve_under_root(manifest_path, data_root, label="Manifest")
    seen_ids: set[str] = set()
    for row_number, row in enumerate(_read_manifest_rows(safe_manifest), start=1):
        audio_ref = _value(row, "audio", "audio_path", "path", "file")
        reference = _value(row, "text", "reference", "transcript", "sentence", "normalized_text")
        if not audio_ref or not reference:
            raise BenchmarkManifestError(
                f"Manifest 第 {row_number} 筆需包含 audio 與 text/reference/transcript"
            )
        audio_candidate = Path(audio_ref)
        if audio_candidate.is_absolute() or ".." in audio_candidate.parts:
            raise BenchmarkManifestError(f"Manifest 第 {row_number} 筆 audio 路徑不安全")
        audio_path = _resolve_under_root(safe_manifest.parent / audio_candidate, data_root, label="音檔")
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            raise BenchmarkManifestError(f"Manifest 第 {row_number} 筆是不支援的音檔格式")
        if not audio_path.is_file():
            raise BenchmarkManifestError(f"Manifest 第 {row_number} 筆找不到音檔：{audio_ref}")
        sample_id = _value(row, "id", "sample_id", "audio_id") or audio_path.stem
        if sample_id in seen_ids:
            sample_id = f"{sample_id}-{row_number}"
        seen_ids.add(sample_id)
        language = _value(row, "language", "lang", "locale") or "und"
        if not _LANGUAGE_RE.fullmatch(language):
            raise BenchmarkManifestError(f"Manifest 第 {row_number} 筆 language 格式不正確")
        yield BenchmarkSample(
            sample_id=sample_id,
            audio_path=audio_path,
            reference=reference,
            language=language,
        )
 

def load_manifest(manifest_path: Path, data_root: Path, *, max_samples: int) -> list[BenchmarkSample]:
    """Load a bounded number of validated audio/reference pairs."""
    if max_samples < 1:
        raise BenchmarkManifestError("測試筆數至少要是 1")
    samples: list[BenchmarkSample] = []
    for sample in _iter_manifest_samples(manifest_path, data_root):
        samples.append(sample)
        if len(samples) >= max_samples:
            break
    if not samples:
        raise BenchmarkManifestError("Manifest 沒有可執行的測試資料")
    return samples


def manifest_stats(manifest_path: Path, data_root: Path) -> dict[str, Any]:
    """Count validated samples and summarize languages for a mounted manifest."""
    count = 0
    languages: dict[str, int] = {}
    for sample in _iter_manifest_samples(manifest_path, data_root):
        count += 1
        languages[sample.language] = languages.get(sample.language, 0) + 1
    if not count:
        raise BenchmarkManifestError("Manifest 沒有可執行的測試資料")
    return {"samples": count, "languages": languages}


def find_manifest_sample(
    manifest_path: Path, data_root: Path, sample_id: str
) -> BenchmarkSample:
    """Find a manifest sample by opaque ID for safe in-browser audio playback."""
    requested_id = (sample_id or "").strip()
    if not requested_id:
        raise BenchmarkManifestError("請指定資料集樣本 ID")
    for sample in _iter_manifest_samples(manifest_path, data_root):
        if sample.sample_id == requested_id:
            return sample
    raise BenchmarkManifestError("找不到指定的資料集樣本")


def normalized_text(text: str) -> str:
    """Normalize casing, compatibility glyphs, punctuation, and spacing for scoring."""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    normalized = "".join(
        char for char in normalized if not unicodedata.category(char).startswith("P")
    )
    return " ".join(normalized.split())


def select_metric(language: str, requested_metric: str = "auto") -> str:
    requested = (requested_metric or "auto").lower()
    if requested in {"wer", "cer"}:
        return requested
    if requested != "auto":
        raise BenchmarkManifestError("評分方式必須是 auto、wer 或 cer")
    primary = (language or "und").lower().replace("_", "-").split("-", 1)[0]
    return "cer" if primary in CHARACTER_METRIC_LANGUAGES else "wer"


def _units(text: str, metric: str) -> list[str]:
    normalized = normalized_text(text)
    if metric == "cer":
        return [character for character in normalized if not character.isspace()]
    return normalized.split()


def edit_distance(reference_units: list[str], hypothesis_units: list[str]) -> int:
    """Memory-efficient Levenshtein distance for word or character units."""
    if len(reference_units) < len(hypothesis_units):
        reference_units, hypothesis_units = hypothesis_units, reference_units
    previous = list(range(len(hypothesis_units) + 1))
    for ref_index, ref_unit in enumerate(reference_units, start=1):
        current = [ref_index]
        for hyp_index, hyp_unit in enumerate(hypothesis_units, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_unit != hyp_unit),
                )
            )
        previous = current
    return previous[-1]


def score_transcript(
    reference: str,
    hypothesis: str,
    *,
    language: str = "und",
    requested_metric: str = "auto",
) -> dict[str, Any]:
    """Score one transcript and expose both conventional error rate and UI accuracy."""
    metric = select_metric(language, requested_metric)
    reference_units = _units(reference, metric)
    hypothesis_units = _units(hypothesis, metric)
    if not reference_units:
        raise BenchmarkManifestError("參考逐字稿正規化後不可為空")
    errors = edit_distance(reference_units, hypothesis_units)
    error_rate = errors / len(reference_units)
    return {
        "metric": metric,
        "errors": errors,
        "reference_units": len(reference_units),
        "hypothesis_units": len(hypothesis_units),
        "error_rate": error_rate,
        "accuracy_percent": max(0.0, (1.0 - error_rate) * 100.0),
        "exact_match": normalized_text(reference) == normalized_text(hypothesis),
    }


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate micro scores and language/metric breakdowns from completed samples."""
    if not rows:
        return {"samples": 0, "accuracy_percent": 0.0, "error_rate": 0.0, "by_language": []}

    def summarize(group: list[dict[str, Any]], language: str | None = None) -> dict[str, Any]:
        errors = sum(int(row["score"]["errors"]) for row in group)
        reference_units = sum(int(row["score"]["reference_units"]) for row in group)
        exact_matches = sum(bool(row["score"]["exact_match"]) for row in group)
        error_rate = errors / reference_units if reference_units else 0.0
        result: dict[str, Any] = {
            "samples": len(group),
            "errors": errors,
            "reference_units": reference_units,
            "error_rate": error_rate,
            "accuracy_percent": max(0.0, (1.0 - error_rate) * 100.0),
            "exact_match_percent": exact_matches / len(group) * 100.0,
        }
        if language is not None:
            result["language"] = language
            result["metric"] = group[0]["score"]["metric"]
        return result

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        score = row["score"]
        key = (str(row.get("language") or "und"), str(score["metric"]))
        groups.setdefault(key, []).append(row)
    language_rows = []
    for (language, _metric), group in sorted(groups.items()):
        language_rows.append(summarize(group, language))
    metrics = {str(row["score"]["metric"]) for row in rows}
    overall = summarize(rows)
    overall["metric"] = next(iter(metrics)) if len(metrics) == 1 else None
    overall["mixed_metrics"] = len(metrics) > 1
    # Word and character units are not commensurable. Preserve per-language
    # aggregates, but do not advertise a pseudo-global WER/CER when a run
    # contains both kinds of languages.
    if overall["mixed_metrics"]:
        overall["errors"] = None
        overall["reference_units"] = None
        overall["error_rate"] = None
        overall["accuracy_percent"] = None
    overall["by_language"] = language_rows
    return overall
