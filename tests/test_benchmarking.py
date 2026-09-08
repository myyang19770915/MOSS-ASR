"""Tests for local ASR benchmark manifest loading and score calculations."""

import json

import pytest

from moss_asr.benchmarking import (
    BenchmarkManifestError,
    list_manifests,
    load_manifest,
    resolve_manifest_path,
    score_transcript,
    summarize_scores,
)


def _make_manifest(tmp_path):
    root = tmp_path / "benchmarks"
    dataset = root / "zh-en"
    audio_dir = dataset / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "zh.wav").write_bytes(b"placeholder")
    (audio_dir / "en.wav").write_bytes(b"placeholder")
    manifest = dataset / "test.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"id": "zh-1", "audio": "audio/zh.wav", "text": "你好，世界！", "language": "zh-TW"}, ensure_ascii=False),
                json.dumps({"id": "en-1", "audio": "audio/en.wav", "reference": "Hello world", "language": "en"}),
            ]
        ),
        encoding="utf-8",
    )
    return root, manifest


def test_load_manifest_only_accepts_audio_under_data_root(tmp_path):
    root, manifest = _make_manifest(tmp_path)
    samples = load_manifest(manifest, root, max_samples=10)
    assert [sample.sample_id for sample in samples] == ["zh-1", "en-1"]
    assert samples[0].language == "zh-TW"
    assert samples[1].audio_path.name == "en.wav"

    unsafe = manifest.with_name("unsafe.jsonl")
    unsafe.write_text('{"audio":"../../outside.wav","text":"unsafe"}\n', encoding="utf-8")
    with pytest.raises(BenchmarkManifestError, match="不安全"):
        load_manifest(unsafe, root, max_samples=1)


def test_manifest_listing_is_relative_and_hides_templates(tmp_path):
    root, _ = _make_manifest(tmp_path)
    (root / "manifest.example.jsonl").write_text("{}\n", encoding="utf-8")
    manifests = list_manifests(root)
    assert len(manifests) == 1
    assert manifests[0]["name"] == "zh-en/test.jsonl"
    assert manifests[0]["format"] == "jsonl"
    assert resolve_manifest_path("zh-en/test.jsonl", root).name == "test.jsonl"
    with pytest.raises(BenchmarkManifestError):
        resolve_manifest_path("../test.jsonl", root)


def test_score_transcript_uses_cer_for_chinese_and_wer_for_english():
    chinese = score_transcript("你好，世界！", "你好世界", language="zh-TW")
    assert chinese["metric"] == "cer"
    assert chinese["errors"] == 0
    assert chinese["accuracy_percent"] == 100

    english = score_transcript("hello world", "hello brave world", language="en")
    assert english["metric"] == "wer"
    assert english["errors"] == 1
    assert english["reference_units"] == 2
    assert english["accuracy_percent"] == 50


def test_summary_reports_per_language_metrics():
    rows = [
        {"language": "zh-TW", "score": score_transcript("你好世界", "你好世界", language="zh-TW")},
        {"language": "en", "score": score_transcript("hello world", "hello there", language="en")},
    ]
    summary = summarize_scores(rows)
    assert summary["samples"] == 2
    assert {item["language"] for item in summary["by_language"]} == {"zh-TW", "en"}
    assert summary["exact_match_percent"] == 50
    assert summary["mixed_metrics"] is True
    assert summary["accuracy_percent"] is None
