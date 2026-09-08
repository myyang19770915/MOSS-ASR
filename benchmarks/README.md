# ASR benchmark datasets

Place each dataset in its own folder below `benchmarks/` and provide a `.jsonl` or `.csv` manifest. The API only reads files under this directory; it never uploads the benchmark audio to an external service.

## JSONL format

```json
{"id":"zh-001","audio":"audio/zh-001.wav","text":"這是一段參考逐字稿。","language":"zh-TW"}
```

`audio` must be a path relative to the manifest file. Required fields are `audio` and one of `text`, `reference`, `transcript`, `sentence`, or `normalized_text`. Optional fields are `id` and `language`.

Locale values such as `en-US`, `zh-CN`, and `cmn_Hans_CN` are accepted. The benchmark preserves that locale in reports, while translating it to a MOSS-supported primary language hint (`en`, `zh`) for inference. If MOSS does not support a language hint, the run falls back to automatic language detection.

## CSV format

```csv
id,audio,text,language
en-001,audio/en-001.wav,This is a reference transcript.,en
```

## Suggested public sources

- [Mozilla Common Voice](https://commonvoice.mozilla.org/datasets): diverse, crowd-sourced speech; download only the desired language and split.
- [Google FLEURS](https://huggingface.co/datasets/google/fleurs): 102-language ASR benchmark under CC BY 4.0.
- [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli): 18 transcribed European languages and accented-English test data under CC0.
- [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14): 14-language customer-service speech under CC BY 4.0; useful for short multi-language smoke tests.
- [LibriSpeech ASR](https://huggingface.co/datasets/openslr/librispeech_asr): English read speech under CC BY 4.0; useful as a conventional English ASR reference.

Always keep the original dataset licence, attribution, and any speaker-privacy requirements when downloading or redistributing a subset.
