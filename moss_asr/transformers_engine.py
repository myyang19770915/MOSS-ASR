"""Local Transformers fallback engine for MOSS-Transcribe-Diarize."""

from __future__ import annotations

import copy
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any

from .config import ASRConfig
from .prompt_builder import PromptBuilder


class TransformersEngine:
    """Direct HuggingFace Transformers model inference runner."""

    _model_cache: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
    _model_cache_lock = threading.Lock()

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.model_id = self.config.model_id
        self.model = None
        self.processor = None
        self.device = None
        self.dtype = None

    def load_model(self) -> None:
        """Lazy load the model and processor into memory/GPU."""
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        cache_key = (self.model_id, str(self.device))

        # One server process may create one pipeline per request. Share the
        # loaded model so the GPU weights are loaded only once.
        with self._model_cache_lock:
            cached = self._model_cache.get(cache_key)
            if cached:
                self.model, self.processor, self.dtype = cached
                return

            # Select best available attention implementation
            attn_impl = "sdpa"
            if self.device.type == "cuda":
                try:
                    import flash_attn  # noqa: F401
                    attn_impl = "flash_attention_2"
                except ImportError:
                    attn_impl = "sdpa"

            print(f"Loading MOSS model `{self.model_id}` on {self.device} (dtype={self.dtype}, attn={attn_impl})...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                dtype="auto",
                attn_implementation=attn_impl,
            ).to(dtype=self.dtype).to(self.device).eval()

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self._model_cache[cache_key] = (self.model, self.processor, self.dtype)

    def transcribe(
        self,
        audio_file_path: str | Path,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Transcribe an audio file using local transformers model."""
        self.load_model()
        import torch

        audio_path = Path(audio_file_path).expanduser().resolve()
        prompt_str = prompt or PromptBuilder.from_config(self.config)
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp_val = temperature if temperature is not None else self.config.temperature

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": prompt_str},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Load audio items
        from moss_transcribe_diarize.inference_utils import process_audio_info
        audios = process_audio_info(messages, sampling_rate=self.processor.feature_extractor.sampling_rate)
        
        audio_kwargs = {"device": str(self.device)} if self.device.type == "cuda" else {}
        inputs = self.processor(
            text=text_input,
            audio=audios,
            max_length=131072,
            audio_kwargs=audio_kwargs,
            return_tensors="pt",
        ).to(self.device)

        prompt_len = int(inputs["attention_mask"][0].sum().item())
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = max_tokens
        generation_config.do_sample = temp_val > 0.0
        if generation_config.do_sample:
            generation_config.temperature = temp_val

        start_time = time.time()
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs["audio_chunk_mapping"],
                generation_config=generation_config,
            )

        elapsed = time.time() - start_time
        generated_ids = outputs[0][prompt_len:]
        decoded_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "text": decoded_text,
            "elapsed_sec": elapsed,
            "generated_tokens": int(generated_ids.numel()),
            "prompt_len": prompt_len,
        }
