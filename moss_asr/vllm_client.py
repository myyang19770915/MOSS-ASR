"""vLLM and SGLang OpenAI-compatible Audio Transcription API Client."""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from .config import ASRConfig
from .prompt_builder import PromptBuilder


class VLLMClient:
    """Client for querying vLLM or SGLang audio transcription endpoint."""

    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()
        self.base_url = self.config.vllm_base_url.rstrip("/")
        self.api_key = self.config.api_key or "EMPTY"
        self.timeout = self.config.timeout

    def check_health(self) -> bool:
        """Check if the vLLM server is reachable and responsive."""
        candidates = [
            f"{self.base_url}/health",
            f"{self.base_url}/v1/models",
            f"{self.base_url}/models",
            self.base_url,
        ]
        for url in candidates:
            try:
                req = urllib.request.Request(url, method="GET")
                if self.api_key and self.api_key != "EMPTY":
                    req.add_header("Authorization", f"Bearer {self.api_key}")
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if resp.status in (200, 204):
                        return True
            except Exception:
                continue
        return False

    def transcribe(
        self,
        audio_file_path: str | Path,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Send audio file to vLLM /v1/audio/transcriptions endpoint.
        
        Returns dict with keys: 'text', 'raw_response', 'elapsed_sec', 'usage'.
        """
        path = Path(audio_file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        prompt_str = prompt or PromptBuilder.from_config(self.config)
        max_tokens = max_new_tokens or self.config.max_new_tokens
        temp_val = temperature if temperature is not None else self.config.temperature

        with open(path, "rb") as f:
            audio_bytes = f.read()

        fields: Dict[str, str] = {
            "model": self.config.model_id,
            "prompt": prompt_str,
            "response_format": "json",
            "stream": "true",
            "stream_include_usage": "true",
            "max_completion_tokens": str(int(max_tokens)),
            "temperature": str(float(temp_val)),
        }

        if self.config.language:
            fields["language"] = self.config.language

        url = self._get_transcriptions_url()
        boundary = f"----moss-asr-{uuid.uuid4().hex}"
        body = self._build_multipart_body(
            boundary=boundary,
            fields=fields,
            file_field="file",
            filename=path.name,
            content_type="audio/wav",
            file_bytes=audio_bytes,
        )

        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "MOSS-ASR-Client/1.0",
        }
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        start_time = time.time()

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    res_data = self._read_sse_stream(resp, progress_callback)
                else:
                    raw = resp.read().decode("utf-8")
                    try:
                        res_data = json.loads(raw)
                    except json.JSONDecodeError:
                        res_data = {"text": raw}

            elapsed = time.time() - start_time
            text = res_data.get("text", "")
            return {
                "text": text,
                "raw_response": res_data,
                "elapsed_sec": elapsed,
                "usage": res_data.get("usage", {}),
            }

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM server returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to connect to vLLM server at {url}: {exc.reason}") from exc

    def _get_transcriptions_url(self) -> str:
        if self.base_url.endswith("/audio/transcriptions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/audio/transcriptions"
        return f"{self.base_url}/v1/audio/transcriptions"

    @staticmethod
    def _build_multipart_body(
        boundary: str,
        fields: Dict[str, str],
        file_field: str,
        filename: str,
        content_type: str,
        file_bytes: bytes,
    ) -> bytes:
        chunks: list[bytes] = []
        for key, val in fields.items():
            chunks.extend([
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"),
                val.encode("utf-8"),
                b"\r\n",
            ])
        chunks.extend([
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ])
        return b"".join(chunks)

    @staticmethod
    def _read_sse_stream(
        resp: Any,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        parts: list[str] = []
        usage: dict[str, Any] = {}

        for line_bytes in resp:
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            if "usage" in chunk and isinstance(chunk["usage"], dict):
                usage = chunk["usage"]

            choices = chunk.get("choices") or []
            for choice in choices:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    parts.append(content)
                    if progress_callback:
                        progress_callback(content)

        return {"text": "".join(parts), "usage": usage}
