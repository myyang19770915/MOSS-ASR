"""Transcript Proofreading and Error Correction Engine (Rule-based & LLM-based)."""

from __future__ import annotations

import difflib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Callable

from .config import ProofreadConfig
from .transcript_parser import TranscriptSegment, TranscriptResult


@dataclass
class ProofreadDiff:
    """Represents a single correction difference."""
    segment_idx: int
    original: str
    corrected: str
    change_type: str  # 'homophone', 'punctuation', 'stutter', 'glossary', 'llm'


# Common Chinese homophone / typo corrections in ASR
DEFAULT_HOMOPHONE_MAP = {
    "佈署": "部署",
    "連繫": "聯繫",
    "因該": "應該",
    "在度": "再度",
    "在次": "再次",
    "在見": "再見",
    "以經": "已經",
    "按裝": "安裝",
    "登陸": "登入",
    "帳號密碼": "帳號密碼",
    "訊號": "信號",
    "資詢": "諮詢",
    "幅射": "輻射",
    "針孔": "針孔",
    "銷毀": "銷毀",
}

# Common speech filler words and stutter patterns
STUTTER_PATTERNS = [
    (re.compile(r"(那個\s*){2,}"), "那個 "),
    (re.compile(r"(就是\s*){2,}"), "就是 "),
    (re.compile(r"(然後\s*){2,}"), "然後 "),
    (re.compile(r"(對\s*){3,}"), "對 "),
    (re.compile(r"(嗯\s*){2,}"), "嗯 "),
    (re.compile(r"(啊\s*){2,}"), "啊 "),
    (re.compile(r"([一-龥])\1{2,}"), r"\1"),  # 3+ repeated Chinese characters (e.g. 看看看 -> 看)
]


class RuleBasedProofreader:
    """Fast local rule-based corrector for punctuation, stuttering, and glossary terms."""

    def __init__(
        self,
        custom_replacements: Optional[Dict[str, str]] = None,
        remove_stutter: bool = True,
        normalize_punctuation: bool = True,
    ):
        self.replacements = {**DEFAULT_HOMOPHONE_MAP, **(custom_replacements or {})}
        self.remove_stutter = remove_stutter
        self.normalize_punctuation = normalize_punctuation

    def clean_text(self, text: str) -> str:
        """Apply rule-based cleanups to a single string."""
        if not text:
            return ""

        res = text

        # 1. Custom & Homophone Dictionary Replacement
        for wrong, right in self.replacements.items():
            if wrong in res:
                res = res.replace(wrong, right)

        # 2. Remove Stutter & Filler repetitions
        if self.remove_stutter:
            # Single character stutter e.g. "這..這..這個" or "我、我、我們"
            res = re.sub(r"([一-龥])(?:[\.、，\s]*\1)+", r"\1", res)
            # For Latin text, require a separator and repeat whole words so
            # legitimate double letters in words such as "hello" survive.
            res = re.sub(
                r"\b([a-zA-Z]+)(?:[\.,、，\s]+\1\b)+",
                r"\1",
                res,
                flags=re.IGNORECASE,
            )
            # Repeated phrase patterns
            for pat, repl in STUTTER_PATTERNS:
                res = pat.sub(repl, res)

        # 3. Punctuation Normalization
        if self.normalize_punctuation:
            # Replace double punctuation
            res = re.sub(r"，{2,}", "，", res)
            res = re.sub(r"。{2,}", "。", res)
            res = re.sub(r"！{2,}", "！", res)
            res = re.sub(r"？{2,}", "？", res)
            res = re.sub(r"([，。！？])\s*([，。！？])", r"\2", res)
            # Ensure space between English words and Chinese characters
            res = re.sub(r"([\u4e00-\u9fa5])([a-zA-Z0-9])", r"\1 \2", res)
            res = re.sub(r"([a-zA-Z0-9])([\u4e00-\u9fa5])", r"\1 \2", res)

        return res.strip()

    def process_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Apply rule cleanups across all segments."""
        cleaned_segments: List[TranscriptSegment] = []
        for seg in segments:
            cleaned_text = self.clean_text(seg.text)
            cleaned_segments.append(
                TranscriptSegment(
                    start=seg.start,
                    end=seg.end,
                    speaker=seg.speaker,
                    text=cleaned_text,
                    confidence=seg.confidence,
                )
            )
        return cleaned_segments


class LLMProofreader:
    """Intelligent context-aware proofreading engine powered by LLM (OpenAI-compatible / Qwen / Claude)."""

    SYSTEM_PROMPT = """你是繁體中文（臺灣）語音轉錄文稿的資深審校員。你的工作是找出並修正 ASR 的「聽錯字」，不是改寫、摘要、翻譯或創作。

校對核心目標（請主動逐段檢查，而不是只修英文空格）：
1. 先讀完全部段落，以前後語境、談話主題與說話脈絡判斷字詞是否合理。
2. 優先修正上下文已明確支持的同音字、近音字、形近字、常用詞誤辨、斷詞錯誤與語意不通的字詞。例如僅在語境吻合時：
   - 「機器學息模型」→「機器學習模型」
   - 「明天開視訊會意」→「明天開視訊會議」
   - 「系統顯示聯線逾時」→「系統顯示連線逾時」
   - 「我們要重新佈署服務」→「我們要重新部署服務」
   這些只是判斷方式示例；不可機械套用，也不可依示例捏造內容。
3. 依臺灣繁體中文的慣用語與語境校訂用字，例如「帳號、資料、網路、軟體、伺服器、連線、會議紀錄」。保留原說話者的口語語氣，不要為了文雅而改寫句子。
4. 英文大小寫、英文與中文之間的空格、標點只是低優先的排版修正；若存在高信心的中文 ASR 錯字，必須優先修正中文錯字，不能只做空格調整就結束。
5. 只有在上下文或常識能提供高信心依據時才替換；「在／再、的／得／地、做／作」等可能改變語意的字，沒有足夠依據時保留原文。

不可違反的輸出規則：
1. 每個輸入 index 必須剛好輸出一個項目；index、順序、段落數不可改變，不可合併、拆分、刪除或新增段落。
2. 時間戳與說話人標籤由系統在外部保留；`corrected_text` 不得包含時間戳、`[S01]` 等說話人標籤或序號。
3. 不得捏造、補充、刪除事實、數字、姓名、引用、網址、程式碼、產品名稱或說話者意圖；保留原有語言與口語語氣。
4. 專有名詞、英文、縮寫、檔名、URL、指令與數字須優先忠實保留；只有轉錄錯誤有明確依據時才改。跨段上下文只能協助判斷，不能移動任何文字到其他段落。
5. `changes_made` 必須精簡指出實際修改類型，例如「同音字：會意→會議」、「上下文：學息→學習」；完全未修改才填「無」。

只輸出一個可解析的 JSON 物件，不要 Markdown、程式碼區塊、前言或額外欄位：
{"corrected_segments":[{"index":0,"corrected_text":"...","changes_made":"..."}],"summary":"..."}"""

    def __init__(self, config: ProofreadConfig):
        self.config = config
        self.api_base = (config.llm_api_base or "https://api.openai.com/v1").rstrip("/")
        self.api_key = config.llm_api_key or ""
        self.model = config.llm_model
        self.temperature = config.temperature

    def proofread(
        self,
        segments: List[TranscriptSegment],
        stream_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[List[TranscriptSegment], str]:
        """Proofread a list of transcript segments using an OpenAI-compatible LLM.

        When ``stream_callback`` is supplied, every generated content delta is
        forwarded immediately while the final JSON is still parsed atomically.
        """
        if not segments:
            return [], "No segments to proofread."

        if not self.api_key:
            raise ValueError("LLM API key is required for LLM proofreading.")

        # Prepare segment payload
        payload_segments = [
            {"index": i, "speaker": seg.speaker, "text": seg.text}
            for i, seg in enumerate(segments)
        ]

        glossary_note = (
            "\n可信任熱詞／專有名詞（若語境相符請使用此精確寫法）："
            + json.dumps(self.config.glossary_terms, ensure_ascii=False)
            if self.config.glossary_terms
            else ""
        )
        user_content = (
            f"請校對以下 {len(segments)} 段語音轉錄文稿。請先用全部段落建立語境，再依 index 逐段輸出；"
            f"務必優先找出高信心的中文同音／近音 ASR 錯字、常用詞誤辨與語意不通處，"
            f"不要只處理英文空格或標點。{glossary_note}\n\n"
            f"{json.dumps(payload_segments, ensure_ascii=False, indent=2)}\n\n"
            f"請以標準 JSON 格式輸出：\n"
            f'{{"corrected_segments": [{{"index": 0, "corrected_text": "...", "changes_made": "..."}}], "summary": "校對摘要說明"}}'
        )

        request_body: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": self.config.max_completion_tokens,
        }
        if stream_callback:
            request_body["stream"] = True

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.config.http_user_agent,
        }

        try:
            requested_effort = (self.config.reasoning_effort or "").strip().lower() or None
            transcript_chars = sum(len(segment.text or "") for segment in segments)
            is_long_document = len(segments) >= 100 or transcript_chars >= 5000
            adaptive_applied = bool(
                self.config.adaptive_reasoning
                and is_long_document
                and requested_effort in {"medium", "high"}
            )
            selected_effort = "low" if adaptive_applied else requested_effort
            if adaptive_applied and progress_callback:
                progress_callback(
                    {
                        "phase": "optimized",
                        "message": (
                            f"偵測到長文（{len(segments)} 段），已將推理強度從 "
                            f"{requested_effort} 調整為 low，避免無效重試。"
                        ),
                        "requested_effort": requested_effort,
                        "effective_effort": selected_effort,
                    }
                )
            fallback_effort = {
                "high": "low",
                "medium": "low",
                "minimal": "none",
                "low": "none",
            }.get(selected_effort, "low" if selected_effort is None else None)
            attempts = [selected_effort]
            if fallback_effort and fallback_effort != selected_effort:
                attempts.append(fallback_effort)

            last_diagnostic = ""
            for attempt_index, effort in enumerate(attempts):
                if progress_callback:
                    progress_callback(
                        {
                            "phase": "request",
                            "attempt": attempt_index + 1,
                            "reasoning_effort": effort or "provider-default",
                            "message": (
                                f"送出第 {attempt_index + 1} 次智慧校對請求"
                                f"（推理強度 {effort or 'provider-default'}）。"
                            ),
                        }
                    )
                attempt_body = dict(request_body)
                if effort:
                    attempt_body["reasoning_effort"] = effort
                content_str, saw_reasoning, finish_reason = self._request_content(
                    url=url,
                    headers=headers,
                    request_body=attempt_body,
                    stream_callback=stream_callback,
                    progress_callback=progress_callback,
                )
                if content_str.strip():
                    corrected, summary = self._apply_llm_result(segments, content_str)
                    call_note = (
                        f"LLM 校對呼叫 {attempt_index + 1} 次；有效推理強度 {effort or 'provider-default'}。"
                    )
                    if adaptive_applied:
                        call_note += " 已啟用長文加速。"
                    if attempt_index:
                        summary = (
                            f"模型首次僅回傳推理內容，系統已自動改用 {effort} 推理強度重試。 "
                            f"{summary}"
                        )
                    return corrected, f"{call_note} {summary}"

                reason = "僅收到推理內容" if saw_reasoning else "沒有收到內容"
                finish_note = f"，finish_reason={finish_reason}" if finish_reason else ""
                last_diagnostic = f"{reason}{finish_note}"
                if attempt_index + 1 < len(attempts) and progress_callback:
                    progress_callback(
                        {
                            "phase": "retry",
                            "attempt": attempt_index + 2,
                            "message": (
                                f"第 {attempt_index + 1} 次只收到推理內容，"
                                f"即將改用 {attempts[attempt_index + 1]} 重試。"
                            ),
                        }
                    )

            raise RuntimeError(
                "LiteLLM 模型未輸出最終校對 JSON（"
                f"{last_diagnostic or '空白回應'}）。請降低 reasoning_effort 或縮短單次文稿。"
            )

        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            detail = self._provider_error_detail(body, self.api_key)
            hint = ""
            if exc.code == 401:
                hint = "；請確認 API Key 是否正確且仍有效"
            elif exc.code == 403:
                hint = "；API Key 可能被停用，或沒有此模型／團隊的使用權限"
            raise RuntimeError(f"LiteLLM HTTP {exc.code}: {detail}{hint}") from exc
        except Exception as exc:
            raise RuntimeError(f"LLM proofreading request failed: {exc}") from exc

    def _request_content(
        self,
        url: str,
        headers: Dict[str, str],
        request_body: Dict[str, Any],
        stream_callback: Optional[Callable[[str], None]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ) -> Tuple[str, bool, Optional[str]]:
        """Run one completion request and return final content diagnostics."""
        req = urllib.request.Request(
            url,
            data=json.dumps(request_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.request_timeout_seconds) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if stream_callback and "text/event-stream" in content_type:
                return self._read_sse_stream(resp, stream_callback, progress_callback)

            resp_json = json.loads(resp.read().decode("utf-8"))
            choice = (resp_json.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = message.get("content") or ""
            saw_reasoning = bool(message.get("reasoning_content") or message.get("reasoning"))
            usage_details = ((resp_json.get("usage") or {}).get("completion_tokens_details") or {})
            saw_reasoning = saw_reasoning or bool(usage_details.get("reasoning_tokens"))
            return str(content), saw_reasoning, choice.get("finish_reason")

    @staticmethod
    def _provider_error_detail(body: str, secret: str = "") -> str:
        """Extract a concise OpenAI-compatible provider error without echoing secrets."""
        detail = body.strip()
        try:
            parsed = json.loads(detail)
            error = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("detail") or error)
            else:
                detail = str(error)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        if secret:
            detail = detail.replace(secret, "***")
        detail = re.sub(r"\s+", " ", detail).strip()
        return detail[:800] or "上游服務拒絕請求"

    @staticmethod
    def _read_sse_stream(
        resp: Any,
        stream_callback: Callable[[str], None],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[str, bool, Optional[str]]:
        """Read SSE content while tracking reasoning-only/empty responses."""
        parts: List[str] = []
        saw_reasoning = False
        finish_reason: Optional[str] = None
        reasoning_chars = 0
        last_reasoning_report = 0
        writing_reported = False
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
            usage_details = ((chunk.get("usage") or {}).get("completion_tokens_details") or {})
            saw_reasoning = saw_reasoning or bool(usage_details.get("reasoning_tokens"))
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning_delta:
                    saw_reasoning = True
                    reasoning_chars += len(str(reasoning_delta))
                    if progress_callback and reasoning_chars - last_reasoning_report >= 256:
                        last_reasoning_report = reasoning_chars
                        progress_callback(
                            {
                                "phase": "reasoning",
                                "reasoning_chars": reasoning_chars,
                                "message": f"模型正在分析上下文與可能的 ASR 誤辨（已處理 {reasoning_chars:,} 個推理字元）。",
                            }
                        )
                content = delta.get("content")
                if content:
                    if progress_callback and not writing_reported:
                        writing_reported = True
                        progress_callback(
                            {
                                "phase": "writing",
                                "message": "模型已完成分析，開始串流輸出校對文稿。",
                            }
                        )
                    parts.append(content)
                    stream_callback(content)
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
        return "".join(parts), saw_reasoning, finish_reason

    @staticmethod
    def _apply_llm_result(
        segments: List[TranscriptSegment],
        content_str: str,
    ) -> Tuple[List[TranscriptSegment], str]:
        """Validate model JSON and apply only text changes to existing segments."""
        cleaned_content = content_str.strip()
        if not cleaned_content:
            raise ValueError("模型回應內容為空，沒有可解析的校對 JSON")
        if cleaned_content.startswith("```"):
            cleaned_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned_content).strip()
        try:
            parsed_result = json.loads(cleaned_content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"模型輸出的校對 JSON 不完整或格式錯誤：{exc.msg}（位置 {exc.pos}）"
            ) from exc
        corrected_items = parsed_result.get("corrected_segments", [])
        summary = str(parsed_result.get("summary", "LLM 校對完成。")).strip() or "LLM 校對完成。"

        index_to_text: Dict[int, str] = {}
        for item in corrected_items:
            if not isinstance(item, dict) or "index" not in item:
                continue
            try:
                index = int(item["index"])
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(segments):
                index_to_text[index] = str(item.get("corrected_text", "")).strip()

        new_segments: List[TranscriptSegment] = []
        for i, orig_seg in enumerate(segments):
            corrected_t = index_to_text.get(i) or orig_seg.text
            new_segments.append(
                TranscriptSegment(
                    start=orig_seg.start,
                    end=orig_seg.end,
                    speaker=orig_seg.speaker,
                    text=corrected_t,
                    confidence=orig_seg.confidence,
                )
            )
        return new_segments, summary


class TranscriptProofreadingPipeline:
    """Unified proofreading pipeline combining rule-based & LLM methods."""

    def __init__(self, config: Optional[ProofreadConfig] = None):
        self.config = config or ProofreadConfig()
        self.rule_engine = RuleBasedProofreader(
            custom_replacements=self.config.custom_replacements,
            remove_stutter=self.config.remove_stutter,
            normalize_punctuation=self.config.normalize_punctuation,
        )
        self.llm_engine = LLMProofreader(self.config) if (self.config.enabled and self.config.llm_api_key) else None

    def process(
        self,
        result: TranscriptResult,
        stream_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> TranscriptResult:
        """Run complete proofreading pipeline on a TranscriptResult."""
        # 1. Rule-based pass
        intermediate_segments = self.rule_engine.process_segments(result.segments)
        notes = "已完成規則字典與贅字標點校對。"
        if progress_callback:
            progress_callback(
                {
                    "phase": "rules_complete",
                    "message": "規則字典與贅字標點校對已完成，先顯示初步結果。",
                    "segments": [segment.to_dict() for segment in intermediate_segments],
                }
            )

        # 2. LLM pass (if enabled). ``is_proofread`` means the requested LLM
        # review actually completed, rather than merely that local rules ran.
        llm_succeeded = False
        if self.config.enabled and self.llm_engine:
            try:
                final_segments, llm_notes = self.llm_engine.proofread(
                    intermediate_segments,
                    stream_callback=stream_callback,
                    progress_callback=progress_callback,
                )
                notes = f"{notes} {llm_notes}"
                llm_succeeded = True
            except Exception as e:
                notes = f"{notes} LLM 智慧校對失敗：{e}"
                final_segments = intermediate_segments
        elif self.config.enabled:
            final_segments = intermediate_segments
            notes = f"{notes} LLM 智慧校對未執行：未提供 API Key。"
        else:
            final_segments = intermediate_segments

        return TranscriptResult(
            raw_text=result.raw_text,
            segments=final_segments,
            audio_duration=result.audio_duration,
            elapsed_time=result.elapsed_time,
            model=result.model,
            is_proofread=llm_succeeded if self.config.enabled else True,
            proofread_notes=notes,
        )
