"use strict";

const $ = (selector) => document.querySelector(selector);
const state = {
  config: null,
  currentResult: null,
  originalResult: null,
  baseSegments: [],
  correctionItems: [],
  transcriptStream: "",
  proofreadStream: "",
  proofreadRequested: false,
  startedAt: 0,
  timerId: null,
  toastId: null,
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  initReaderSize();
  bindEvents();
  await loadConfig();
  await checkReadiness();
}

function bindEvents() {
  const input = $("#mediaFile");
  const dropZone = $("#dropZone");

  dropZone.addEventListener("click", () => input.click());
  input.addEventListener("change", () => selectFile(input.files[0]));
  for (const name of ["dragenter", "dragover"]) {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.add("dragging");
    });
  }
  for (const name of ["dragleave", "drop"]) {
    dropZone.addEventListener(name, (event) => {
      event.preventDefault();
      dropZone.classList.remove("dragging");
    });
  }
  dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    selectFile(file);
  });

  $("#proofread").addEventListener("change", updateProofreadOptions);
  $("#toggleKey").addEventListener("click", toggleApiKey);
  $("#testLlmButton").addEventListener("click", testLlmConnection);
  $("#dismissError").addEventListener("click", clearError);
  $("#changedOnly").addEventListener("change", renderDiff);
  $("#transcribeForm").addEventListener("submit", submitTranscription);
  $("#copyResultButton").addEventListener("click", copyResult);
  $("#downloadSrt").addEventListener("click", () => downloadFormat("srt"));
  $("#downloadJson").addEventListener("click", () => downloadFormat("json"));
  document.querySelectorAll("[data-reader-size]").forEach((button) => {
    button.addEventListener("click", () => setReaderSize(button.dataset.readerSize));
  });
}

function initReaderSize() {
  let saved = "comfortable";
  try { saved = localStorage.getItem("moss-reader-size") || saved; } catch (_) {}
  if (!["compact", "comfortable", "large"].includes(saved)) saved = "comfortable";
  setReaderSize(saved, false);
}

function setReaderSize(size, persist = true) {
  document.documentElement.dataset.readerSize = size;
  document.querySelectorAll("[data-reader-size]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.readerSize === size));
  });
  if (persist) {
    try { localStorage.setItem("moss-reader-size", size); } catch (_) {}
  }
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    if (!response.ok) throw new Error("無法讀取服務設定");
    state.config = await response.json();
    $("#vllmUrl").value = state.config.vllm_url || "";
    $("#llmUrl").value = state.config.litellm_url || "";
    $("#llmModel").value = state.config.litellm_model || "";
    $("#versionBadge").textContent = `API v${state.config.version}`;
    $("#dropHint").textContent = `或點擊選擇檔案 · 上限 ${state.config.max_upload_mb} MB · 最長 ${state.config.max_audio_minutes} 分鐘`;
  } catch (error) {
    showError(error.message);
  }
}

async function checkReadiness() {
  const badge = $("#serviceBadge");
  const label = $("#serviceText");
  try {
    const response = await fetch("/readyz", { cache: "no-store" });
    if (!response.ok) throw new Error("not ready");
    badge.className = "service-badge ready";
    label.textContent = "模型服務就緒";
  } catch (_) {
    badge.className = "service-badge error";
    label.textContent = "模型服務未就緒";
  }
}

function selectFile(file) {
  clearError();
  const zone = $("#dropZone");
  const meta = $("#fileMeta");
  if (!file) {
    zone.classList.remove("has-file");
    meta.classList.add("hidden");
    return;
  }

  const maxBytes = state.config?.max_upload_bytes || 100 * 1024 * 1024;
  if (file.size > maxBytes) {
    showError(`檔案大小為 ${formatBytes(file.size)}，超過 ${formatBytes(maxBytes)} 上限。`);
    $("#mediaFile").value = "";
    zone.classList.remove("has-file");
    meta.classList.add("hidden");
    return;
  }

  zone.classList.add("has-file");
  $("#dropTitle").textContent = file.name;
  meta.textContent = `${formatBytes(file.size)} · ${file.type || "媒體檔案"}`;
  meta.classList.remove("hidden");
  setWorkflow("upload");
}

function updateProofreadOptions() {
  const enabled = $("#proofread").checked;
  $("#proofreadOptions").classList.toggle("hidden", !enabled);
  $("#submitText").textContent = enabled ? "開始轉錄與校對" : "開始轉錄";
}

function toggleApiKey() {
  const input = $("#llmKey");
  const show = input.type === "password";
  input.type = show ? "text" : "password";
  $("#toggleKey").textContent = show ? "隱藏" : "顯示";
}

async function testLlmConnection() {
  clearError();
  const button = $("#testLlmButton");
  const status = $("#llmTestStatus");
  const form = new FormData();
  form.append("llm_url", $("#llmUrl").value.trim());
  form.append("llm_key", $("#llmKey").value);
  form.append("llm_model", $("#llmModel").value.trim());
  button.disabled = true;
  status.className = "connection-status";
  status.textContent = "正在驗證 Key 與模型清單…";
  try {
    const response = await fetch("/api/proofread/check", { method: "POST", body: form });
    if (!response.ok) throw new Error(await responseError(response));
    const result = await response.json();
    status.className = `connection-status ${result.model_available ? "success" : "warning"}`;
    status.textContent = result.model_available
      ? `連線成功，${result.model} 可使用。`
      : `Key 有效，但找不到 ${result.model}；請改用帳號允許的模型。`;
  } catch (error) {
    status.className = "connection-status error";
    status.textContent = error.message || "LiteLLM 連線測試失敗";
    showError(status.textContent);
  } finally {
    button.disabled = false;
  }
}

async function submitTranscription(event) {
  event.preventDefault();
  clearError();
  const file = $("#mediaFile").files[0];
  if (!file) {
    showError("請先選擇音訊或影片檔案。");
    $("#dropZone").focus();
    return;
  }

  state.proofreadRequested = $("#proofread").checked;
  resetResults();
  setBusy(true);
  setWorkflow("transcribe");
  updateProgress("GPU 佇列", "正在上傳並建立工作…", 2);

  const form = new FormData();
  form.append("file", file);
  form.append("vllm_url", $("#vllmUrl").value.trim());
  form.append("include_timestamps", String($("#timestamps").checked));
  form.append("include_speakers", String($("#diarize").checked));
  form.append("hotwords", $("#hotwords").value.trim());
  form.append("proofread", String(state.proofreadRequested));
  form.append("llm_url", $("#llmUrl").value.trim());
  form.append("llm_key", $("#llmKey").value);
  form.append("llm_model", $("#llmModel").value.trim());
  form.append("reasoning_effort", $("#reasoningEffort").value);
  form.append("adaptive_proofread", String($("#adaptiveProofread").checked));

  try {
    const response = await fetch("/api/transcribe/stream", { method: "POST", body: form });
    if (!response.ok) throw new Error(await responseError(response));
    await consumeSse(response);
  } catch (error) {
    showError(error.message || "轉錄請求失敗");
    setPanelState("transcriptState", "處理失敗", "error");
    if (state.proofreadRequested) setPanelState("proofreadState", "未完成", "error");
    updateProgress("發生錯誤", "工作未完成，請檢查設定後重試。", 0);
  } finally {
    setBusy(false);
  }
}

async function responseError(response) {
  const text = await response.text();
  try {
    const parsed = JSON.parse(text);
    return parsed.detail || text || `HTTP ${response.status}`;
  } catch (_) {
    return text || `HTTP ${response.status}`;
  }
}

function resetResults() {
  state.currentResult = null;
  state.originalResult = null;
  state.baseSegments = [];
  state.correctionItems = [];
  state.transcriptStream = "";
  state.proofreadStream = "";

  $("#resultCard").classList.remove("hidden");
  $("#proofreadPanel").classList.toggle("hidden", !state.proofreadRequested);
  $("#diffCard").classList.add("hidden");
  $("#diffOutput").innerHTML = "";
  $("#diffSummary").textContent = "等待校對串流";
  $("#originalOutput").textContent = "模型產生的原始內容會即時顯示在這裡。";
  $("#proofreadOutput").textContent = "完整轉錄完成後，將開始智慧校對。";
  $("#resultStats").textContent = "工作處理中";
  setPanelState("transcriptState", "等待模型輸出", "waiting");
  setPanelState("proofreadState", "等待原始逐字稿", "waiting");
  setActionButtons(false);
  state.startedAt = Date.now();
  clearInterval(state.timerId);
  state.timerId = setInterval(updateElapsed, 1000);
  $("#resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

function setBusy(busy) {
  $("#submitBtn").disabled = busy;
  $("#submitText").textContent = busy ? "處理中…" : (state.proofreadRequested ? "開始轉錄與校對" : "開始轉錄");
  $("#progressPanel").classList.toggle("hidden", !busy && !state.currentResult);
  if (!busy) {
    clearInterval(state.timerId);
    state.timerId = null;
  }
}

function updateElapsed() {
  if (!state.startedAt) return;
  const seconds = Math.max(0, Math.floor((Date.now() - state.startedAt) / 1000));
  $("#elapsedText").textContent = `已經過 ${formatDuration(seconds)}`;
}

function updateProgress(stage, message, percent) {
  const normalized = Math.max(0, Math.min(100, Math.round(percent || 0)));
  $("#progressPanel").classList.remove("hidden");
  $("#progressStage").textContent = stage;
  $("#statusText").textContent = message;
  $("#progressPercent").textContent = `${normalized}%`;
  $("#progressBar").style.width = `${normalized}%`;
}

async function consumeSse(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, "\n");
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      processSseEvent(block);
    }
    if (done) break;
  }
  if (buffer.trim()) processSseEvent(buffer);
}

function processSseEvent(block) {
  const lines = block.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event:")) || "event: message";
  const event = eventLine.slice(6).trim();
  const data = lines.filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
  if (!data) return;

  let payload;
  try {
    payload = JSON.parse(data);
  } catch (_) {
    throw new Error("服務回傳了無法解析的串流資料");
  }

  if (event === "queued") {
    updateProgress("GPU 佇列", payload.message || "等待 GPU 工作槽", 3);
    return;
  }
  if (event === "status") {
    const progress = Number(payload.progress || 0) * 100;
    updateProgress(progress >= 88 ? "後處理" : "語音轉錄", payload.message || "處理中", progress);
    return;
  }
  if (event === "transcription_delta") {
    state.transcriptStream += payload.delta || "";
    const output = $("#originalOutput");
    output.textContent = state.transcriptStream;
    output.scrollTop = output.scrollHeight;
    setPanelState("transcriptState", "即時串流中", "streaming");
    return;
  }
  if (event === "transcription_chunk_complete") {
    const total = payload.chunks_total || 1;
    const current = payload.chunk_index || 1;
    $("#resultStats").textContent = `已完成音訊分段 ${current} / ${total}`;
    return;
  }
  if (event === "transcription_complete") {
    state.originalResult = payload;
    state.baseSegments = payload.segments || [];
    $("#originalOutput").textContent = formatTranscript(payload);
    setPanelState("transcriptState", "原始轉錄完成", "complete");
    $("#resultStats").textContent = `${state.baseSegments.length} 段 · 音訊 ${formatDuration(payload.audio_duration || 0)}`;
    if (state.proofreadRequested) {
      setWorkflow("proofread");
      updateProgress("智慧校對", "原始轉錄完成，準備送往校對模型…", 88);
    }
    return;
  }
  if (event === "proofread_started") {
    $("#proofreadPanel").classList.remove("hidden");
    $("#proofreadOutput").textContent = "正在建立上下文並檢查中文 ASR 誤辨…";
    setPanelState("proofreadState", `${payload.model || "LLM"} 串流中`, "streaming");
    updateProgress("智慧校對", `正在使用 ${payload.model || "LLM"} 逐段校訂`, 91);
    return;
  }
  if (event === "proofread_progress") {
    const phase = payload.phase || "working";
    const message = payload.message || "智慧校對處理中…";
    if (phase === "rules_complete") {
      state.correctionItems = (payload.segments || []).map((segment, index) => ({
        index,
        corrected_text: segment.text || "",
        changes_made: segment.text === state.baseSegments[index]?.text ? "無" : "規則字典／標點初步校對",
      }));
      renderProofreadPreview();
      renderDiff();
      setPanelState("proofreadState", "規則初稿已完成", "streaming");
      updateProgress("智慧校對", message, 90);
      return;
    }
    if (phase === "writing") {
      state.proofreadStream = "";
      state.correctionItems = [];
      $("#proofreadOutput").textContent = message;
      setPanelState("proofreadState", "正在串流校對文字", "streaming");
      updateProgress("智慧校對", message, 94);
      return;
    }
    if (phase === "reasoning") {
      if (!state.proofreadStream && state.correctionItems.length) {
        setPanelState("proofreadState", "模型正在分析上下文", "streaming");
      }
      const activity = Math.min(2, Math.log10(Number(payload.reasoning_chars || 0) + 1));
      updateProgress("智慧校對", message, 91 + activity);
      return;
    }
    if (phase === "retry") {
      setPanelState("proofreadState", `自動重試第 ${payload.attempt || 2} 次`, "streaming");
      updateProgress("智慧校對", message, 92);
      return;
    }
    if (phase === "optimized") {
      setPanelState("proofreadState", "已啟用長文加速", "streaming");
      updateProgress("智慧校對", message, 91);
      return;
    }
    if (phase === "request") {
      setPanelState("proofreadState", `校對請求 #${payload.attempt || 1}`, "streaming");
      updateProgress("智慧校對", message, 92);
      return;
    }
    updateProgress("智慧校對", message, 92);
    return;
  }
  if (event === "proofread_unavailable") {
    const detail = payload.detail || "智慧校對服務無法使用";
    $("#proofreadOutput").textContent = detail;
    setPanelState("proofreadState", "無法開始校對", "error");
    showError(detail);
    return;
  }
  if (event === "proofread_delta") {
    state.proofreadStream += payload.delta || "";
    const parsed = extractCompletedCorrectionItems(state.proofreadStream);
    if (parsed.length > state.correctionItems.length) {
      state.correctionItems = parsed;
      renderProofreadPreview();
      renderDiff();
    } else if (!state.correctionItems.length) {
      $("#proofreadOutput").textContent = state.proofreadStream;
    }
    const ratio = state.baseSegments.length ? state.correctionItems.length / state.baseSegments.length : 0;
    updateProgress("智慧校對", `已收到 ${state.correctionItems.length} / ${state.baseSegments.length || "?"} 段校對結果`, 91 + ratio * 7);
    return;
  }
  if (event === "complete") {
    completeResult(payload);
    return;
  }
  if (event === "error") {
    throw new Error(payload.detail || "轉錄請求失敗");
  }
}

function completeResult(payload) {
  state.currentResult = payload;
  state.currentResult.display_text = formatTranscript(payload);

  if (!state.originalResult) {
    state.originalResult = payload;
    state.baseSegments = payload.segments || [];
    $("#originalOutput").textContent = formatTranscript(payload);
  }

  if (state.proofreadRequested) {
    if (payload.is_proofread) {
      state.correctionItems = (payload.segments || []).map((segment, index) => ({
        index,
        corrected_text: segment.text || "",
        changes_made: state.correctionItems[index]?.changes_made || (segment.text === state.baseSegments[index]?.text ? "無" : "模型校訂"),
      }));
      $("#proofreadOutput").textContent = formatTranscript(payload);
      setPanelState("proofreadState", "智慧校對完成", "complete");
      renderDiff();
    } else {
      const notes = payload.proofread_notes || "智慧校對未完成";
      $("#proofreadOutput").textContent = notes;
      setPanelState("proofreadState", "校對未完成", "error");
      showError(notes);
    }
  }

  setPanelState("transcriptState", "原始轉錄完成", "complete");
  setWorkflow("complete");
  updateProgress("完成", state.proofreadRequested && payload.is_proofread ? "轉錄與智慧校對已完成" : "語音轉錄已完成", 100);
  $("#resultStats").textContent = [
    `音訊 ${formatDuration(payload.audio_duration || 0)}`,
    `處理 ${formatDuration(payload.elapsed_time || 0)}`,
    `${(payload.segments || []).length} 段`,
    state.proofreadRequested ? (payload.is_proofread ? "智慧校對完成" : "智慧校對未完成") : "未啟用智慧校對",
  ].join(" · ");
  setActionButtons(true);
  showToast("處理完成，結果已可複製或下載。");
}

function extractCompletedCorrectionItems(stream) {
  const keyAt = stream.indexOf('"corrected_segments"');
  if (keyAt < 0) return [];
  const arrayAt = stream.indexOf("[", keyAt);
  if (arrayAt < 0) return [];

  const items = [];
  let cursor = arrayAt + 1;
  while (cursor < stream.length) {
    while (/[\s,]/.test(stream[cursor] || "")) cursor += 1;
    if (stream[cursor] === "]") break;
    if (stream[cursor] !== "{") break;

    const start = cursor;
    let depth = 0;
    let inString = false;
    let escaped = false;
    let end = -1;
    for (; cursor < stream.length; cursor += 1) {
      const char = stream[cursor];
      if (inString) {
        if (escaped) escaped = false;
        else if (char === "\\") escaped = true;
        else if (char === '"') inString = false;
        continue;
      }
      if (char === '"') inString = true;
      else if (char === "{") depth += 1;
      else if (char === "}") {
        depth -= 1;
        if (depth === 0) { end = cursor + 1; cursor = end; break; }
      }
    }
    if (end < 0) break;
    try {
      const item = JSON.parse(stream.slice(start, end));
      const index = Number(item.index);
      if (Number.isInteger(index) && typeof item.corrected_text === "string") {
        items[index] = {
          index,
          corrected_text: item.corrected_text,
          changes_made: String(item.changes_made || "").trim() || "未提供說明",
        };
      }
    } catch (_) {
      break;
    }
  }
  return items.filter(Boolean);
}

function renderProofreadPreview() {
  const lines = state.correctionItems.map((item) => {
    const reason = item.changes_made && item.changes_made !== "無" ? `\n   修改：${item.changes_made}` : "";
    return `[段落 ${item.index + 1}] ${item.corrected_text}${reason}`;
  });
  const output = $("#proofreadOutput");
  output.textContent = lines.join("\n\n");
  output.scrollTop = output.scrollHeight;
}

function renderDiff() {
  if (!state.baseSegments.length || !state.correctionItems.length) return;
  const changedOnly = $("#changedOnly").checked;
  const rows = [];
  let changed = 0;
  let compared = 0;

  for (const item of state.correctionItems) {
    const source = state.baseSegments[item.index];
    if (!source) continue;
    compared += 1;
    const before = source.text || "";
    const after = item.corrected_text || "";
    const isChanged = before !== after;
    if (isChanged) changed += 1;
    if (changedOnly && !isChanged) continue;

    const diff = buildTextDiff(before, after);
    const stamp = source.start != null && source.end != null
      ? `${formatClock(source.start)} → ${formatClock(source.end)}`
      : `第 ${item.index + 1} 段`;
    const speaker = source.speaker ? ` · ${escapeHtml(source.speaker)}` : "";
    const reason = item.changes_made && item.changes_made !== "無"
      ? `<span class="diff-reason">${escapeHtml(item.changes_made)}</span>` : "";

    rows.push(`<article class="diff-row ${isChanged ? "changed" : "unchanged"}">
      <div class="diff-meta">
        <span>#${item.index + 1} · ${stamp}${speaker}</span>
        <span class="diff-status">${isChanged ? "已修正" : "無變更"}${reason}</span>
      </div>
      <div class="diff-columns">
        <div class="diff-text"><small>原始</small>${diff.before}</div>
        <div class="diff-text"><small>校對後</small>${diff.after}</div>
      </div>
    </article>`);
  }

  $("#diffCard").classList.remove("hidden");
  $("#diffOutput").innerHTML = rows.length ? rows.join("") : '<div class="diff-empty">目前沒有符合篩選條件的段落。</div>';
  $("#diffSummary").textContent = `已比對 ${compared} / ${state.baseSegments.length} 段 · ${changed} 段修改`;
}

function buildTextDiff(before, after) {
  if (before === after) {
    const safe = escapeHtml(before);
    return { before: safe, after: safe };
  }
  const left = tokenize(before);
  const right = tokenize(after);
  if (left.length * right.length > 160000) return simpleTextDiff(before, after);

  const matrix = Array.from({ length: left.length + 1 }, () => new Uint16Array(right.length + 1));
  for (let i = left.length - 1; i >= 0; i -= 1) {
    for (let j = right.length - 1; j >= 0; j -= 1) {
      matrix[i][j] = left[i] === right[j]
        ? matrix[i + 1][j + 1] + 1
        : Math.max(matrix[i + 1][j], matrix[i][j + 1]);
    }
  }

  const parts = [];
  let i = 0;
  let j = 0;
  while (i < left.length || j < right.length) {
    if (i < left.length && j < right.length && left[i] === right[j]) {
      pushDiffPart(parts, "same", left[i]); i += 1; j += 1;
    } else if (j < right.length && (i === left.length || matrix[i][j + 1] >= matrix[i + 1][j])) {
      pushDiffPart(parts, "add", right[j]); j += 1;
    } else {
      pushDiffPart(parts, "delete", left[i]); i += 1;
    }
  }

  const beforeHtml = parts.filter((part) => part.type !== "add").map((part) =>
    part.type === "delete" ? `<span class="diff-delete">${escapeHtml(part.text)}</span>` : escapeHtml(part.text)
  ).join("");
  const afterHtml = parts.filter((part) => part.type !== "delete").map((part) =>
    part.type === "add" ? `<span class="diff-add">${escapeHtml(part.text)}</span>` : escapeHtml(part.text)
  ).join("");
  return { before: beforeHtml, after: afterHtml };
}

function tokenize(text) {
  if (typeof Intl.Segmenter === "function") {
    const segmenter = new Intl.Segmenter("zh-Hant", { granularity: "word" });
    return Array.from(segmenter.segment(text), (entry) => entry.segment);
  }
  return Array.from(text);
}

function pushDiffPart(parts, type, text) {
  const last = parts[parts.length - 1];
  if (last?.type === type) last.text += text;
  else parts.push({ type, text });
}

function simpleTextDiff(before, after) {
  let prefix = 0;
  while (prefix < before.length && prefix < after.length && before[prefix] === after[prefix]) prefix += 1;
  let leftEnd = before.length - 1;
  let rightEnd = after.length - 1;
  while (leftEnd >= prefix && rightEnd >= prefix && before[leftEnd] === after[rightEnd]) {
    leftEnd -= 1; rightEnd -= 1;
  }
  const head = escapeHtml(before.slice(0, prefix));
  const tail = escapeHtml(before.slice(leftEnd + 1));
  const removed = escapeHtml(before.slice(prefix, leftEnd + 1));
  const added = escapeHtml(after.slice(prefix, rightEnd + 1));
  return {
    before: `${head}<span class="diff-delete">${removed}</span>${tail}`,
    after: `${head}<span class="diff-add">${added}</span>${tail}`,
  };
}

function formatTranscript(payload) {
  return $("#timestamps").checked
    ? (payload.srt_text || payload.formatted_text || payload.full_text || "")
    : (payload.formatted_text || payload.full_text || "");
}

function setPanelState(id, label, status) {
  const element = $(`#${id}`);
  element.textContent = label;
  element.className = `panel-state ${status}`;
}

function setWorkflow(step) {
  const order = ["upload", "transcribe", "proofread"];
  const activeIndex = step === "complete" ? order.length : order.indexOf(step);
  document.querySelectorAll(".workflow-step").forEach((element, index) => {
    element.classList.toggle("complete", index < activeIndex || step === "complete");
    element.classList.toggle("active", index === activeIndex);
  });
}

function setActionButtons(enabled) {
  $("#copyResultButton").disabled = !enabled;
  $("#downloadSrt").disabled = !enabled;
  $("#downloadJson").disabled = !enabled;
}

async function copyResult() {
  if (!state.currentResult) return;
  try {
    await navigator.clipboard.writeText(state.currentResult.display_text || state.currentResult.formatted_text || "");
    showToast($("#timestamps").checked ? "已複製 SRT。" : "已複製逐字稿。");
  } catch (_) {
    showError("瀏覽器未允許剪貼簿操作，請手動選取結果複製。");
  }
}

function downloadFormat(format) {
  if (!state.currentResult) return;
  const file = $("#mediaFile").files[0];
  const base = (file?.name || "transcript").replace(/\.[^.]+$/, "").replace(/[^\w\u3400-\u9fff-]+/g, "_");
  let text;
  let mime;
  if (format === "json") {
    text = JSON.stringify(state.currentResult, null, 2);
    mime = "application/json;charset=utf-8";
  } else {
    text = state.currentResult.srt_text || state.currentResult.formatted_text || "";
    mime = "application/x-subrip;charset=utf-8";
  }
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${base}.${format}`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function showError(message) {
  $("#errorText").textContent = message;
  $("#errorBanner").classList.remove("hidden");
}

function clearError() {
  $("#errorBanner").classList.add("hidden");
  $("#errorText").textContent = "";
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(state.toastId);
  state.toastId = setTimeout(() => toast.classList.remove("show"), 2600);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[character]);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function formatDuration(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  if (hours) return `${hours} 小時 ${minutes} 分 ${remainder} 秒`;
  if (minutes) return `${minutes} 分 ${remainder} 秒`;
  return `${remainder} 秒`;
}

function formatClock(seconds) {
  const total = Math.max(0, Number(seconds) || 0);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = (total % 60).toFixed(2).padStart(5, "0");
  return hours ? `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${secs}` : `${String(minutes).padStart(2, "0")}:${secs}`;
}
