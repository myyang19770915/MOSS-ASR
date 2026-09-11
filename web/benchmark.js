"use strict";

const $ = (selector) => document.querySelector(selector);
const state = { config: null, catalog: null, preview: null, activeManifest: "", rows: [], finalPayload: null, startedAt: 0, timerId: null };

document.addEventListener("DOMContentLoaded", init);

async function init() {
  bindEvents();
  try {
    const [configResponse, catalogResponse] = await Promise.all([fetch("/api/config"), fetch("/api/benchmark/catalog")]);
    if (!configResponse.ok || !catalogResponse.ok) throw new Error("無法取得服務設定");
    state.config = await configResponse.json();
    state.catalog = await catalogResponse.json();
    $("#versionBadge").textContent = `API v${state.config.version}`;
    $("#vllmUrl").value = state.config.vllm_url || "";
    $("#modelId").value = state.config.model_id || "";
    $("#maxSamples").max = state.catalog.max_samples;
    $("#maxSamplesHint").textContent = `單次最多 ${state.catalog.max_samples} 筆；先以小樣本驗證。`;
    renderCatalog();
    renderDatasetCards();
    renderAdvancedSummary();
    setPipelineStage("dataset");
  } catch (error) {
    showError(error.message || "初始化 Benchmark 頁面失敗");
  }
}

function bindEvents() {
  $("#benchmarkForm").addEventListener("submit", runBenchmark);
  $("#refreshCatalog").addEventListener("click", refreshCatalog);
  $("#manifest").addEventListener("change", () => {
    renderDatasetLanguageOptions();
    updateRunAvailability();
    setPipelineStage("dataset");
    loadPreview();
  });
  $("#datasetLanguage").addEventListener("change", loadPreview);
  $("#languageOverride").addEventListener("change", renderOverrideNotice);
  ["languageOverride", "maxChunkSec", "vllmUrl", "modelId"].forEach((id) => $("#" + id).addEventListener("input", renderAdvancedSummary));
  $("#dismissError").addEventListener("click", clearError);
  $("#downloadJson").addEventListener("click", () => download("json"));
  $("#downloadCsv").addEventListener("click", () => download("csv"));
}

async function refreshCatalog() {
  try {
    $("#refreshCatalog").disabled = true;
    const response = await fetch("/api/benchmark/catalog");
    if (!response.ok) throw new Error(await responseText(response));
    state.catalog = await response.json();
    renderCatalog();
    renderDatasetCards();
    if ($("#manifest").value) await loadPreview();
  } catch (error) {
    showError(error.message || "重新掃描資料集失敗");
  } finally {
    $("#refreshCatalog").disabled = false;
  }
}

function renderCatalog() {
  const manifests = state.catalog?.manifests || [];
  const select = $("#manifest");
  select.innerHTML = '<option value="">選擇已掛載的 manifest</option>' + manifests.map((item) =>
    `<option value="${escapeHtml(item.name)}">${escapeHtml(item.name)} · ${formatSampleCount(item.samples)} · ${formatBytes(item.bytes)}${item.invalid ? " · 格式需修正" : ""}</option>`
  ).join("");
  const total = manifests.reduce((sum, item) => sum + (Number(item.samples) || 0), 0);
  $("#manifestHint").textContent = manifests.length
    ? `找到 ${manifests.length} 個資料集 manifest，共 ${total} 筆可跑分樣本。音檔不會離開本機掛載目錄。`
    : "尚未找到 manifest。請將資料集放到 benchmarks/ 後按「重新掃描」。";
  $("#manifestHint").classList.toggle("ready", manifests.length > 0);
  $("#runReadiness").textContent = manifests.length
    ? `✓ ${manifests.length} 個 manifest 已就緒 · ${formatSampleCount(total)}`
    : "尚未發現可用 manifest";
  $("#runReadiness").classList.toggle("ready", manifests.length > 0);
  renderDatasetLanguageOptions();
  updateRunAvailability();
}

function updateRunAvailability() {
  const button = $("#runButton");
  const manifest = $("#manifest").value;
  const info = selectedManifestInfo();
  const ready = Boolean(manifest && info && !info.invalid);
  button.disabled = !ready;
  button.title = ready ? "使用目前設定開始跑分" : "請先選擇一個 benchmark manifest";
}

function renderAdvancedSummary() {
  const language = $("#languageOverride").value;
  const duration = $("#maxChunkSec").value || "—";
  $("#advancedSummary").textContent = `${language === "auto" ? "自動提示" : language} · ${Number(duration).toLocaleString("zh-TW")} 秒切片`;
}

function setPipelineStage(stage) {
  const order = ["dataset", "transcribing", "scoring"];
  const activeIndex = order.indexOf(stage);
  document.querySelectorAll("[data-pipeline-step]").forEach((element, index) => {
    element.classList.toggle("active", index === activeIndex);
    element.classList.toggle("complete", activeIndex > index || stage === "complete");
  });
}

function selectedManifestInfo() {
  const manifest = $("#manifest").value;
  return (state.catalog?.manifests || []).find((item) => item.name === manifest) || null;
}

function renderDatasetLanguageOptions() {
  const select = $("#datasetLanguage");
  const hint = $("#datasetLanguageHint");
  const selected = select.value;
  const info = selectedManifestInfo();
  if (!info || info.invalid) {
    select.innerHTML = '<option value="auto">先選擇 manifest</option>';
    select.disabled = true;
    hint.textContent = "只會跑選定語言的樣本。";
    return;
  }
  const languages = Object.entries(info.languages || {}).sort(([left], [right]) => left.localeCompare(right));
  select.disabled = false;
  select.innerHTML = '<option value="auto">全部語言（不篩選）</option>' + languages.map(([language, count]) =>
    `<option value="${escapeAttribute(language)}">${escapeHtml(language)} · ${formatSampleCount(count)}</option>`
  ).join("");
  select.value = languages.some(([language]) => language === selected) ? selected : "auto";
  hint.textContent = languages.length
    ? `此資料集有 ${languages.length} 種語言；選取後只跑該語言的樣本。`
    : "此資料集沒有可用的語言標記。";
}

async function loadPreview() {
  const manifest = $("#manifest").value;
  state.preview = null;
  $("#datasetPreview").classList.add("hidden");
  if (!manifest) return;
  try {
    const language = $("#datasetLanguage").value;
    const response = await fetch(`/api/benchmark/preview?manifest=${encodeURIComponent(manifest)}&language=${encodeURIComponent(language)}&limit=4`);
    if (!response.ok) throw new Error(await responseText(response));
    state.preview = await response.json();
    renderPreview();
  } catch (error) {
    showError(error.message || "無法讀取資料集預覽");
  }
}

function renderPreview() {
  const preview = state.preview;
  if (!preview) return;
  $("#datasetPreview").classList.remove("hidden");
  $("#previewTitle").textContent = preview.dataset || preview.manifest;
  const languageSummary = Object.entries(preview.languages || {}).map(([language, count]) => `${language} × ${count}`).join(" · ");
  const filterText = preview.selected_language === "auto" ? "全部語言" : `僅 ${preview.selected_language}`;
  $("#previewMeta").textContent = `${filterText} · ${formatSampleCount(preview.samples)}／共 ${formatSampleCount(preview.total_samples)} · ${languageSummary || "未標記語言"}`;
  $("#previewCount").textContent = formatSampleCount(preview.samples);
  $("#previewSamples").innerHTML = preview.preview.map((sample) => `<article class="preview-sample">
    <div class="preview-sample-top"><strong title="${escapeAttribute(sample.id)}">${escapeHtml(sample.id)}</strong><span>${escapeHtml(sample.language)}</span></div>
    <p>${escapeHtml(sample.reference)}</p>
    <audio controls preload="metadata" src="${escapeAttribute(sample.audio_url)}">此瀏覽器不支援音訊播放。</audio>
  </article>`).join("");
  renderOverrideNotice();
}

function renderOverrideNotice() {
  const notice = $("#languageOverrideNotice");
  const override = $("#languageOverride").value;
  const datasetLanguage = $("#datasetLanguage").value;
  const languages = Object.keys(state.preview?.languages || {});
  const isMultilingual = languages.length > 1;
  if (override !== "auto" && datasetLanguage !== "auto" && override !== datasetLanguage) {
    notice.textContent = `目前資料只篩選「${datasetLanguage}」，但模型提示指定為「${override}」。若非刻意測試錯誤提示，建議改為「自動」或相同語言。`;
    notice.classList.remove("hidden");
  } else if (override !== "auto" && datasetLanguage === "auto" && isMultilingual) {
    notice.textContent = `目前選擇「${override}」全域語言提示，會套用至 ${languages.length} 種資料集語言，也會影響 WER／CER 判定；若要依每筆語言跑分，請選擇「自動」。`;
    notice.classList.remove("hidden");
  } else {
    notice.classList.add("hidden");
    notice.textContent = "";
  }
}

function renderDatasetCards() {
  const datasets = state.catalog?.datasets || [];
  $("#datasetCards").innerHTML = datasets.map((dataset) => `<article class="dataset-card">
    <h3>${escapeHtml(dataset.name)}</h3>
    <dl><dt>語言</dt><dd>${escapeHtml(dataset.languages)}</dd><dt>授權</dt><dd>${escapeHtml(dataset.license)}</dd><dt>適用情境</dt><dd>${escapeHtml(dataset.best_for)}</dd></dl>
    <a href="${escapeAttribute(dataset.url)}" target="_blank" rel="noopener">前往資料集來源 ↗</a>
  </article>`).join("") || '<p class="empty-copy">目前沒有資料集建議。</p>';
}

async function runBenchmark(event) {
  event.preventDefault();
  clearError();
  const manifest = $("#manifest").value;
  if (!manifest) return showError("請先選擇一個已掛載的 benchmark manifest。");
  const maxSamples = Number($("#maxSamples").value);
  if (!Number.isInteger(maxSamples) || maxSamples < 1) return showError("請輸入有效的測試筆數。");
  const form = new FormData();
  form.append("manifest", manifest);
  form.append("vllm_url", $("#vllmUrl").value.trim());
  form.append("model_id", $("#modelId").value.trim());
  form.append("dataset_language", $("#datasetLanguage").value);
  form.append("language_override", $("#languageOverride").value);
  form.append("metric", $("#metric").value);
  form.append("max_samples", String(maxSamples));
  form.append("max_chunk_sec", $("#maxChunkSec").value);
  state.activeManifest = manifest;
  resetRun();
  setPipelineStage("transcribing");
  setBusy(true);
  try {
    const response = await fetch("/api/benchmark/run", { method: "POST", body: form });
    if (!response.ok) throw new Error(await responseText(response));
    await consumeSse(response);
  } catch (error) {
    setPipelineStage("dataset");
    showError(error.message || "Benchmark 執行失敗");
  } finally {
    setBusy(false);
  }
}

function resetRun() {
  state.rows = [];
  state.finalPayload = null;
  state.startedAt = Date.now();
  clearInterval(state.timerId);
  state.timerId = setInterval(updateElapsed, 1000);
  $("#progressPanel").classList.remove("hidden");
  $("#resultCard").classList.remove("hidden");
  $("#resultStats").textContent = "Benchmark 執行中";
  $("#summaryCards").innerHTML = scoreCards(null);
  $("#languageRows").innerHTML = '<tr><td colspan="6">等待首筆結果。</td></tr>';
  $("#sampleRows").innerHTML = '<tr><td colspan="7">正在等待模型輸出。</td></tr>';
  $("#downloadJson").disabled = true;
  $("#downloadCsv").disabled = true;
  updateProgress("GPU 佇列", "Benchmark 已準備完成", 0, 0, 0);
  $("#resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
}

function setBusy(busy) {
  if (busy) $("#runButton").disabled = true;
  else updateRunAvailability();
  $("#runText").textContent = busy ? "跑分中…" : "開始跑分";
  if (!busy) { clearInterval(state.timerId); state.timerId = null; }
}

function updateElapsed() {
  if (!state.startedAt) return;
  $("#elapsedText").textContent = `已經過 ${formatDuration((Date.now() - state.startedAt) / 1000)}`;
}

function updateProgress(stage, message, position, total, fallbackPercent) {
  $("#statusText").textContent = message;
  $("#progressPercent").textContent = total ? `${position} / ${total}` : "等待中";
  const percent = total ? Math.max(0, Math.min(100, position / total * 100)) : (fallbackPercent || 0);
  $("#progressBar").style.width = `${percent}%`;
  $("#progressPanel .progress-stage").textContent = stage;
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
      processSseBlock(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
    }
    if (done) break;
  }
  if (buffer.trim()) processSseBlock(buffer);
}

function processSseBlock(block) {
  const event = (block.split("\n").find((line) => line.startsWith("event:")) || "event: message").slice(6).trim();
  const data = block.split("\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trimStart()).join("\n");
  if (!data) return;
  let payload;
  try { payload = JSON.parse(data); } catch (_) { throw new Error("服務回傳了無法解析的 Benchmark 串流資料"); }
  if (event === "queued") {
    updateProgress("GPU 佇列", payload.message || "等待 GPU 工作槽", 0, payload.samples_total || 0, 2);
    return;
  }
  if (event === "sample_started") {
    updateProgress("TRANSCRIBING", payload.message || "轉錄中", payload.position - 1, payload.samples_total, 0);
    return;
  }
  if (event === "sample_complete") {
    state.rows.push(payload.row);
    $("#resultStats").textContent = `已完成 ${payload.position} / ${payload.samples_total} 筆 · ${payload.row.language} · ${payload.row.score.metric.toUpperCase()}`;
    updateProgress("SCORING", `已完成 ${payload.row.id}，正在計算累積分數`, payload.position, payload.samples_total, 0);
    renderSummary(payload.summary);
    renderRows();
    return;
  }
  if (event === "complete") {
    state.finalPayload = payload;
    state.rows = payload.rows || state.rows;
    $("#resultStats").textContent = `完成 ${state.rows.length} 筆 · 總耗時 ${formatDuration(payload.elapsed_time || 0)}`;
    updateProgress("COMPLETE", "Benchmark 評分完成", state.rows.length, state.rows.length, 100);
    renderSummary(payload.summary);
    renderRows();
    $("#downloadJson").disabled = false;
    $("#downloadCsv").disabled = false;
    setPipelineStage("complete");
    return;
  }
  if (event === "error") throw new Error(payload.detail || "Benchmark 執行失敗");
}

function renderSummary(summary) {
  $("#summaryCards").innerHTML = scoreCards(summary);
  const groups = summary?.by_language || [];
  $("#languageRows").innerHTML = groups.length ? groups.map((item) => `<tr>
    <td>${escapeHtml(item.language)}</td><td>${escapeHtml(item.metric.toUpperCase())}</td><td>${item.samples}</td>
    <td>${formatPercent(item.error_rate * 100)}</td><td class="${scoreClass(item.accuracy_percent)}">${formatPercent(item.accuracy_percent)}</td><td>${formatPercent(item.exact_match_percent)}</td>
  </tr>`).join("") : '<tr><td colspan="6">尚無完成樣本。</td></tr>';
}

function scoreCards(summary) {
  if (!summary) return ["整體正確率", "單位錯誤率", "Exact match", "完成樣本"].map((label, index) => `<article class="score-card ${index === 0 ? "primary" : ""}"><span>${label}</span><strong>—</strong></article>`).join("");
  if (summary.mixed_metrics) return `<article class="score-card primary"><span>整體正確率</span><strong>—</strong><small>WER 與 CER 不可合併</small></article>
    <article class="score-card"><span>單位錯誤率</span><strong>—</strong><small>請比較各語言列</small></article>
    <article class="score-card"><span>Exact match</span><strong>${formatPercent(summary.exact_match_percent)}</strong></article>
    <article class="score-card"><span>完成樣本</span><strong>${summary.samples}</strong></article>`;
  return `<article class="score-card primary"><span>整體正確率</span><strong>${formatPercent(summary.accuracy_percent)}</strong></article>
    <article class="score-card"><span>單位錯誤率</span><strong>${formatPercent(summary.error_rate * 100)}</strong></article>
    <article class="score-card"><span>Exact match</span><strong>${formatPercent(summary.exact_match_percent)}</strong></article>
    <article class="score-card"><span>完成樣本</span><strong>${summary.samples}</strong></article>`;
}

function renderRows() {
  $("#sampleRows").innerHTML = state.rows.length ? state.rows.map((row, index) => `<tr>
    <td>${index + 1}</td><td><audio class="sample-audio" controls preload="none" src="${escapeAttribute(audioUrl(row.id))}">無法播放音檔。</audio></td>
    <td>${escapeHtml(row.language)}</td><td>${escapeHtml(row.score.metric.toUpperCase())}</td>
    <td class="${scoreClass(row.score.accuracy_percent)}">${formatPercent(row.score.accuracy_percent)}</td>
    <td>${escapeHtml(row.reference)}</td><td>${escapeHtml(row.hypothesis)}</td>
  </tr>`).join("") : '<tr><td colspan="7">尚無完成樣本。</td></tr>';
}

function audioUrl(sampleId) {
  return `/api/benchmark/audio?manifest=${encodeURIComponent(state.activeManifest)}&sample_id=${encodeURIComponent(sampleId)}`;
}

function download(format) {
  if (!state.finalPayload) return;
  const text = format === "json" ? JSON.stringify(state.finalPayload, null, 2) : toCsv(state.rows);
  const blob = new Blob([text], { type: format === "json" ? "application/json;charset=utf-8" : "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `moss-asr-benchmark-${new Date().toISOString().replace(/[:.]/g, "-")}.${format}`;
  document.body.appendChild(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function toCsv(rows) {
  const header = ["id", "language", "metric", "accuracy_percent", "error_rate", "errors", "reference_units", "reference", "hypothesis"];
  const escape = (value) => `"${String(value ?? "").replace(/"/g, '""')}"`;
  return [header, ...rows.map((row) => [row.id, row.language, row.score.metric, row.score.accuracy_percent.toFixed(4), row.score.error_rate.toFixed(6), row.score.errors, row.score.reference_units, row.reference, row.hypothesis])].map((row) => row.map(escape).join(",")).join("\n");
}

async function responseText(response) {
  try { const payload = await response.json(); return payload.detail || "請求失敗"; } catch (_) { return `請求失敗（HTTP ${response.status}）`; }
}
function clearError() { $("#errorBanner").classList.add("hidden"); $("#errorText").textContent = ""; }
function showError(message) { $("#errorText").textContent = message; $("#errorBanner").classList.remove("hidden"); }
function formatPercent(value) { return value == null ? "—" : `${Number(value).toFixed(2)}%`; }
function scoreClass(value) { return Number(value) >= 80 ? "score-good" : "score-warn"; }
function formatBytes(bytes) { if (!Number.isFinite(bytes) || bytes <= 0) return "0 B"; const units = ["B", "KB", "MB", "GB"]; const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1); return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`; }
function formatSampleCount(value) { const count = Number(value); return Number.isFinite(count) ? `${count.toLocaleString("zh-TW")} 筆` : "樣本數未知"; }
function formatDuration(seconds) { const total = Math.max(0, Math.round(Number(seconds) || 0)); const minutes = Math.floor(total / 60); const remainder = total % 60; return minutes ? `${minutes} 分 ${remainder} 秒` : `${remainder} 秒`; }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character]); }
function escapeAttribute(value) { return escapeHtml(value); }
