const $ = (id) => document.getElementById(id);
const FIELDS = ["emotional_tone", "emotional_intensity", "background_noise_present", "background_noise_type",
  "background_noise_severity", "audio_quality", "speaker_overlap_present", "long_silence_present", "confidence"];
const SHORT = { emotional_tone: "tone", emotional_intensity: "intensity", background_noise_present: "noise",
  background_noise_type: "noise type", background_noise_severity: "noise sev.", audio_quality: "quality",
  speaker_overlap_present: "overlap", long_silence_present: "long silence", confidence: "conf." };
let picked = [];      // [{file, path}]
let batchId = null;
let pollTimer = null;

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  if (r.status === 401) { location.href = "/login"; throw new Error("auth"); }
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || `HTTP ${r.status}`);
  return j;
}

// ------------------------------------------------------------ config / header
api("/api/config").then((c) => {
  $("cfg").textContent = `model ${c.model} · up to ${c.max_samples} samples/clip · budget $${c.target_cost_per_min}/audio-min` +
    (c.llm_available ? "" : " · Gemini key missing (DSP-only available)");
  if (!c.llm_available) $("backend").value = "dsp";
});
$("logout").onclick = async () => { await fetch("/api/logout", { method: "POST" }); location.href = "/login"; };

// ------------------------------------------------------------ picking files
function setPicked(list) {
  picked = list;
  const zips = list.filter((x) => x.file.name.toLowerCase().endsWith(".zip")).length;
  const csvs = list.filter((x) => x.file.name.toLowerCase().endsWith(".csv")).length;
  const size = list.reduce((a, x) => a + x.file.size, 0) / 1048576;
  $("picked").textContent = list.length
    ? `${list.length} file(s) selected · ${zips} ZIP · ${csvs} CSV · ${size.toFixed(1)} MB` : "";
  $("uploadBtn").disabled = !list.length;
}
$("folderInput").onchange = (e) => setPicked([...e.target.files].map((f) => ({ file: f, path: f.webkitRelativePath || f.name })));
$("fileInput").onchange = (e) => setPicked([...e.target.files].map((f) => ({ file: f, path: f.name })));

const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async (e) => {
  const items = [...e.dataTransfer.items].map((i) => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  const out = [];
  async function walk(entry, prefix) {
    if (entry.isFile) {
      const f = await new Promise((res, rej) => entry.file(res, rej));
      out.push({ file: f, path: prefix + f.name });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const ch of batch) await walk(ch, prefix + entry.name + "/");
      } while (batch.length);
    }
  }
  for (const it of items) await walk(it, "");
  setPicked(out);
});

// ------------------------------------------------------------ upload
$("uploadBtn").onclick = () => {
  const fd = new FormData();
  picked.forEach((x) => fd.append("files", x.file, x.path));
  fd.append("label", $("label").value);
  const xhr = new XMLHttpRequest();
  $("upProg").classList.remove("hidden");
  $("uploadBtn").disabled = true;
  $("uploadMsg").textContent = "Uploading…";
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) $("upProg").firstElementChild.style.width = `${(100 * e.loaded) / e.total}%`; };
  xhr.onload = () => {
    $("uploadBtn").disabled = false;
    if (xhr.status === 401) { location.href = "/login"; return; }
    let j = {};
    try { j = JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status !== 200) { $("uploadMsg").innerHTML = `<span class="err">${esc(j.detail || "Upload failed")}</span>`; return; }
    $("uploadMsg").innerHTML = `<span class="ok">Uploaded</span>`;
    batchId = j.batch_id;
    showReport(j.report);
  };
  xhr.onerror = () => { $("uploadBtn").disabled = false; $("uploadMsg").innerHTML = `<span class="err">Network error</span>`; };
  xhr.open("POST", "/api/batches");
  xhr.send(fd);
};

function list(title, items, cls) {
  if (!items || !items.length) return "";
  return `<p class="${cls}"><b>${esc(title)} (${items.length})</b></p><ul class="report">${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
}
function showReport(r) {
  $("validatePanel").classList.remove("hidden");
  $("resultsPanel").classList.add("hidden");
  const ok = !r.errors.length || r.to_process > 0;
  $("report").innerHTML =
    `<p>${r.manifest ? `Manifest: <b>${esc(r.manifest)}</b>` : `<span class="warn">No manifest</span>`} · ` +
    `<b>${r.to_process}</b> audio file(s) will be processed · ${r.labeled} with labels</p>` +
    list("Errors", r.errors, "err") +
    list("In CSV but audio file missing", r.missing_audio, "err") +
    list("Audio not listed in CSV (will still be processed)", r.unlisted_audio, "warn") +
    list("Unsupported / non-audio files (skipped)", r.unsupported, "warn") +
    list("Rows with malformed result_json (scored as unlabeled)", r.malformed_labels, "warn") +
    list("Notes", r.warnings, "muted");
  $("runBtn").disabled = !ok || !r.to_process;
  $("runMsg").textContent = "";
}

// ------------------------------------------------------------ run + poll
$("runBtn").onclick = async () => {
  const fd = new FormData();
  fd.append("backend", $("backend").value);
  $("runBtn").disabled = true;
  try {
    await api(`/api/batches/${batchId}/run`, { method: "POST", body: fd });
    openBatch(batchId);
  } catch (e) {
    $("runMsg").innerHTML = `<span class="err">${esc(e.message)}</span>`;
    $("runBtn").disabled = false;
  }
};
$("cancelBtn").onclick = async () => { await api(`/api/batches/${batchId}/cancel`, { method: "POST" }); };
$("deleteBtn").onclick = () => deleteBatch(batchId, $("deleteBtn").dataset.label, $("deleteBtn").dataset.total);

// Returns true when the batch was deleted. Shows the server's reason (e.g. still processing) on failure.
async function deleteBatch(id, label, total) {
  if (!confirm(`Delete batch "${label}" and its ${total} result(s)? This cannot be undone.`)) return false;
  try {
    await api(`/api/batches/${id}`, { method: "DELETE" });
  } catch (e) {
    alert(`Could not delete: ${e.message}`);
    return false;
  }
  if (id === batchId) {
    clearInterval(pollTimer);
    batchId = null;
    $("resultsPanel").classList.add("hidden");
    $("validatePanel").classList.add("hidden");
  }
  if (!$("historyPanel").classList.contains("hidden")) loadHistory();
  return true;
}

function openBatch(id) {
  batchId = id;
  $("resultsPanel").classList.remove("hidden");
  $("dlCsv").href = `/api/batches/${id}/results.csv`;
  $("dlJson").href = `/api/batches/${id}/results.json`;
  clearInterval(pollTimer);
  poll();
  pollTimer = setInterval(poll, 1500);
}

async function poll() {
  let s;
  try { s = await api(`/api/batches/${batchId}`); } catch (e) { return; }
  const c = s.counts, finished = c.done + c.error;
  $("resultsTitle").textContent = `3 · Results — ${s.batch.label}`;
  $("bar").style.width = `${s.total ? (100 * finished) / s.total : 0}%`;
  $("progressText").textContent = `${finished}/${s.total} finished · ${c.processing} processing · ${c.queued} queued · ` +
    `${c.done} succeeded · ${c.error} failed · status: ${s.batch.status}` +
    (s.batch.config.backend ? ` · backend ${s.batch.config.backend}` : "");
  $("cancelBtn").classList.toggle("hidden", s.batch.status !== "processing");
  $("deleteBtn").classList.toggle("hidden", s.batch.status === "processing");
  $("deleteBtn").dataset.label = s.batch.label;
  $("deleteBtn").dataset.total = s.total;
  const sm = s.summary;
  $("stats").innerHTML = [
    ["Audio analyzed", `${sm.audio_minutes} min`],
    ["API cost", `$${sm.cost_usd.toFixed(4)}`],
    ["Cost / audio-min", sm.cost_per_audio_min == null ? "–" : `$${sm.cost_per_audio_min.toFixed(5)}`],
    ["Processing s / audio-min", sm.processing_s_per_audio_min ?? "–"],
  ].map(([k, v]) => `<div class="stat"><div class="v">${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join("");
  renderMetrics(s.metrics);
  renderTable(s.files);
  if (s.batch.status === "completed") {
    clearInterval(pollTimer);
    if (!$("historyPanel").classList.contains("hidden")) loadHistory();  // re-enable its Delete button
  }
}

function renderMetrics(m) {
  if (!m) { $("metrics").innerHTML = ""; return; }
  const rows = Object.entries(m.fields).map(([f, v]) =>
    `<tr><td>${esc(f)}</td><td>${v.n ?? m.n}</td><td>${v.accuracy.toFixed(3)}</td><td>${v.macro_f1.toFixed(3)}</td></tr>`).join("");
  const fx = (v) => (v == null ? "–" : v.toFixed(3));
  const cls = m.tone_confusion.classes;
  const cm = m.tone_confusion.matrix.map((row, i) =>
    `<tr><th>${cls[i]}</th>${row.map((x, j) => `<td class="${i === j ? "diag" : ""}">${x}</td>`).join("")}</tr>`).join("");
  $("metrics").innerHTML = `
    <details open><summary>Accuracy vs. provided labels (n=${m.n})</summary>
      <div class="stats">
        <div class="stat"><div class="v">${fx(m.tone_accuracy)}</div><div class="k">Tone accuracy</div></div>
        <div class="stat"><div class="v">${fx(m.tone_macro_f1)}</div><div class="k">Tone macro-F1</div></div>
        <div class="stat"><div class="v">${fx(m.mean_field_accuracy)}</div><div class="k">Mean field accuracy</div></div>
        <div class="stat"><div class="v">${fx(m.tone_brier)}</div><div class="k">Tone Brier (conf.)</div></div>
      </div>
      <div class="row" style="align-items:flex-start;gap:24px">
        <div class="table-wrap"><table><tr><th>Field</th><th>Labeled</th><th>Accuracy</th><th>Macro-F1</th></tr>${rows}</table></div>
        <div class="table-wrap"><table class="cm"><tr><th>label ↓ / pred →</th>${cls.map((c) => `<th>${c}</th>`).join("")}</tr>${cm}</table></div>
      </div>
    </details>`;
}

function cell(f, p, label) {
  if (!p) return "<td></td>";
  let v = p[f];
  const shown = typeof v === "boolean" ? (v ? "yes" : "no") : v === "" ? "—" : v;
  if (label && f in label && f !== "confidence" && f !== "background_noise_type") {
    const ok = String(label[f]).toLowerCase() === String(v).toLowerCase();
    const title = ok ? "matches label" : `label: ${label[f]}`;
    return `<td class="${ok ? "match" : "miss"}" title="${esc(title)}">${esc(shown)}</td>`;
  }
  return `<td>${esc(shown)}</td>`;
}

function renderTable(files) {
  const head = `<tr><th>File</th><th>Status</th>${FIELDS.map((f) => `<th>${SHORT[f]}</th>`).join("")}<th>Details</th></tr>`;
  const body = files.map((f) => {
    const st = { done: "ok", error: "err", processing: "warn", queued: "", pending: "" }[f.status] || "";
    const meta = f.meta || {};
    const det = f.status === "error"
      ? `<td class="wrap err">${esc(f.error)}</td>`
      : f.status === "done"
        ? `<td class="wrap"><details><summary>${meta.duration_s}s · ${meta.total_s}s proc · $${(meta.cost_usd || 0).toFixed(5)}</summary>` +
          `<pre>${esc(JSON.stringify({ evidence: meta.evidence, diagnostics: meta.diagnostics, features: meta.features, label: f.label }, null, 2))}</pre></details></td>`
        : "<td></td>";
    return `<tr><td>${esc(f.name)}</td><td><span class="chip ${st}">${f.status}</span></td>` +
      FIELDS.map((k) => cell(k, f.prediction, f.label)).join("") + det + "</tr>";
  }).join("");
  $("tbl").innerHTML = head + body;
}

// ------------------------------------------------------------ history
$("historyBtn").onclick = () => {
  const p = $("historyPanel");
  p.classList.toggle("hidden");
  if (!p.classList.contains("hidden")) loadHistory();
};

async function loadHistory() {
  const rows = await api("/api/batches");
  if (!rows.length) { $("hist").innerHTML = `<tr><td class="muted">No batches yet.</td></tr>`; return; }
  $("hist").innerHTML = `<tr><th>Batch</th><th>Created</th><th>Status</th><th>Done</th><th>Failed</th><th>Total</th><th></th></tr>` +
    rows.map((b) => `<tr><td>${esc(b.label)}</td><td>${new Date(b.created * 1000).toLocaleString()}</td><td>${esc(b.status)}</td>` +
      `<td>${b.done || 0}</td><td>${b.failed || 0}</td><td>${b.total}</td>` +
      `<td class="row"><button class="open" data-id="${esc(b.id)}" data-status="${esc(b.status)}">Open</button>` +
      `<button class="del" data-id="${esc(b.id)}" data-label="${esc(b.label)}" data-total="${b.total}"` +
      `${b.status === "processing" ? " disabled title=\"Cancel the batch before deleting\"" : ""}>Delete</button></td></tr>`).join("");
  $("hist").querySelectorAll("button.open").forEach((btn) => btn.onclick = async () => {
    if (btn.dataset.status === "validated") {
      batchId = btn.dataset.id;
      const s = await api(`/api/batches/${batchId}`);
      showReport(s.batch.report);
    } else openBatch(btn.dataset.id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  $("hist").querySelectorAll("button.del").forEach((btn) => btn.onclick = () =>
    deleteBatch(btn.dataset.id, btn.dataset.label, btn.dataset.total));
}
