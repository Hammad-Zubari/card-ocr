"use strict";

/**
 * Ecommerce Gateway Pakistan - Visitor Card OCR Front-end Application
 * Real-time scanning, engine selection, duplicate verification, direct click-to-edit fields, and full database CRUD.
 */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const FIELDS = ["name", "job_title", "company_name", "address", "telephone", "mobile_phone", "email", "business_type", "business_confidence"];

const state = {
  job: null,
  cards: new Map(),
  order: [],
  edits: new Map(),
  rowErrors: new Map(),
  selected: null,
  polling: false,
  records: [],
  selectedRecordIds: new Set(),
};

// ------------------------------------------------------------------ helpers
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) throw new Error((body && body.error) || `${res.status} ${res.statusText}`);
  return body;
}

function toast(msg, kind = "") {
  const t = $("#toast");
  if (!t) return;
  t.textContent = msg;
  t.className = `toast-3d ${kind}`;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), kind === "bad" ? 8000 : 4500);
}

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));

function mainClassOf(value) {
  const text = String(value || "").trim();
  const labeled = text.match(/(?:^|\|\s*)Main Class:\s*([^|]+)/i);
  if (labeled) return labeled[1].trim();
  return text.split(" — ")[0].trim();
}

const secs = (v) => (v == null ? "–" : `${Number(v).toFixed(1)}s`);
const value = (id, f) => {
  const e = state.edits.get(id);
  return e && f in e ? e[f] : (state.cards.get(id) ? state.cards.get(id)[f] : "");
};

// ------------------------------------------------------------------ engine selector
function getSelectedProvider() {
  const select = $("#provider-select");
  if (select) return select.value;
  const checked = $('input[name="provider_choice"]:checked');
  return checked ? checked.value : "qwen";
}

function setupEngines() {
  const testActiveBtn = $("#btn-test-active-engine");
  if (testActiveBtn) {
    testActiveBtn.onclick = async () => {
      const provider = getSelectedProvider();
      const providers = {
        qwen: { path: "/api/qwen/check", name: "Qwen 2.5 Vision (OpenRouter 72B)" },
        nvidia: { path: "/api/nvidia/check", name: "NVIDIA Vision (Llama 3.2 11B NIM)" },
      };
      const selected = providers[provider] || providers.qwen;
      
      testActiveBtn.disabled = true;
      const oldText = testActiveBtn.textContent;
      testActiveBtn.textContent = "Testing…";
      try {
        const r = await api(selected.path);
        toast(r.ok ? `✓ ${selected.name} Connected (${r.model})` : `⚠ ${selected.name} Unavailable: ${r.error}`, r.ok ? "ok" : "bad");
      } catch (err) {
        toast(`${selected.name} check failed: ${err.message}`, "bad");
      } finally {
        testActiveBtn.disabled = false;
        testActiveBtn.textContent = oldText;
      }
    };
  }

  const geminiBtn = $("#btn-gemini-check");
  if (geminiBtn) {
    geminiBtn.onclick = async () => {
      geminiBtn.disabled = true;
      const oldText = geminiBtn.textContent;
      geminiBtn.textContent = "Testing…";
      try {
        const r = await api("/api/gemini/check");
        toast(r.ok ? `✓ Gemini Verifier Online (${r.model})` : `⚠ Gemini Unavailable: ${r.error}`, r.ok ? "ok" : "bad");
      } catch (err) {
        toast(`Gemini check failed: ${err.message}`, "bad");
      } finally {
        geminiBtn.disabled = false;
        geminiBtn.textContent = oldText;
      }
    };
  }
}

// ------------------------------------------------------------------ upload
function setupUpload() {
  const input = $("#file-input"), dz = $("#dropzone");
  $("#btn-choose").onclick = () => input.click();
  input.onchange = () => input.files.length && upload([...input.files]);

  dz.ondragover = (e) => { e.preventDefault(); dz.classList.add("over"); };
  dz.ondragleave = () => dz.classList.remove("over");
  dz.ondrop = (e) => {
    e.preventDefault();
    dz.classList.remove("over");
    const files = [...e.dataTransfer.files].filter((f) => /\.pdf$/i.test(f.name) || f.type === "application/pdf");
    if (!files.length) return toast("Only PDF files are supported", "bad");
    upload(files);
  };

  $("#btn-process").onclick = startProcessing;
  $("#btn-reset").onclick = resetAll;
}

async function upload(files) {
  if (state.polling) return toast("A batch is still currently processing", "bad");
  resetAll();
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f, f.name));
  $("#status-msg").textContent = `Uploading ${files.length} file(s)…`;
  try {
    fd.append("provider", getSelectedProvider());
    const job = await api("/api/upload", { method: "POST", body: fd });
    applySnapshot(job);
    const valid = job.files.filter((f) => !f.error).length;
    $("#btn-process").disabled = valid === 0;
    $("#status-msg").textContent = valid
      ? `${valid} PDF(s) ready · ${job.stats.pages} page(s). Press Start Processing Batch.`
      : "No valid PDF files found.";
  } catch (e) {
    $("#status-msg").textContent = `Upload failed: ${e.message}`;
    toast(`Upload failed: ${e.message}`, "bad");
  } finally {
    $("#file-input").value = "";
  }
}

function renderFiles(files) {
  $("#file-list").innerHTML =
    files.map((f) => `<li>
      <span>📄 <strong>${esc(f.name)}</strong></span>
      ${f.error ? `<span class="err">✖ ${esc(f.error)}</span>`
                : `<span class="muted">${f.pages} page(s)${f.cards ? ` · ${f.cards} card(s)` : ""}</span>`}
    </li>`).join("") || `<li class="muted empty-item">No PDF files uploaded yet.</li>`;
}

function resetAll() {
  state.job = null;
  state.cards.clear();
  state.order = [];
  state.edits.clear();
  state.rowErrors.clear();
  state.selected = null;
  renderFiles([]);
  renderTable();
  renderDetail();
  renderStats({}, {});
  setProgress(0);
  $("#errors-box").hidden = true;
  $("#status-msg").textContent = "Upload one or more PDFs to begin processing.";
  $("#scanner-laser").hidden = true;

  const image = $("#live-scan-image");
  const placeholder = $("#live-scan-placeholder");
  if (image) {
    image.hidden = true;
    image.style.display = "none";
    image.src = "";
  }
  if (placeholder) {
    placeholder.hidden = false;
    placeholder.style.display = "flex";
  }

  ["#btn-process", "#btn-save-all", "#btn-save", "#btn-save-review", "#btn-csv"].forEach(
    (b) => { const el = $(b); if (el) el.disabled = true; }
  );
}

// ------------------------------------------------------------------ processing
async function startProcessing() {
  if (!state.job) return;
  $("#btn-process").disabled = true;
  $("#scanner-laser").hidden = false;
  try {
    applySnapshot(await api(`/api/jobs/${state.job.id}/start`, { method: "POST" }));
    poll();
  } catch (e) {
    toast(e.message, "bad");
  }
}

async function poll() {
  state.polling = true;
  $("#scanner-laser").hidden = false;
  try {
    while (true) {
      const snap = await api(`/api/jobs/${state.job.id}?since=0`);
      applySnapshot(snap);
      if (snap.status !== "running") break;
      await new Promise((r) => setTimeout(r, 600));
    }
    applySnapshot(await api(`/api/jobs/${state.job.id}?since=0`), true);
    $("#scanner-laser").hidden = true;
    if (state.job.status === "done") {
      toast(state.job.message, "ok");
      ["#btn-save-all", "#btn-save", "#btn-save-review", "#btn-csv"].forEach(
        (b) => { const el = $(b); if (el) el.disabled = false; }
      );
    } else {
      toast(state.job.message, "bad");
    }
  } catch (e) {
    $("#scanner-laser").hidden = true;
    toast(`Lost connection: ${e.message}`, "bad");
  } finally {
    state.polling = false;
  }
}

function applySnapshot(snap, replaceAll = false) {
  state.job = { ...snap, cards: undefined };
  if (replaceAll) {
    state.order = [];
  }
  let newestId = null;
  for (const c of snap.cards || []) {
    if (!state.cards.has(c.id) || replaceAll) {
      if (!state.order.includes(c.id)) state.order.push(c.id);
      newestId = c.id;
    }
    state.cards.set(c.id, c);
  }
  if (newestId != null && snap.status === "running") {
    state.selected = newestId;
  }
  renderFiles(snap.files);
  setProgress(snap.progress);
  $("#status-msg").textContent = snap.message;
  renderStats(snap.stats, snap.timings);

  const errs = snap.errors || [];
  $("#errors-box").hidden = !errs.length;
  $("#errors-count").textContent = errs.length;
  $("#errors").innerHTML = errs.map((e) => `<li>${esc(e)}</li>`).join("");

  renderTable();
  renderDetail();
  renderLiveScan();
}

function renderLiveScan() {
  const image = $("#live-scan-image");
  const placeholder = $("#live-scan-placeholder");
  const fields = $("#live-scan-fields");
  const msg = $("#live-scan-msg");
  const c = state.selected && state.cards.get(state.selected);

  if (!c || !state.job) {
    if (image) {
      image.hidden = true;
      image.style.display = "none";
    }
    if (placeholder) {
      placeholder.hidden = false;
      placeholder.style.display = "flex";
    }
    msg.textContent = "Waiting for scan to start…";
    fields.innerHTML = [
      ["Name", "—"], ["Designation", "—"], ["Address", "—"],
      ["Telephone", "—"], ["Mobile Phone", "—"], ["Email", "—"],
      ["Business Type", "—"],
    ].map(([l, v]) => `<div class="field-card-3d"><span>${l}</span><b>${v}</b></div>`).join("");
    return;
  }

  msg.textContent = `${c.ocr_status} · Card #${c.id} · ${c.pdf_name || "PDF"} (Page ${c.page_number})`;
  if (c.has_image) {
    image.onload = () => {
      image.hidden = false;
      image.style.display = "block";
      if (placeholder) {
        placeholder.hidden = true;
        placeholder.style.display = "none";
      }
    };
    image.onerror = () => {
      image.hidden = true;
      image.style.display = "none";
      if (placeholder) {
        placeholder.hidden = false;
        placeholder.style.display = "flex";
      }
    };
    image.src = `/api/jobs/${state.job.id}/cards/${c.id}/image?live=${Date.now()}`;
  } else {
    image.hidden = true;
    image.style.display = "none";
    if (placeholder) {
      placeholder.hidden = false;
      placeholder.style.display = "flex";
    }
  }

  fields.innerHTML = [
    ["Name", value(c.id, "name")],
    ["Designation", value(c.id, "job_title")],
    ["Company Name", value(c.id, "company_name")],
    ["Address", value(c.id, "address")],
    ["Telephone", value(c.id, "telephone")],
    ["Mobile Phone", value(c.id, "mobile_phone")],
    ["Email", value(c.id, "email")],
    ["Main Class", `${mainClassOf(value(c.id, "business_type"))}${value(c.id, "business_confidence") ? ` (${value(c.id, "business_confidence")}%)` : ""}`],
  ].map(([label, val]) =>
    `<div class="field-card-3d"><span>${label}</span><b>${esc(val || (c.ocr_status === 'Scanning' ? 'Scanning…' : '—'))}</b></div>`
  ).join("");
}

function setProgress(p) {
  const pct = Math.round((p || 0) * 100);
  $("#progress-bar").style.width = `${pct}%`;
  $("#progress-pct").textContent = `${pct}%`;
}

function renderStats(s, t) {
  $$("#stats [data-k]").forEach((el) => {
    const k = el.dataset.k;
    el.textContent = s[k] ?? "–";
  });
  $$("#stats [data-t]").forEach((el) => (el.textContent = secs(t && t[el.dataset.t])));
}

// ------------------------------------------------------------------ table
function statusBadges(c) {
  const cls = {
    "OK": "b-ok",
    "Needs Review": "b-review",
    "Failed": "b-failed",
    "Scanning": "b-scanning",
  }[c.ocr_status] || "b-review";

  let html = `<span class="badge ${cls}">${esc(c.ocr_status)}</span>`;
  if (c.db_id) html += ` <span class="badge b-saved">Saved #${c.db_id}</span>`;
  return html;
}

function visibleIds() {
  const f = $("#filter").value;
  const q = $("#search").value.trim().toLowerCase();
  return state.order.filter((id) => {
    const c = state.cards.get(id);
    if (!c) return false;
    if (f === "unsaved" && c.db_id) return false;
    if (!["all", "dup", "unsaved"].includes(f) && c.ocr_status !== f) return false;
    if (q && [c.pdf_name, ...FIELDS.map((k) => value(id, k))].join(" ").toLowerCase().indexOf(q) < 0) return false;
    return true;
  });
}

function renderTable() {
  const tbody = $("#results tbody");
  const ids = visibleIds();
  if (!ids.length) {
    tbody.innerHTML = `<tr class="empty"><td colspan="10">
      <div class="empty-state">
        <p>${state.order.length ? "No cards match the active filter/search." : "Scanned visitor card records will stream here in real-time."}</p>
      </div>
    </td></tr>`;
    return;
  }

  tbody.innerHTML = ids.map((id) => {
    const c = state.cards.get(id);
    const e = state.edits.get(id) || {};
    const cells = FIELDS.map((f) => {
      const shown = f === "business_type" ? mainClassOf(value(id, f)) : value(id, f);
      return `<td contenteditable="plaintext-only" spellcheck="false" data-f="${f}"
        class="${f in e ? "edited" : ""}">${esc(shown)}</td>`;
    }).join("");
    return `<tr data-id="${id}" class="${state.selected === id ? "selected" : ""} ${state.rowErrors.has(id) ? "row-error" : ""}"
              title="${esc(state.rowErrors.get(id) || "")}">
      <td class="c-num">${id}</td>
      <td class="card-preview">${c.has_image
        ? `<img src="/api/jobs/${state.job.id}/cards/${c.id}/image" alt="card ${id}">`
        : `<span class="muted">Scanning…</span>`}</td>
      ${cells}
    </tr>`;
  }).join("");
}

function setupTable() {
  const tbody = $("#results tbody");
  tbody.addEventListener("click", (ev) => {
    const tr = ev.target.closest("tr[data-id]");
    if (!tr) return;
    const id = Number(tr.dataset.id);
    if (state.selected !== id) {
      state.selected = id;
      $$("tr.selected", tbody).forEach((r) => r.classList.remove("selected"));
      tr.classList.add("selected");
      renderDetail();
      renderLiveScan();
    }
  });

  tbody.addEventListener("keydown", (ev) => {
    if (ev.target.matches("td[contenteditable]") && ev.key === "Enter") {
      ev.preventDefault();
      ev.target.blur();
    }
    if (ev.target.matches("td[contenteditable]") && ev.key === "Escape") {
      const tr = ev.target.closest("tr");
      const id = Number(tr.dataset.id);
      ev.target.textContent = value(id, ev.target.dataset.f);
      ev.target.blur();
    }
  });

  tbody.addEventListener("focusout", (ev) => {
    const td = ev.target;
    if (!td.matches("td[contenteditable]")) return;
    const id = Number(td.closest("tr").dataset.id);
    const f = td.dataset.f;
    const v = td.textContent.replace(/\s*\n\s*/g, ", ").trim();
    const orig = (state.cards.get(id) && state.cards.get(id)[f]) || "";
    const e = state.edits.get(id) || {};
    if (v === orig) delete e[f]; else e[f] = v;
    if (Object.keys(e).length) state.edits.set(id, e); else state.edits.delete(id);
    td.classList.toggle("edited", f in e);
    if (state.selected === id) renderDetail();
  });

  $("#filter").onchange = renderTable;
  $("#search").oninput = renderTable;
  $("#btn-save-all").onclick = saveAll;
  $("#btn-save").onclick = saveSelected;
  $("#btn-save-review").onclick = saveReviewed;
  $("#btn-csv").onclick = exportCsv;
}

// ------------------------------------------------------------------ interactive detail sidebar
function renderDetail() {
  const box = $("#detail");
  const c = state.selected && state.cards.get(state.selected);
  if (!c) {
    box.innerHTML = `<div class="detail-placeholder">
      <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
      <p class="muted">Select any table row to inspect card preview and click any field to edit directly.</p>
    </div>`;
    return;
  }

  const e = state.edits.get(c.id) || {};
  const img = c.has_image
    ? `<div class="detail-img-wrap"><img src="/api/jobs/${state.job.id}/cards/${c.id}/image" alt="card ${c.id}"></div>`
    : `<p class="muted text-center p-2">No card crop image available.</p>`;

  const fieldConfig = [
    { key: "name", label: "Full Name", type: "input", placeholder: "e.g. John Doe" },
    { key: "job_title", label: "Designation / Role", type: "input", placeholder: "e.g. Managing Director" },
    { key: "company_name", label: "Company Name", type: "input", placeholder: "e.g. Avanza Solutions" },
    { key: "address", label: "Address", type: "textarea", placeholder: "e.g. Office address / city" },
    { key: "telephone", label: "Telephone (Landline)", type: "input", placeholder: "e.g. +92-21-35112709" },
    { key: "mobile_phone", label: "Mobile Phone", type: "input", placeholder: "e.g. 0300-2162152" },
    { key: "email", label: "Email Address", type: "input", placeholder: "e.g. name@domain.com" },
    { key: "business_type", label: "Main Class", type: "textarea", placeholder: "Main Class..." },
  ];

  const formFieldsHtml = fieldConfig.map((fc) => {
    const isEdited = fc.key in e;
    const currVal = fc.key === "business_type"
      ? mainClassOf(value(c.id, fc.key))
      : value(c.id, fc.key);
    if (fc.type === "textarea") {
      return `
        <div class="detail-field-group">
          <label class="detail-field-label">
            <span>${fc.label}</span>
            <span class="detail-field-hint">${isEdited ? "✓ Edited" : "✏️ Click to edit"}</span>
          </label>
          <textarea class="detail-input-3d detail-textarea-3d ${isEdited ? "edited" : ""}" data-card-id="${c.id}" data-f="${fc.key}" placeholder="${fc.placeholder}">${esc(currVal)}</textarea>
        </div>`;
    }
    return `
      <div class="detail-field-group">
        <label class="detail-field-label">
          <span>${fc.label}</span>
          <span class="detail-field-hint">${isEdited ? "✓ Edited" : "✏️ Click to edit"}</span>
        </label>
        <input type="text" class="detail-input-3d ${isEdited ? "edited" : ""}" data-card-id="${c.id}" data-f="${fc.key}" value="${esc(currVal)}" placeholder="${fc.placeholder}">
      </div>`;
  }).join("");

  box.innerHTML = `
    <div class="detail-header-3d">
      <span class="detail-header-title" title="${esc(c.pdf_name || "")}">📄 ${esc(c.pdf_name || "Card #" + c.id)}</span>
      <span class="muted font-11">Page ${c.page_number}</span>
    </div>
    ${img}
    <div class="detail-card-meta-bar">
      <span class="font-mono"><strong>Card #${c.id}</strong></span>
      <div>${c.db_id ? `Saved #${c.db_id}` : ""}</div>
    </div>
    <form class="detail-fields-form" onsubmit="return false;">
      ${formFieldsHtml}
    </form>
    <div class="detail-card-actions-3d">
      <button class="btn-3d btn-sm primary btn-detail-save-one" data-id="${c.id}" type="button" ${c.db_id ? "disabled" : ""}>
        ${c.db_id ? "✓ Saved (#" + c.db_id + ")" : "💾 Save This Card"}
      </button>
      <button class="btn-3d btn-sm secondary btn-detail-mark-ok" data-id="${c.id}" type="button">
        Mark as OK
      </button>
    </div>
    ${c.issues && c.issues.length ? `
      <dl class="detail-meta-list">
        ${c.issues && c.issues.length ? `<dt>OCR Issues</dt><dd class="muted">${esc(c.issues.join("; "))}</dd>` : ""}
      </dl>
    ` : ""}
  `;

  // Bind input listeners for real-time synchronization
  $$(".detail-input-3d", box).forEach((input) => {
    const onUpdate = () => {
      const cardId = Number(input.dataset.cardId);
      const f = input.dataset.f;
      const v = input.value.trim();
      const orig = (state.cards.get(cardId) && state.cards.get(cardId)[f]) || "";
      const ed = state.edits.get(cardId) || {};
      if (v === orig) delete ed[f]; else ed[f] = v;
      if (Object.keys(ed).length) state.edits.set(cardId, ed); else state.edits.delete(cardId);
      input.classList.toggle("edited", f in ed);
      
      // Update table cell
      const td = $(`#results tbody tr[data-id="${cardId}"] td[data-f="${f}"]`);
      if (td) {
        td.textContent = v;
        td.classList.toggle("edited", f in ed);
      }
      // Update live monitor if current
      renderLiveScan();
    };

    input.addEventListener("input", onUpdate);
    input.addEventListener("change", onUpdate);
  });

  const saveOneBtn = $(".btn-detail-save-one", box);
  if (saveOneBtn) {
    saveOneBtn.onclick = async () => {
      const card = collectCards((x) => x.id === c.id);
      if (!card.length) return toast("Card already saved or invalid.", "bad");
      await pushToDatabase(card);
    };
  }

  const markOkBtn = $(".btn-detail-mark-ok", box);
  if (markOkBtn) {
    markOkBtn.onclick = () => {
      c.ocr_status = "OK";
      renderTable();
      renderDetail();
      toast(`Card #${c.id} marked as Verified OK`, "ok");
    };
  }
}

// ------------------------------------------------------------------ save & export
function collectCards(filterFn) {
  const out = [];
  for (const id of state.order) {
    const c = state.cards.get(id);
    if (!c || !filterFn(c)) continue;
    const merged = { ...c };
    const e = state.edits.get(id);
    if (e) Object.assign(merged, e);
    out.push(merged);
  }
  return out;
}

async function saveAll() {
  const cards = collectCards((c) => !c.db_id);
  if (!cards.length) return toast("No unsaved cards in batch to save.", "bad");
  await pushToDatabase(cards);
}

async function saveSelected() {
  const cards = collectCards((c) => c.ocr_status === "OK" && !c.db_id);
  if (!cards.length) return toast("No unsaved 'OK' cards to save.", "bad");
  await pushToDatabase(cards);
}

async function saveReviewed() {
  const cards = collectCards((c) => (c.ocr_status === "Needs Review" || state.edits.has(c.id)) && !c.db_id);
  if (!cards.length) return toast("No reviewed / edited cards to save.", "bad");
  await pushToDatabase(cards);
}

async function pushToDatabase(cards) {
  state.rowErrors.clear();
  try {
    const r = await api("/api/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cards }),
    });
    for (const res of r.results || []) {
      const c = state.cards.get(res.id);
      if (c && res.db_id) c.db_id = res.db_id;
      if (res.error) state.rowErrors.set(res.id, res.error);
    }
    toast(`Saved ${r.saved} card(s) to database. ${r.failed ? `${r.failed} failed.` : ""}`, r.failed ? "bad" : "ok");
  } catch (e) {
    toast(`Save failed: ${e.message}`, "bad");
  } finally {
    renderTable();
    renderDetail();
    loadDrawerRecords();
  }
}

async function recheckDuplicates() {
  if (!state.order.length) return;
  const cards = collectCards(() => true);
  try {
    const r = await api("/api/check-duplicates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cards }),
    });
    for (const res of r.results || []) {
      const c = state.cards.get(res.id);
      if (c) {
        c.duplicate_match_id = res.duplicate_match_id;
      }
    }
    toast(`Duplicate check complete (${r.duplicates_found} found)`, "ok");
  } catch (e) {
    toast(e.message, "bad");
  } finally {
    renderTable();
    renderDetail();
  }
}

function exportCsv() {
  const cards = collectCards(() => true);
  if (!cards.length) return toast("No cards to export.", "bad");
  const head = ["Card ID", "PDF Name", "Page", "Name", "Designation", "Address", "Telephone", "Mobile Phone", "Email", "Main Class", "Status", "Database ID"];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = [head.map(q).join(",")].concat(
    cards.map((c) => [
      c.id, c.pdf_name, c.page_number,
      value(c.id, "name"), value(c.id, "job_title"), value(c.id, "address"),
      value(c.id, "telephone"), value(c.id, "mobile_phone"), value(c.id, "email"),
      value(c.id, "business_type"),
      c.ocr_status, c.db_id,
    ].map(q).join(","))
  );
  const blob = new Blob(["\uFEFF" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob),
    download: `Visitor_Cards_Export_${new Date().toISOString().slice(0,10)}.csv`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
}

// ------------------------------------------------------------------ database tab
async function loadRecords() {
  const tbody = $("#records tbody");
  tbody.innerHTML = `<tr class="empty"><td colspan="12">Loading records from database…</td></tr>`;
  
  const q = $("#rec-search").value.trim();
  const from = $("#rec-date-from").value;
  const to = $("#rec-date-to").value;
  const sort = $("#rec-sort").value;

  const params = new URLSearchParams({
    limit: 500,
    q, date_from: from, date_to: to, sort
  });

  try {
    const res = await api(`/api/records?${params.toString()}`);
    state.records = res.rows || [];
    state.selectedRecordIds.clear();
    updateBulkActions();
    $("#rec-count").textContent = `${res.total} total record(s)`;

    if (!state.records.length) {
      tbody.innerHTML = `<tr class="empty"><td colspan="13"><div class="empty-state"><p>No records found in database.</p></div></td></tr>`;
      return;
    }

    tbody.innerHTML = state.records.map((r) => `
      <tr data-id="${r.id}">
        <td class="c-chk"><input type="checkbox" class="chk-record" data-id="${r.id}"></td>
        <td class="c-num">#${r.id}</td>
        <td class="c-name"><strong>${esc(r.name || "—")}</strong></td>
        <td class="c-title">${esc(r.job_title || "—")}</td>
        <td class="c-name">${esc(r.company_name || "—")}</td>
        <td class="c-addr">${esc(r.address || "—")}</td>
        <td class="c-tel">${esc(r.telephone || "—")}</td>
        <td class="c-mobile">${esc(r.mobile_phone || "—")}</td>
        <td class="c-email">${esc(r.email || "—")}</td>
        <td class="c-biz">${r.business_type
          ? `<span class="badge b-biz">${esc(mainClassOf(r.business_type))}${r.business_confidence ? ` (${esc(r.business_confidence)}%)` : ""}</span>`
          : "—"}</td>
        <td class="c-date"><span class="font-mono muted">${esc((r.created_at || "").slice(0, 10))}</span></td>
        <td class="c-act">
          <button class="btn-3d btn-sm secondary btn-edit-rec" data-id="${r.id}">Edit</button>
          <button class="btn-3d btn-sm bad btn-del-rec" data-id="${r.id}">Delete</button>
        </td>
      </tr>
    `).join("");

    $$(".chk-record", tbody).forEach((chk) => {
      chk.onchange = (e) => {
        const id = Number(e.target.dataset.id);
        if (e.target.checked) state.selectedRecordIds.add(id);
        else state.selectedRecordIds.delete(id);
        updateBulkActions();
      };
    });

    $$(".btn-edit-rec", tbody).forEach((btn) => {
      btn.onclick = () => openEditModal(Number(btn.dataset.id));
    });

    $$(".btn-del-rec", tbody).forEach((btn) => {
      btn.onclick = () => deleteSingleRecord(Number(btn.dataset.id));
    });

  } catch (e) {
    tbody.innerHTML = `<tr class="empty"><td colspan="13" class="err">Failed to load records: ${esc(e.message)}</td></tr>`;
  }
}

function updateBulkActions() {
  const count = state.selectedRecordIds.size;
  const bar = $("#rec-bulk-actions");
  if (count > 0) {
    bar.hidden = false;
    $("#bulk-selected-count").textContent = `${count} selected`;
  } else {
    bar.hidden = true;
  }
  const chkAll = $("#chk-all-records");
  if (chkAll) {
    chkAll.checked = count > 0 && count === state.records.length;
  }
}

async function bulkDeleteRecords() {
  const ids = Array.from(state.selectedRecordIds);
  if (!ids.length) return;
  if (!confirm(`Are you sure you want to delete ${ids.length} selected record(s)?`)) return;
  try {
    const r = await api("/api/records/bulk-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    toast(`Deleted ${r.deleted} record(s).`, "ok");
    loadRecords();
    loadDrawerRecords();
  } catch (e) {
    toast(`Delete failed: ${e.message}`, "bad");
  }
}

async function deleteSingleRecord(id) {
  if (!confirm(`Are you sure you want to delete record #${id}?`)) return;
  try {
    await api(`/api/records/${id}`, { method: "DELETE" });
    toast(`Record #${id} deleted successfully.`, "ok");
    loadRecords();
    loadDrawerRecords();
  } catch (e) {
    toast(`Delete failed: ${e.message}`, "bad");
  }
}

// ------------------------------------------------------------------ edit modal
function setupModal() {
  const modal = $("#edit-modal");
  const form = $("#modal-edit-form");
  const closeBtn = $("#btn-modal-close");
  const cancelBtn = $("#btn-modal-cancel");
  const deleteBtn = $("#btn-modal-delete");

  const closeModal = () => modal.close();
  closeBtn.onclick = closeModal;
  cancelBtn.onclick = closeModal;

  form.onsubmit = async (e) => {
    e.preventDefault();
    const id = Number($("#edit-rec-id").value);
    const payload = {
      name: $("#edit-name").value.trim(),
      job_title: $("#edit-job-title").value.trim(),
      company_name: $("#edit-company-name").value.trim(),
      address: $("#edit-address").value.trim(),
      telephone: $("#edit-telephone").value.trim(),
      mobile_phone: $("#edit-mobile").value.trim(),
      email: $("#edit-email").value.trim(),
      business_type: $("#edit-business-type").value.trim(),
      business_confidence: Number($("#edit-business-confidence")?.value || 0),
    };

    try {
      await api(`/api/records/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      toast(`Record #${id} updated successfully.`, "ok");
      closeModal();
      loadRecords();
      loadDrawerRecords();
    } catch (err) {
      toast(`Update failed: ${err.message}`, "bad");
    }
  };

  deleteBtn.onclick = async () => {
    const id = Number($("#edit-rec-id").value);
    if (!id) return;
    if (confirm(`Delete record #${id}?`)) {
      closeModal();
      await deleteSingleRecord(id);
    }
  };
}

async function openEditModal(id) {
  const modal = $("#edit-modal");
  try {
    const rec = await api(`/api/records/${id}`);
    const r = rec.record || rec;
    $("#edit-rec-id").value = r.id;
    $("#modal-title").textContent = `Edit Record #${r.id}`;
    $("#edit-name").value = r.name || "";
    $("#edit-job-title").value = r.job_title || "";
    $("#edit-company-name").value = r.company_name || "";
    $("#edit-address").value = r.address || "";
    $("#edit-telephone").value = r.telephone || "";
    $("#edit-mobile").value = r.mobile_phone || "";
    $("#edit-email").value = r.email || "";
    $("#edit-business-type").value = r.business_type || "";
    if ($("#edit-business-confidence")) $("#edit-business-confidence").value = r.business_confidence || 0;
    modal.showModal();
  } catch (e) {
    toast(`Failed to load record #${id}: ${e.message}`, "bad");
  }
}

// ------------------------------------------------------------------ slide drawer
function setupDrawer() {
  const drawer = $("#records-drawer");
  const overlay = $("#drawer-overlay");
  const toggleBtn = $("#btn-toggle-drawer");
  const closeBtn = $("#btn-close-drawer");

  const openDrawer = () => {
    drawer.classList.add("open");
    overlay.hidden = false;
    loadDrawerRecords();
  };

  const closeDrawer = () => {
    drawer.classList.remove("open");
    overlay.hidden = true;
  };

  if (toggleBtn) toggleBtn.onclick = openDrawer;
  if (closeBtn) closeBtn.onclick = closeDrawer;
  if (overlay) overlay.onclick = closeDrawer;

  const refreshBtn = $("#btn-drawer-refresh");
  if (refreshBtn) refreshBtn.onclick = loadDrawerRecords;

  let dtm;
  const searchInput = $("#drawer-search");
  if (searchInput) {
    searchInput.oninput = () => {
      clearTimeout(dtm);
      dtm = setTimeout(loadDrawerRecords, 300);
    };
  }
}

async function loadDrawerRecords() {
  const searchInput = $("#drawer-search");
  const q = searchInput ? searchInput.value.trim() : "";
  const list = $("#drawer-records-list");
  if (!list) return;
  list.innerHTML = `<p class="muted p-4">Loading saved records…</p>`;
  try {
    const r = await api(`/api/records?limit=100&q=${encodeURIComponent(q)}`);
    const countEl = $("#drawer-rec-count");
    if (countEl) countEl.textContent = `${r.total} records`;
    renderDrawerList(r.rows);
  } catch (e) {
    list.innerHTML = `<p class="err p-4">${esc(e.message)}</p>`;
  }
}

function renderDrawerList(rows) {
  const list = $("#drawer-records-list");
  if (!list) return;
  if (!rows || !rows.length) {
    list.innerHTML = `<p class="muted text-center p-4">No records found.</p>`;
    return;
  }
  list.innerHTML = rows.map((x) => `
    <div class="drawer-card" data-id="${x.id}">
      <div class="drawer-card-header">
        <span class="drawer-card-name">${esc(x.name || "Unnamed")}</span>
        <span class="badge b-saved">#${x.id}</span>
      </div>
      <div class="drawer-card-meta">${esc(x.job_title || "No Designation")}</div>
      <div class="drawer-card-contacts">
        ${x.telephone ? `<span>📞 ${esc(x.telephone)}</span>` : ""}
        ${x.mobile_phone ? `<span>📱 ${esc(x.mobile_phone)}</span>` : ""}
        ${x.email ? `<span>✉️ ${esc(x.email)}</span>` : ""}
      </div>
      <div class="drawer-card-actions">
        <button class="btn-3d btn-sm secondary btn-drawer-edit" data-id="${x.id}">Edit</button>
        <button class="btn-3d btn-sm bad btn-drawer-del" data-id="${x.id}">Delete</button>
      </div>
    </div>
  `).join("");

  $$(".btn-drawer-edit", list).forEach((btn) => {
    btn.onclick = () => openEditModal(Number(btn.dataset.id));
  });
  $$(".btn-drawer-del", list).forEach((btn) => {
    btn.onclick = () => deleteSingleRecord(Number(btn.dataset.id));
  });
}

// ------------------------------------------------------------------ tabs & filters
function setupTabs() {
  $$(".tab-btn, .tab").forEach((t) => {
    t.onclick = () => {
      $$(".tab-btn, .tab").forEach((x) => x.classList.toggle("active", x === t));
      $$(".tab-panel").forEach((p) =>
        p.classList.toggle("active", p.id === `tab-${t.dataset.tab}`)
      );
      if (t.dataset.tab === "records") loadRecords();
    };
  });

  const refreshBtn = $("#btn-rec-refresh");
  if (refreshBtn) refreshBtn.onclick = loadRecords;

  const chkAll = $("#chk-all-records");
  if (chkAll) {
    chkAll.onchange = (e) => {
      const checked = e.target.checked;
      $$(".chk-record").forEach((chk) => {
        chk.checked = checked;
        const id = Number(chk.dataset.id);
        if (checked) state.selectedRecordIds.add(id);
        else state.selectedRecordIds.delete(id);
      });
      updateBulkActions();
    };
  }

  const bulkDelBtn = $("#btn-bulk-delete");
  if (bulkDelBtn) bulkDelBtn.onclick = bulkDeleteRecords;

  let tm;
  const recSearch = $("#rec-search");
  if (recSearch) {
    recSearch.oninput = () => {
      clearTimeout(tm);
      tm = setTimeout(loadRecords, 300);
    };
  }


  const dateFrom = $("#rec-date-from");
  if (dateFrom) dateFrom.onchange = loadRecords;

  const dateTo = $("#rec-date-to");
  if (dateTo) dateTo.onchange = loadRecords;

  const sortSelect = $("#rec-sort");
  if (sortSelect) sortSelect.onchange = loadRecords;

  const resetFiltersBtn = $("#btn-rec-reset-filters");
  if (resetFiltersBtn) {
    resetFiltersBtn.onclick = () => {
      if (recSearch) recSearch.value = "";
      if (dateFrom) dateFrom.value = "";
      if (dateTo) dateTo.value = "";
      if (sortSelect) sortSelect.value = "id_desc";
      loadRecords();
    };
  }

  const exportBtn = $("#btn-rec-export");
  if (exportBtn) {
    exportBtn.onclick = () => {
      if (!state.records.length) return toast("No records to export.", "bad");
      const head = ["ID", "Name", "Designation", "Address", "Telephone", "Mobile Phone", "Email", "Created At"];
      const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
      const lines = [head.map(q).join(",")].concat(
        state.records.map((r) => [
          r.id, r.name, r.job_title, r.address, r.telephone, r.mobile_phone, r.email, r.created_at
        ].map(q).join(","))
      );
      const blob = new Blob(["\uFEFF" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
      const a = Object.assign(document.createElement("a"), {
        href: URL.createObjectURL(blob),
        download: `Visitor_Database_Export_${new Date().toISOString().slice(0,10)}.csv`,
      });
      a.click();
      URL.revokeObjectURL(a.href);
    };
  }
}

// ------------------------------------------------------------------ initialization
function init() {
  setupEngines();
  setupUpload();
  setupTable();
  setupTabs();
  setupModal();
  setupDrawer();
}

init();
