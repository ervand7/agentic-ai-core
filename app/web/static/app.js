"use strict";

/**
 * Tool registry — one entry per backend handler.
 * The UI is generated entirely from this config, so every endpoint
 * is covered and adding a new one is a single object.
 */
const TOOLS = [
  {
    id: "ask",
    group: "AI Tasks",
    name: "Ask",
    icon: "✦",
    method: "POST",
    path: "/ask",
    kind: "json",
    title: "Ask a question",
    desc: "Send a question and get a single AI-generated answer.",
    fields: [
      { name: "question", label: "Question", type: "textarea", required: true,
        placeholder: "Explain async/await in Python in simple words." },
    ],
  },
  {
    id: "ask-stream",
    group: "AI Tasks",
    name: "Ask (Stream)",
    icon: "≈",
    method: "POST",
    path: "/ask-stream",
    kind: "stream",
    title: "Ask — streaming",
    desc: "Same as Ask, but the answer streams in token by token.",
    fields: [
      { name: "question", label: "Question", type: "textarea", required: true,
        placeholder: "Write a short paragraph about FastAPI streaming." },
    ],
  },
  {
    id: "classify",
    group: "AI Tasks",
    name: "Classify",
    icon: "◑",
    method: "POST",
    path: "/classify",
    kind: "json",
    title: "Classify sentiment",
    desc: "Classify text into sentiment, plus a short summary and keywords.",
    fields: [
      { name: "text", label: "Text", type: "textarea", required: true,
        placeholder: "The product is good but shipping was slow." },
    ],
  },
  {
    id: "summarize",
    group: "AI Tasks",
    name: "Summarize",
    icon: "❡",
    method: "POST",
    path: "/summarize",
    kind: "json",
    title: "Summarize text",
    desc: "Condense longer text into 2-3 clear sentences.",
    fields: [
      { name: "text", label: "Text", type: "textarea", required: true,
        placeholder: "Paste a longer paragraph here…" },
    ],
  },
  {
    id: "extract-keywords",
    group: "AI Tasks",
    name: "Keywords",
    icon: "#",
    method: "POST",
    path: "/extract-keywords",
    kind: "json",
    title: "Extract keywords",
    desc: "Pull the most relevant key terms out of the text.",
    fields: [
      { name: "text", label: "Text", type: "textarea", required: true,
        placeholder: "FastAPI is often used for building async APIs in Python." },
    ],
  },
  {
    id: "translate",
    group: "AI Tasks",
    name: "Translate",
    icon: "⇄",
    method: "POST",
    path: "/translate",
    kind: "json",
    title: "Translate text",
    desc: "Translate input text into a target language.",
    fields: [
      { name: "text", label: "Text", type: "textarea", required: true,
        placeholder: "Good morning, how are you?" },
      { name: "target_language", label: "Target language", type: "text", required: true,
        placeholder: "Spanish" },
    ],
  },
  {
    id: "analyze-text",
    group: "AI Tasks",
    name: "Analyze",
    icon: "❖",
    method: "POST",
    path: "/analyze-text",
    kind: "json",
    title: "Analyze text",
    desc: "Combined NLP: summary, sentiment, keywords and detected language.",
    fields: [
      { name: "text", label: "Text", type: "textarea", required: true,
        placeholder: "FastAPI is great for building APIs quickly, but onboarding juniors needs better docs." },
    ],
  },
  {
    id: "upload",
    group: "Documents",
    name: "Upload",
    icon: "⬆",
    method: "POST",
    path: "/documents/upload",
    kind: "multipart",
    title: "Upload a document",
    desc: "Upload a .txt or .pdf file. It is chunked, embedded, and stored in Qdrant.",
    fields: [
      { name: "file", label: "Document (.txt, .pdf)", type: "file", accept: ".txt,.pdf", required: true },
    ],
  },
  {
    id: "search",
    group: "Documents",
    name: "Search",
    icon: "⌕",
    method: "POST",
    path: "/documents/search",
    kind: "json",
    title: "Semantic search",
    desc: "Search stored chunks by meaning. Optional filters refine the results.",
    fields: [
      { name: "query", label: "Query", type: "textarea", required: true,
        placeholder: "How does retry logic work?" },
      { name: "top_k", label: "Top K", type: "number", default: 3, min: 1, max: 10,
        hint: "How many results to return (1-10)." },
      { name: "filename", label: "Filename filter", type: "text", required: false,
        placeholder: "sample.txt", hint: "Optional — search inside one file only." },
      { name: "keyword", label: "Keyword", type: "text", required: false,
        placeholder: "retry", hint: "Optional — require this word (hybrid search)." },
      { name: "min_similarity", label: "Min similarity", type: "number", required: false,
        min: 0, max: 1, step: 0.05, placeholder: "0.5", hint: "Optional — drop weak matches (0-1)." },
    ],
  },
  {
    id: "ask-docs",
    group: "Documents",
    name: "Chat with Docs",
    icon: "✸",
    method: "POST",
    path: "/documents/ask",
    kind: "json",
    title: "Chat with your documents (RAG)",
    desc: "Ask a question. The answer is generated only from your uploaded documents, with citations.",
    fields: [
      { name: "question", label: "Question", type: "textarea", required: true,
        placeholder: "How does retry logic work in this project?" },
      { name: "top_k", label: "Top K", type: "number", required: false, min: 1, max: 20,
        placeholder: "4", hint: "Optional — how many chunks to use as context." },
      { name: "filename", label: "Filename filter", type: "text", required: false,
        placeholder: "sample.txt", hint: "Optional — answer from one file only." },
      { name: "min_similarity", label: "Min similarity", type: "number", required: false,
        min: 0, max: 1, step: 0.05, placeholder: "0.2", hint: "Optional — ignore weakly related context (0-1)." },
    ],
  },
  {
    id: "health",
    group: "System",
    name: "Health",
    icon: "♥",
    method: "GET",
    path: "/health",
    kind: "none",
    title: "Health check",
    desc: "Quick liveness check for the backend.",
    fields: [],
  },
];

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

let activeTool = TOOLS[0];
let lastResultJson = null;

/* ---------------- Navigation ---------------- */
function buildNav() {
  const nav = $("#nav");
  nav.innerHTML = "";
  const groups = [...new Set(TOOLS.map((t) => t.group))];
  for (const group of groups) {
    nav.appendChild(el("div", "nav-group-label", group));
    for (const tool of TOOLS.filter((t) => t.group === group)) {
      const btn = el("button", "nav-item");
      btn.type = "button";
      btn.dataset.id = tool.id;
      btn.appendChild(el("span", "nav-ico", tool.icon));
      btn.appendChild(el("span", "nav-name", tool.name));
      btn.appendChild(el("span", "nav-method", tool.method));
      btn.addEventListener("click", () => selectTool(tool.id));
      nav.appendChild(btn);
    }
  }
}

function selectTool(id) {
  activeTool = TOOLS.find((t) => t.id === id) || TOOLS[0];
  document.querySelectorAll(".nav-item").forEach((n) =>
    n.classList.toggle("active", n.dataset.id === activeTool.id)
  );
  $("#toolTitle").textContent = activeTool.title;
  $("#toolDesc").textContent = activeTool.desc;
  buildForm();
  resetResult();
}

/* ---------------- Form building ---------------- */
function buildForm() {
  const form = $("#toolForm");
  form.innerHTML = "";

  if (!activeTool.fields.length) {
    const note = el("p", "tool-desc");
    note.textContent = `No input needed — just press Run to call ${activeTool.method} ${activeTool.path}.`;
    form.appendChild(note);
  }

  for (const field of activeTool.fields) {
    const wrap = el("div", "field");
    const label = el("label", null, field.label);
    label.htmlFor = `f_${field.name}`;
    wrap.appendChild(label);

    if (field.type === "textarea") {
      const ta = el("textarea");
      ta.id = `f_${field.name}`;
      ta.name = field.name;
      if (field.placeholder) ta.placeholder = field.placeholder;
      if (field.required) ta.required = true;
      wrap.appendChild(ta);
    } else if (field.type === "file") {
      wrap.appendChild(buildFileDrop(field));
    } else {
      const input = el("input");
      input.type = field.type === "number" ? "number" : "text";
      input.id = `f_${field.name}`;
      input.name = field.name;
      if (field.placeholder) input.placeholder = field.placeholder;
      if (field.required) input.required = true;
      if (field.default !== undefined) input.value = field.default;
      if (field.min !== undefined) input.min = field.min;
      if (field.max !== undefined) input.max = field.max;
      if (field.step !== undefined) input.step = field.step;
      wrap.appendChild(input);
    }

    if (field.hint) wrap.appendChild(el("span", "hint", field.hint));
    form.appendChild(wrap);
  }
}

function buildFileDrop(field) {
  const drop = el("label", "file-drop");
  drop.innerHTML = `<div>Drop a file here or <strong>browse</strong></div>`;
  const input = el("input");
  input.type = "file";
  input.id = `f_${field.name}`;
  input.name = field.name;
  if (field.accept) input.accept = field.accept;
  if (field.required) input.required = true;
  const nameEl = el("div", "file-name");
  drop.appendChild(input);
  drop.appendChild(nameEl);

  input.addEventListener("change", () => {
    nameEl.textContent = input.files[0] ? input.files[0].name : "";
  });
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("drag"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("drag");
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      nameEl.textContent = input.files[0].name;
    }
  });
  return drop;
}

/* ---------------- Submit ---------------- */
async function handleSubmit(e) {
  e.preventDefault();
  const tool = activeTool;
  setLoading(true);

  try {
    if (tool.kind === "stream") {
      await runStream(tool);
    } else {
      await runStandard(tool);
    }
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    setLoading(false);
  }
}

function collectPayload(tool) {
  const payload = {};
  for (const field of tool.fields) {
    const node = $(`#f_${field.name}`);
    if (!node) continue;
    let value = node.value;
    if (field.type === "number") {
      if (value === "" || value === null) continue;
      value = Number(value);
    } else {
      if (typeof value === "string") value = value.trim();
      if (value === "" && !field.required) continue;
    }
    payload[field.name] = value;
  }
  return payload;
}

async function runStandard(tool) {
  let options = { method: tool.method };

  if (tool.kind === "json") {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(collectPayload(tool));
  } else if (tool.kind === "multipart") {
    const fd = new FormData();
    const fileInput = $(`#f_file`);
    if (!fileInput || !fileInput.files.length) throw new Error("Please choose a file first.");
    fd.append("file", fileInput.files[0]);
    options.body = fd;
  }

  const res = await fetch(tool.path, options);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }

  setStatus(res.status, res.ok);
  if (!res.ok) {
    showError(data.detail || `Request failed (${res.status}).`);
    return;
  }
  lastResultJson = data;
  renderResult(tool, data);
}

async function runStream(tool) {
  const res = await fetch(tool.path, {
    method: tool.method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(collectPayload(tool)),
  });
  setStatus(res.status, res.ok);

  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try { detail = JSON.parse(text).detail; } catch {}
    showError(detail || `Request failed (${res.status}).`);
    return;
  }

  const body = $("#resultBody");
  body.innerHTML = "";
  const block = el("div", "result-block");
  const answer = el("div", "answer-text");
  const cursor = el("span", "stream-cursor");
  block.appendChild(answer);
  block.appendChild(cursor);
  body.appendChild(block);
  $("#copyBtn").hidden = true;

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    full += decoder.decode(value, { stream: true });
    answer.textContent = full;
    body.scrollTop = body.scrollHeight;
  }
  cursor.remove();
  lastResultJson = { answer: full };
  $("#copyBtn").hidden = false;
}

/* ---------------- Result rendering ---------------- */
function renderResult(tool, data) {
  const body = $("#resultBody");
  body.innerHTML = "";
  const block = el("div", "result-block");

  // Primary text-like fields
  const textField = data.answer ?? data.summary ?? data.translation;
  if (textField !== undefined && tool.id !== "analyze-text") {
    block.appendChild(sectionLabel(labelForTool(tool)));
    block.appendChild(answerText(textField));
  }

  if (data.sentiment) {
    block.appendChild(sectionLabel("Sentiment"));
    const s = el("span", `sentiment ${data.sentiment}`, data.sentiment);
    block.appendChild(s);
  }

  if (tool.id === "analyze-text" && data.summary) {
    block.appendChild(sectionLabel("Summary"));
    block.appendChild(answerText(data.summary));
  }

  if (Array.isArray(data.keywords) && data.keywords.length) {
    block.appendChild(sectionLabel("Keywords"));
    const chips = el("div", "chips");
    data.keywords.forEach((k) => chips.appendChild(el("span", "chip", k)));
    block.appendChild(chips);
  }

  if (data.language) {
    block.appendChild(sectionLabel("Detected language"));
    block.appendChild(answerText(data.language));
  }

  // Document upload
  if (data.chunks_stored !== undefined) {
    block.appendChild(metaRow([
      ["File", data.filename],
      ["Chunks stored", data.chunks_stored],
      ["Characters", data.total_characters],
    ]));
  }

  // Search results
  if (Array.isArray(data.results)) {
    block.appendChild(sectionLabel(`Results (${data.results.length})`));
    if (!data.results.length) {
      block.appendChild(answerText("No matching chunks."));
    }
    data.results.forEach((hit) => block.appendChild(searchHit(hit)));
  }

  // RAG: grounding indicator + citations
  if (data.used_context !== undefined) {
    const grounded = el("span", `grounding ${data.used_context ? "ok" : "none"}`,
      data.used_context ? "Grounded in documents" : "No relevant context — abstained");
    block.appendChild(grounded);
  }
  if (Array.isArray(data.citations) && data.citations.length) {
    block.appendChild(sectionLabel(`Citations (${data.citations.length})`));
    data.citations.forEach((c) => block.appendChild(citationCard(c)));
  }

  // Health
  if (data.status && Object.keys(data).length === 1) {
    block.appendChild(metaRow([["Status", data.status]]));
  }

  // Model / token meta footer
  const meta = [];
  if (data.model) meta.push(["Model", data.model]);
  if (data.tokens_used !== undefined) meta.push(["Tokens", data.tokens_used]);
  if (data.prompt_version) meta.push(["Prompt", data.prompt_version]);
  if (meta.length) {
    block.appendChild(el("hr", "divider"));
    block.appendChild(metaRow(meta));
  }

  // Raw JSON
  block.appendChild(rawJson(data));
  body.appendChild(block);
  $("#copyBtn").hidden = false;
}

function labelForTool(tool) {
  if (tool.id === "translate") return "Translation";
  if (tool.id === "summarize") return "Summary";
  return "Answer";
}

function sectionLabel(text) { return el("div", "section-label", text); }
function answerText(text) { return el("div", "answer-text", String(text)); }

function metaRow(pairs) {
  const row = el("div", "meta-row");
  for (const [k, v] of pairs) {
    if (v === undefined || v === null) continue;
    const m = el("div", "meta");
    m.appendChild(el("span", "meta-k", k));
    m.appendChild(el("span", "meta-v", String(v)));
    row.appendChild(m);
  }
  return row;
}

function searchHit(hit) {
  const card = el("div", "search-hit");
  const top = el("div", "hit-top");
  const score = typeof hit.similarity === "number" ? hit.similarity : 0;
  top.appendChild(el("span", "hit-score", `similarity ${score.toFixed(4)}`));
  if (hit.filename) top.appendChild(el("span", "hit-file", hit.filename));
  card.appendChild(top);
  card.appendChild(el("div", "hit-text", hit.text || ""));
  const bar = el("div", "score-bar");
  const fill = el("div", "score-fill");
  fill.style.width = `${Math.max(0, Math.min(1, score)) * 100}%`;
  bar.appendChild(fill);
  card.appendChild(bar);
  return card;
}

function citationCard(c) {
  const card = el("div", "search-hit");
  const top = el("div", "hit-top");
  const left = el("div", "cite-left");
  left.appendChild(el("span", "cite-badge", `[${c.index}]`));
  if (c.filename) left.appendChild(el("span", "hit-file", c.filename));
  top.appendChild(left);
  const score = typeof c.similarity === "number" ? c.similarity : 0;
  top.appendChild(el("span", "hit-score", `similarity ${score.toFixed(4)}`));
  card.appendChild(top);
  card.appendChild(el("div", "hit-text", c.text || ""));
  return card;
}

function rawJson(data) {
  const details = el("details", "raw");
  details.appendChild(el("summary", null, "Raw JSON"));
  const pre = el("pre", "code", JSON.stringify(data, null, 2));
  details.appendChild(pre);
  return details;
}

/* ---------------- UI helpers ---------------- */
function setLoading(loading) {
  const btn = $("#submitBtn");
  btn.disabled = loading;
  btn.querySelector(".btn-label").textContent = loading ? "Running" : "Run";
  btn.querySelector(".spinner").hidden = !loading;
}

function setStatus(code, ok) {
  const pill = $("#statusPill");
  pill.hidden = false;
  pill.textContent = `${code}`;
  pill.className = `status-pill ${ok ? "ok" : "err"}`;
}

function showError(message) {
  const body = $("#resultBody");
  body.innerHTML = "";
  const box = el("div", "error-box", message);
  body.appendChild(box);
  $("#copyBtn").hidden = true;
}

function resetResult() {
  const body = $("#resultBody");
  body.innerHTML = "";
  const empty = el("div", "empty-state");
  empty.appendChild(el("div", "empty-glyph", "⟡"));
  empty.appendChild(el("p", null, "Run a tool to see results here."));
  body.appendChild(empty);
  $("#statusPill").hidden = true;
  $("#copyBtn").hidden = true;
  lastResultJson = null;
}

/* ---------------- Health ---------------- */
async function checkHealth() {
  const dot = $("#healthDot");
  const text = $("#healthText");
  try {
    const res = await fetch("/health");
    if (res.ok) {
      dot.className = "health-dot ok";
      text.textContent = "online";
    } else {
      dot.className = "health-dot down";
      text.textContent = "degraded";
    }
  } catch {
    dot.className = "health-dot down";
    text.textContent = "offline";
  }
}

/* ---------------- Init ---------------- */
function init() {
  buildNav();
  selectTool(TOOLS[0].id);
  $("#toolForm").addEventListener("submit", handleSubmit);
  $("#clearBtn").addEventListener("click", () => { buildForm(); resetResult(); });
  $("#copyBtn").addEventListener("click", async () => {
    if (!lastResultJson) return;
    await navigator.clipboard.writeText(JSON.stringify(lastResultJson, null, 2));
    const btn = $("#copyBtn");
    const prev = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => (btn.textContent = prev), 1200);
  });
  checkHealth();
  setInterval(checkHealth, 15000);
}

document.addEventListener("DOMContentLoaded", init);
