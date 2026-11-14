// LearnBox 공용 웹 UI (PC 브라우저 + 휴대폰). 같은 코드, 같은 기능, 같은 데이터.
"use strict";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort.min.js";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const DAYS = ["월", "화", "수", "목", "금", "토", "일"];

function store(key, val) {
  try {
    if (val === undefined) return localStorage.getItem(key);
    if (val === null) localStorage.removeItem(key);
    else localStorage.setItem(key, val);
  } catch (e) { return null; }
}

const S = { token: store("lb_token"), role: store("lb_role"), offline: false, docsOffset: 0, timer: null, tab: "search" };

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function hl(s) { return esc(s).replace(/\u0002/g, "<mark>").replace(/\u0003/g, "</mark>"); }
function toast(msg, ms = 3500) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), ms);
}
function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString("ko-KR", { dateStyle: "short", timeStyle: "short" });
}
function fmtBytes(n) {
  if (n > 1e9) return (n / 1e9).toFixed(2) + "GB";
  if (n > 1e6) return (n / 1e6).toFixed(1) + "MB";
  return Math.round(n / 1e3) + "KB";
}

// ---------------- API ----------------
async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (S.token) headers.Authorization = "Bearer " + S.token;
  if (opts.body && typeof opts.body !== "string") { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(opts.body); }
  const r = await fetch(path, { ...opts, headers });
  if (r.status === 401 && !path.endsWith("/login")) { logout(true); throw new Error("다시 로그인하세요."); }
  if (opts.raw) { if (!r.ok) throw new Error("HTTP " + r.status); return r; }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || "HTTP " + r.status);
  return data;
}
const post = (p, body) => api(p, { method: "POST", body: body || {} });

// ---------------- 기기 저장소 (IndexedDB: HTTP에서도 동작) ----------------
const idb = {
  db: null,
  async open() {
    if (this.db) return this.db;
    this.db = await new Promise((res, rej) => {
      const r = indexedDB.open("learnbox", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("kv");
      r.onsuccess = () => res(r.result);
      r.onerror = () => rej(r.error);
    });
    return this.db;
  },
  async get(k) {
    const db = await this.open();
    return new Promise((res, rej) => { const r = db.transaction("kv").objectStore("kv").get(k); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
  },
  async set(k, v) {
    const db = await this.open();
    return new Promise((res, rej) => { const t = db.transaction("kv", "readwrite"); t.objectStore("kv").put(v, k); t.oncomplete = () => res(); t.onerror = () => rej(t.error); });
  },
};

// ---------------- 로그인 ----------------
function showApp() {
  $("#login").hidden = true;
  $("#tabs").hidden = false;
  $("#main").hidden = false;
  $("#logout").hidden = S.offline;
  $("#who").textContent = S.offline ? "오프라인" : S.role === "admin" ? "관리자" : "사용자";
  $$(".admin-only").forEach((el) => (el.hidden = S.role !== "admin" || S.offline));
  $$(".admin-hint").forEach((el) => (el.hidden = S.role === "admin"));
  $$("#tabs button").forEach((b) => (b.disabled = S.offline && !["search", "models"].includes(b.dataset.tab)));
  switchTab(S.tab);
  refreshSnapInfo();
  refreshLocalModelInfo();
}
function showLogin(msg) {
  $("#login").hidden = false; $("#tabs").hidden = true; $("#main").hidden = true; $("#logout").hidden = true;
  $("#who").textContent = "";
  $("#login-msg").textContent = msg || "";
  idb.get("snapshot").then((s) => ($("#offline-enter").hidden = !s)).catch(() => {});
}
function logout(expired) {
  if (S.token && !expired) api("/api/logout", { method: "POST" }).catch(() => {});
  S.token = null; S.role = null; store("lb_token", null); store("lb_role", null);
  showLogin(expired ? "세션이 만료되었습니다." : "");
}
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const r = await post("/api/login", { pin: $("#pin").value });
    S.token = r.token; S.role = r.role; S.offline = false;
    store("lb_token", r.token); store("lb_role", r.role);
    $("#pin").value = "";
    showApp();
  } catch (err) { $("#login-msg").textContent = err.message; }
});
$("#logout").addEventListener("click", () => logout(false));
$("#offline-enter").addEventListener("click", () => { S.offline = true; S.role = null; showApp(); });

// ---------------- 탭 ----------------
function switchTab(name) {
  S.tab = name;
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => (t.hidden = t.id !== "tab-" + name));
  clearInterval(S.timer);
  if (S.offline) return;
  const loaders = { search: loadModelsSelect, collect: loadCollect, train: loadTrain, models: loadModels, jobs: loadJobs, admin: loadAdmin };
  (loaders[name] || (() => {}))();
  if (name === "jobs" || name === "collect" || name === "train") S.timer = setInterval(() => (name === "jobs" ? loadJobs() : name === "collect" ? loadGate() : loadTrain(true)), 3000);
}
$$("#tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));

function updateNet() {
  const on = navigator.onLine;
  const n = $("#net");
  n.textContent = on ? "온라인" : "오프라인";
  n.classList.toggle("off", !on);
}
window.addEventListener("online", updateNet);
window.addEventListener("offline", updateNet);

// ---------------- 검색 / 질문 ----------------
function renderResults(rows, offline) {
  $("#search-mode").textContent = offline ? "이 기기에 저장된 사본에서 검색했습니다." : "";
  if (!rows.length) { $("#results").innerHTML = '<div class="card muted">결과가 없습니다.</div>'; return; }
  $("#results").innerHTML = '<div class="card">' + rows.map((r) => `
    <div class="result">
      <a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.title || r.url)}</a>
      <div class="url">${esc(r.url)}</div>
      <div class="snip">${hl(r.snippet)}</div>
    </div>`).join("") + "</div>";
}

async function offlineSearch(q) {
  const snap = await idb.get("snapshot");
  if (!snap) throw new Error("이 기기에 저장된 사본이 없습니다.");
  const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
  const scored = [];
  for (const d of snap.documents) {
    const text = (d.title + "\n" + d.text).toLowerCase();
    let score = 0;
    for (const t of terms) { let i = -1, c = 0; while ((i = text.indexOf(t, i + 1)) !== -1 && c < 50) c++; if (!c) { score = 0; break; } score += c; }
    if (score) {
      const pos = d.text.toLowerCase().indexOf(terms[0]);
      const start = Math.max(0, pos - 60);
      let snip = d.text.slice(start, start + 200);
      for (const t of terms) snip = snip.replace(new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), (m) => "\u0002" + m + "\u0003");
      scored.push({ ...d, score, snippet: (start ? "…" : "") + snip + "…" });
    }
  }
  return scored.sort((a, b) => b.score - a.score).slice(0, 20);
}

$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#q").value.trim();
  $("#answer").hidden = true;
  if (S.offline || !navigator.onLine) {
    try { renderResults(await offlineSearch(q), true); } catch (err) { toast(err.message); }
    return;
  }
  try {
    const r = await api("/api/search?q=" + encodeURIComponent(q));
    renderResults(r.results, false);
  } catch (err) {
    try { renderResults(await offlineSearch(q), true); } catch (_) { toast(err.message); }
  }
});

$("#ask-btn").addEventListener("click", async () => {
  const q = $("#q").value.trim();
  if (!q) return toast("질문을 입력하세요.");
  if (S.offline) return toast("오프라인에서는 [모델] 탭의 온디바이스 실행을 쓰세요.");
  const box = $("#answer");
  box.hidden = false;
  box.innerHTML = '<p class="muted">답변 생성 중… (CPU에서는 시간이 걸릴 수 있습니다)</p>';
  try {
    const r = await post("/api/ask", { q, model_id: $("#ask-model").value || null });
    box.innerHTML = (r.answer ? `<h3>답변 <span class="muted small">${esc(r.model || "")}</span></h3><div class="out">${esc(r.answer)}</div>` : "<h3>관련 자료</h3>")
      + (r.note ? `<p class="muted small">${esc(r.note)}</p>` : "");
    renderResults(r.sources, false);
  } catch (err) { box.innerHTML = `<p class="error">${esc(err.message)}</p>`; }
});

async function loadModelsSelect() {
  try {
    const { models } = await api("/api/models");
    const sel = $("#ask-model");
    const cur = sel.value;
    sel.innerHTML = '<option value="">없음 (검색만)</option>' + models.map((m) => `<option value="${m.id}">${esc(m.name)} (${m.kind === "pc" ? "PC" : "모바일"})</option>`).join("");
    sel.value = cur;
  } catch (e) { /* 무시 */ }
}

async function refreshSnapInfo() {
  try {
    const s = await idb.get("snapshot");
    $("#snap-info").textContent = s ? `저장됨: 문서 ${s.documents.length}개 · ${new Date(s.saved_at).toLocaleString("ko-KR")}` : "아직 저장된 사본이 없습니다.";
  } catch (e) { $("#snap-info").textContent = "이 브라우저는 기기 저장을 지원하지 않습니다."; }
}
$("#snap-btn").addEventListener("click", async () => {
  try {
    const r = await api("/api/snapshot");
    await idb.set("snapshot", { documents: r.documents, saved_at: Date.now() });
    refreshSnapInfo();
    toast(`문서 ${r.documents.length}개를 이 기기에 저장했습니다.`);
  } catch (err) { toast(err.message); }
});

// ---------------- 수집 ----------------
async function loadGate() {
  try {
    const st = await api("/api/status");
    const g = st.gate;
    const el = $("#gate");
    el.className = "banner " + (g.allowed ? "ok" : "warn");
    el.textContent = g.allowed
      ? `지금은 수집 허용 시간입니다 (${g.windows}, ${g.timezone}).`
      : `지금은 수집할 수 없습니다. 허용 시간: ${g.windows}. 다음 허용: ${fmtTime(g.next_allowed)}`;
    const names = { wikipedia: "위키백과", searxng: "SearXNG", brave: "Brave Search", none: "사용 안 함" };
    $("#provider-info").textContent = `검색 제공자: ${names[st.search_provider] || st.search_provider}. 결과 페이지를 가져와 자료로 저장합니다.`;
    const ac = st.auto_collect;
    $("#auto-info").textContent = ac.enabled ? `자동 수집 켜짐: 허용 시간대에 ${ac.interval_min}분마다 목록에서 하나씩 수집합니다.` : "자동 수집이 꺼져 있습니다 (관리 > 설정에서 켤 수 있습니다).";
    $("#doc-count").textContent = `문서 ${st.counts.documents}개 · ${fmtBytes(st.counts.chars * 2)}`;
  } catch (e) { /* 무시 */ }
}
async function loadWatch() {
  const { items } = await api("/api/watch");
  $("#watch-list").innerHTML = items.map((w) => `
    <li><span class="grow">${w.kind === "search" ? "🔎" : "🔗"} ${esc(w.value)} <span class="muted small">${w.last_run ? "최근 " + fmtTime(w.last_run) : "아직 안 함"}</span></span>
    <button class="danger small" data-del-watch="${w.id}">삭제</button></li>`).join("") || '<li class="muted">비어 있음</li>';
}
async function loadDocs(append) {
  if (!append) S.docsOffset = 0;
  const { documents } = await api("/api/documents?offset=" + S.docsOffset);
  const html = documents.map((d) => `
    <li><span class="grow"><a href="${esc(d.url)}" target="_blank" rel="noopener noreferrer">${esc(d.title || d.url)}</a>
      <br><span class="muted small">${esc(d.source)}${d.query ? " · " + esc(d.query) : ""} · ${d.chars.toLocaleString()}자 · ${fmtTime(d.fetched_at)}</span></span>
      ${S.role === "admin" ? `<button class="danger small" data-del-doc="${d.id}">삭제</button>` : ""}</li>`).join("");
  $("#docs").innerHTML = append ? $("#docs").innerHTML + html : html || '<li class="muted">아직 없음</li>';
  S.docsOffset += documents.length;
  $("#docs-more").hidden = documents.length < 50;
}
function loadCollect() { loadGate(); loadWatch().catch(() => {}); loadDocs(false).catch(() => {}); }
$("#docs-more").addEventListener("click", () => loadDocs(true));

$("#links-btn").addEventListener("click", async () => {
  const urls = $("#links").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!urls.length) return toast("링크를 입력하세요.");
  try {
    const r = await post("/api/collect/links", { urls, depth: +$("#depth").value });
    $("#links").value = "";
    toast(`수집 작업 #${r.job_id} 시작. [작업] 탭에서 진행 상황을 볼 수 있습니다.`);
  } catch (err) { toast(err.message, 6000); }
});
$("#websearch-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const r = await post("/api/collect/search", { query: $("#wq").value.trim() });
    toast(`검색 수집 작업 #${r.job_id} 시작.`);
    $("#wq").value = "";
  } catch (err) { toast(err.message, 6000); }
});
$("#watch-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await post("/api/watch", { kind: $("#watch-kind").value, value: $("#watch-value").value.trim() });
    $("#watch-value").value = "";
    loadWatch();
  } catch (err) { toast(err.message); }
});
document.addEventListener("click", async (e) => {
  const w = e.target.closest("[data-del-watch]");
  if (w) { await api("/api/watch/" + w.dataset.delWatch, { method: "DELETE" }).catch((x) => toast(x.message)); loadWatch(); }
  const d = e.target.closest("[data-del-doc]");
  if (d && confirm("이 문서를 삭제할까요?")) { await api("/api/documents/" + d.dataset.delDoc, { method: "DELETE" }).catch((x) => toast(x.message)); loadDocs(false); }
  const c = e.target.closest("[data-cancel]");
  if (c) { await post(`/api/jobs/${c.dataset.cancel}/cancel`).catch((x) => toast(x.message)); loadJobs(); }
  const g = e.target.closest("[data-get-model]");
  if (g) downloadModel(+g.dataset.getModel, g);
  const dr = e.target.closest("[data-drive]");
  if (dr) $("#target").value = dr.dataset.drive;
});

// ---------------- 학습 ----------------
async function loadTrain(quiet) {
  try {
    const st = await api("/api/status");
    const c = st.capabilities;
    const ok = (b) => (b ? "있음" : "없음");
    $("#caps").innerHTML = `연산 장치: <b>${esc(c.device)}</b> · PyTorch ${ok(c.torch)} · transformers ${ok(c.transformers)} · peft ${ok(c.peft)}<br>`
      + `학습 자료: 문서 ${st.counts.documents}개, 조각 ${st.counts.chunks}개`
      + (c.torch ? "" : '<br><span class="error">학습하려면 PC에서 <code>pip install -r requirements-train.txt</code>를 실행하세요.</span>');
    if (!quiet) {
      const s = await api("/api/settings").catch(() => null);
      if (s) { $("#pc-base").placeholder = s.pc_training.base_model; $("#pc-steps").placeholder = s.pc_training.max_steps; $("#mb-steps").placeholder = s.mobile_training.steps; }
    }
  } catch (e) { /* 무시 */ }
}
$("#pc-train").addEventListener("click", async () => {
  const body = {};
  if ($("#pc-base").value.trim()) body.base_model = $("#pc-base").value.trim();
  if ($("#pc-steps").value) body.max_steps = +$("#pc-steps").value;
  try { const r = await post("/api/train/pc", body); toast(`PC 학습 작업 #${r.job_id} 시작.`); } catch (err) { toast(err.message, 6000); }
});
$("#mb-train").addEventListener("click", async () => {
  const body = { preset: $("#mb-preset").value };
  if ($("#mb-steps").value) body.steps = +$("#mb-steps").value;
  try { const r = await post("/api/train/mobile", body); toast(`휴대폰용 학습 작업 #${r.job_id} 시작.`); } catch (err) { toast(err.message, 6000); }
});

// ---------------- 모델 ----------------
function infoList(i) {
  const rows = [
    ["기반 모델", i.base_model], ["크기", i.preset], ["파라미터", i.params ? i.params.toLocaleString() : i.trainable_params ? `${i.trainable_params.toLocaleString()} 학습 / ${i.total_params.toLocaleString()} 전체` : null],
    ["학습 단계", i.steps], ["장치", i.device], ["평가 손실 (전 → 후)", i.eval_loss_before != null ? `${i.eval_loss_before} → ${i.eval_loss_after}` : null],
    ["학습 시간", i.seconds != null ? `${Math.round(i.seconds / 60)}분 ${Math.round(i.seconds % 60)}초` : null],
  ].filter((r) => r[1] != null && r[1] !== "");
  return '<dl class="kv">' + rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("") + "</dl>";
}
async function loadModels() {
  try {
    const { models } = await api("/api/models");
    $("#models").innerHTML = models.map((m) => `
      <div class="card">
        <h3>${esc(m.name)} <span class="status">${m.kind === "pc" ? "PC용 LoRA" : "휴대폰용 ONNX"}</span></h3>
        <p class="muted small">${fmtTime(m.created_at)}</p>
        ${infoList(m.info)}
        ${m.kind === "mobile" ? (m.info.onnx ? `<button class="secondary small" data-get-model="${m.id}">이 기기로 받기</button>` : '<p class="error small">ONNX 변환 실패</p>') : '<p class="muted small">[검색] 탭의 "답변 모델"에서 선택해 PC에서 답변을 생성합니다.</p>'}
      </div>`).join("") || '<div class="card muted">아직 학습한 모델이 없습니다.</div>';
  } catch (e) { toast(e.message); }
}
async function downloadModel(id, btn) {
  btn.disabled = true; btn.textContent = "받는 중…";
  try {
    const [onnx, vocab, info] = await Promise.all([
      api(`/api/models/${id}/files/model.onnx`, { raw: true }).then((r) => r.arrayBuffer()),
      api(`/api/models/${id}/files/vocab.json`),
      api(`/api/models/${id}/files/info.json`),
    ]);
    await idb.set("local_model", { id, onnx, itos: vocab.itos, info, saved_at: Date.now() });
    refreshLocalModelInfo();
    toast(`모델을 이 기기에 저장했습니다 (${fmtBytes(onnx.byteLength)}).`);
  } catch (err) { toast(err.message); }
  btn.disabled = false; btn.textContent = "이 기기로 받기";
}
async function refreshLocalModelInfo() {
  try {
    const m = await idb.get("local_model");
    $("#local-model-info").textContent = m ? `저장된 모델: #${m.id} (${m.info.preset}, ${fmtBytes(m.onnx.byteLength)}). 인터넷 없이 이 기기에서 실행됩니다. 처음 한 번은 실행 엔진을 받아야 합니다.` : "받은 모델이 없습니다. 위 목록에서 휴대폰용 모델을 받으세요.";
  } catch (e) { /* 무시 */ }
}
function loadScript(src) {
  return new Promise((res, rej) => { const s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = () => rej(new Error("실행 엔진을 불러오지 못했습니다 (인터넷 필요).")); document.head.appendChild(s); });
}
let ortSession = null, ortModelId = null;
$("#local-gen-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const out = $("#local-out");
  out.hidden = false; out.textContent = "준비 중…";
  try {
    const m = await idb.get("local_model");
    if (!m) throw new Error("받은 모델이 없습니다.");
    if (!window.ort) { await loadScript(ORT_URL); ort.env.wasm.wasmPaths = ORT_WASM; }
    if (!ortSession || ortModelId !== m.id) { ortSession = await ort.InferenceSession.create(new Uint8Array(m.onnx), { executionProviders: ["wasm"] }); ortModelId = m.id; }
    const prompt = $("#local-prompt").value;
    const stoi = new Map(m.itos.map((c, i) => [c, i]));
    const V = m.itos.length, ctxLen = m.info.seq_len || 128;
    const ids = [2, ...Array.from(prompt).map((c) => stoi.get(c) ?? 1)];
    let text = prompt;
    out.textContent = text;
    for (let step = 0; step < 120; step++) {
      const ctx = ids.slice(-ctxLen);
      const input = new ort.Tensor("int64", BigInt64Array.from(ctx.map((x) => BigInt(x))), [1, ctx.length]);
      const res = await ortSession.run({ input_ids: input });
      const logits = res.logits.data.subarray((ctx.length - 1) * V, ctx.length * V);
      const next = sampleTopK(logits, 20, 0.8);
      if (next === 3) break;
      ids.push(next);
      if (next >= 4) { text += m.itos[next]; out.textContent = text; }
    }
  } catch (err) { out.textContent = err.message; }
});
function sampleTopK(logits, k, temp) {
  const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, k);
  const mx = logits[idx[0]];
  const p = idx.map((i) => Math.exp((logits[i] - mx) / temp));
  const sum = p.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum;
  for (let j = 0; j < idx.length; j++) { r -= p[j]; if (r <= 0) return idx[j]; }
  return idx[0];
}

// ---------------- 작업 ----------------
const KIND = { collect_links: "링크 수집", collect_search: "검색 수집", train_pc: "PC 모델 학습", train_mobile: "휴대폰용 모델 학습", export: "외장 저장 내보내기" };
const STATUS = { queued: "대기", running: "진행 중", done: "완료", failed: "실패", cancelled: "취소", blocked: "시간 밖 중지", interrupted: "중단" };
function summary(j) {
  const r = j.result || {};
  if (j.kind.startsWith("collect")) return r.new != null ? `새 문서 ${r.new} · 갱신 ${r.updated} · 변화 없음 ${r.unchanged} · 짧아서 제외 ${r.skipped} · 오류 ${(r.errors || []).length}` : "";
  if (j.kind.startsWith("train")) return r.eval_loss_after != null ? `평가 손실 ${r.eval_loss_before} → ${r.eval_loss_after}` : "";
  if (j.kind === "export") return r.path ? `${r.path} (${r.files}개 파일, ${fmtBytes(r.bytes)})` : "";
  return "";
}
async function loadJobs() {
  try {
    const { jobs } = await api("/api/jobs");
    $("#jobs").innerHTML = jobs.map((j) => {
      const active = j.status === "running" || j.status === "queued";
      const canCancel = active && (S.role === "admin" || j.kind.startsWith("collect"));
      const errs = (j.result && j.result.errors && j.result.errors.length) ? `<details><summary>오류 ${j.result.errors.length}건</summary><pre>${esc(j.result.errors.join("\n"))}</pre></details>` : "";
      const tb = j.result && j.result.traceback ? `<details><summary>자세한 오류</summary><pre>${esc(j.result.traceback)}</pre></details>` : "";
      return `<div class="card job">
        <div class="row wrap gap"><b>#${j.id} ${KIND[j.kind] || j.kind}</b> <span class="status ${j.status}">${STATUS[j.status] || j.status}</span>
          <span class="muted small">${esc(j.actor)} · ${fmtTime(j.created_at)}</span>
          ${canCancel ? `<button class="danger small" data-cancel="${j.id}">취소</button>` : ""}</div>
        ${active ? `<div class="bar"><div style="width:${Math.round((j.progress || 0) * 100)}%"></div></div>` : ""}
        <div class="small">${esc(j.message || "")}</div>
        <div class="small muted">${esc(summary(j))}</div>${errs}${tb}
      </div>`;
    }).join("") || '<div class="card muted">작업이 없습니다.</div>';
  } catch (e) { /* 무시 */ }
}

// ---------------- 관리 ----------------
function windowRow(w) {
  const div = document.createElement("div");
  div.className = "window-row";
  div.innerHTML = `<span class="days">${DAYS.map((d, i) => `<label><input type="checkbox" value="${i}" ${w.days.includes(i) ? "checked" : ""}>${d}</label>`).join("")}</span>
    <input type="time" class="w-start" value="${w.start}"> ~ <input type="time" class="w-end" value="${w.end === "24:00" ? "23:59" : w.end}">
    <button class="danger small" type="button">삭제</button>`;
  div.querySelector("button").addEventListener("click", () => div.remove());
  return div;
}
async function loadAdmin() {
  if (S.role !== "admin") return;
  try {
    const [s, d, a] = await Promise.all([api("/api/settings"), api("/api/drives"), api("/api/audit")]);
    $("#drives").innerHTML = d.drives.length ? d.drives.map((x) => `<button class="secondary small" data-drive="${esc(x.path)}">${esc(x.path)} · 남은 ${x.free_gb}GB / ${x.total_gb}GB</button>`).join("") : '<span class="muted small">자동으로 찾은 외장 장치가 없습니다. 경로를 직접 입력하세요.</span>';
    const wl = $("#windows"); wl.innerHTML = "";
    s.allowed_windows.forEach((w) => wl.appendChild(windowRow(w)));
    $("#s-tz").value = s.timezone;
    $("#s-allowed").value = s.allowed_domains.join("\n");
    $("#s-blocked").value = s.blocked_domains.join("\n");
    $("#s-interval").value = s.request_interval_sec;
    $("#s-maxpages").value = s.max_pages_per_job;
    $("#s-provider").value = s.search.provider;
    $("#s-wikilang").value = s.search.wikipedia_lang;
    $("#s-results").value = s.search.results_to_fetch;
    $("#s-searx").value = s.search.searxng_url;
    $("#s-brave").value = s.search.brave_api_key;
    $("#s-auto").checked = s.auto_collect.enabled;
    $("#s-autoint").value = s.auto_collect.interval_min;
    $("#audit").innerHTML = a.audit.map((x) => `<li><span class="grow">${fmtTime(x.ts)} · <b>${esc(x.action)}</b> · ${esc(x.actor)}<br><span class="muted">${esc(x.detail)}</span></span></li>`).join("");
  } catch (e) { toast(e.message); }
}
$("#add-window").addEventListener("click", () => $("#windows").appendChild(windowRow({ days: [0, 1, 2, 3, 4, 5, 6], start: "01:00", end: "06:00" })));
$("#save-settings").addEventListener("click", async () => {
  const lines = (s) => s.split("\n").map((x) => x.trim()).filter(Boolean);
  const windows = $$("#windows .window-row").map((r) => {
    const end = r.querySelector(".w-end").value;
    return { days: Array.from(r.querySelectorAll(".days input:checked")).map((x) => +x.value), start: r.querySelector(".w-start").value, end: end === "23:59" ? "24:00" : end };
  }).filter((w) => w.days.length && w.start && w.end);
  const body = {
    allowed_windows: windows, timezone: $("#s-tz").value.trim(),
    allowed_domains: lines($("#s-allowed").value), blocked_domains: lines($("#s-blocked").value),
    request_interval_sec: +$("#s-interval").value, max_pages_per_job: +$("#s-maxpages").value,
    search: { provider: $("#s-provider").value, wikipedia_lang: $("#s-wikilang").value.trim(), results_to_fetch: +$("#s-results").value,
      searxng_url: $("#s-searx").value.trim(), brave_api_key: $("#s-brave").value },
    auto_collect: { enabled: $("#s-auto").checked, interval_min: +$("#s-autoint").value },
  };
  try { await post("/api/settings", body); $("#settings-msg").textContent = "저장했습니다."; loadAdmin(); } catch (err) { $("#settings-msg").textContent = err.message; }
});
$("#export-btn").addEventListener("click", async () => {
  const target = $("#target").value.trim();
  if (!target) return toast("저장 위치를 입력하세요.");
  const include = $$(".inc:checked").map((x) => x.value);
  if (!confirm(`${target} 에 내보낼까요?`)) return;
  try { const r = await post("/api/export", { target, include }); toast(`내보내기 작업 #${r.job_id} 시작. [작업] 탭에서 확인하세요.`); } catch (err) { toast(err.message, 6000); }
});

// ---------------- 시작 ----------------
async function boot() {
  updateNet();
  if ("serviceWorker" in navigator && window.isSecureContext) navigator.serviceWorker.register("/sw.js").catch(() => {});
  if (!S.token) return showLogin();
  try {
    const st = await api("/api/status");
    S.role = st.role; store("lb_role", st.role);
    showApp();
  } catch (e) {
    if (!navigator.onLine || e instanceof TypeError) { S.offline = true; showApp(); }
    else showLogin();
  }
}
boot();
