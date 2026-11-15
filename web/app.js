// LearnBox 공용 웹 UI v2 (PC 브라우저 + 휴대폰). 같은 코드, 같은 기능, 같은 데이터.
// - 서버 API(docs/SPEC.md §5)와 온보드 엔진(window.Onboard, /onboard.js)을 쓴다. 프레임워크 없음.
// - 서버에서 온 문자열은 모두 esc()로 거른 뒤 HTML에 넣는다. localStorage 접근은 store()로만 (try/catch).
"use strict";

const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
const OB = () => (typeof window !== "undefined" && window.Onboard) || null;
const DAYS = ["월", "화", "수", "목", "금", "토", "일"];

const SOURCE_LABEL = { link: "웹", search: "검색수집", file: "파일", auto: "자율" };
const MATCH_LABEL = { keyword: "키워드", semantic: "의미", both: "키워드+의미" };
const ENGINE_LABEL = { search: "검색만", tiny: "소형 AI", lora: "LoRA", onboard: "온보드", online: "온라인" };
const MODE_LABEL = { hybrid: "통합", keyword: "키워드", semantic: "의미" };
const KIND = {
  collect_links: "링크 수집", collect_search: "검색 수집", ingest_path: "파일 가져오기",
  index_embeddings: "의미 검색 색인", build_qa: "Q&A 데이터 생성", train_tiny: "소형 AI 학습", train_lora: "LoRA 학습",
  autolearn_train: "자율 학습(학습)", finalize_learning: "학습 종료·결과 내보내기", export: "외장 저장 내보내기",
  train_mobile: "소형 AI 학습", train_pc: "LoRA 학습",
};
const STATUS = { queued: "대기", running: "진행 중", done: "완료", failed: "실패", cancelled: "취소", blocked: "시간 밖 중지", interrupted: "중단" };
const PHASE = {
  idle: "대기", collecting: "수집 중", training: "학습 중", waiting_window: "허용 시간 기다리는 중",
  finishing: "학습 종료 중", cooldown: "다음 수집까지 쉬는 중",
};
const FIT_LABEL = { "phone-low": "휴대폰 저사양", "phone-high": "휴대폰 고사양", "pc-cpu": "PC CPU", "pc-gpu": "PC GPU" };
const TIER_LABEL = { "pc-cpu": "PC CPU", "pc-gpu": "PC GPU", "pc-gpu-high": "고성능 GPU", low: "저사양", mid: "중급", high: "고성능" };
const TARGET_LABEL = { tiny: "소형 AI 모델", lora: "LoRA 모델", embeddings: "의미 검색 색인", qa: "Q&A 학습 데이터" };
const ARCH_KEYS = ["d_model", "n_layer", "n_head", "seq_len", "vocab_size"];
const ARCH_LABEL = { d_model: "차원 (d_model)", n_layer: "층 (n_layer)", n_head: "헤드 (n_head)", seq_len: "문맥 (seq_len)", vocab_size: "글자 수 (vocab)" };
const FALLBACK_LIMITS = { d_model: [32, 2048], n_layer: [1, 48], n_head: [1, 32], seq_len: [16, 2048], vocab_size: [100, 20000] };
const FALLBACK_PRESETS = {
  nano: { d_model: 128, n_layer: 2, n_head: 4, seq_len: 128, vocab_size: 2000, label: "나노 (저사양 휴대폰)", target: "phone-low" },
  mini: { d_model: 256, n_layer: 4, n_head: 4, seq_len: 192, vocab_size: 3000, label: "미니 (고사양 휴대폰)", target: "phone-high" },
  small: { d_model: 384, n_layer: 6, n_head: 6, seq_len: 256, vocab_size: 4000, label: "스몰 (PC CPU)", target: "pc-cpu" },
  base: { d_model: 512, n_layer: 8, n_head: 8, seq_len: 384, vocab_size: 6000, label: "베이스 (PC GPU)", target: "pc-gpu" },
  large: { d_model: 768, n_layer: 12, n_head: 12, seq_len: 512, vocab_size: 8000, label: "라지 (고성능 GPU)", target: "pc-gpu-high" },
};
// config.py DEFAULTS 와 같은 값 (관리자 설정을 못 읽을 때)
const DEFAULTS = {
  tiny: { steps: 1500, batch_size: 32 },
  lora: { base_model: "Qwen/Qwen2.5-0.5B-Instruct", max_steps: 200, seq_len: 512, batch_size: 2, lora_r: 8, use_qa: true },
  autolearn: {
    targets: { tiny: true, lora: false, embeddings: true, qa: false }, collect_interval_min: 10, pages_per_cycle: 5,
    max_pages_per_hour: 60, train_min_new_chars: 20000, train_interval_min: 60, steps_per_cycle: 300, train_outside_window: false,
  },
};
const SUPPORTED = [".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm", ".xml", ".pdf", ".docx", ".srt", ".log"];
const LORA_CUSTOM = "__custom";

// ---------------- 작은 도우미 ----------------
function store(key, val) {
  try {
    if (val === undefined) return localStorage.getItem(key);
    if (val === null) localStorage.removeItem(key);
    else localStorage.setItem(key, String(val));
  } catch (e) { /* 저장소를 못 쓰는 브라우저(사생활 보호 모드 등) */ }
  return null;
}
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
// 서버 스니펫의 강조 표시(\u0002 ... \u0003)를 <mark>로. 짝이 안 맞으면 버린다.
function hl(s) {
  let open = false;
  return esc(s).replace(/[\u0002\u0003]/g, (c) => {
    if (c === "\u0002" && !open) { open = true; return "<mark>"; }
    if (c === "\u0003" && open) { open = false; return "</mark>"; }
    return "";
  }) + (open ? "</mark>" : "");
}
const num = (x) => (Number.isFinite(Number(x)) ? Number(x) : 0);
const cls = (x) => String(x ?? "").replace(/[^a-zA-Z0-9_-]/g, "");
function fmtNum(n) { return n == null || n === "" || !Number.isFinite(Number(n)) ? "-" : Number(n).toLocaleString("ko-KR"); }
function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? String(iso) : d.toLocaleString("ko-KR", { dateStyle: "short", timeStyle: "short" });
}
function relTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return "";
  const min = Math.round((d - Date.now()) / 60000);
  if (Math.abs(min) < 1) return "곧";
  const a = Math.abs(min);
  const t = a < 60 ? `${a}분` : a < 1440 ? `${Math.round(a / 60)}시간` : `${Math.round(a / 1440)}일`;
  return min > 0 ? `${t} 후` : `${t} 전`;
}
function fmtBytes(n) {
  n = Number(n);
  if (!Number.isFinite(n) || n < 0) return "-";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "GB";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "MB";
  if (n >= 1e3) return Math.round(n / 1e3) + "KB";
  return n + "B";
}
function fmtMB(mb) {
  mb = Number(mb);
  if (!Number.isFinite(mb)) return "-";
  if (mb >= 1024) return (mb / 1024).toFixed(1) + "GB";
  if (mb >= 10) return Math.round(mb) + "MB";
  return mb.toFixed(1) + "MB";
}
function fmtCount(n) {
  n = Number(n);
  if (!Number.isFinite(n)) return "-";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return Math.round(n / 1e3) + "K";
  return String(n);
}
function fmtLoss(x) { return x == null || !Number.isFinite(Number(x)) ? "-" : Number(x).toFixed(3); }
function fmtDur(sec) {
  sec = Number(sec);
  if (!Number.isFinite(sec)) return "";
  const m = Math.floor(sec / 60), s = Math.round(sec % 60);
  return m ? `${m}분 ${s}초` : `${s}초`;
}
function isHttp(u) { return /^https?:\/\//i.test(String(u || "")); }
function fileLabel(u) {
  const m = /^file:\/\/\d+\/([^#]*)(#\d+)?$/.exec(String(u || ""));
  if (!m) return String(u || "");
  let name = m[1];
  try { name = decodeURIComponent(name); } catch (e) { /* 그대로 */ }
  return "파일: " + name + (m[2] ? ` (${m[2]})` : "");
}
function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
function setBar(bar, p) { const d = bar && bar.firstElementChild; if (d) d.style.width = Math.round(Math.max(0, Math.min(1, p)) * 100) + "%"; }
function intVal(sel) { const v = $(sel).value.trim(); if (v === "") return undefined; const n = Number(v); return Number.isFinite(n) ? n : undefined; }
function setVal(sel, v) { const el = $(sel); if (el && v != null) el.value = v; }

function toast(msg, opts) {
  const o = typeof opts === "number" ? { ms: opts } : opts || {};
  const t = $("#toast"), act = $("#toast-act");
  $("#toast-msg").textContent = msg;
  if (o.action) {
    act.hidden = false; act.textContent = o.action.label;
    act.onclick = () => { t.hidden = true; o.action.fn(); };
  } else { act.hidden = true; act.onclick = null; }
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), o.ms || (o.action ? 8000 : 3500));
}

// ---------------- 상태 ----------------
const S = {
  token: store("lb_token"), role: store("lb_role"), offline: false, tab: store("lb_tab") || "search",
  status: null, device: null, myDevice: null, models: [], settings: null, auto: null, snapInfo: null, localModel: null,
  docsOffset: 0, notifSeen: num(store("lb_notif_seen")), notifInit: false, uploadQueue: [], uploading: false,
  formsBuilt: false, defaultsApplied: false, est: { tiny: 0, lora: 0, al: 0 }, lastResultKey: "", genAbort: null,
};
const SS = { q: "", engine: store("lb_engine") || "search", sources: [], mode: "hybrid", offset: 0, limit: 10, total: 0, seq: 0, askSeq: 0, local: false };
if (!ENGINE_LABEL[SS.engine]) SS.engine = "search";

// ---------------- API ----------------
async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (S.token) headers.Authorization = "Bearer " + S.token;
  let body = opts.body;
  if (body != null && typeof body !== "string" && !(body instanceof Blob) && !(body instanceof ArrayBuffer)) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(body);
  }
  let r;
  try {
    r = await fetch(path, { method: opts.method || "GET", headers, body, signal: opts.signal, cache: "no-store" });
  } catch (e) {
    if (e && e.name === "AbortError") throw e;
    const err = new Error("PC의 LearnBox에 연결할 수 없습니다.");
    err.network = true;
    throw err;
  }
  if (r.status === 401 && !path.endsWith("/login")) { logout(true); throw new Error("다시 로그인하세요."); }
  if (opts.raw) {
    if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.error || "HTTP " + r.status); }
    return r;
  }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) { const err = new Error(data.error || "HTTP " + r.status); err.status = r.status; throw err; }
  return data;
}
const post = (p, body) => api(p, { method: "POST", body: body || {} });

// 인증이 필요한 다운로드: fetch(Bearer) -> blob -> object URL -> <a download>
async function downloadAuth(path, filename, btn) {
  const label = btn ? btn.textContent : "";
  if (btn) { btn.disabled = true; btn.textContent = "받는 중…"; }
  try {
    const r = await api(path, { raw: true });
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename || "download"; a.rel = "noopener"; a.hidden = true;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch (e) { toast(e.message, 6000); }
  finally { if (btn) { btn.disabled = false; btn.textContent = label; } }
}
async function fetchBuffer(path, onProgress) {
  const r = await api(path, { raw: true });
  const total = Number(r.headers.get("Content-Length")) || 0;
  if (!r.body || typeof r.body.getReader !== "function" || !total) return r.arrayBuffer();
  const reader = r.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    if (onProgress) onProgress(Math.min(1, got / total));
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out.buffer;
}
function uploadFile(file, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open("POST", "/api/files");
    if (S.token) x.setRequestHeader("Authorization", "Bearer " + S.token);
    x.setRequestHeader("X-File-Name", encodeURIComponent(file.name));
    x.setRequestHeader("Content-Type", "application/octet-stream");
    x.upload.onprogress = (e) => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
    x.upload.onload = () => onProgress && onProgress(1);
    x.onload = () => {
      let d = {};
      try { d = JSON.parse(x.responseText || "{}"); } catch (e) { d = {}; }
      if (x.status === 401) { logout(true); reject(new Error("다시 로그인하세요.")); return; }
      if (x.status >= 200 && x.status < 300) resolve(d);
      else reject(new Error(d.error || (x.status === 413 ? "파일이 너무 큽니다." : "HTTP " + x.status)));
    };
    x.onerror = () => reject(new Error("올리지 못했습니다 (연결이 끊겼습니다)."));
    x.send(file);
  });
}

// ---------------- 주기적 갱신 (화면이 보일 때만) ----------------
const Poll = {
  t: {},
  set(name, fn, ms) { this.clear(name); this.t[name] = setInterval(() => { if (!document.hidden) fn(); }, ms); },
  clear(name) { clearInterval(this.t[name]); delete this.t[name]; },
  clearAll() { Object.keys(this.t).forEach((k) => this.clear(k)); },
};

// ---------------- 로그인 ----------------
function showApp() {
  $("#login").hidden = true;
  $("#tabs").hidden = false;
  $("#main").hidden = false;
  $("#logout").hidden = false;
  $("#logout .lbl").textContent = S.offline ? "로그인" : "로그아웃";
  $("#logout").setAttribute("aria-label", S.offline ? "로그인 화면으로" : "로그아웃");
  $("#bell-wrap").hidden = S.offline;
  $("#who").textContent = S.offline ? "오프라인" : S.role === "admin" ? "관리자" : "사용자";
  $$(".admin-only").forEach((el) => (el.hidden = S.role !== "admin" || S.offline));
  $$(".admin-hint").forEach((el) => (el.hidden = S.role === "admin" || S.offline));
  $$("#tabs button").forEach((b) => (b.disabled = S.offline && !["search", "models"].includes(b.dataset.tab)));
  let tab = S.tab;
  if (S.offline && !["search", "models"].includes(tab)) tab = "search";
  if (tab === "admin" && S.role !== "admin") tab = "search";
  if (!$("#tab-" + tab)) tab = "search";
  updateNet();
  applyEngineUI();
  switchTab(tab);
  refreshSnapInfo();
  refreshLocalModelInfo();
  if (!S.offline) {
    S.notifInit = false;
    pollNotifications();
    Poll.set("notif", pollNotifications, 15000);
  }
}
function showLogin(msg) {
  Poll.clearAll();
  closeViewer();
  $("#login").hidden = false; $("#tabs").hidden = true; $("#main").hidden = true; $("#logout").hidden = true;
  $("#bell-wrap").hidden = true; $("#notif-panel").hidden = true;
  $("#who").textContent = "";
  $("#login-msg").textContent = msg || "";
  const ob = OB();
  if (ob) {
    Promise.all([ob.snapshotInfo().catch(() => null), ob.modelInfo().catch(() => null)])
      .then(([s, m]) => ($("#offline-enter").hidden = !(s || m))).catch(() => {});
  }
}
function logout(expired) {
  if (S.token && !expired && !S.offline) api("/api/logout", { method: "POST" }).catch(() => {});
  S.token = null; S.role = null; S.offline = false; S.device = null; S.settings = null; S.formsBuilt = false; S.defaultsApplied = false;
  store("lb_token", null); store("lb_role", null);
  showLogin(expired ? "세션이 만료되었습니다. 다시 로그인하세요." : "");
}
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#login-msg").textContent = "";
  try {
    const r = await post("/api/login", { pin: $("#pin").value });
    S.token = r.token; S.role = r.role; S.offline = false;
    store("lb_token", r.token); store("lb_role", r.role);
    $("#pin").value = "";
    showApp();
  } catch (err) { $("#login-msg").textContent = err.message; }
});
$("#logout").addEventListener("click", () => {
  if (S.offline) { S.offline = false; showLogin(""); return; }
  logout(false);
});
$("#offline-enter").addEventListener("click", () => { S.offline = true; showApp(); });

// ---------------- 탭 ----------------
function switchTab(name) {
  S.tab = name;
  store("lb_tab", name);
  $$("#tabs button").forEach((b) => {
    const on = b.dataset.tab === name;
    b.classList.toggle("active", on);
    if (on) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
  });
  $$(".tab").forEach((t) => (t.hidden = t.id !== "tab-" + name));
  Poll.clear("tab");
  if (S.offline) { if (name === "models") loadModels(); return; }
  const loaders = { search: loadSearchTab, data: loadData, train: () => loadTrain(false), models: loadModels, jobs: loadJobs, admin: loadAdmin };
  (loaders[name] || (() => {}))();
  const polls = { jobs: [loadJobs, 3000], data: [loadGate, 10000], train: [() => loadTrain(true), 4000] };
  if (polls[name]) Poll.set("tab", polls[name][0], polls[name][1]);
}
$$("#tabs button").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));

function updateNet() {
  const on = navigator.onLine;
  const n = $("#net");
  n.textContent = !on ? "오프라인" : S.offline ? "PC 연결 안 됨" : "온라인";
  n.classList.toggle("off", !on || S.offline);
}
window.addEventListener("online", async () => {
  updateNet();
  if (S.offline && S.token) {
    try { const st = await api("/api/status"); S.status = st; S.offline = false; showApp(); } catch (e) { /* 아직 연결 안 됨 */ }
  }
});
window.addEventListener("offline", updateNet);
document.addEventListener("visibilitychange", () => {
  if (document.hidden || !S.token || S.offline) return;
  pollNotifications();
  if (S.tab === "jobs") loadJobs();
  else if (S.tab === "train") loadTrain(true);
});

async function refreshStatus() {
  if (S.offline || !S.token) return null;
  try {
    S.status = await api("/api/status");
    if (typeof S.status.unread === "number") setBadge(S.status.unread);
    applyEngineUI();
    return S.status;
  } catch (e) { return null; }
}
async function loadModelsList() {
  try { const r = await api("/api/models"); S.models = r.models || []; } catch (e) { /* 이전 목록 유지 */ }
  return S.models;
}
function modelKind(m) { return m.kind === "lora" || m.kind === "pc" ? "lora" : "tiny"; }

// ---------------- 알림 ----------------
function setBadge(n) {
  n = num(n);
  const b = $("#bell-badge");
  b.hidden = n <= 0;
  b.textContent = n > 99 ? "99+" : String(n);
  $("#bell").setAttribute("aria-label", n > 0 ? `알림 (안 읽음 ${n}개)` : "알림");
}
function isLearningEnd(n) {
  return n.level === "success" && ((n.data && n.data.kind === "learning_end") || /학습이 종료/.test(n.title || ""));
}
async function pollNotifications() {
  if (!S.token || S.offline) return;
  let r;
  try { r = await api("/api/notifications?unread=1&limit=20"); } catch (e) { return; }
  setBadge(r.unread || 0);
  const list = r.notifications || [];
  const maxId = list.reduce((m, n) => Math.max(m, num(n.id)), 0);
  const first = !S.notifInit;
  S.notifInit = true;
  if (first && store("lb_notif_seen") == null) { // 이 기기에서 처음: 지난 알림은 조용히 표시만
    S.notifSeen = Math.max(S.notifSeen, maxId);
    store("lb_notif_seen", S.notifSeen);
    return;
  }
  const fresh = list.filter((n) => num(n.id) > S.notifSeen).sort((a, b) => a.id - b.id);
  if (!fresh.length) return;
  S.notifSeen = Math.max(S.notifSeen, maxId);
  store("lb_notif_seen", S.notifSeen);
  const end = fresh.filter(isLearningEnd).pop();
  if (end) {
    toast(`${end.title}\n${String(end.body || "").split("\n").slice(-1)[0]}`, {
      ms: 12000, action: { label: "결과 보기", fn: () => openResult(end.data && end.data.result) },
    });
    browserNotify(end.title, end.body || "", "learnbox-" + end.id);
    S.lastResultKey = "";
    if (S.tab === "train") loadTrain(false);
  } else {
    const last = fresh[fresh.length - 1];
    toast(last.title + (fresh.length > 1 ? ` (외 ${fresh.length - 1}건)` : ""), last.level === "error" || last.level === "warning" ? 7000 : 4000);
  }
  if (!$("#notif-panel").hidden) loadNotifPanel();
}
async function browserNotify(title, body, tag) {
  try {
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    let reg = null;
    if (navigator.serviceWorker && navigator.serviceWorker.getRegistration) {
      reg = await Promise.race([navigator.serviceWorker.getRegistration(), new Promise((r) => setTimeout(() => r(null), 800))]);
    }
    if (reg && reg.showNotification) await reg.showNotification(title, { body, icon: "/icon.svg", tag });
    else new Notification(title, { body, icon: "/icon.svg", tag });
  } catch (e) { /* 휴대폰 브라우저는 페이지에서 알림을 못 만들 수 있다: 화면 안 알림으로 충분 */ }
}
function askNotifyPermission() {
  try {
    if ("Notification" in window && Notification.permission === "default") {
      const p = Notification.requestPermission();
      if (p && p.then) p.then(updatePermButton).catch(() => {});
    }
  } catch (e) { /* 무시 */ }
}
function updatePermButton() {
  let show = false;
  try { show = "Notification" in window && Notification.permission === "default"; } catch (e) { show = false; }
  $("#notif-perm").hidden = !show;
}
async function loadNotifPanel() {
  const ul = $("#notif-list");
  updatePermButton();
  try {
    const r = await api("/api/notifications?limit=30");
    setBadge(r.unread || 0);
    ul.innerHTML = (r.notifications || []).map((n) => `
      <li class="${n.read ? "" : "unread"}">
        <span class="lv ${cls(n.level)}" aria-hidden="true"></span>
        <div class="grow">
          <div class="t">${esc(n.title)}</div>
          ${n.body ? `<div class="body">${esc(n.body)}</div>` : ""}
          <div class="small muted">${esc(fmtTime(n.ts))}${n.read ? "" : " · 안 읽음"}</div>
          <div class="row wrap gap">
            ${n.read ? "" : `<button type="button" class="ghost small" data-act="notif-read" data-id="${num(n.id)}">읽음</button>`}
            ${n.data && n.data.result ? `<button type="button" class="secondary small" data-act="notif-result" data-name="${esc(n.data.result)}">결과 보기</button>` : ""}
          </div>
        </div>
      </li>`).join("") || '<li class="muted small">알림이 없습니다.</li>';
  } catch (e) { ul.innerHTML = `<li class="error small">${esc(e.message)}</li>`; }
}
function toggleNotif(open) {
  const p = $("#notif-panel");
  const show = open == null ? p.hidden : open;
  p.hidden = !show;
  $("#bell").setAttribute("aria-expanded", String(show));
  if (show) loadNotifPanel();
}
$("#bell").addEventListener("click", (e) => { e.stopPropagation(); toggleNotif(); });
$("#notif-readall").addEventListener("click", async () => {
  try { await post("/api/notifications/read", {}); setBadge(0); loadNotifPanel(); } catch (e) { toast(e.message); }
});
$("#notif-perm").addEventListener("click", askNotifyPermission);
document.addEventListener("click", (e) => {
  const p = $("#notif-panel");
  if (!p.hidden && !e.target.closest("#bell-wrap")) toggleNotif(false);
});
function openResult(name) {
  toggleNotif(false);
  S.openResult = name || true;
  S.lastResultKey = "";
  switchTab("train");
  setTimeout(() => { const c = $("#auto-card"); if (c) c.scrollIntoView({ behavior: "smooth", block: "start" }); }, 150);
}

// ---------------- 검색 ----------------
function loadSearchTab() {
  refreshStatus();
  loadModelsList().then(fillAskModels);
  refreshSnapInfo();
}
function applyEngineUI() {
  const e = SS.engine;
  $$('#engine input[name="engine"]').forEach((i) => (i.checked = i.value === e));
  const ask = e === "tiny" || e === "lora";
  $("#ask-model-wrap").hidden = !ask || S.offline;
  $("#mode-wrap").hidden = e === "onboard" || S.offline;
  $("#search-btn").textContent = e === "search" ? "검색" : "질문";
  const st = S.status || {};
  const online = st.online || null;
  const hasKind = (k) => S.models.some((m) => modelKind(m) === k);
  const notes = {
    search: "모은 자료 안에서 찾습니다. 답변 엔진을 고르면 찾은 자료를 근거로 답합니다.",
    tiny: "소형 AI 모델이 모은 자료를 흉내 내어 이어 씁니다. 사실 확인은 출처를 보세요."
      + (S.models.length && !hasKind("tiny") ? " 아직 학습한 소형 AI 모델이 없습니다." : ""),
    lora: "LoRA 모델이 검색한 자료를 근거로 답합니다 (PC에서 실행, CPU만 있으면 느립니다)."
      + (S.models.length && !hasKind("lora") ? " 아직 학습한 LoRA 모델이 없습니다." : ""),
    onboard: "이 기기 안에서 인터넷 없이 찾고, 받은 소형 AI 모델로 이어 씁니다."
      + (S.snapInfo ? "" : " 먼저 아래 '이 기기에 저장'으로 자료 사본을 받으세요."),
    online: "온라인 모델(Claude)이 찾은 자료만 근거로 답합니다. 질문과 자료 일부가 Anthropic API로 전송됩니다."
      + (online && !online.available ? ` 지금은 쓸 수 없습니다: ${online.reason || "설정을 확인하세요."}` : ""),
  };
  let note = notes[e] || "";
  if (S.offline && e !== "onboard") note = "PC에 연결되어 있지 않아 이 기기에 저장된 사본에서 찾습니다 (온보드).";
  $("#engine-note").textContent = note;
  const na = { online: online && !online.available, onboard: !S.snapInfo, tiny: S.offline, lora: S.offline };
  $$("#engine label").forEach((l) => {
    const v = l.querySelector("input").value;
    l.querySelector("span").classList.toggle("na", !!na[v]);
  });
  fillAskModels();
}
function fillAskModels() {
  const sel = $("#ask-model");
  const want = SS.engine === "lora" ? "lora" : "tiny";
  const cur = sel.value;
  const list = S.models.filter((m) => modelKind(m) === want);
  sel.innerHTML = '<option value="">가장 최근 모델</option>' + list.map((m) =>
    `<option value="${num(m.id)}">#${num(m.id)} ${esc(m.name)}</option>`).join("");
  if (list.some((m) => String(m.id) === cur)) sel.value = cur;
}
$("#engine").addEventListener("change", (e) => {
  if (e.target.name !== "engine") return;
  SS.engine = e.target.value;
  store("lb_engine", SS.engine);
  applyEngineUI();
});
$("#src-chips").addEventListener("click", (e) => {
  const b = e.target.closest(".chip");
  if (!b) return;
  const v = b.dataset.src;
  if (!v) SS.sources = [];
  else if (SS.sources.includes(v)) SS.sources = SS.sources.filter((x) => x !== v);
  else SS.sources = SS.sources.concat(v);
  $$("#src-chips .chip").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.src ? SS.sources.includes(c.dataset.src) : !SS.sources.length)));
  if (SS.q) runSearch(0, { noAsk: true });
});
$("#mode").addEventListener("change", () => { SS.mode = $("#mode").value; if (SS.q) runSearch(0, { noAsk: true }); });
$("#search-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch(0); });
$("#prev-page").addEventListener("click", () => { runSearch(Math.max(0, SS.offset - SS.limit), { noAsk: true, scroll: true }); });
$("#next-page").addEventListener("click", () => { runSearch(SS.offset + SS.limit, { noAsk: true, scroll: true }); });

function setMeta(html) { $("#search-meta").innerHTML = html; }
function srcBadge(s) { return s ? `<span class="badge src-${cls(s)}">${esc(SOURCE_LABEL[s] || s)}</span>` : ""; }
function matchBadge(m) { return m ? `<span class="badge m-${cls(m)}">${esc(MATCH_LABEL[m] || m)}</span>` : ""; }
function docButton(row, extra) {
  const title = row.title || (row.url ? fileLabel(row.url) : "") || "제목 없음";
  if (row.doc_id != null && Number.isFinite(Number(row.doc_id))) {
    return `<button type="button" class="link ${extra || ""}" data-act="view" data-doc="${num(row.doc_id)}">${esc(title)}</button>`;
  }
  if (isHttp(row.url)) return `<a href="${esc(row.url)}" target="_blank" rel="noopener noreferrer">${esc(title)}</a>`;
  return `<span>${esc(title)}</span>`;
}
function urlLine(u) {
  if (!u) return "";
  if (isHttp(u)) return `<div class="url"><a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(u)}</a></div>`;
  return `<div class="url">${esc(fileLabel(u))}</div>`;
}

// 검색어에서 강조할 낱말 ("구문", 일반 낱말; -제외어는 뺀다)
function queryTerms(q) {
  const out = [];
  const s = String(q || "").replace(/(-?)"([^"]*)"/g, (m, neg, p) => { if (!neg && p.trim()) out.push(p.trim()); return " "; });
  for (const w of s.split(/\s+/)) {
    if (!w || w.startsWith("-")) continue;
    const t = w.replace(/^["'“”‘’(\[{<]+|["'“”‘’)\]}>,.!?;:]+$/g, "");
    if (t) out.push(t);
  }
  return Array.from(new Set(out)).sort((a, b) => b.length - a.length);
}
function findSpans(text, terms, max) {
  max = max || 500;
  const low = text.toLowerCase();
  const spans = [];
  for (const t of terms) {
    const tl = t.toLowerCase();
    if (!tl) continue;
    let i = 0;
    while ((i = low.indexOf(tl, i)) !== -1 && spans.length < max * 4) { spans.push([i, i + tl.length]); i += tl.length; }
  }
  spans.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  const merged = [];
  for (const sp of spans) {
    const last = merged[merged.length - 1];
    if (last && sp[0] <= last[1]) last[1] = Math.max(last[1], sp[1]);
    else merged.push([sp[0], sp[1]]);
  }
  return merged.slice(0, max);
}
function spansHtml(text, spans, markCls) {
  let h = "", pos = 0, k = 0;
  for (const [a, b] of spans) {
    if (a < pos || b <= a || b > text.length) continue;
    h += esc(text.slice(pos, a)) + `<mark${markCls ? ` class="${markCls}" data-i="${k++}"` : ""}>${esc(text.slice(a, b))}</mark>`;
    pos = b;
  }
  return h + esc(text.slice(pos));
}
function markTerms(text, terms) { return spansHtml(String(text || ""), findSpans(String(text || ""), terms, 50)); }

async function runSearch(offset, opts) {
  opts = opts || {};
  const q = $("#q").value.trim();
  if (!q) { toast("검색어를 입력하세요."); return; }
  const seq = ++SS.seq;
  const fresh = !opts.noAsk && offset === 0;
  SS.q = q; SS.offset = offset;
  const engine = SS.engine;
  const local = S.offline || engine === "onboard";
  if (fresh) { $("#answer").hidden = true; $("#extract-card").hidden = true; }
  setMeta('<span class="spinner" aria-hidden="true"></span>찾는 중…');
  if (opts.scroll) $("#search-meta").scrollIntoView({ block: "start", behavior: "smooth" });
  if (local) return localSearch(q, offset, seq, fresh && engine !== "search");
  if (fresh && engine !== "search") askServer(q, engine);
  const p = new URLSearchParams({ q, limit: String(SS.limit), offset: String(offset), mode: SS.mode });
  if (SS.sources.length) p.set("sources", SS.sources.join(","));
  try {
    const r = await api("/api/search?" + p.toString());
    if (seq !== SS.seq) return;
    renderSearch(r.results || [], r.total, r, false);
    if (offset === 0) renderExtract(r.extract || [], q);
  } catch (e) {
    if (seq !== SS.seq) return;
    if (e.network) {
      toast("PC에 연결할 수 없어 이 기기에 저장된 사본에서 찾습니다.");
      return localSearch(q, offset, seq, fresh && engine !== "search");
    }
    setMeta("");
    $("#results").innerHTML = `<div class="card error">${esc(e.message)}</div>`;
    $("#pager").hidden = true;
  }
}
function renderSearch(rows, total, r, local) {
  SS.local = local;
  SS.total = Number.isFinite(Number(total)) ? Number(total) : rows.length;
  const from = SS.offset + 1, to = SS.offset + rows.length;
  let meta = rows.length ? `${fmtNum(from)}–${fmtNum(to)}번째 · 약 ${fmtNum(Math.max(SS.total, to))}개 문서` : "결과가 없습니다.";
  if (local) meta += " · 이 기기에 저장된 사본에서 찾았습니다";
  else if (r && r.mode_used) {
    if (SS.mode !== "keyword" && r.mode_used === "keyword") meta += " · 의미 검색 색인이 없어 키워드로 찾았습니다";
    else meta += ` · ${MODE_LABEL[r.mode_used] || r.mode_used} 검색`;
  }
  setMeta(esc(meta));
  if (!rows.length) {
    $("#results").innerHTML = '<div class="card muted small">찾은 자료가 없습니다. 다른 낱말로 찾거나 자료 종류를 "전체"로 바꿔 보세요.</div>';
    $("#pager").hidden = SS.offset === 0;
    $("#prev-page").disabled = SS.offset === 0;
    $("#next-page").disabled = true;
    return;
  }
  $("#results").innerHTML = '<div class="card">' + rows.map((x, i) => `
    <article class="result" id="res-${SS.offset + i + 1}">
      <div class="head"><span class="num">${SS.offset + i + 1}.</span>${docButton(x)}${srcBadge(x.source)}${matchBadge(x.match)}</div>
      ${urlLine(x.url)}
      <div class="snip">${hl(x.snippet || "")}</div>
    </article>`).join("") + "</div>";
  const pages = Math.max(1, Math.ceil(Math.max(SS.total, to) / SS.limit));
  const page = Math.floor(SS.offset / SS.limit) + 1;
  $("#pager").hidden = pages <= 1 && SS.offset === 0;
  $("#prev-page").disabled = SS.offset === 0;
  $("#next-page").disabled = to >= SS.total || rows.length < SS.limit;
  $("#page-info").textContent = `${page} / ${Math.max(page, pages)}쪽`;
}
function renderExtract(list, q) {
  const card = $("#extract-card");
  if (!list || !list.length) { card.hidden = true; return; }
  const terms = queryTerms(q);
  $("#extract").innerHTML = list.map((s) =>
    `<li>${markTerms(s.sentence, terms)} <span class="from muted">— ${docButton({ doc_id: s.doc_id, title: s.title, url: s.url }, "small")}</span></li>`).join("");
  card.hidden = false;
}
function answerHead(engine, model) {
  return `<div class="answer-head"><h3>답변</h3><span class="badge">${esc(ENGINE_LABEL[engine] || engine || "")}</span>${model ? `<span class="muted small">${esc(model)}</span>` : ""}</div>`;
}
async function askServer(q, engine) {
  const seq = ++SS.askSeq;
  const box = $("#answer");
  box.hidden = false;
  const wait = { online: "온라인 모델에 묻는 중…", lora: "LoRA 모델이 답변을 만드는 중… (CPU에서는 몇 분 걸릴 수 있습니다)", tiny: "소형 AI 모델이 글을 만드는 중…" };
  box.innerHTML = answerHead(engine) + `<p class="muted"><span class="spinner" aria-hidden="true"></span>${esc(wait[engine] || "답변을 만드는 중…")}</p>`;
  const body = { q, engine };
  const mid = $("#ask-model").value;
  if ((engine === "tiny" || engine === "lora") && mid) body.model_id = Number(mid);
  if (SS.sources.length) body.sources = SS.sources.slice();
  try {
    const r = await post("/api/ask", body);
    if (seq !== SS.askSeq) return;
    renderAnswer(r, engine, q);
    if ($("#extract-card").hidden && r.extract && r.extract.length) renderExtract(r.extract, q);
  } catch (e) {
    if (seq !== SS.askSeq) return;
    if (e.network) { box.innerHTML = answerHead(engine) + '<p class="muted">PC에 연결할 수 없어 이 기기에서 답합니다.</p>'; return localAnswer(q, SS.seq, seq); }
    box.innerHTML = answerHead(engine) + `<p class="error">${esc(e.message)}</p>`;
  }
}
function linkCites(text, n) {
  return esc(text).replace(/\[(\d{1,3})\]/g, (m, d) => {
    const i = Number(d);
    return i >= 1 && i <= n ? `<a href="#ans-src-${i}" class="cite" data-act="cite" data-n="${i}">[${i}]</a>` : m;
  });
}
function groupCitations(cites) {
  const by = new Map();
  for (const c of cites || []) {
    const i = num(c.index);
    if (!by.has(i)) by.set(i, { index: i, title: c.title, url: c.url, texts: [] });
    const g = by.get(i);
    const t = String(c.cited_text || "").trim();
    if (t && g.texts.length < 2 && !g.texts.includes(t)) g.texts.push(t.length > 300 ? t.slice(0, 300) + "…" : t);
  }
  return Array.from(by.values()).sort((a, b) => a.index - b.index);
}
function renderAnswer(r, engine, q) {
  const box = $("#answer");
  const srcs = r.sources || [];
  const cites = groupCitations(r.citations);
  let h = answerHead(r.engine || engine, r.model);
  if (r.answer) h += `<div class="out answer-text">${linkCites(r.answer, srcs.length)}</div>`;
  else h += '<p class="muted">답변을 만들지 못했습니다. 아래 근거 자료와 검색 결과를 보세요.</p>';
  if (r.note) h += `<p class="muted small">${esc(r.note)}</p>`;
  if (cites.length) {
    h += "<h4>인용</h4><ul class=\"srcs\">" + cites.map((c) => {
      const s = srcs[c.index - 1] || {};
      const ref = c.index >= 1 && c.index <= srcs.length
        ? `<a href="#ans-src-${c.index}" class="cite" data-act="cite" data-n="${c.index}">[${c.index}]</a>` : `<span class="n">[${c.index}]</span>`;
      return `<li>${ref} ${docButton({ doc_id: s.doc_id, title: c.title || s.title, url: c.url || s.url })}${c.texts.map((t) => `<span class="quote">${esc(t)}</span>`).join("")}</li>`;
    }).join("") + "</ul>";
  }
  if (srcs.length) {
    const terms = queryTerms(q);
    h += "<h4>근거 자료</h4><ol class=\"srcs\">" + srcs.map((s, i) => `
      <li id="ans-src-${i + 1}"><span class="n">[${i + 1}]</span>${docButton(s)} ${srcBadge(s.source)}
        ${s.snippet ? `<div class="small muted">${hl(s.snippet)}</div>` : s.content ? `<div class="small muted">${markTerms(String(s.content).slice(0, 240), terms)}…</div>` : ""}</li>`).join("") + "</ol>";
  }
  box.innerHTML = h;
  box.hidden = false;
}

// 이 기기 안에서 찾기 (오프라인 또는 온보드 엔진)
async function localSearch(q, offset, seq, withAnswer) {
  const ob = OB();
  if (!ob) {
    setMeta("");
    $("#results").innerHTML = '<div class="card error">온보드 엔진(onboard.js)을 불러오지 못했습니다.</div>';
    return;
  }
  try {
    const opt = { limit: SS.limit, offset, sources: SS.sources.length ? SS.sources.slice() : null };
    const rows = await ob.search(q, opt);
    if (seq !== SS.seq) return;
    renderSearch(rows, rows.total, null, true);
    if (offset === 0) renderExtract(ob.extract(q, rows, 3), q);
    if (withAnswer) localAnswer(q, seq, ++SS.askSeq, rows);
  } catch (e) {
    if (seq !== SS.seq) return;
    setMeta("");
    $("#results").innerHTML = `<div class="card error">${esc(e.message)}</div>`;
    $("#pager").hidden = true;
  }
}
async function localAnswer(q, seq, askSeq, rows) {
  const ob = OB();
  const box = $("#answer");
  if (!ob) return;
  box.hidden = false;
  box.innerHTML = answerHead("onboard") + '<p class="muted"><span class="spinner" aria-hidden="true"></span>이 기기에서 찾는 중…</p>';
  try {
    let ex;
    if (rows) ex = ob.extract(q, rows, 3);
    else {
      const res = await ob.answer(q, { limit: SS.limit, sources: SS.sources.length ? SS.sources.slice() : null, generate: false });
      rows = res.rows; ex = res.extract;
      if (seq === SS.seq) { renderSearch(rows, rows.total, null, true); renderExtract(ex, q); }
    }
    if (askSeq !== SS.askSeq) return;
    const info = await ob.modelInfo();
    if (!ex || !ex.length) {
      box.innerHTML = answerHead("onboard") + '<p class="muted">이 기기에 저장된 자료에서 찾지 못했습니다.</p>';
      return;
    }
    if (!info) {
      box.innerHTML = answerHead("onboard") + `<div class="out answer-text">${esc(ex[0].sentence)}</div>
        <p class="muted small">이 기기에 받은 모델이 없어 가장 알맞은 문장만 보여 줍니다. [모델] 탭에서 소형 AI 모델을 받으면 이어 쓰기를 합니다.</p>`;
      return;
    }
    const best = ex[0].sentence;
    box.innerHTML = answerHead("onboard", info.name) +
      `<div class="out answer-text"><span class="prompt">${esc(best)}</span><span class="gen"> …</span></div>
       <p class="muted small">이 기기의 소형 AI 모델(${esc(info.variant || "")})이 가장 알맞은 문장에 이어 쓴 글입니다. 사실 확인은 출처를 보세요.</p>`;
    const gen = $(".gen", box);
    const text = await ob.generate(best, {
      onToken: (piece, t) => { if (askSeq === SS.askSeq && gen.isConnected) gen.textContent = t; },
    });
    if (askSeq === SS.askSeq && gen.isConnected) gen.textContent = text || "";
  } catch (e) {
    if (askSeq !== SS.askSeq) return;
    box.innerHTML = answerHead("onboard") + `<p class="error">${esc(e.message)}</p>`;
  }
}

// ---------------- 문서 보기 ----------------
const V = { hits: [], cur: -1, lastFocus: null, snap: null, snapAt: null };
async function openViewer(docId, q) {
  V.lastFocus = document.activeElement;
  $("#viewer").hidden = false;
  document.body.classList.add("noscroll");
  $("#viewer-title").textContent = "불러오는 중…";
  $("#viewer-meta").textContent = "";
  $("#viewer-open").hidden = true;
  $("#hit-pos").textContent = "";
  $("#hit-prev").disabled = $("#hit-next").disabled = true;
  $("#viewer-body").innerHTML = '<p class="muted"><span class="spinner" aria-hidden="true"></span>문서를 불러오는 중…</p>';
  $("#viewer-close").focus();
  try {
    let d;
    if (S.offline || SS.local) d = await localDoc(docId, q);
    else {
      try { d = await api(`/api/documents/${docId}?q=${encodeURIComponent(q || "")}`); }
      catch (e) { if (e.network) d = await localDoc(docId, q); else throw e; }
    }
    if (!d) throw new Error("문서를 찾을 수 없습니다.");
    renderViewer(d, q);
  } catch (e) {
    $("#viewer-title").textContent = "문서를 열 수 없습니다";
    $("#viewer-body").innerHTML = `<p class="error">${esc(e.message)}</p>`;
  }
}
async function localDoc(docId, q) {
  const ob = OB();
  if (!ob) throw new Error("온보드 엔진을 불러오지 못했습니다.");
  const info = await ob.snapshotInfo();
  if (!V.snap || !info || V.snapAt !== info.saved_at) {
    V.snap = await ob.db.get("snapshot");
    V.snapAt = info && info.saved_at;
  }
  const snap = V.snap;
  if (!snap) throw new Error("이 기기에 저장된 사본이 없습니다.");
  let text = "", title = "", url = "", source = "";
  if (Array.isArray(snap.chunks)) {
    const parts = snap.chunks.filter((c) => Number(c.d) === Number(docId)).sort((a, b) => a.c - b.c);
    if (!parts.length) return null;
    text = parts.map((c) => String(c.x || "").replace(/[\u0002\u0003]/g, "")).join("\n");
    title = parts[0].t; url = parts[0].u; source = parts[0].s;
  } else if (Array.isArray(snap.documents)) {
    const d = snap.documents.find((x) => Number(x.id) === Number(docId));
    if (!d) return null;
    text = d.text || ""; title = d.title; url = d.url;
  }
  return { id: docId, title, url, source, text, chars: text.length, matches: findSpans(text, queryTerms(q)), partial: true };
}
function renderViewer(d, q) {
  const text = String(d.text || "");
  $("#viewer-title").textContent = d.title || fileLabel(d.url) || "제목 없음";
  const meta = [SOURCE_LABEL[d.source] || d.source, d.fetched_at ? fmtTime(d.fetched_at) : "", d.chars != null ? fmtNum(d.chars) + "자" : "",
    d.query ? "검색어: " + d.query : "", d.partial ? "이 기기 사본에 있는 부분만 보입니다" : "", !isHttp(d.url) && d.url ? fileLabel(d.url) : ""].filter(Boolean);
  $("#viewer-meta").textContent = meta.join(" · ");
  const open = $("#viewer-open");
  if (isHttp(d.url)) { open.href = d.url; open.hidden = false; } else { open.hidden = true; open.removeAttribute("href"); }
  let spans = Array.isArray(d.matches_utf16) ? d.matches_utf16 : Array.isArray(d.matches) ? d.matches : [];
  spans = spans.filter((s) => Array.isArray(s) && s.length >= 2 && Number.isInteger(s[0]) && Number.isInteger(s[1]))
    .map((s) => [s[0], s[1]]).sort((a, b) => a[0] - b[0]);
  if (!spans.length && q) spans = findSpans(text, queryTerms(q));
  const body = $("#viewer-body");
  body.innerHTML = text ? spansHtml(text, spans, "hit") : '<p class="muted">본문이 비어 있습니다.</p>';
  body.scrollTop = 0;
  V.hits = $$("mark.hit", body);
  V.cur = -1;
  $("#hit-prev").disabled = $("#hit-next").disabled = !V.hits.length;
  if (V.hits.length) gotoHit(0);
  else $("#hit-pos").textContent = q ? "일치하는 곳이 없습니다" : "";
}
function gotoHit(i) {
  const n = V.hits.length;
  if (!n) return;
  if (V.cur >= 0 && V.hits[V.cur]) V.hits[V.cur].classList.remove("current");
  V.cur = ((i % n) + n) % n;
  const m = V.hits[V.cur];
  m.classList.add("current");
  m.scrollIntoView({ block: "center" });
  $("#hit-pos").textContent = `${V.cur + 1} / ${n} 일치`;
}
function closeViewer() {
  if ($("#viewer").hidden) return;
  $("#viewer").hidden = true;
  document.body.classList.remove("noscroll");
  $("#viewer-body").innerHTML = "";
  V.hits = [];
  if (V.lastFocus && V.lastFocus.isConnected && typeof V.lastFocus.focus === "function") V.lastFocus.focus();
}
$("#hit-prev").addEventListener("click", () => gotoHit(V.cur - 1));
$("#hit-next").addEventListener("click", () => gotoHit(V.cur + 1));
document.addEventListener("keydown", (e) => {
  if (!$("#viewer").hidden) {
    if (e.key === "Escape") { e.preventDefault(); closeViewer(); }
    else if (e.key === "Enter" && e.target === $("#viewer-body")) { e.preventDefault(); gotoHit(V.cur + (e.shiftKey ? -1 : 1)); }
    else if (e.key === "Tab") { // 대화 상자 안에서만 초점 이동
      const f = $$("#viewer button:not([disabled]), #viewer a[href]:not([hidden]), #viewer-body").filter((x) => x.offsetParent !== null);
      if (f.length) {
        const i = f.indexOf(document.activeElement);
        if (e.shiftKey && i <= 0) { e.preventDefault(); f[f.length - 1].focus(); }
        else if (!e.shiftKey && i === f.length - 1) { e.preventDefault(); f[0].focus(); }
      }
    }
    return;
  }
  if (e.key === "Escape" && !$("#notif-panel").hidden) { toggleNotif(false); $("#bell").focus(); }
});

// ---------------- 이 기기에 저장 (오프라인 사본) ----------------
async function refreshSnapInfo() {
  const ob = OB();
  if (!ob) {
    $("#snap-info").textContent = "온보드 엔진(onboard.js)을 불러오지 못했습니다.";
    $("#snap-btn").disabled = true;
    return;
  }
  try {
    const dev = S.myDevice || await ob.detectDevice();
    S.myDevice = dev;
    $("#snap-device").textContent = `이 기기: ${dev.label} → 자동으로 조각 ${fmtNum(dev.limits.snapshotChunks)}개까지 저장`;
    $("#snap-size").options[0].textContent = `자동 (조각 ${fmtNum(dev.limits.snapshotChunks)}개)`;
  } catch (e) { /* 무시 */ }
  try {
    const info = await ob.snapshotInfo();
    S.snapInfo = info;
    $("#snap-info").textContent = info
      ? `저장됨: 조각 ${fmtNum(info.count)}개${info.docs ? ` · 문서 ${fmtNum(info.docs)}개` : ""}${info.saved_at ? " · " + fmtTime(info.saved_at) : ""}`
      : "아직 저장된 사본이 없습니다.";
    $("#snap-del").hidden = !info || typeof ob.deleteSnapshot !== "function";
  } catch (e) { $("#snap-info").textContent = "이 브라우저는 기기 저장을 지원하지 않습니다."; }
  applyEngineUI();
}
$("#snap-btn").addEventListener("click", async () => {
  if (S.offline) return toast("PC에 연결되어 있을 때 받을 수 있습니다.");
  const ob = OB();
  if (!ob) return toast("온보드 엔진을 불러오지 못했습니다.");
  const btn = $("#snap-btn"), bar = $("#snap-bar");
  btn.disabled = true; bar.hidden = false; setBar(bar, 0.05);
  try {
    const dev = S.myDevice || await ob.detectDevice();
    const sel = $("#snap-size").value;
    const n = sel === "auto" ? dev.limits.snapshotChunks : Number(sel);
    $("#snap-info").textContent = "자료를 받는 중…";
    const r = await api("/api/snapshot?max_chunks=" + encodeURIComponent(n));
    const rows = r.chunks || r.documents || [];
    if (!rows.length) throw new Error("저장할 자료가 없습니다. 먼저 자료를 모으세요.");
    setBar(bar, 0.3);
    const meta = await ob.saveSnapshot(rows, {
      onProgress: (p, msg) => { setBar(bar, 0.3 + 0.7 * p); $("#snap-info").textContent = "저장 중…" + (msg ? " " + msg : ""); },
    });
    V.snap = null;
    if (typeof ob.requestPersist === "function") ob.requestPersist().catch(() => {});
    toast(`조각 ${fmtNum(meta && meta.count != null ? meta.count : rows.length)}개를 이 기기에 저장했습니다.`);
  } catch (e) { toast(e.message, 6000); }
  finally { btn.disabled = false; bar.hidden = true; refreshSnapInfo(); }
});
$("#snap-del").addEventListener("click", async () => {
  const ob = OB();
  if (!ob || !confirm("이 기기에 저장된 자료 사본을 지울까요?")) return;
  try { await ob.deleteSnapshot(); V.snap = null; toast("사본을 지웠습니다."); } catch (e) { toast(e.message); }
  refreshSnapInfo();
});

// ---------------- 자료 ----------------
function loadData() { loadGate(); loadFiles(); loadWatch(); loadDocs(false); }
async function loadGate() {
  const st = await refreshStatus();
  if (!st) return;
  const g = st.gate || {};
  const el = $("#gate");
  el.className = "banner " + (g.allowed ? "ok" : "warn");
  el.textContent = g.allowed
    ? `지금은 수집 허용 시간입니다 (${g.windows || ""}, ${g.timezone || ""}).`
    : `지금은 수집할 수 없습니다. 허용 시간: ${g.windows || "-"}. 다음 허용: ${fmtTime(g.next_allowed) || "-"}`;
  const names = { wikipedia: "위키백과", searxng: "SearXNG", brave: "Brave Search", none: "사용 안 함" };
  $("#provider-info").textContent = `검색 제공자: ${names[st.search_provider] || st.search_provider || "-"}. 결과 페이지를 가져와 자료로 저장합니다.`;
  const ac = st.auto_collect || {};
  const al = st.autolearn || {};
  let info = ac.enabled ? `자동 수집 켜짐: 허용 시간대에 ${ac.interval_min}분마다 목록에서 하나씩 수집합니다.` : "자동 수집이 꺼져 있습니다 (관리 > 설정).";
  if (al.active) info += " 자율 학습이 진행 중이라 목록과 자동으로 찾은 주제를 조금씩 수집합니다.";
  $("#auto-info").textContent = info;
  if (st.counts) $("#doc-count").textContent = `문서 ${fmtNum(st.counts.documents)}개 · ${fmtNum(st.counts.chars)}자`;
}
async function loadFiles() {
  try {
    const { files } = await api("/api/files");
    $("#files-count").textContent = files.length ? `${fmtNum(files.length)}개` : "";
    $("#files").innerHTML = files.map((f) => `
      <li><span class="grow"><b>${esc(f.name)}</b><br>
        <span class="muted small">${fmtBytes(f.size)} · 문서 ${fmtNum(f.docs)}개 · ${esc(fmtTime(f.created_at))}${f.actor ? " · " + esc(f.actor) : ""}</span></span></li>`).join("")
      || '<li class="muted small">아직 가져온 파일이 없습니다.</li>';
  } catch (e) { $("#files").innerHTML = `<li class="error small">${esc(e.message)}</li>`; }
}
function addFiles(list) {
  const files = Array.from(list || []);
  if (!files.length) return;
  if (S.offline) return toast("PC에 연결되어 있을 때 가져올 수 있습니다.");
  for (const f of files) {
    const li = document.createElement("li");
    li.innerHTML = '<span class="grow"><b></b> <span class="muted small size"></span><br><span class="small st">대기 중</span></span><div class="bar"><div></div></div>';
    $("b", li).textContent = f.name;
    $(".size", li).textContent = fmtBytes(f.size);
    $("#upload-list").prepend(li);
    S.uploadQueue.push({ f, li });
  }
  if (!S.uploading) processUploads();
}
async function processUploads() {
  S.uploading = true;
  let added = false;
  while (S.uploadQueue.length) {
    const { f, li } = S.uploadQueue.shift();
    const st = $(".st", li), bar = $(".bar", li);
    const ext = ((f.name.match(/\.[^.]+$/) || [""])[0] || "").toLowerCase();
    if (!SUPPORTED.includes(ext)) { st.textContent = "지원하지 않는 형식입니다."; st.className = "small error"; bar.hidden = true; continue; }
    st.textContent = "올리는 중… 0%";
    try {
      const r = await uploadFile(f, (p) => {
        setBar(bar, p);
        st.textContent = p < 1 ? `올리는 중… ${Math.round(p * 100)}%` : "내용을 읽는 중…";
      });
      bar.hidden = true;
      if (r.status === "duplicate") { st.textContent = "이미 가져온 파일입니다 (같은 내용)."; st.className = "small warn-text"; }
      else {
        const skipped = Array.isArray(r.skipped) ? r.skipped.length : num(r.skipped);
        st.textContent = `완료: 문서 ${fmtNum(r.docs)}개 · ${fmtNum(r.chars)}자` + (skipped ? ` · 짧아서 건너뜀 ${fmtNum(skipped)}` : "");
        st.className = "small ok-text";
        added = true;
      }
    } catch (e) { bar.hidden = true; st.textContent = "실패: " + e.message; st.className = "small error"; }
  }
  S.uploading = false;
  loadFiles();
  if (added) { loadDocs(false); loadGate(); }
}
(function setupDrop() {
  const dz = $("#dropzone");
  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files");
  dz.addEventListener("dragenter", (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; dz.classList.add("over"); });
  dz.addEventListener("dragover", (e) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = "copy"; });
  dz.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) dz.classList.remove("over"); });
  dz.addEventListener("drop", (e) => { e.preventDefault(); depth = 0; dz.classList.remove("over"); addFiles(e.dataTransfer.files); });
  // 밖에 떨어뜨린 파일을 브라우저가 열어 버리지 않도록
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => { if (hasFiles(e) && !e.target.closest("#dropzone")) e.preventDefault(); });
  $("#file-input").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
})();
$("#ingest-path-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const r = await post("/api/ingest/path", { path: $("#ingest-path").value.trim() });
    $("#ingest-path").value = "";
    jobToast(`파일 가져오기 작업 #${r.job_id} 시작.`);
  } catch (err) { toast(err.message, 6000); }
});
function jobToast(msg) { toast(msg + " [작업] 탭에서 진행 상황을 볼 수 있습니다.", { ms: 6000, action: { label: "작업 보기", fn: () => switchTab("jobs") } }); }

async function loadWatch() {
  try {
    const { items } = await api("/api/watch");
    const user = items.filter((w) => w.origin !== "auto").length;
    $("#watch-list").innerHTML = (items.length ? `<li class="muted small">사용자 ${fmtNum(user)}개 · 자동 발견 ${fmtNum(items.length - user)}개</li>` : "") + items.map((w) => `
      <li><span class="grow"><span class="badge">${w.kind === "search" ? "검색어" : "링크"}</span>
        ${w.origin === "auto" ? '<span class="badge origin-auto" title="자율 학습이 찾은 주제">자동 발견</span>' : '<span class="badge">사용자</span>'}
        ${esc(w.value)}<br><span class="muted small">${w.last_run ? "최근 수집 " + esc(fmtTime(w.last_run)) : "아직 수집 안 함"}${num(w.depth) ? ` · 링크 ${num(w.depth)}단계` : ""}</span></span>
        <button type="button" class="danger small" data-act="del-watch" data-id="${num(w.id)}" aria-label="${esc("삭제: " + w.value)}">삭제</button></li>`).join("")
      || '<li class="muted small">비어 있습니다.</li>';
  } catch (e) { /* 무시 */ }
}
async function loadDocs(append) {
  if (!append) S.docsOffset = 0;
  const src = $("#docs-source").value;
  try {
    const r = await api(`/api/documents?offset=${S.docsOffset}${src ? "&source=" + encodeURIComponent(src) : ""}`);
    const got = r.documents || [];
    const docs = src ? got.filter((d) => d.source === src) : got; // 서버가 거르지 않아도 맞게
    const html = docs.map((d) => `
      <li><span class="grow">${docButton({ doc_id: d.id, title: d.title, url: d.url })} ${srcBadge(d.source)}
        <br><span class="muted small">${d.query ? "검색어 " + esc(d.query) + " · " : ""}${fmtNum(d.chars)}자 · ${esc(fmtTime(d.fetched_at))}</span>
        ${urlLine(d.url)}</span>
        ${S.role === "admin" ? `<button type="button" class="danger small" data-act="del-doc" data-id="${num(d.id)}" aria-label="${esc("문서 삭제: " + (d.title || d.url))}">삭제</button>` : ""}</li>`).join("");
    if (append) $("#docs").insertAdjacentHTML("beforeend", html);
    else $("#docs").innerHTML = html || '<li class="muted small">아직 없습니다.</li>';
    S.docsOffset += got.length;
    $("#docs-more").hidden = got.length < 50;
    if (r.counts) $("#doc-count").textContent = `문서 ${fmtNum(r.counts.documents)}개 · ${fmtNum(r.counts.chars)}자`;
  } catch (e) { if (!append) $("#docs").innerHTML = `<li class="error small">${esc(e.message)}</li>`; }
}
$("#docs-more").addEventListener("click", () => loadDocs(true));
$("#docs-source").addEventListener("change", () => loadDocs(false));
$("#links-btn").addEventListener("click", async () => {
  const urls = $("#links").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!urls.length) return toast("링크를 입력하세요.");
  try {
    const r = await post("/api/collect/links", { urls, depth: Number($("#depth").value) });
    $("#links").value = "";
    jobToast(`수집 작업 #${r.job_id} 시작.`);
  } catch (err) { toast(err.message, 6000); }
});
$("#websearch-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const r = await post("/api/collect/search", { query: $("#wq").value.trim() });
    $("#wq").value = "";
    jobToast(`검색 수집 작업 #${r.job_id} 시작.`);
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

// ---------------- 학습 ----------------
async function loadTrain(quiet) {
  loadAutolearn();
  if (quiet) return;
  loadDevice();
  loadModelsList().then(fillResumeSelects);
  refreshStatus().then(renderIndexInfo);
  if (S.role === "admin") { loadTrainDefaults(); loadResults(); }
}
function presetsOf() { return (S.device && S.device.presets) || FALLBACK_PRESETS; }
function limitsOf() { return (S.device && S.device.limits) || FALLBACK_LIMITS; }
function presetLabel(name) {
  const p = presetsOf()[name];
  if (name === "custom") return "직접 설정";
  if (name === "auto") return "자동";
  return p ? p.label : name || "-";
}
async function loadDevice(force) {
  const box = $("#device-card");
  let pcHtml = "";
  try {
    if (!S.device || force) S.device = await api("/api/device");
    pcHtml = pcCardHtml(S.device);
  } catch (e) { pcHtml = `<div><h4>이 PC</h4><p class="error small">${esc(e.message)}</p></div>`; }
  initTrainForms();
  let me = "";
  try {
    const ob = OB();
    if (ob) { S.myDevice = S.myDevice || await ob.detectDevice(); me = myDeviceHtml(S.myDevice); }
  } catch (e) { me = ""; }
  box.innerHTML = pcHtml + me;
}
function pcCardHtml(dev) {
  const pc = dev.pc || {}, gpu = pc.gpu || {}, rec = dev.recommend || {};
  const pk = ["torch", "transformers", "peft", "onnxruntime", "anthropic", "pypdf"];
  return `<div><h4>이 PC <span class="badge">${esc(TIER_LABEL[pc.tier] || pc.tier || "")}</span></h4>
    <dl class="kv">
      <dt>운영체제</dt><dd>${esc(pc.os || "-")}</dd>
      <dt>CPU</dt><dd>${fmtNum(pc.cpu_count)}코어</dd>
      <dt>메모리</dt><dd>${pc.ram_gb != null ? esc(pc.ram_gb) + "GB" : "알 수 없음"}</dd>
      <dt>GPU</dt><dd>${gpu.type ? esc(gpu.name || gpu.type) + (gpu.vram_gb ? ` · VRAM ${esc(gpu.vram_gb)}GB` : "") : "없음 (CPU로 학습)"}</dd>
      <dt>추천</dt><dd>소형 AI ${esc(presetLabel(rec.tiny_preset))}${rec.tiny_steps_per_cycle ? ` · 주기당 ${fmtNum(rec.tiny_steps_per_cycle)}단계` : ""}
        · LoRA ${rec.lora_base_model ? esc(rec.lora_base_model) : "권장 안 함"}</dd>
    </dl>
    <div class="pkgs">${pk.map((k) => `<span class="fit ${pc[k] ? "yes" : "no"}">${k} ${pc[k] ? "있음" : "없음"}</span>`).join("")}</div>
    ${(rec.notes || []).map((n) => `<p class="muted small">${esc(n)}</p>`).join("")}
    ${pc.torch === false ? '<p class="error small">학습하려면 PC에서 <code>pip install -r requirements-train.txt</code>를 실행하세요.</p>' : ""}
  </div>`;
}
function myDeviceHtml(d) {
  const l = d.limits || {};
  return `<div><h4>이 기기 (브라우저) <span class="badge">${esc(TIER_LABEL[d.tier] || d.tier || "")}</span></h4>
    <p class="small">${esc(d.label || "")}</p>
    <dl class="kv">
      <dt>오프라인 사본</dt><dd>조각 ${fmtNum(l.snapshotChunks)}개</dd>
      <dt>모델 형식</dt><dd>${l.preferInt8 ? "int8 (작고 빠름)" : "fp32 (정확)"} · ${fmtNum(l.maxModelMB)}MB 이하 권장</dd>
      <dt>생성 길이</dt><dd>${fmtNum(l.maxNewTokens)}자</dd>
      ${d.storageQuotaMB ? `<dt>저장 공간</dt><dd>약 ${fmtMB(d.storageQuotaMB)}</dd>` : ""}
    </dl></div>`;
}
function archFieldsHtml(prefix) {
  const L = limitsOf();
  return ARCH_KEYS.map((k) => {
    const [lo, hi] = L[k] || FALLBACK_LIMITS[k];
    return `<label>${ARCH_LABEL[k]}<input type="number" id="${prefix}-${k}" data-arch="${k}" min="${num(lo)}" max="${num(hi)}" step="1" inputmode="numeric"></label>`;
  }).join("");
}
function presetOptions() {
  const P = presetsOf();
  const rec = S.device && S.device.recommend && S.device.recommend.tiny_preset;
  return `<option value="auto">자동${rec ? ` (이 PC 추천: ${esc(presetLabel(rec))})` : ""}</option>` +
    Object.keys(P).map((k) => {
      const e = P[k].estimate;
      return `<option value="${esc(k)}">${esc(P[k].label || k)}${e && e.params ? ` · ${fmtCount(e.params)}` : ""}</option>`;
    }).join("") + '<option value="custom">직접 설정</option>';
}
function initTrainForms() {
  if (!S.formsBuilt || (S.device && !S.formsFromDevice)) {
    for (const id of ["#tiny-preset", "#al-preset"]) { const cur = $(id).value; $(id).innerHTML = presetOptions(); if (cur) $(id).value = cur; if (!$(id).value) $(id).value = "auto"; }
    $("#tiny-custom").innerHTML = archFieldsHtml("tiny");
    $("#al-custom").innerHTML = archFieldsHtml("al");
    const bm = (S.device && S.device.base_models) || [{ id: DEFAULTS.lora.base_model, size: "0.5B" }];
    const curB = $("#lora-base").value;
    $("#lora-base").innerHTML = bm.map((m) =>
      `<option value="${esc(m.id)}">${esc(m.size || "")} · ${esc(m.id)}${m.min_ram_gb ? ` (RAM ${esc(m.min_ram_gb)}GB+)` : ""}${m.license && !/^Apache/.test(m.license) ? " · " + esc(m.license) : ""}</option>`).join("")
      + `<option value="${LORA_CUSTOM}">직접 입력…</option>`;
    if (curB) $("#lora-base").value = curB;
    if (!$("#lora-base").value) $("#lora-base").selectedIndex = 0;
    S.formsBuilt = true;
    S.formsFromDevice = !!S.device;
    if (!$("#tiny-steps").value) setVal("#tiny-steps", DEFAULTS.tiny.steps);
    if (!$("#tiny-batch").value) setVal("#tiny-batch", DEFAULTS.tiny.batch_size);
    if (!$("#lora-steps").value) setVal("#lora-steps", DEFAULTS.lora.max_steps);
    if (!$("#lora-seq").value) setVal("#lora-seq", DEFAULTS.lora.seq_len);
    if (!$("#lora-batch").value) setVal("#lora-batch", DEFAULTS.lora.batch_size);
    if (!$("#al-steps").value) applyAutolearnDefaults(DEFAULTS.autolearn);
    if (S.device && S.device.recommend) {
      const rec = S.device.recommend;
      if (rec.lora_base_model && !S.defaultsApplied && $(`#lora-base option[value="${CSS.escape(rec.lora_base_model)}"]`)) $("#lora-base").value = rec.lora_base_model;
      if (rec.tiny_steps_per_cycle && !S.defaultsApplied) setVal("#al-steps", rec.tiny_steps_per_cycle);
    }
    onPresetChange("tiny", true);
    onPresetChange("al", true);
    onLoraBaseChange(true);
  }
  if (S.settings && !S.defaultsApplied) applySettingsDefaults(S.settings);
}
function applyAutolearnDefaults(a) {
  const t = a.targets || {};
  $$('input[name="al-target"]').forEach((c) => { if (c.value in t) c.checked = !!t[c.value]; });
  setVal("#al-steps", a.steps_per_cycle); setVal("#al-minchars", a.train_min_new_chars); setVal("#al-train-int", a.train_interval_min);
  setVal("#al-collect-int", a.collect_interval_min); setVal("#al-pages", a.pages_per_cycle); setVal("#al-maxhour", a.max_pages_per_hour);
  $("#al-outside").checked = !!a.train_outside_window;
}
function applySettingsDefaults(s) {
  S.defaultsApplied = true;
  const tt = s.tiny_training || {}, lt = s.lora_training || {};
  setVal("#tiny-steps", tt.steps); setVal("#tiny-batch", tt.batch_size);
  if (tt.preset && $(`#tiny-preset option[value="${CSS.escape(tt.preset)}"]`)) {
    $("#tiny-preset").value = tt.preset;
    if (tt.preset === "custom") ARCH_KEYS.forEach((k) => { if (tt[k]) setVal(`#tiny-${k}`, tt[k]); });
  }
  setVal("#lora-steps", lt.max_steps); setVal("#lora-seq", lt.seq_len); setVal("#lora-batch", lt.batch_size);
  if (lt.lora_r && $(`#lora-r option[value="${CSS.escape(String(lt.lora_r))}"]`)) $("#lora-r").value = String(lt.lora_r);
  if (lt.use_qa != null) $("#lora-qa").checked = !!lt.use_qa;
  if (lt.base_model) {
    if ($(`#lora-base option[value="${CSS.escape(lt.base_model)}"]`)) $("#lora-base").value = lt.base_model;
    else { $("#lora-base").value = LORA_CUSTOM; $("#lora-custom").value = lt.base_model; }
  }
  if (s.autolearn) applyAutolearnDefaults(s.autolearn);
  onPresetChange("tiny", true);
  onPresetChange("al", true);
  onLoraBaseChange(true);
}
async function loadTrainDefaults() {
  if (S.settings) return initTrainForms();
  try { S.settings = await api("/api/settings"); } catch (e) { return; }
  initTrainForms();
}
function readArch(prefix) {
  const out = {};
  for (const k of ARCH_KEYS) { const v = intVal(`#${prefix}-${k}`); if (v != null) out[k] = Math.round(v); }
  return out;
}
function fillArch(prefix, arch) { ARCH_KEYS.forEach((k) => { if (arch && arch[k] != null) setVal(`#${prefix}-${k}`, arch[k]); }); }
function checkArch(a) {
  const L = limitsOf();
  for (const k of ARCH_KEYS) {
    if (a[k] == null) return `${ARCH_LABEL[k]} 값을 입력하세요.`;
    const [lo, hi] = L[k] || FALLBACK_LIMITS[k];
    if (a[k] < lo || a[k] > hi) return `${ARCH_LABEL[k]}는 ${lo}~${hi} 사이여야 합니다.`;
  }
  if (a.d_model % a.n_head) return `차원(d_model ${a.d_model})은 헤드 수(${a.n_head})로 나누어떨어져야 합니다.`;
  return null;
}
function onPresetChange(prefix, silent) {
  const sel = $(`#${prefix}-preset`);
  const v = sel.value;
  const custom = v === "custom";
  $(`#${prefix}-custom`).hidden = !custom;
  if (custom && !ARCH_KEYS.some((k) => $(`#${prefix}-${k}`).value)) {
    const rec = S.device && S.device.recommend && S.device.recommend.tiny_preset;
    fillArch(prefix, presetsOf()[rec] || presetsOf().mini || FALLBACK_PRESETS.mini);
  }
  if (prefix === "tiny") {
    const p = presetsOf()[v];
    $("#tiny-preset-desc").textContent = p && p.desc ? p.desc : v === "auto" ? "이 PC 성능에 맞는 크기를 자동으로 고릅니다." : custom ? "크기를 직접 정합니다. 차원은 헤드 수로 나누어떨어져야 합니다." : "";
  }
  if (!silent || prefix !== "al" || custom) runEstimate(prefix);
}
// 실시간 크기 예상 (POST /api/estimate). 입력이 멈추면 보낸다.
const estimateDebounced = { tiny: null, al: null, lora: null };
function runEstimate(prefix) {
  if (S.role !== "admin" || S.offline) return;
  if (!estimateDebounced[prefix]) estimateDebounced[prefix] = debounce(() => doEstimate(prefix), 350);
  estimateDebounced[prefix]();
}
function tinyParams(prefix) {
  const preset = $(`#${prefix}-preset`).value;
  const body = { preset };
  if (preset === "custom" || (prefix === "tiny" && $("#tiny-resume").value)) Object.assign(body, readArch(prefix));
  return body;
}
async function doEstimate(prefix) {
  const box = $(prefix === "al" ? "#al-estimate" : `#${prefix}-estimate`);
  const btn = { tiny: "#tiny-start", lora: "#lora-start", al: "#auto-start" }[prefix];
  const seq = ++S.est[prefix];
  let body;
  if (prefix === "lora") {
    const base = loraBase();
    if (!base) { box.innerHTML = '<p class="error">모델 이름을 입력하세요.</p>'; return; }
    body = { kind: "lora", base_model: base, lora_r: Number($("#lora-r").value), seq_len: intVal("#lora-seq") || DEFAULTS.lora.seq_len, batch_size: intVal("#lora-batch") || 1 };
  } else {
    body = { kind: "tiny", ...tinyParams(prefix), batch_size: (prefix === "tiny" ? intVal("#tiny-batch") : null) || DEFAULTS.tiny.batch_size };
    if (body.preset === "custom") {
      const err = checkArch(body);
      if (err) { box.innerHTML = `<p class="error">${esc(err)}</p>`; $(btn).disabled = true; return; }
    }
  }
  box.innerHTML = '<span class="muted small"><span class="spinner" aria-hidden="true"></span>크기를 계산하는 중…</span>';
  try {
    const r = await post("/api/estimate", body);
    if (seq !== S.est[prefix]) return;
    box.innerHTML = prefix === "lora" ? loraEstimateHtml(r) : tinyEstimateHtml(r, prefix === "al");
    $(btn).disabled = false;
  } catch (e) {
    if (seq !== S.est[prefix]) return;
    box.innerHTML = `<p class="error">${esc(e.message)}</p>`;
    $(btn).disabled = !e.network;
  }
}
function fitBadges(fits) {
  return Object.keys(FIT_LABEL).map((k) => {
    if (!fits || !(k in fits)) return "";
    const ok = !!fits[k];
    return `<span class="fit ${ok ? "yes" : "no"}"><span aria-hidden="true">${ok ? "✓" : "×"}</span><span class="sr-only">${ok ? "적합:" : "부적합:"}</span>${FIT_LABEL[k]}</span>`;
  }).join("");
}
function tinyEstimateHtml(r, compact) {
  const e = r.estimate || r, a = r.arch || {};
  const arch = ARCH_KEYS.every((k) => a[k] != null)
    ? `<div class="small muted">${esc(presetLabel(a.preset))} · 차원 ${fmtNum(a.d_model)} · 층 ${fmtNum(a.n_layer)} · 헤드 ${fmtNum(a.n_head)} · 문맥 ${fmtNum(a.seq_len)} · 글자 ${fmtNum(a.vocab_size)}</div>` : "";
  return `${arch}<div>파라미터 <b>${fmtCount(e.params)}</b> · 파일 ${fmtMB(e.fp32_mb)} (fp32) / ${fmtMB(e.int8_mb)} (int8)</div>
    ${compact ? "" : `<div class="small">학습 메모리 약 ${fmtMB(e.train_ram_mb)} · 실행 메모리 약 ${fmtMB(e.infer_ram_mb)}</div>`}
    <div>${fitBadges(e.fits)}</div>`;
}
function loraEstimateHtml(e) {
  const pc = (S.device && S.device.pc) || {}, gpu = pc.gpu || {};
  const ramOk = pc.ram_gb != null ? pc.ram_gb >= e.ram_gb_est : null;
  const vramOk = gpu.type && gpu.vram_gb != null ? gpu.vram_gb >= e.vram_gb_est : gpu.type ? null : false;
  const badge = (ok, yes, no, na) => `<span class="fit ${ok == null ? "na" : ok ? "yes" : "no"}">${esc(ok == null ? na : ok ? yes : no)}</span>`;
  return `<div>기반 모델 ${esc(e.size || (e.base_params_b ? e.base_params_b + "B" : "크기 알 수 없음"))}${e.license ? ` · 사용 조건 ${esc(e.license)}` : ""}</div>
    <div>학습 파라미터 약 <b>${fmtCount(e.trainable_params_est)}</b> · 어댑터 약 ${fmtMB(e.adapter_mb_est)}</div>
    <div class="small">필요 메모리: CPU 학습 시 RAM 약 ${esc(e.ram_gb_est)}GB · GPU 학습 시 VRAM 약 ${esc(e.vram_gb_est)}GB</div>
    <div>${badge(ramOk, `RAM 충분 (이 PC ${pc.ram_gb}GB)`, `RAM 부족 (이 PC ${pc.ram_gb}GB)`, "RAM 확인 불가")}${badge(vramOk, `GPU 충분 (${gpu.vram_gb}GB)`, gpu.type ? `GPU 메모리 부족 (${gpu.vram_gb}GB)` : "GPU 없음 (CPU로 매우 느림)", "GPU 메모리 확인 불가")}</div>
    ${e.note ? `<p class="muted small">${esc(e.note)}</p>` : ""}`;
}
function loraBase() {
  const v = $("#lora-base").value;
  return v === LORA_CUSTOM ? $("#lora-custom").value.trim() : v;
}
function onLoraBaseChange(silent) {
  $("#lora-custom-wrap").hidden = $("#lora-base").value !== LORA_CUSTOM;
  if (!silent || S.role === "admin") runEstimate("lora");
}
function archOf(m) {
  const i = (m && m.info) || {};
  const a = i.arch || i.config || i;
  return ARCH_KEYS.every((k) => a[k] != null) ? { preset: a.preset || i.preset, ...Object.fromEntries(ARCH_KEYS.map((k) => [k, a[k]])) } : null;
}
function fillResumeSelects() {
  const mk = (kind, sel) => {
    const cur = $(sel).value;
    const list = S.models.filter((m) => modelKind(m) === kind);
    $(sel).innerHTML = '<option value="">새로 학습</option>' + list.map((m) => {
      const i = m.info || {};
      const extra = kind === "tiny" ? (i.preset || (i.arch && i.arch.preset) || "") : (i.base_model || "");
      return `<option value="${num(m.id)}">#${num(m.id)} ${esc(m.name)}${extra ? " · " + esc(extra) : ""} · ${fmtNum(i.total_steps || i.steps || 0)}단계</option>`;
    }).join("");
    if (list.some((m) => String(m.id) === cur)) $(sel).value = cur;
  };
  mk("tiny", "#tiny-resume");
  mk("lora", "#lora-resume");
  onResumeChange("tiny", true);
  onResumeChange("lora", true);
}
function onResumeChange(kind, silent) {
  const id = $(`#${kind}-resume`).value;
  const m = id ? S.models.find((x) => String(x.id) === id) : null;
  if (kind === "tiny") {
    const a = archOf(m);
    const lock = !!(m && a);
    $("#tiny-preset").disabled = lock;
    $$("#tiny-custom input").forEach((i) => (i.disabled = lock));
    if (lock) {
      const P = presetsOf();
      const same = a.preset && P[a.preset] && ARCH_KEYS.every((k) => Number(P[a.preset][k]) === Number(a[k]));
      $("#tiny-preset").value = same ? a.preset : "custom";
      fillArch("tiny", a);
      $("#tiny-custom").hidden = !(!same);
      $("#tiny-preset-desc").textContent = "이어 학습은 원래 모델과 같은 크기로만 할 수 있습니다.";
    } else if (!silent) onPresetChange("tiny");
    if (!silent || lock) runEstimate("tiny");
  } else {
    const base = m && m.info && m.info.base_model;
    $("#lora-base").disabled = !!base;
    $("#lora-custom").disabled = !!base;
    if (base) {
      if ($(`#lora-base option[value="${CSS.escape(base)}"]`)) $("#lora-base").value = base;
      else { $("#lora-base").value = LORA_CUSTOM; $("#lora-custom").value = base; }
      onLoraBaseChange(true);
    }
    if (!silent) runEstimate("lora");
  }
}
$("#tiny-preset").addEventListener("change", () => onPresetChange("tiny"));
$("#al-preset").addEventListener("change", () => onPresetChange("al"));
$("#tiny-custom").addEventListener("input", () => runEstimate("tiny"));
$("#al-custom").addEventListener("input", () => runEstimate("al"));
$("#tiny-batch").addEventListener("input", () => runEstimate("tiny"));
$("#tiny-resume").addEventListener("change", () => onResumeChange("tiny"));
$("#lora-resume").addEventListener("change", () => onResumeChange("lora"));
$("#lora-base").addEventListener("change", () => onLoraBaseChange());
["#lora-custom", "#lora-seq", "#lora-batch"].forEach((s) => $(s).addEventListener("input", () => runEstimate("lora")));
$("#lora-r").addEventListener("change", () => runEstimate("lora"));

$("#tiny-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = tinyParams("tiny");
  if (body.preset === "custom") { const err = checkArch(body); if (err) return toast(err); }
  const steps = intVal("#tiny-steps"), batch = intVal("#tiny-batch");
  if (steps) body.steps = steps;
  if (batch) body.batch_size = batch;
  const rid = $("#tiny-resume").value;
  if (rid) body.resume_from_model_id = Number(rid);
  try { const r = await post("/api/train/tiny", body); jobToast(`소형 AI 모델 학습 작업 #${r.job_id} 시작.`); } catch (err) { toast(err.message, 7000); }
});
$("#lora-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const base = loraBase();
  if (!base) return toast("기반 모델을 고르거나 입력하세요.");
  const body = { base_model: base, lora_r: Number($("#lora-r").value), use_qa: $("#lora-qa").checked };
  const steps = intVal("#lora-steps"), seq = intVal("#lora-seq"), batch = intVal("#lora-batch");
  if (steps) body.max_steps = steps;
  if (seq) body.seq_len = seq;
  if (batch) body.batch_size = batch;
  const rid = $("#lora-resume").value;
  if (rid) body.resume_from_model_id = Number(rid);
  try { const r = await post("/api/train/lora", body); jobToast(`LoRA 학습 작업 #${r.job_id} 시작.`); } catch (err) { toast(err.message, 7000); }
});
function renderIndexInfo(st) {
  if (!st) return;
  const c = st.counts || {};
  $("#index-info").textContent = `의미 검색 색인: 조각 ${fmtNum(c.chunks)}개 중 ${fmtNum(c.embeddings)}개 색인됨. 새 자료를 넣은 뒤 색인을 만들면 뜻이 비슷한 글도 찾습니다.`;
  const o = st.online;
  $("#qa-info").textContent = o
    ? (o.available ? `온라인 모델 사용 가능 (${o.model || ""}). 자료 조각마다 질문·답변을 만들어 LoRA 학습에 씁니다.` : `온라인 모델을 쓸 수 없습니다: ${o.reason || "관리 > 설정을 확인하세요."}`)
    : "온라인 모델로 자료 조각마다 질문·답변을 만들어 LoRA 학습에 씁니다.";
}
$("#emb-btn").addEventListener("click", async () => {
  try { const r = await post("/api/index/embeddings"); jobToast(`의미 검색 색인 작업 #${r.job_id} 시작.`); } catch (e) { toast(e.message, 6000); }
});
$("#qa-btn").addEventListener("click", async () => {
  try {
    const r = await post("/api/qa/build", { limit_chunks: intVal("#qa-chunks") || 50, per_chunk: intVal("#qa-per") || 2 });
    jobToast(`Q&A 데이터 생성 작업 #${r.job_id} 시작.`);
  } catch (e) { toast(e.message, 6000); }
});

// 학습 결과 파일
function resultPath(name, rel) {
  const base = "/api/results/" + encodeURIComponent(name);
  return rel == null ? base + "/download" : base + "/files/" + String(rel).split("/").map(encodeURIComponent).join("/");
}
function filesHtml(name, files, canDl) {
  return '<ul class="files">' + (files || []).map((f) => `
    <li><span class="grow">${esc(f.path)}${f.dir ? ` <span class="muted">(파일 ${fmtNum(f.count)}개)</span>` : ""}</span>
      <span class="size">${fmtBytes(f.bytes)}</span>
      ${canDl && !f.dir ? `<button type="button" class="ghost small" data-act="dl-file" data-name="${esc(name)}" data-path="${esc(f.path)}" aria-label="${esc("받기: " + f.path)}">받기</button>` : ""}</li>`).join("") + "</ul>";
}
async function loadResults() {
  const box = $("#results-list");
  try {
    const { results } = await api("/api/results");
    box.innerHTML = (results || []).map((r) => `
      <div class="results-item">
        <div class="row wrap gap between">
          <span class="grow"><b>${esc(r.name)}</b><br><span class="muted small">${esc(fmtTime(r.created_at))} · 파일 ${fmtNum(r.file_count != null ? r.file_count : (r.files || []).length)}개 · ${fmtBytes(r.total_bytes)}${r.zip_bytes ? " · ZIP " + fmtBytes(r.zip_bytes) : ""}</span></span>
          ${r.zip ? `<button type="button" class="small" data-act="dl-zip" data-name="${esc(r.name)}">ZIP 받기</button>` : ""}
        </div>
        <details><summary>파일 목록</summary>${filesHtml(r.name, r.files, true)}</details>
      </div>`).join("") || '<p class="muted small">아직 결과가 없습니다. 자율 학습에서 [학습 종료]를 누르면 만들어집니다.</p>';
  } catch (e) { box.innerHTML = `<p class="error small">${esc(e.message)}</p>`; }
}

// 자율 학습
async function loadAutolearn() {
  let st;
  try { st = await api("/api/autolearn"); } catch (e) {
    if (!S.auto) $("#auto-status").innerHTML = `<p class="error small">${esc(e.message)}</p>`;
    return;
  }
  const prev = S.auto;
  S.auto = st;
  renderAutolearn(st);
  if (prev && prev.active && !st.active) { loadModelsList().then(fillResumeSelects); if (S.role === "admin") loadResults(); }
}
function renderAutolearn(st) {
  const s = st.session;
  const phase = st.phase || "idle";
  const ph = $("#auto-phase");
  ph.textContent = PHASE[phase] || phase;
  ph.className = "status " + cls(phase);
  const running = !!(st.active && s && s.status === "running");
  const finishing = !!(st.active && s && s.status === "finishing") || phase === "finishing";
  $("#auto-stop").hidden = !running;
  $("#auto-stop").disabled = false;
  $("#auto-form").hidden = !!st.active;
  if (!st.active && !$("#al-estimate").innerHTML && S.role === "admin") runEstimate("al");
  const w = st.window || {};
  let h = `<p class="small">허용 시간: ${esc(w.windows || "-")} · ${w.allowed ? '<span class="ok-text">지금 수집 가능</span>' : `<span class="warn-text">지금은 수집 불가</span>${w.next_allowed ? ` (다음: ${esc(fmtTime(w.next_allowed))})` : ""}`}</p>`;
  if (st.active && s) {
    const x = s.stats || {}, c = s.config || {};
    const stat = (v, label) => `<div class="stat"><b>${v}</b><span>${label}</span></div>`;
    const targets = Object.keys(TARGET_LABEL).filter((k) => c.targets && c.targets[k]).map((k) => TARGET_LABEL[k]);
    h += `<div class="stats">
      ${stat(fmtNum(x.cycles || 0), "수집 횟수")}${stat(fmtNum(x.pages_collected || 0), "수집한 페이지")}
      ${stat(fmtNum(x.docs_new || 0), "새 문서")}${stat(fmtNum(x.chars_new || 0), "새 글자")}
      ${stat(fmtNum(x.trainings || 0), "학습 횟수")}${stat(fmtNum(x.discovered_topics || 0), "자동 발견 주제")}
      ${stat(`${fmtNum(st.hour_pages || 0)} / ${fmtNum(c.max_pages_per_hour)}`, "최근 1시간 페이지")}
      ${stat(`${fmtNum(st.new_chars_since_train || 0)} / ${fmtNum(c.train_min_new_chars)}`, "학습 후 새 글자 / 기준")}
    </div>
    <dl class="kv">
      <dt>시작</dt><dd>${esc(fmtTime(s.started_at))}${s.actor ? " · " + esc(s.actor) : ""}</dd>
      <dt>학습 대상</dt><dd>${esc(targets.join(", ") || "없음")}${c.tiny_preset && c.targets && c.targets.tiny ? ` · 소형 AI 크기 ${esc(presetLabel(c.tiny_preset))}` : ""}</dd>
      <dt>마지막 수집</dt><dd>${x.last_collect_at ? esc(fmtTime(x.last_collect_at)) : "아직 없음"}</dd>
      <dt>마지막 학습</dt><dd>${x.last_train_at ? esc(fmtTime(x.last_train_at)) : "아직 없음"}</dd>
      ${running ? `<dt>다음 수집</dt><dd>${st.next_collect_at ? `${esc(fmtTime(st.next_collect_at))} (${esc(relTime(st.next_collect_at))})` : "허용 시간을 기다리는 중"}</dd>
      <dt>다음 학습 확인</dt><dd>${st.next_train_check_at ? `${esc(fmtTime(st.next_train_check_at))} (${esc(relTime(st.next_train_check_at))})` : "-"}</dd>` : ""}
    </dl>`;
    const hist = (x.train_history || []).slice(-3).reverse();
    if (hist.length) {
      h += '<ul class="files">' + hist.map((t) => `<li><span class="grow">${esc(fmtTime(t.ts))} · ${esc(TARGET_LABEL[t.kind] || t.kind || "")}</span><span class="size">평가 손실 ${fmtLoss(t.eval_loss_before)} → ${fmtLoss(t.eval_loss_after)}</span></li>`).join("") + "</ul>";
    }
    if (x.note) h += `<p class="muted small">${esc(x.note)}</p>`;
  } else {
    h += `<p class="muted small">진행 중인 자율 학습이 없습니다.${S.role === "admin" ? "" : " 관리자가 시작할 수 있습니다."}</p>`;
  }
  $("#auto-status").innerHTML = h;
  $("#auto-finishing").hidden = !finishing;
  if (finishing) updateFinishing(st);
  const key = st.active ? "active" : resultKey(st);
  if (key !== S.lastResultKey) { S.lastResultKey = key; $("#auto-result").innerHTML = st.active ? "" : lastResultHtml(st); }
}
function lastSession(st) { return (st.recent_sessions || []).find((x) => x.status !== "running" && x.status !== "finishing"); }
function resultKey(st) { const s = lastSession(st); return s ? `${s.id}:${s.status}:${S.role}` : "none"; }
async function updateFinishing(st) {
  const jid = st.jobs && st.jobs.finalize;
  if (!jid) { $("#auto-fin-msg").textContent = "준비 중…"; return; }
  try {
    const { jobs } = await api("/api/jobs");
    const j = (jobs || []).find((x) => x.id === jid);
    if (!j) return;
    $("#auto-fin-bar").style.width = Math.round(num(j.progress) * 100) + "%";
    $("#auto-fin-msg").textContent = `${Math.round(num(j.progress) * 100)}% · ${j.message || STATUS[j.status] || j.status || ""}`;
  } catch (e) { /* 무시 */ }
}
function summaryHtml(sm) {
  if (!sm || typeof sm !== "object") return "";
  const parts = [];
  if (sm.documents != null) parts.push(`문서 ${fmtNum(sm.documents)}개 (${fmtNum(sm.chars)}자)`);
  if (sm.session_documents != null) parts.push(`이번 세션 ${fmtNum(sm.session_documents)}개`);
  if (sm.files_ingested) parts.push(`가져온 파일 ${fmtNum(sm.files_ingested)}개`);
  if (sm.pages_collected != null) parts.push(`수집 페이지 ${fmtNum(sm.pages_collected)}`);
  if (sm.trainings != null) parts.push(`학습 ${fmtNum(sm.trainings)}회`);
  if (Array.isArray(sm.models) && sm.models.length) parts.push(`모델 ${fmtNum(sm.models.length)}개`);
  if (sm.qa_pairs) parts.push(`Q&A ${fmtNum(sm.qa_pairs)}개`);
  return parts.length ? `<p class="small">${esc(parts.join(" · "))}</p>` : "";
}
function lastResultHtml(st) {
  const s = lastSession(st);
  if (!s) return "";
  const r = s.result || {};
  if (s.status === "ended" && r.name) {
    const admin = S.role === "admin";
    const open = S.openResult && (S.openResult === true || S.openResult === r.name);
    S.openResult = null;
    return `<div class="result-card" id="last-result">
      <h4>학습 종료 결과 · ${esc(fmtTime(s.ended_at))}</h4>
      <p class="small"><b>${esc(r.name)}</b><br>파일 ${fmtNum(r.file_count)}개 · 전체 ${fmtBytes(r.total_bytes)}${r.zip_bytes ? " · ZIP " + fmtBytes(r.zip_bytes) : ""}</p>
      ${summaryHtml(r.summary)}
      ${r.dir ? `<p class="small muted">PC 폴더: ${esc(r.dir)}</p>` : ""}
      ${r.copied_to ? `<p class="small">외장 저장장치에 복사함: ${esc(r.copied_to)}</p>` : ""}
      ${r.copy_error ? `<p class="small error">외장 저장장치 복사 실패: ${esc(r.copy_error)}</p>` : ""}
      ${r.final_train && r.final_train.error ? `<p class="small error">마지막 학습 실패: ${esc(r.final_train.error)}</p>` : ""}
      ${admin && r.zip ? `<button type="button" class="small" data-act="dl-zip" data-name="${esc(r.name)}">ZIP 받기${r.zip_bytes ? ` (${fmtBytes(r.zip_bytes)})` : ""}</button>` : ""}
      ${admin ? "" : '<p class="muted small">파일 받기는 관리자만 할 수 있습니다.</p>'}
      <details${open ? " open" : ""}><summary>파일 목록</summary>${filesHtml(r.name, r.files, admin)}</details>
    </div>`;
  }
  if (s.status === "failed" || s.status === "cancelled") {
    return `<div class="banner warn">최근 학습 종료가 ${s.status === "failed" ? "실패했습니다" : "취소되었습니다"}${r.error ? ": " + esc(r.error) : "."}</div>`;
  }
  return "";
}
$("#auto-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const targets = {};
  $$('input[name="al-target"]').forEach((c) => (targets[c.value] = c.checked));
  if (!Object.values(targets).some(Boolean)) return toast("학습 대상을 하나 이상 고르세요.");
  const preset = $("#al-preset").value;
  const body = { targets, tiny_preset: preset, train_outside_window: $("#al-outside").checked };
  if (preset === "custom") {
    const a = readArch("al");
    const err = checkArch(a);
    if (err && targets.tiny) return toast(err);
    body.tiny_arch = a;
  }
  const nums = { steps_per_cycle: "#al-steps", train_min_new_chars: "#al-minchars", train_interval_min: "#al-train-int",
    collect_interval_min: "#al-collect-int", pages_per_cycle: "#al-pages", max_pages_per_hour: "#al-maxhour" };
  for (const [k, sel] of Object.entries(nums)) { const v = intVal(sel); if (v != null) body[k] = v; }
  askNotifyPermission();
  const btn = $("#auto-start");
  btn.disabled = true;
  try { await post("/api/autolearn/start", body); toast("자율 학습을 시작했습니다. 허용 시간에 조금씩 수집하고 학습합니다."); S.lastResultKey = ""; loadAutolearn(); }
  catch (err) { toast(err.message, 7000); }
  finally { btn.disabled = false; }
});
$("#auto-stop").addEventListener("click", async () => {
  if (!confirm("자율 학습을 끝낼까요?\n\n마지막 학습을 한 뒤, 배운 내용을 파일(JSONL·CSV·Markdown·TXT, 검색 색인, 모델, 보고서, ZIP)로 내보냅니다. 끝나면 알림으로 알려 드립니다.")) return;
  askNotifyPermission();
  const btn = $("#auto-stop");
  btn.disabled = true;
  try { await post("/api/autolearn/stop"); toast("학습 종료를 시작했습니다. 끝나면 알려 드립니다."); loadAutolearn(); }
  catch (err) { toast(err.message, 7000); btn.disabled = false; }
});

// ---------------- 모델 ----------------
function modelFiles(m) { const f = (m.info && m.info.files) || {}; return typeof f === "object" && !Array.isArray(f) ? f : {}; }
function hasInt8(m) { const i = m.info || {}; return !!(i.onnx_int8 || modelFiles(m)["model.int8.onnx"]); }
function hasOnnx(m) { const i = m.info || {}; return !!(i.onnx || modelFiles(m)["model.onnx"]); }
function autoVariant(m) { const d = S.myDevice; return d && d.limits && d.limits.preferInt8 && hasInt8(m) ? "int8" : "fp32"; }
function infoList(m) {
  const i = m.info || {}, kind = modelKind(m);
  const a = archOf(m);
  const parent = i.parent ? String(i.parent).split(/[\\/]/).filter(Boolean).pop() : null;
  const rows = kind === "tiny" ? [
    ["크기", i.preset ? presetLabel(i.preset) : null],
    ["구조", a ? `차원 ${a.d_model} · 층 ${a.n_layer} · 헤드 ${a.n_head} · 문맥 ${a.seq_len} · 글자 ${a.vocab_size}` : null],
    ["파라미터", i.params ? `${fmtCount(i.params)} (${fmtNum(i.params)})` : null],
  ] : [
    ["기반 모델", i.base_model],
    ["LoRA", i.lora_r ? `r ${i.lora_r} · alpha ${i.lora_alpha ?? "-"}` : null],
    ["파라미터", i.trainable_params ? `${fmtCount(i.trainable_params)} 학습 / ${fmtCount(i.total_params)} 전체` : null],
    ["Q&A 예시", i.qa_examples ? fmtNum(i.qa_examples) : null],
  ];
  rows.push(
    ["학습 단계", i.steps != null ? `${fmtNum(i.steps)}${i.total_steps && i.total_steps !== i.steps ? ` (누적 ${fmtNum(i.total_steps)})` : ""}` : null],
    ["이어 학습한 모델", parent],
    ["평가 손실 (전 → 후)", i.eval_loss_before != null || i.eval_loss_after != null ? `${fmtLoss(i.eval_loss_before)} → ${fmtLoss(i.eval_loss_after)}` : null],
    ["퍼플렉시티", i.perplexity_after != null ? fmtLoss(i.perplexity_after) : null],
    ["장치", i.device ? `${i.device}${i.threads ? ` · ${i.threads}스레드` : ""}` : null],
    ["학습 시간", i.seconds != null ? fmtDur(i.seconds) : null],
    ["ONNX", kind === "tiny" ? `${hasOnnx(m) ? "있음" : "없음"}${hasInt8(m) ? " · int8 있음" : ""}${i.onnx_verified ? " · 검증됨" : ""}` : null],
    ["예시 생성", i.sample || null],
  );
  return '<dl class="kv">' + rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("") + "</dl>";
}
function modelCardHtml(m) {
  const kind = modelKind(m), i = m.info || {}, id = num(m.id);
  const files = modelFiles(m);
  const fl = Object.keys(files);
  let h = `<div class="card model">
    <div class="row wrap gap"><h3 class="grow">${esc(m.name)}</h3><span class="badge kind-${kind}">${kind === "tiny" ? "소형 AI" : "LoRA"}</span></div>
    <p class="muted small">#${id} · ${esc(fmtTime(m.created_at))}</p>
    ${infoList(m)}
    ${i.onnx_error ? `<p class="error small">ONNX 변환 오류: ${esc(i.onnx_error)}</p>` : ""}
    ${fl.length ? `<details><summary>파일 ${fl.length}개</summary><ul class="files">${fl.map((n) => `<li><span class="grow">${esc(n)}</span><span class="size">${fmtBytes(files[n])}</span></li>`).join("")}</ul></details>` : ""}`;
  if (kind === "tiny") {
    if (hasOnnx(m)) {
      const av = autoVariant(m);
      h += `<div class="model-actions"><div class="row wrap gap">
        <label class="inline small">형식
          <select id="variant-${id}">
            <option value="auto">자동 (${av})</option>
            <option value="int8"${hasInt8(m) ? "" : " disabled"}>int8${files["model.int8.onnx"] ? ` · ${fmtBytes(files["model.int8.onnx"])}` : ""}</option>
            <option value="fp32">fp32${files["model.onnx"] ? ` · ${fmtBytes(files["model.onnx"])}` : ""}</option>
          </select></label>
        <button type="button" class="secondary small" data-act="get-model" data-id="${id}">이 기기로 받기</button>
      </div></div>`;
    } else h += '<p class="error small">ONNX 파일이 없어 이 기기로 받을 수 없습니다.</p>';
    h += `<form class="pc-gen" data-id="${id}">
        <div class="row"><label class="sr-only" for="pcgen-${id}">PC에서 생성할 시작 문장</label>
          <input id="pcgen-${id}" name="prompt" placeholder="시작 문장" required>
          <button type="submit" class="secondary small">PC에서 생성</button></div>
        <pre class="out" hidden aria-live="polite"></pre>
      </form>`;
  } else {
    h += `<div class="model-actions"><button type="button" class="secondary small" data-act="ask-with" data-id="${id}">이 모델로 질문하기</button>
      <span class="muted small">PC에서 검색한 자료를 근거로 답합니다.</span></div>`;
  }
  return h + "</div>";
}
async function loadModels() {
  refreshLocalModelInfo();
  if (S.offline) {
    $("#models").innerHTML = '<div class="card muted small">오프라인에서는 PC의 모델 목록을 볼 수 없습니다. 이 기기에 받은 모델은 위에서 실행할 수 있습니다.</div>';
    return;
  }
  const ob = OB();
  if (ob && !S.myDevice) { try { S.myDevice = await ob.detectDevice(); } catch (e) { /* 무시 */ } }
  await loadModelsList();
  $("#models").innerHTML = S.models.map(modelCardHtml).join("") || '<div class="card muted small">아직 학습한 모델이 없습니다. [학습] 탭에서 학습하세요.</div>';
}
async function downloadModelToDevice(id, btn) {
  const ob = OB();
  if (!ob) return toast("온보드 엔진을 불러오지 못했습니다.");
  const m = S.models.find((x) => x.id === id);
  if (!m) return;
  let variant = ($(`#variant-${id}`) || {}).value || "auto";
  if (variant === "auto") variant = autoVariant(m);
  if (variant === "int8" && !hasInt8(m)) variant = "fp32";
  const file = variant === "int8" ? "model.int8.onnx" : "model.onnx";
  const label = btn.textContent;
  btn.disabled = true; btn.textContent = "받는 중…";
  try {
    const [buf, vocab] = await Promise.all([
      fetchBuffer(`/api/models/${id}/files/${file}`, (p) => (btn.textContent = `받는 중… ${Math.round(p * 100)}%`)),
      api(`/api/models/${id}/files/vocab.json`),
    ]);
    btn.textContent = "저장 중…";
    const saved = await ob.saveModel({ id, name: m.name, info: m.info }, buf, vocab.itos || vocab, variant);
    const warn = saved && saved.warning;
    toast(`모델을 이 기기에 저장했습니다 (${variant}, ${fmtBytes(buf.byteLength)}).` + (warn ? "\n" + warn : ""), warn ? 9000 : 4000);
    refreshLocalModelInfo();
    applyEngineUI();
  } catch (e) { toast(e.message, 7000); }
  finally { btn.disabled = false; btn.textContent = label; }
}
async function refreshLocalModelInfo() {
  const ob = OB();
  if (!ob) { $("#local-model-info").textContent = "온보드 엔진(onboard.js)을 불러오지 못했습니다."; $("#local-gen-btn").disabled = true; return; }
  try {
    const dev = S.myDevice || await ob.detectDevice();
    S.myDevice = dev;
    $("#local-device").textContent = `이 기기: ${dev.label} · 권장 형식 ${dev.limits.preferInt8 ? "int8" : "fp32"}`;
    if (!$("#local-max").value) $("#local-max").value = dev.limits.maxNewTokens;
  } catch (e) { /* 무시 */ }
  try {
    const m = await ob.modelInfo();
    S.localModel = m;
    $("#local-model-info").textContent = m
      ? `저장된 모델: ${m.name || "#" + m.id} (${m.variant || "fp32"} · ${fmtBytes(m.bytes)}${m.saved_at ? " · " + fmtTime(m.saved_at) : ""}). 인터넷 없이 이 기기에서 실행됩니다. 처음 한 번은 실행 엔진을 받아야 합니다.`
      : "받은 모델이 없습니다. 아래 목록에서 소형 AI 모델의 [이 기기로 받기]를 누르세요.";
    $("#local-del").hidden = !m || typeof ob.deleteModel !== "function";
    $("#local-gen-btn").disabled = !m;
  } catch (e) { /* 무시 */ }
}
$("#local-gen-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const ob = OB();
  if (!ob) return;
  const prompt = $("#local-prompt").value;
  const out = $("#local-out");
  out.hidden = false;
  out.innerHTML = `<span class="prompt">${esc(prompt)}</span><span class="gen"> (실행 엔진 준비 중…)</span>`;
  const gen = $(".gen", out);
  if (S.genAbort) S.genAbort.abort();
  const ctl = (S.genAbort = new AbortController());
  $("#local-stop").hidden = false;
  $("#local-gen-btn").disabled = true;
  let started = false;
  try {
    const text = await ob.generate(prompt, {
      maxNew: intVal("#local-max") || undefined, signal: ctl.signal,
      onToken: (piece, t) => { started = true; gen.textContent = t; },
    });
    gen.textContent = text || (started ? gen.textContent : " (생성된 글자가 없습니다)");
  } catch (err) { gen.textContent = ""; out.insertAdjacentHTML("beforeend", `\n<span class="error">${esc(err.message)}</span>`); }
  finally {
    if (S.genAbort === ctl) S.genAbort = null;
    $("#local-stop").hidden = true;
    $("#local-gen-btn").disabled = !S.localModel;
  }
});
$("#local-stop").addEventListener("click", () => { if (S.genAbort) S.genAbort.abort(); });
$("#local-del").addEventListener("click", async () => {
  const ob = OB();
  if (!ob || !confirm("이 기기에 저장된 모델을 지울까요?")) return;
  try { await ob.deleteModel(); toast("모델을 지웠습니다."); } catch (e) { toast(e.message); }
  refreshLocalModelInfo();
});
document.addEventListener("submit", async (e) => {
  const f = e.target.closest && e.target.closest(".pc-gen");
  if (!f) return;
  e.preventDefault();
  const out = $("pre", f), btn = $("button", f);
  out.hidden = false;
  out.textContent = "PC에서 생성하는 중…";
  btn.disabled = true;
  try {
    const r = await post("/api/generate", { model_id: Number(f.dataset.id), prompt: f.elements.prompt.value, max_new: 200 });
    out.innerHTML = `<span class="prompt">${esc(f.elements.prompt.value)}</span>${esc(r.text || "")}`;
  } catch (err) { out.innerHTML = `<span class="error">${esc(err.message)}</span>`; }
  finally { btn.disabled = false; }
});

// ---------------- 작업 ----------------
function jobSummary(j) {
  const r = j.result || {};
  const k = j.kind;
  if (k.startsWith("collect")) return r.new != null ? `새 문서 ${r.new} · 갱신 ${r.updated} · 변화 없음 ${r.unchanged} · 짧아서 제외 ${r.skipped} · 오류 ${(r.errors || []).length}` : "";
  if (k === "ingest_path") return r.files != null ? `파일 ${r.files} · 문서 ${r.docs} · 중복 ${r.duplicates || 0} · 오류 ${(r.errors || []).length}` : "";
  if (k === "index_embeddings") return r.indexed != null ? `새로 색인 ${fmtNum(r.indexed)} · 전체 ${fmtNum(r.total)}${r.model ? " · " + r.model : ""}` : "";
  if (k === "build_qa") return r.added != null ? `Q&A ${fmtNum(r.added)}개 추가 · 전체 ${fmtNum(r.total)}${r.errors ? ` · 오류 ${Array.isArray(r.errors) ? r.errors.length : r.errors}` : ""}` : "";
  if (k === "finalize_learning") return r.name ? `결과 ${r.name} · 파일 ${fmtNum(r.file_count != null ? r.file_count : (r.files || []).length)}개 · ${fmtBytes(r.total_bytes)}` : "";
  if (k === "autolearn_train") {
    const ev = r.eval || {};
    const parts = Object.keys(ev).map((t) => `${TARGET_LABEL[t] || t} ${fmtLoss(ev[t] && ev[t].before)} → ${fmtLoss(ev[t] && ev[t].after)}`);
    return parts.length ? "평가 손실: " + parts.join(" · ") : r.models ? `모델 ${r.models.length}개` : "";
  }
  if (k.startsWith("train")) return r.eval_loss_after != null ? `평가 손실 ${fmtLoss(r.eval_loss_before)} → ${fmtLoss(r.eval_loss_after)}` : "";
  if (k === "export") return r.path ? `${r.path} (${r.files}개 파일, ${fmtBytes(r.bytes)})` : "";
  return "";
}
function jobParams(j) {
  const p = j.params || {};
  if (j.kind === "collect_search" && p.query) return "검색어: " + p.query;
  if (j.kind === "collect_links" && p.urls) return p.urls[0] + (p.urls.length > 1 ? ` 외 ${p.urls.length - 1}개` : "");
  if (j.kind === "ingest_path" && p.path) return p.path;
  if ((j.kind === "train_tiny" || j.kind === "train_mobile") && p.preset) return "크기: " + presetLabel(p.preset) + (p.resume_from_model_id ? ` · #${p.resume_from_model_id}에서 이어서` : "");
  if ((j.kind === "train_lora" || j.kind === "train_pc") && p.base_model) return p.base_model + (p.resume_from_model_id ? ` · #${p.resume_from_model_id}에서 이어서` : "");
  if (j.kind === "export" && p.target) return p.target;
  return "";
}
async function loadJobs() {
  try {
    const { jobs } = await api("/api/jobs");
    $("#jobs").innerHTML = (jobs || []).map((j) => {
      const active = j.status === "running" || j.status === "queued";
      const canCancel = active && (S.role === "admin" || String(j.kind).startsWith("collect"));
      const r = j.result || {};
      const errs = Array.isArray(r.errors) && r.errors.length ? `<details><summary>오류 ${r.errors.length}건</summary><pre>${esc(r.errors.map((x) => typeof x === "string" ? x : JSON.stringify(x)).join("\n"))}</pre></details>` : "";
      const tb = r.traceback ? `<details><summary>자세한 오류</summary><pre>${esc(r.traceback)}</pre></details>` : "";
      const params = jobParams(j);
      return `<div class="card job">
        <div class="row wrap gap"><b>#${num(j.id)} ${esc(KIND[j.kind] || j.kind)}</b> <span class="status ${cls(j.status)}">${esc(STATUS[j.status] || j.status)}</span>
          <span class="muted small">${esc(j.actor === "autolearn" ? "자율 학습" : j.actor || "")} · ${esc(fmtTime(j.created_at))}</span>
          ${canCancel ? `<button type="button" class="danger small" data-act="cancel-job" data-id="${num(j.id)}">취소</button>` : ""}</div>
        ${params ? `<div class="small muted">${esc(params)}</div>` : ""}
        ${active ? `<div class="bar"><div style="width:${Math.round(num(j.progress) * 100)}%"></div></div>` : ""}
        <div class="small">${esc(j.message || "")}</div>
        <div class="small muted">${esc(jobSummary(j))}</div>${errs}${tb}
      </div>`;
    }).join("") || '<div class="card muted">작업이 없습니다.</div>';
  } catch (e) { /* 무시 */ }
}

// ---------------- 관리 ----------------
function windowRow(w) {
  const div = document.createElement("div");
  div.className = "window-row";
  const days = Array.isArray(w.days) ? w.days : [];
  div.innerHTML = `<span class="days" role="group" aria-label="요일">${DAYS.map((d, i) => `<label><input type="checkbox" value="${i}" ${days.includes(i) ? "checked" : ""}>${d}</label>`).join("")}</span>
    <input type="time" class="w-start" aria-label="시작 시각"> ~ <input type="time" class="w-end" aria-label="끝 시각">
    <button class="danger small" type="button">삭제</button>`;
  $(".w-start", div).value = w.start || "01:00";
  $(".w-end", div).value = w.end === "24:00" ? "23:59" : w.end || "06:00";
  $("button", div).addEventListener("click", () => div.remove());
  return div;
}
async function loadAdmin() {
  if (S.role !== "admin") return;
  try {
    const [s, d, a] = await Promise.all([api("/api/settings"), api("/api/drives"), api("/api/audit")]);
    S.settings = s;
    $("#drives").innerHTML = (d.drives || []).length ? d.drives.map((x) => `<button type="button" class="secondary small" data-act="drive" data-path="${esc(x.path)}">${esc(x.path)} · 남은 ${esc(x.free_gb)}GB / ${esc(x.total_gb)}GB</button>`).join("") : '<span class="muted small">자동으로 찾은 외장 장치가 없습니다. 경로를 직접 입력하세요.</span>';
    const wl = $("#windows");
    wl.innerHTML = "";
    (s.allowed_windows || []).forEach((w) => wl.appendChild(windowRow(w)));
    $("#s-tz").value = s.timezone || "";
    $("#s-allowed").value = (s.allowed_domains || []).join("\n");
    $("#s-blocked").value = (s.blocked_domains || []).join("\n");
    setVal("#s-interval", s.request_interval_sec);
    setVal("#s-maxpages", s.max_pages_per_job);
    const sr = s.search || {};
    $("#s-provider").value = sr.provider || "wikipedia";
    $("#s-wikilang").value = sr.wikipedia_lang || "";
    setVal("#s-results", sr.results_to_fetch);
    $("#s-searx").value = sr.searxng_url || "";
    $("#s-brave").value = sr.brave_api_key || "";
    const ac = s.auto_collect || {};
    $("#s-auto").checked = !!ac.enabled;
    setVal("#s-autoint", ac.interval_min);
    const on = s.online || {};
    $("#s-online-enabled").checked = !!on.enabled;
    $("#s-online-model").value = on.model || "";
    $("#s-online-key").value = on.api_key || "";
    $("#s-online-effort").value = on.effort || "medium";
    const em = s.embeddings || {};
    $("#s-emb-enabled").checked = !!em.enabled;
    $("#s-emb-model").value = em.model || "";
    const al = s.autolearn || {};
    $("#s-al-target").value = al.results_export_target || "";
    $("#s-al-outside").checked = !!al.train_outside_window;
    setVal("#s-al-maxhour", al.max_pages_per_hour);
    const fm = al.formats || [];
    $$(".al-fmt").forEach((c) => (c.checked = fm.includes(c.value)));
    $("#audit").innerHTML = (a.audit || []).map((x) => `<li><span class="grow">${esc(fmtTime(x.ts))} · <b>${esc(x.action)}</b> · ${esc(x.actor)}<br><span class="muted">${esc(x.detail)}</span></span></li>`).join("") || '<li class="muted">기록이 없습니다.</li>';
  } catch (e) { toast(e.message); }
  const st = S.status || await refreshStatus();
  const o = st && st.online;
  $("#online-status").textContent = o ? (o.available ? `지금 사용 가능 (키: ${{ settings: "설정", env: "환경변수", profile: "로그인 프로필" }[o.key_source] || "확인됨"}).` : `지금은 쓸 수 없음: ${o.reason || ""}`) : "";
}
$("#add-window").addEventListener("click", () => $("#windows").appendChild(windowRow({ days: [0, 1, 2, 3, 4, 5, 6], start: "01:00", end: "06:00" })));
$("#save-settings").addEventListener("click", async () => {
  const lines = (s) => s.split("\n").map((x) => x.trim()).filter(Boolean);
  const windows = $$("#windows .window-row").map((r) => {
    const end = $(".w-end", r).value;
    return { days: $$(".days input:checked", r).map((x) => Number(x.value)), start: $(".w-start", r).value, end: end === "23:59" ? "24:00" : end };
  }).filter((w) => w.days.length && w.start && w.end);
  const formats = $$(".al-fmt:checked").map((c) => c.value);
  if (!formats.length) { $("#settings-msg").textContent = "결과 문서 형식을 하나 이상 고르세요."; return; }
  const body = {
    allowed_windows: windows, timezone: $("#s-tz").value.trim(),
    allowed_domains: lines($("#s-allowed").value), blocked_domains: lines($("#s-blocked").value),
    request_interval_sec: Number($("#s-interval").value), max_pages_per_job: Number($("#s-maxpages").value),
    search: { provider: $("#s-provider").value, wikipedia_lang: $("#s-wikilang").value.trim(), results_to_fetch: Number($("#s-results").value),
      searxng_url: $("#s-searx").value.trim(), brave_api_key: $("#s-brave").value },
    auto_collect: { enabled: $("#s-auto").checked, interval_min: Number($("#s-autoint").value) },
    online: { enabled: $("#s-online-enabled").checked, model: $("#s-online-model").value.trim(), api_key: $("#s-online-key").value.trim(),
      effort: $("#s-online-effort").value },
    embeddings: { enabled: $("#s-emb-enabled").checked, model: $("#s-emb-model").value.trim() },
    autolearn: { results_export_target: $("#s-al-target").value.trim(), train_outside_window: $("#s-al-outside").checked,
      max_pages_per_hour: Number($("#s-al-maxhour").value) || 1, formats },
  };
  try {
    await post("/api/settings", body);
    $("#settings-msg").textContent = "저장했습니다.";
    S.settings = null; S.defaultsApplied = false;
    loadAdmin();
    refreshStatus();
  } catch (err) { $("#settings-msg").textContent = err.message; }
});
$("#export-btn").addEventListener("click", async () => {
  const target = $("#target").value.trim();
  if (!target) return toast("저장 위치를 입력하세요.");
  const include = $$(".inc:checked").map((x) => x.value);
  if (!confirm(`${target} 에 내보낼까요?`)) return;
  try { const r = await post("/api/export", { target, include }); jobToast(`내보내기 작업 #${r.job_id} 시작.`); } catch (err) { toast(err.message, 6000); }
});

// ---------------- 공용 클릭 처리 (data-act) ----------------
const ACTIONS = {
  "view": (el) => openViewer(Number(el.dataset.doc), S.tab === "search" ? SS.q : ""),
  "viewer-close": () => closeViewer(),
  "cite": (el, e) => {
    e.preventDefault();
    const t = $("#ans-src-" + Number(el.dataset.n));
    if (!t) return;
    t.scrollIntoView({ behavior: "smooth", block: "center" });
    t.classList.remove("flash"); void t.offsetWidth; t.classList.add("flash");
  },
  "del-watch": async (el) => { await api("/api/watch/" + Number(el.dataset.id), { method: "DELETE" }).catch((x) => toast(x.message)); loadWatch(); },
  "del-doc": async (el) => {
    if (!confirm("이 문서를 삭제할까요?")) return;
    await api("/api/documents/" + Number(el.dataset.id), { method: "DELETE" }).catch((x) => toast(x.message));
    loadDocs(false);
  },
  "cancel-job": async (el) => { await post(`/api/jobs/${Number(el.dataset.id)}/cancel`).catch((x) => toast(x.message)); loadJobs(); },
  "get-model": (el) => downloadModelToDevice(Number(el.dataset.id), el),
  "ask-with": (el) => {
    SS.engine = "lora"; store("lb_engine", "lora");
    switchTab("search");
    applyEngineUI();
    $("#ask-model").value = String(Number(el.dataset.id));
    $("#q").focus();
  },
  "drive": (el) => { $("#target").value = el.dataset.path; },
  "dl-zip": (el) => downloadAuth(resultPath(el.dataset.name), el.dataset.name + ".zip", el),
  "dl-file": (el) => downloadAuth(resultPath(el.dataset.name, el.dataset.path), el.dataset.path.split("/").pop() || "file", el),
  "notif-read": async (el) => { await post("/api/notifications/read", { ids: [Number(el.dataset.id)] }).catch((x) => toast(x.message)); loadNotifPanel(); },
  "notif-result": (el) => openResult(el.dataset.name),
};
document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-act]");
  if (!el) return;
  const fn = ACTIONS[el.dataset.act];
  if (fn) fn(el, e);
});

// ---------------- 시작 ----------------
async function boot() {
  updateNet();
  if ("serviceWorker" in navigator && window.isSecureContext) navigator.serviceWorker.register("/sw.js").catch(() => {});
  applyEngineUI();
  if (!S.token) return showLogin();
  try {
    const st = await api("/api/status");
    S.status = st;
    S.role = st.role || S.role;
    store("lb_role", S.role);
    showApp();
  } catch (e) {
    if (!S.token) return; // 401 -> 이미 로그인 화면
    if (e.network || !navigator.onLine) { S.offline = true; showApp(); }
    else showLogin(e.message);
  }
}
boot();
