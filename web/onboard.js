// LearnBox 온보드 엔진: 이 기기(휴대폰/PC 브라우저) 안에서 인터넷 없이 검색하고 소형 AI 모델을 실행한다.
// - 기기 등급(저사양/중급/고성능)에 따라 사본 크기, 생성 길이, 모델 형식(int8/fp32), 스레드 수를 정한다.
// - 자료 사본(조각 단위)을 IndexedDB("learnbox" v2, 저장소 "kv")에 저장하고 BM25 색인(한글 두 글자 묶음 +
//   영문/숫자 단어)으로 검색한다. "구문"은 붙어 있는 그대로 걸러 내고 -단어는 그 단어가 든 문서를 뺀다.
// - 소형 AI 모델(ONNX)을 onnxruntime-web 1.18.0(jsDelivr)으로 실행한다. 처음 한 번은 실행 엔진을 받아야 한다.
// 모듈/빌드 없이 <script src="/onboard.js">로 읽는 일반 스크립트이며 window.Onboard 를 만든다.
// Node에서도 읽을 수 있다(테스트용): window/indexedDB/navigator 는 쓸 때만 찾는다.
(function (root) {
  "use strict";

  const ORT_VERSION = "1.18.0";
  const ORT_DIST = "https://cdn.jsdelivr.net/npm/onnxruntime-web@" + ORT_VERSION + "/dist/";
  const ORT_URL = ORT_DIST + "ort.min.js";
  const DB_NAME = "learnbox";
  const DB_VERSION = 2;
  const STORE = "kv";
  const K_SNAPSHOT = "snapshot";
  const K_SNAPSHOT_META = "snapshot_meta";
  const K_MODEL = "model";
  const K_MODEL_META = "model_meta";
  const K_LEGACY_MODEL = "local_model"; // v1 app.js 가 쓰던 키
  const INDEX_VERSION = 1;
  const K1 = 1.2;
  const B = 0.75;
  const SNIPPET_CHARS = 200;
  const CANDIDATE_CAP = 5000;
  const MAX_QUERY_CHARS = 512;
  const MAX_TERMS = 20;
  const MAX_PHRASES = 10;
  const MAX_EXCLUDE = 10;
  const MAX_WORD = 40;
  const PAD = 0, UNK = 1, BOS = 2, EOS = 3, N_SPECIAL = 4;

  const TIER_LIMITS = {
    low: { snapshotChunks: 1500, maxNewTokens: 60, preferInt8: true, ortThreads: 1, maxModelMB: 8 },
    mid: { snapshotChunks: 6000, maxNewTokens: 120, preferInt8: true, ortThreads: 2, maxModelMB: 40 },
    high: { snapshotChunks: 30000, maxNewTokens: 200, preferInt8: false, ortThreads: 4, maxModelMB: 400 },
  };
  const TIER_NAMES = { low: "저사양", mid: "중급", high: "고성능" };

  const SOURCE_ALIASES = { web: "link", "웹": "link", "검색수집": "search", "검색": "search", "파일": "file",
    "자율": "auto", all: null, "전체": null };

  // 서버(search.py)와 같은 규칙
  const OPERATORS = new Set(["AND", "OR", "NOT", "NEAR"]);
  const EDGE_STRIP = "\"'`*^:()[]{}<>,;.!?~|\\/=“”‘’「」『』《》〈〉【】·…- \t";
  const INNER_SPLIT = /[()\[\]{}*^"“”]+/g;
  const PHRASE_RE = /(-?)"([^"]*)"/g;
  const KO_SUFFIXES = [
    "으로부터", "에서부터", "이라는", "이라고", "에게서", "으로써", "으로서", "에서는", "에서도", "이란", "라는",
    "라고", "에서", "에게", "한테", "으로", "부터", "까지", "처럼", "보다", "이나", "이랑", "하고", "과의", "와의",
    "에는", "에도", "로서", "로써", "입니다", "이다", "이에요", "예요", "인가요", "인가", "인지", "이란?", "란",
    "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만", "나", "랑", "요",
  ].sort((a, b) => b.length - a.length);
  const STOPWORDS = new Set([
    "무엇", "무엇인가", "무엇인가요", "무엇인지", "무엇이", "뭐", "뭐야", "뭔가요", "뭔지", "뭘", "어떻게", "어떤",
    "왜", "언제", "어디", "어디서", "누구", "누가", "얼마나", "무슨", "대해", "대해서", "대한", "관해", "관해서",
    "관한", "설명", "설명해", "설명해줘", "설명해주세요", "알려줘", "알려주세요", "알려", "주세요", "해줘", "있나요",
    "있는", "하는", "있다", "이다", "합니다", "하나요", "인가요", "그리고", "또는", "그", "이", "저", "것",
    "what", "is", "are", "was", "were", "the", "a", "an", "of", "to", "in", "how", "why", "who", "when",
    "where", "which", "does", "do", "did", "and", "or", "for", "on", "with", "about", "please", "explain",
    "tell", "me", "can", "you", "i", "it", "this", "that", "be",
  ]);

  let UNI_WORD = null; // 라틴 확장/키릴 등 ASCII 밖 글자 (오래된 브라우저면 없음)
  try { UNI_WORD = new RegExp("[\\p{L}\\p{N}\\p{M}]", "u"); } catch (e) { UNI_WORD = null; }

  // ------------------------------------------------------------------------------------------------
  // 작은 도우미
  // ------------------------------------------------------------------------------------------------
  function nav() { return (root && root.navigator) || (typeof navigator !== "undefined" ? navigator : null); }
  function idbFactory() {
    if (root && root.indexedDB) return root.indexedDB;
    if (typeof indexedDB !== "undefined") return indexedDB; // eslint-disable-line no-undef
    return null;
  }
  function nowIso() { return new Date().toISOString(); }
  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
  function escRe(s) { return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&"); }
  function collapse(s) { return String(s || "").replace(/\s+/g, " ").trim(); }
  function stripMarks(s) { return String(s == null ? "" : s).replace(/[\u0002\u0003]/g, ""); }
  function hasCase(s) { return s.toLowerCase() !== s.toUpperCase(); }
  function uniqCI(items) {
    const seen = new Set(), out = [];
    for (const x of items) {
      if (!x) continue;
      const k = x.toLowerCase();
      if (!seen.has(k)) { seen.add(k); out.push(x); }
    }
    return out;
  }
  function stripEdges(t) {
    let a = 0, b = t.length;
    while (a < b && EDGE_STRIP.indexOf(t[a]) !== -1) a++;
    while (b > a && EDGE_STRIP.indexOf(t[b - 1]) !== -1) b--;
    return t.slice(a, b);
  }
  function round(x, d) { const m = Math.pow(10, d); return Math.round(x * m) / m; }

  // ------------------------------------------------------------------------------------------------
  // 토큰: 한중일 글자는 두 글자 묶음(bigram, 한 글자짜리 덩어리는 그 글자 하나), 그 밖의 글자/숫자는 소문자 단어
  // ------------------------------------------------------------------------------------------------
  function isCJK(c) {
    return (c >= 0xAC00 && c <= 0xD7A3) || (c >= 0x1100 && c <= 0x11FF) || (c >= 0x3130 && c <= 0x318F) ||
      (c >= 0xA960 && c <= 0xA97F) || (c >= 0xD7B0 && c <= 0xD7FF) || (c >= 0x3040 && c <= 0x30FF) ||
      (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0xF900 && c <= 0xFAFF);
  }
  const ASCII_WORD = new Uint8Array(128);
  for (let c = 48; c <= 57; c++) ASCII_WORD[c] = 1;
  for (let c = 65; c <= 90; c++) ASCII_WORD[c] = 1;
  for (let c = 97; c <= 122; c++) ASCII_WORD[c] = 1;
  function isWord(c, s, i) {
    if (c < 128) return ASCII_WORD[c] === 1;
    if (c < 0xC0 || (c >= 0xD800 && c <= 0xDFFF) || isCJK(c)) return false;
    if (UNI_WORD) return UNI_WORD.test(s[i]);
    return c <= 0x24F && c !== 0xD7 && c !== 0xF7;
  }
  function forEachToken(text, cb) {
    const s = text == null ? "" : String(text);
    const n = s.length;
    let i = 0;
    while (i < n) {
      const c = s.charCodeAt(i);
      if (isCJK(c)) {
        let j = i + 1;
        while (j < n && isCJK(s.charCodeAt(j))) j++;
        if (j - i === 1) cb(s[i]);
        else for (let k = i; k < j - 1; k++) cb(s.slice(k, k + 2));
        i = j;
      } else if (isWord(c, s, i)) {
        let j = i + 1;
        while (j < n && isWord(s.charCodeAt(j), s, j)) j++;
        let w = s.slice(i, j).toLowerCase();
        if (w.length > MAX_WORD) w = w.slice(0, MAX_WORD);
        cb(w);
        i = j;
      } else i++;
    }
  }
  function tokenize(text) {
    const out = [];
    forEachToken(text, (t) => out.push(t));
    return out;
  }
  // 한 글자짜리 한중일 덩어리만 있는 단어 ("책", "수 있") 는 색인으로 다 찾을 수 없다 -> 직접 훑는다
  function weakTerm(term) {
    let hasLong = false, any = false;
    const s = String(term);
    let i = 0;
    while (i < s.length) {
      const c = s.charCodeAt(i);
      if (isCJK(c)) {
        let j = i + 1;
        while (j < s.length && isCJK(s.charCodeAt(j))) j++;
        any = true;
        if (j - i >= 2) hasLong = true;
        i = j;
      } else if (isWord(c, s, i)) { any = true; hasLong = true; i++; } else i++;
    }
    return any && !hasLong;
  }

  // ------------------------------------------------------------------------------------------------
  // BM25 색인 (조각 = 문서 단위). 저장 형식은 작게: 용어는 줄바꿈으로 이은 문자열 하나, 게시 목록은 형식 배열.
  //   {v, n, avgdl, terms: "t0\nt1\n…", off: Uint32Array(T+1), docs: Uint16Array|Uint32Array(P),
  //    tfs: Uint8Array(P), dl: Uint32Array(n)}
  // 용어 id 의 게시 목록 = docs[off[id] .. off[id+1]) (조각 번호 오름차순), 빈도 tfs (255에서 자름).
  // ------------------------------------------------------------------------------------------------
  function chunkText(row) {
    const t = row && row.t ? String(row.t) : "";
    const x = row && row.x ? String(row.x) : "";
    return t ? t + "\n" + x : x;
  }
  function makeBuilder(chunks) {
    const n = chunks.length;
    return {
      n, chunks, ids: new Map(), df: [], dl: new Uint32Array(n), perIds: new Array(n), perTf: new Array(n), total: 0,
    };
  }
  function builderStep(bd, i) {
    const counts = new Map();
    let len = 0;
    forEachToken(chunkText(bd.chunks[i]), (t) => { counts.set(t, (counts.get(t) || 0) + 1); len++; });
    const ids = new Uint32Array(counts.size), tfs = new Uint8Array(counts.size);
    let k = 0;
    for (const [t, c] of counts) {
      let id = bd.ids.get(t);
      if (id === undefined) { id = bd.df.length; bd.ids.set(t, id); bd.df.push(0); }
      bd.df[id]++;
      ids[k] = id; tfs[k] = c > 255 ? 255 : c; k++;
    }
    bd.perIds[i] = ids; bd.perTf[i] = tfs;
    bd.dl[i] = len; bd.total += len;
  }
  function builderFinish(bd) {
    const T = bd.df.length;
    const off = new Uint32Array(T + 1);
    for (let id = 0; id < T; id++) off[id + 1] = off[id] + bd.df[id];
    const P = off[T];
    const docs = bd.n <= 0xFFFF ? new Uint16Array(P) : new Uint32Array(P);
    const tfs = new Uint8Array(P);
    const cur = off.slice(0, T);
    for (let i = 0; i < bd.n; i++) {
      const ids = bd.perIds[i], tf = bd.perTf[i];
      for (let k = 0; k < ids.length; k++) { const p = cur[ids[k]]++; docs[p] = i; tfs[p] = tf[k]; }
      bd.perIds[i] = null; bd.perTf[i] = null;
    }
    const terms = new Array(T);
    for (const [t, id] of bd.ids) terms[id] = t;
    return { v: INDEX_VERSION, n: bd.n, avgdl: bd.n ? bd.total / bd.n : 0, terms: terms.join("\n"), off, docs, tfs,
      dl: bd.dl };
  }
  function buildIndex(chunks, onProgress) {
    const bd = makeBuilder(chunks || []);
    for (let i = 0; i < bd.n; i++) {
      builderStep(bd, i);
      if (onProgress && (i & 1023) === 1023) onProgress((i + 1) / bd.n);
    }
    return builderFinish(bd);
  }
  async function buildIndexAsync(chunks, onProgress) {
    const bd = makeBuilder(chunks || []);
    let last = Date.now();
    for (let i = 0; i < bd.n; i++) {
      builderStep(bd, i);
      if (Date.now() - last > 40) { // 화면이 멈추지 않게 가끔 쉰다
        if (onProgress) { try { onProgress(0.95 * (i + 1) / bd.n, "색인 만드는 중"); } catch (e) { /* 무시 */ } }
        await sleep(0);
        last = Date.now();
      }
    }
    const idx = builderFinish(bd);
    if (onProgress) { try { onProgress(0.97, "색인 완료"); } catch (e) { /* 무시 */ } }
    return idx;
  }
  function validIndex(idx, n) {
    return !!(idx && idx.v === INDEX_VERSION && idx.n === n && typeof idx.terms === "string" && idx.off && idx.docs &&
      idx.tfs && idx.dl && idx.dl.length === n);
  }
  // 검색에 쓰도록 메모리에 펼친다 (용어 -> id Map, 문서 -> 조각 목록)
  function prepare(chunks, index) {
    chunks = chunks || [];
    if (!validIndex(index, chunks.length)) index = buildIndex(chunks);
    const map = new Map();
    if (index.terms.length) {
      const terms = index.terms.split("\n");
      for (let id = 0; id < terms.length; id++) map.set(terms[id], id);
    }
    const docChunks = new Map();
    for (let i = 0; i < chunks.length; i++) {
      const d = chunks[i].d;
      const arr = docChunks.get(d);
      if (arr) arr.push(i); else docChunks.set(d, [i]);
    }
    return { chunks, index, map, docChunks, byChunkId: null };
  }
  function ensurePrepared(p) {
    if (p && p.map && p.index && p.chunks) return p;
    if (p && Array.isArray(p.chunks)) return prepare(p.chunks, p.index);
    if (Array.isArray(p)) return prepare(p, null);
    return prepare([], null);
  }

  // ------------------------------------------------------------------------------------------------
  // 검색어 해석 (서버 search.parse_query 와 같은 규칙)
  // ------------------------------------------------------------------------------------------------
  function normQuery(q) {
    let s = q == null ? "" : String(q);
    try { s = s.normalize("NFC"); } catch (e) { /* 무시 */ }
    s = s.slice(0, MAX_QUERY_CHARS).replace(/[“”＂]/g, '"').replace(/[\u0002\u0003]/g, " ");
    return s.replace(/[\x00-\x08\x0b-\x1f\x7f]/g, " ");
  }
  function parseQuery(q) {
    const s = normQuery(q);
    const phrases = [], exclude = [], terms = [];
    const rest = s.replace(PHRASE_RE, (m, neg, body) => {
      const b = stripEdges(collapse(body.replace(INNER_SPLIT, " ")));
      if (b) (neg ? exclude : phrases).push(b.slice(0, 200));
      return " ";
    });
    for (let raw of rest.split(/\s+/)) {
      if (!raw) continue;
      const neg = raw[0] === "-" && raw.length > 1;
      if (neg) raw = raw.slice(1);
      for (const part of raw.replace(INNER_SPLIT, " ").split(/\s+/)) {
        const t = stripEdges(part);
        if (!t || OPERATORS.has(t)) continue;
        (neg ? exclude : terms).push(t.slice(0, 100));
      }
    }
    return { terms: uniqCI(terms).slice(0, MAX_TERMS), phrases: uniqCI(phrases).slice(0, MAX_PHRASES),
      exclude: uniqCI(exclude).slice(0, MAX_EXCLUDE) };
  }
  function stripSuffix(t) {
    for (const s of KO_SUFFIXES) {
      if (t.length - s.length >= 2 && t.endsWith(s)) return t.slice(0, t.length - s.length);
    }
    return t;
  }
  // 넓혀 찾기용 단어: 조사/어미를 떼고 의문사·불용어를 뺀다 (서버 relaxed_terms 와 같음)
  function relaxedTerms(pq) {
    let out = [];
    for (const t of pq.terms || []) {
      if (STOPWORDS.has(t.toLowerCase())) continue;
      const s = stripSuffix(t);
      if (STOPWORDS.has(s.toLowerCase())) continue;
      out.push(s);
    }
    const multi = out.filter((t) => t.length >= 2);
    out = multi.length ? multi : out;
    out = uniqCI(out.concat(pq.phrases || []));
    if (!out.length) out = uniqCI((pq.terms || []).concat(pq.phrases || []));
    return out.slice(0, MAX_TERMS);
  }
  // 대소문자 무시 포함 검사 (한글처럼 대소문자가 없으면 indexOf 로 빠르게)
  function matcher(needle) {
    const n = String(needle);
    if (/\s/.test(n)) { // 구문: 공백은 아무 공백 여러 개와도 맞는다
      const re = new RegExp(n.trim().split(/\s+/).map(escRe).join("\\s+"), "i");
      return (s) => re.test(s);
    }
    if (!hasCase(n)) return (s) => s.indexOf(n) !== -1;
    const re = new RegExp(escRe(n), "i");
    return (s) => re.test(s);
  }
  // 일치 횟수 세기 (cap 에서 멈춤). 정규식은 한 번만 만든다.
  function counter(needle) {
    const n = String(needle);
    if (!hasCase(n) && !/\s/.test(n)) {
      return (s, cap) => {
        let c = 0, i = s.indexOf(n);
        while (i !== -1 && c < cap) { c++; i = s.indexOf(n, i + n.length); }
        return c;
      };
    }
    const re = new RegExp(n.trim().split(/\s+/).map(escRe).join("\\s+"), "gi");
    return (s, cap) => {
      let c = 0;
      re.lastIndex = 0;
      while (c < cap && re.exec(s)) c++;
      return c;
    };
  }
  function needleRegex(needles) {
    const ns = uniqCI(needles.map((x) => String(x || "").trim()).filter(Boolean)).sort((a, b) => b.length - a.length);
    if (!ns.length) return null;
    try { return new RegExp(ns.map((n) => n.split(/\s+/).map(escRe).join("\\s+")).join("|"), "gi"); } catch (e) { return null; }
  }

  // ------------------------------------------------------------------------------------------------
  // 미리보기 글: 첫 일치 주변 약 200자, 일치 부분은 \u0002 … \u0003 으로 감싼다
  // ------------------------------------------------------------------------------------------------
  function makeSnippet(text, re, width) {
    width = width || SNIPPET_CHARS;
    text = stripMarks(text);
    const len = text.length;
    let first = -1;
    if (re) { re.lastIndex = 0; const m = re.exec(text); if (m) first = m.index; }
    let s = first < 0 ? 0 : Math.max(0, first - Math.floor(width / 4));
    let e = Math.min(len, s + width);
    s = Math.max(0, Math.min(s, e - width));
    if (s > 0 && s < len && text.charCodeAt(s) >= 0xDC00 && text.charCodeAt(s) <= 0xDFFF) s++;
    if (e < len && e > 0 && text.charCodeAt(e - 1) >= 0xD800 && text.charCodeAt(e - 1) <= 0xDBFF) e--;
    const spans = [];
    if (re && first >= 0) {
      re.lastIndex = s;
      let m;
      while ((m = re.exec(text)) && m.index < e) {
        if (!m[0].length) { re.lastIndex++; continue; }
        const a = Math.max(m.index, s), b = Math.min(m.index + m[0].length, e);
        if (b > a) {
          if (spans.length && a <= spans[spans.length - 1][1]) spans[spans.length - 1][1] = Math.max(spans[spans.length - 1][1], b);
          else spans.push([a, b]);
        }
      }
    }
    let out = "", pos = s;
    for (const [a, b] of spans) { out += text.slice(pos, a) + "\u0002" + text.slice(a, b) + "\u0003"; pos = b; }
    out += text.slice(pos, e);
    return (s > 0 ? "…" : "") + collapse(out) + (e < len ? "…" : "");
  }

  // ------------------------------------------------------------------------------------------------
  // BM25 검색 (k1=1.2, b=0.75). 엄격(모든 단어 포함) -> 넓혀 찾기(조사 뗀 단어 하나 이상) -> 느슨(두 글자 묶음 절반 이상).
  // "구문"은 항상 들어 있어야 하고, -제외어가 든 문서는 뺀다. 문서당 가장 잘 맞는 조각 하나.
  // ------------------------------------------------------------------------------------------------
  function normSources(sources) {
    if (sources == null) return null;
    if (typeof sources === "string") sources = sources.split(",");
    if (!Array.isArray(sources)) return null;
    const out = [];
    for (let s of sources) {
      s = String(s == null ? "" : s).trim();
      if (!s) continue;
      if (Object.prototype.hasOwnProperty.call(SOURCE_ALIASES, s)) {
        const v = SOURCE_ALIASES[s];
        if (v === null) return null;
        s = v;
      }
      if (out.indexOf(s) === -1) out.push(s);
    }
    return out.length ? out : null;
  }

  function bm25Search(snap, q, opts) {
    opts = opts || {};
    const limit = Math.max(1, Math.min(200, parseInt(opts.limit, 10) || 20));
    const offset = Math.max(0, parseInt(opts.offset, 10) || 0);
    const prep = ensurePrepared(snap);
    const empty = Object.defineProperties([], { total: { value: 0 }, mode: { value: "none" } });
    const chunks = prep.chunks, idx = prep.index, N = chunks.length;
    if (!N) return empty;
    const pq = parseQuery(q);
    if (!pq.terms.length && !pq.phrases.length) return empty;

    const strict = pq.terms.slice();
    const phrases = pq.phrases.slice();
    const relaxed = relaxedTerms(pq);
    const scoringTerms = uniqCI(relaxed.concat(phrases, strict.filter((t) => {
      const low = t.toLowerCase();
      return !STOPWORDS.has(low) && !STOPWORDS.has(stripSuffix(t).toLowerCase());
    })));

    const srcs = normSources(opts.sources);
    let allowed = null;
    if (srcs) {
      allowed = new Uint8Array(N);
      for (let i = 0; i < N; i++) if (srcs.indexOf(chunks[i].s) !== -1) allowed[i] = 1;
    }

    const scores = new Float64Array(N);
    const hits = new Uint8Array(N);
    const touched = [];
    const avgdl = idx.avgdl || 1;
    const add = (i, s) => {
      if (scores[i] === 0) touched.push(i);
      scores[i] += s;
      if (hits[i] < 255) hits[i]++;
    };
    const bm = (tf, i, idf) => idf * (tf * (K1 + 1)) / (tf + K1 * (1 - B + B * idx.dl[i] / avgdl));

    // 1) 색인으로: 두 글자 묶음 / 단어
    const qtokens = [];
    const seenTok = new Set();
    for (const t of scoringTerms) forEachToken(t, (tok) => { if (!seenTok.has(tok)) { seenTok.add(tok); qtokens.push(tok); } });
    let units = 0;
    for (const tok of qtokens) {
      if (tok.length === 1 && isCJK(tok.charCodeAt(0))) continue; // 한 글자는 아래에서 직접 훑는다
      units++;
      const id = prep.map.get(tok);
      if (id === undefined) continue;
      const a = idx.off[id], z = idx.off[id + 1];
      const df = z - a;
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
      for (let p = a; p < z; p++) {
        const i = idx.docs[p];
        if (allowed && !allowed[i]) continue;
        add(i, bm(idx.tfs[p], i, idf));
      }
    }
    // 2) 색인으로 못 찾는 단어 (한 글자 한글, 단어 일부분 등)는 조각을 직접 훑는다
    for (const t of scoringTerms) {
      let scan = weakTerm(t);
      if (!scan) {
        let known = false;
        forEachToken(t, (tok) => { if (prep.map.has(tok)) known = true; });
        scan = !known;
      }
      if (!scan) continue;
      units++;
      const count = counter(t);
      const found = [];
      for (let i = 0; i < N; i++) {
        if (allowed && !allowed[i]) continue;
        const c = count(chunkText(chunks[i]), 20);
        if (c) found.push(i, c);
      }
      const df = found.length / 2;
      if (!df) continue;
      const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
      for (let k = 0; k < found.length; k += 2) add(found[k], bm(found[k + 1], found[k], idf));
    }
    if (!touched.length) return empty;

    touched.sort((x, y) => scores[y] - scores[x] || x - y);
    const cands = touched.length > CANDIDATE_CAP ? touched.slice(0, CANDIDATE_CAP) : touched;

    const phraseM = phrases.map(matcher);
    const strictM = strict.map(matcher);
    const relaxedM = relaxed.map(matcher);
    const exclM = pq.exclude.map(matcher);
    const excludedDoc = new Map();
    const isExcluded = (d) => {
      if (!exclM.length) return false;
      if (excludedDoc.has(d)) return excludedDoc.get(d);
      let bad = false;
      for (const i of prep.docChunks.get(d) || []) {
        const hay = chunkText(chunks[i]);
        if (exclM.some((m) => m(hay))) { bad = true; break; }
      }
      excludedDoc.set(d, bad);
      return bad;
    };
    const minHits = Math.max(1, Math.ceil(units / 2));
    const tiers = { strict: [], relaxed: [], loose: [] };
    for (const i of cands) {
      const hay = chunkText(chunks[i]);
      if (!phraseM.every((m) => m(hay))) continue;
      if (isExcluded(chunks[i].d)) continue; // 제외어는 단계(엄격/넓혀) 고르기 전에 뺀다 (서버와 같음)
      if (strictM.every((m) => m(hay))) tiers.strict.push(i);
      else if (relaxedM.some((m) => m(hay))) tiers.relaxed.push(i);
      else if (hits[i] >= minHits) tiers.loose.push(i);
    }
    const mode = tiers.strict.length ? "strict" : tiers.relaxed.length ? "relaxed" : "loose";
    const picked = [];
    const seenDoc = new Set();
    for (const i of tiers[mode]) {
      const d = chunks[i].d;
      if (seenDoc.has(d)) continue;
      seenDoc.add(d);
      picked.push(i);
    }
    let re = needleRegex(strict.concat(phrases, relaxed));
    const tokRe = needleRegex(qtokens.filter((t) => t.length >= 2));
    const rows = picked.slice(offset, offset + limit).map((i) => {
      const c = chunks[i];
      const body = stripMarks(c.x);
      let r = re;
      if (r) { r.lastIndex = 0; if (!r.test(body)) r = tokRe; }
      return { doc_id: c.d, chunk_id: c.c, title: c.t || "", url: c.u || "", source: c.s || "",
        snippet: makeSnippet(body, r || re), score: round(scores[i], 4), content: body, match: "keyword" };
    });
    return Object.defineProperties(rows, { total: { value: picked.length }, mode: { value: mode } });
  }

  // ------------------------------------------------------------------------------------------------
  // 문장 나누기 / 핵심 문장 (서버 extractive_answer 와 같은 생각). 뒤돌아보기 정규식은 오래된 Safari에서
  // 스크립트 전체가 깨지므로 쓰지 않는다.
  // ------------------------------------------------------------------------------------------------
  const SENT_END = ".!?。！？";
  const KO_END = "다요죠음함임됨";
  function sentences(text) {
    const out = [];
    for (const line of String(text == null ? "" : text).split(/\n+/)) {
      let start = 0, i = 0;
      const n = line.length;
      while (i < n) {
        const ch = line[i];
        if (SENT_END.indexOf(ch) !== -1 && i + 1 < n) {
          const nx = line[i + 1];
          if (/\s/.test(nx)) {
            const s = line.slice(start, i + 1).trim();
            if (s) out.push(s);
            let j = i + 1;
            while (j < n && /\s/.test(line[j])) j++;
            start = j; i = j; continue;
          }
          if (ch === "." && i > 0 && KO_END.indexOf(line[i - 1]) !== -1) {
            const s = line.slice(start, i + 1).trim();
            if (s) out.push(s);
            start = i + 1;
          }
        }
        i++;
      }
      const s = line.slice(start).trim();
      if (s) out.push(s);
    }
    return out;
  }

  function extract(q, rows, n, lookup) {
    n = n == null ? 3 : n;
    const pq = parseQuery(q);
    const strict = pq.terms.map((t) => t.toLowerCase());
    const phrases = pq.phrases.map((p) => p.toLowerCase());
    const relaxed = pq.terms.length || pq.phrases.length ? relaxedTerms(pq).map((t) => t.toLowerCase()) : [];
    if (!(strict.length || phrases.length || relaxed.length) || !rows || !rows.length) return [];
    const words = Array.from(new Set(strict.concat(relaxed)));
    const re = needleRegex(pq.terms.concat(pq.phrases, relaxedTerms(pq)));
    const cands = [];
    const seen = new Set();
    let order = 0;
    rows.forEach((r, ri) => {
      let content = r && r.content;
      if (content == null && lookup) content = lookup(r);
      if (content == null) content = r && r.snippet;
      for (let s of sentences(stripMarks(content || ""))) {
        order++;
        const low = s.toLowerCase();
        let score = 0;
        for (const t of words) if (t && low.indexOf(t) !== -1) score += Math.min(t.length, 4);
        for (const p of phrases) if (p && low.indexOf(p) !== -1) score += 2 * Math.min(p.length, 8);
        if (score <= 0) continue;
        const key = collapse(low);
        if (seen.has(key)) continue;
        seen.add(key);
        if (s.length < 15) score *= 0.5;
        if (s.length > 400) s = stripMarks(makeSnippet(s, re, 300));
        score += 0.5 / (1 + ri);
        cands.push([score, order, { sentence: s, doc_id: r.doc_id, title: r.title || "", url: r.url || "" }]);
      }
    });
    cands.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    return cands.slice(0, Math.max(0, parseInt(n, 10) || 0)).map((c) => c[2]);
  }

  // ------------------------------------------------------------------------------------------------
  // 기기 등급
  // ------------------------------------------------------------------------------------------------
  function tierFor(d) {
    const mem = d.memoryGB, cores = d.cores;
    if ((mem != null && mem <= 3) || cores <= 4) return "low";
    if ((mem != null && mem >= 8 && cores >= 8) || (!d.mobile && d.webgpu)) return "high";
    return "mid";
  }
  function fmtGB(x) { return (Math.round(x * 10) / 10).toString(); }
  function deviceLabel(tier, kind, mem, cores, webgpu) {
    const memTxt = mem == null ? "메모리 알 수 없음" : "메모리 " + fmtGB(mem) + "GB";
    return TIER_NAMES[tier] + " " + kind + " · " + memTxt + " · " + cores + "코어" + (webgpu ? " · WebGPU" : "");
  }
  async function probeWebGPU(n) {
    try {
      if (!n || !n.gpu || typeof n.gpu.requestAdapter !== "function") return false;
      const adapter = await Promise.race([n.gpu.requestAdapter(), sleep(1500).then(() => null)]);
      return !!adapter;
    } catch (e) { return false; }
  }
  async function detect() {
    const n = nav() || {};
    const ua = String(n.userAgent || "");
    const uad = n.userAgentData;
    const ipadOS = /Macintosh/.test(ua) && (n.maxTouchPoints || 0) > 1;
    const tablet = /iPad|Tablet/i.test(ua) || ipadOS || (/Android/i.test(ua) && !/Mobile/i.test(ua));
    let mobile = /Android|iPhone|iPad|iPod|Mobile|Windows Phone/i.test(ua) || ipadOS;
    if (uad && typeof uad.mobile === "boolean") mobile = uad.mobile || tablet;
    const mem = typeof n.deviceMemory === "number" && n.deviceMemory > 0 ? n.deviceMemory : null;
    const cores = typeof n.hardwareConcurrency === "number" && n.hardwareConcurrency > 0 ? n.hardwareConcurrency : 2;
    const webgpu = await probeWebGPU(n);
    let quota = null, usage = null;
    try {
      if (n.storage && typeof n.storage.estimate === "function") {
        const est = await n.storage.estimate();
        if (est && typeof est.quota === "number") quota = Math.round(est.quota / 1048576);
        if (est && typeof est.usage === "number") usage = Math.round(est.usage / 1048576);
      }
    } catch (e) { /* 무시 */ }
    const tier = tierFor({ memoryGB: mem, cores, mobile, webgpu });
    const kind = !mobile ? "PC" : tablet ? "태블릿" : "휴대폰";
    return {
      tier, mobile, memoryGB: mem, cores, webgpu, storageQuotaMB: quota, storageUsageMB: usage,
      crossOriginIsolated: !!(root && root.crossOriginIsolated),
      label: deviceLabel(tier, kind, mem, cores, webgpu),
      limits: Object.assign({}, TIER_LIMITS[tier]),
    };
  }

  // ------------------------------------------------------------------------------------------------
  // IndexedDB 키-값 저장소 ("learnbox" v2, "kv"). v1 데이터베이스(같은 "kv" 저장소)는 그대로 올린다.
  // ------------------------------------------------------------------------------------------------
  const kv = {
    _p: null,
    open() {
      if (this._p) return this._p;
      const factory = idbFactory();
      if (!factory) return Promise.reject(new Error("이 브라우저는 기기 저장(IndexedDB)을 지원하지 않습니다."));
      const self = this;
      const attempt = (version) => new Promise((resolve, reject) => {
        let done = false, timer = null;
        let req;
        try { req = version ? factory.open(DB_NAME, version) : factory.open(DB_NAME); } catch (e) { reject(e); return; }
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
        };
        req.onsuccess = () => {
          const db = req.result;
          if (done) { try { db.close(); } catch (e) { /* 무시 */ } return; }
          done = true; clearTimeout(timer);
          db.onversionchange = () => { try { db.close(); } catch (e) { /* 무시 */ } self._p = null; };
          db.onclose = () => { self._p = null; };
          if (!db.objectStoreNames.contains(STORE)) {
            try { db.close(); } catch (e) { /* 무시 */ }
            reject(new Error("기기 저장소 형식이 맞지 않습니다."));
            return;
          }
          resolve(db);
        };
        req.onerror = () => { if (done) return; done = true; clearTimeout(timer); reject(req.error || new Error("기기 저장소를 열 수 없습니다.")); };
        req.onblocked = () => {
          clearTimeout(timer);
          timer = setTimeout(() => {
            if (done) return;
            done = true;
            reject(new Error("다른 탭에 열린 LearnBox 때문에 기기 저장소를 열 수 없습니다. 다른 LearnBox 탭을 닫고 다시 시도하세요."));
          }, 4000);
        };
      });
      this._p = attempt(DB_VERSION).catch((e) => {
        // 더 새 버전으로 이미 열린 적이 있으면 버전 없이 연다
        if (e && e.name === "VersionError") return attempt(0);
        throw e;
      });
      this._p.catch(() => { self._p = null; });
      return this._p;
    },
    async _tx(mode, fn) {
      const db = await this.open();
      return new Promise((resolve, reject) => {
        let t, req;
        try { t = db.transaction(STORE, mode); } catch (e) { this._p = null; reject(e); return; }
        const st = t.objectStore(STORE);
        try { req = fn(st); } catch (e) { try { t.abort(); } catch (_) { /* 무시 */ } reject(quotaError(e)); return; }
        t.oncomplete = () => resolve(req ? req.result : undefined);
        t.onerror = () => reject(quotaError(t.error));
        t.onabort = () => reject(quotaError(t.error) || new Error("기기 저장이 취소되었습니다."));
      });
    },
    get(k) { return this._tx("readonly", (st) => st.get(k)); },
    set(k, v) { return this._tx("readwrite", (st) => { st.put(v, k); return null; }); },
    del(k) { return this._tx("readwrite", (st) => { st.delete(k); return null; }); },
    // 여러 키를 한 번에 (한 트랜잭션). 값이 undefined 면 지운다.
    setMany(obj) {
      return this._tx("readwrite", (st) => {
        for (const k of Object.keys(obj)) { if (obj[k] === undefined) st.delete(k); else st.put(obj[k], k); }
        return null;
      });
    },
    close() {
      const p = this._p;
      this._p = null;
      if (p) return p.then((db) => db.close(), () => {});
      return Promise.resolve();
    },
  };
  function quotaError(err) {
    if (err && (err.name === "QuotaExceededError" || /quota/i.test(String(err.message || "")))) {
      return new Error("기기 저장 공간이 부족합니다. 더 작은 사본이나 모델을 받거나 저장 공간을 비우세요.");
    }
    return err;
  }

  // ------------------------------------------------------------------------------------------------
  // 자료 사본 저장 / 불러오기
  // ------------------------------------------------------------------------------------------------
  let snapCache = null;      // 펼친 사본 (prepare 결과 + saved_at)
  let snapLoading = null;
  let snapGen = 0;           // 저장/삭제할 때마다 올린다 (늦게 끝난 불러오기가 새 사본을 덮지 않게)

  function normalizeChunk(r, k) {
    let x = stripMarks(r && (r.x != null ? r.x : r.content != null ? r.content : r.text));
    let t = stripMarks(r && (r.t != null ? r.t : r.title));
    if (/[ᄀ-ᇿ]/.test(x) || /[ᄀ-ᇿ]/.test(t)) { // 풀어 쓴 한글(NFD)은 모아 쓴다
      try { x = x.normalize("NFC"); t = t.normalize("NFC"); } catch (e) { /* 무시 */ }
    }
    return {
      c: r && (r.c != null ? r.c : r.chunk_id != null ? r.chunk_id : k),
      d: r && (r.d != null ? r.d : r.doc_id != null ? r.doc_id : k),
      t, x,
      u: String((r && (r.u != null ? r.u : r.url)) || ""),
      s: String((r && (r.s != null ? r.s : r.source)) || ""),
    };
  }
  function splitText(text, size) { // db.chunk_text 와 같은 방식 (옛 v1 사본 변환용)
    const out = [];
    let cur = "";
    for (let p of String(text || "").split(/\n+/).map((x) => x.trim()).filter(Boolean)) {
      while (p.length > size) { if (cur) { out.push(cur); cur = ""; } out.push(p.slice(0, size)); p = p.slice(size); }
      if (cur && cur.length + p.length + 1 > size) { out.push(cur); cur = p; } else cur = cur ? cur + "\n" + p : p;
    }
    if (cur) out.push(cur);
    return out;
  }
  function legacyChunks(documents) {
    const out = [];
    for (const doc of documents || []) {
      const parts = splitText(doc.text, 800);
      parts.forEach((x, k) => out.push({ c: doc.id * 1000 + k, d: doc.id, t: doc.title || "", x, u: doc.url || "", s: doc.source || "link" }));
    }
    return out;
  }

  async function loadSnapshot() {
    if (snapCache) return snapCache;
    if (snapLoading) return snapLoading;
    const gen = snapGen;
    snapLoading = (async () => {
      const snap = await kv.get(K_SNAPSHOT);
      if (!snap) return null;
      let chunks, index = null, rebuilt = false;
      if (Array.isArray(snap.chunks)) { chunks = snap.chunks; index = snap.index; }
      else if (Array.isArray(snap.documents)) { chunks = legacyChunks(snap.documents); }
      else return null;
      if (!validIndex(index, chunks.length)) { index = await buildIndexAsync(chunks); rebuilt = Array.isArray(snap.chunks); }
      const prep = prepare(chunks, index);
      prep.saved_at = snap.saved_at || null;
      if (gen !== snapGen) return snapCache; // 그사이 새로 저장/삭제됨
      if (rebuilt) { // 색인 형식이 바뀌었으면 새 색인을 저장해 둔다 (실패해도 검색은 된다)
        try { await kv.set(K_SNAPSHOT, Object.assign({}, snap, { index })); } catch (e) { /* 무시 */ }
      }
      if (gen !== snapGen) return snapCache;
      snapCache = prep;
      return prep;
    })();
    try { return await snapLoading; } finally { snapLoading = null; }
  }

  // ------------------------------------------------------------------------------------------------
  // 모델 저장 / 실행 (onnxruntime-web)
  // ------------------------------------------------------------------------------------------------
  let rt = null;            // {key, session, itos, info, ctx}
  let ortPromise = null;
  let runLock = Promise.resolve();

  function loadScriptDOM(src) {
    return new Promise((resolve, reject) => {
      const doc = root && root.document;
      if (!doc || typeof doc.createElement !== "function") { reject(new Error("이 환경에서는 실행 엔진을 불러올 수 없습니다.")); return; }
      const s = doc.createElement("script");
      s.src = src;
      s.async = true;
      s.crossOrigin = "anonymous"; // 서비스 워커가 정상 응답(opaque 아님)으로 캐시하도록
      s.onload = () => resolve();
      s.onerror = () => { s.remove && s.remove(); reject(new Error("실행 엔진(onnxruntime-web)을 불러오지 못했습니다. 처음 한 번은 인터넷이 필요합니다.")); };
      (doc.head || doc.documentElement).appendChild(s);
    });
  }
  async function ensureOrt(threads) {
    if (!(root && root.ort)) {
      if (!ortPromise) ortPromise = Onboard._loadScript(Onboard.ORT_URL).catch((e) => { ortPromise = null; throw e; });
      await ortPromise;
    }
    const ort = root.ort;
    if (!ort || !ort.InferenceSession) throw new Error("실행 엔진(onnxruntime-web)을 불러오지 못했습니다.");
    if (!ensureOrt._configured) {
      try {
        if (ort.env && ort.env.wasm) {
          if (!ort.env.wasm.wasmPaths) ort.env.wasm.wasmPaths = Onboard.ORT_DIST;
          ort.env.wasm.numThreads = root.crossOriginIsolated ? Math.max(1, threads | 0) : 1;
        }
        if (ort.env) ort.env.logLevel = "error";
      } catch (e) { /* 무시 */ }
      ensureOrt._configured = true;
    }
    return ort;
  }
  function toArrayBuffer(buf) {
    if (buf instanceof ArrayBuffer) return buf;
    if (buf && ArrayBuffer.isView(buf)) return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    if (buf && typeof buf.byteLength === "number" && typeof buf.slice === "function") return buf.slice(0);
    return null;
  }
  async function loadModelRecord() {
    let rec = await kv.get(K_MODEL);
    if (rec && rec.onnx) return rec;
    const old = await kv.get(K_LEGACY_MODEL); // v1: {id, onnx, itos, info, saved_at}
    if (old && old.onnx) {
      return { meta: { id: old.id, name: (old.info && old.info.name) || "모델 #" + old.id, variant: "fp32",
        bytes: old.onnx.byteLength, info: old.info || {}, saved_at: old.saved_at || null }, onnx: old.onnx, itos: old.itos };
    }
    return null;
  }
  function modelCtx(info) {
    info = info || {};
    const arch = info.arch || {};
    const seq = parseInt(info.seq_len || arch.seq_len, 10) || 128;
    const trained = parseInt(info.train_seq_len, 10) || seq;
    return Math.max(2, Math.min(seq, trained));
  }
  function runtimeKey(meta) { return meta.id + ":" + meta.variant + ":" + (meta.saved_at || ""); }
  async function getRuntime(dev) {
    const small = await kv.get(K_MODEL_META); // 큰 모델 파일을 매번 읽지 않도록 작은 정보부터
    if (small && rt && rt.key === runtimeKey(small)) return rt;
    const rec = await loadModelRecord();
    if (!rec) throw new Error("이 기기에 받은 모델이 없습니다. [모델] 탭에서 소형 AI 모델을 먼저 받으세요.");
    const meta = rec.meta || {};
    const key = runtimeKey(meta);
    if (rt && rt.key === key) return rt;
    const ort = await ensureOrt(dev.limits.ortThreads);
    if (rt && rt.session && typeof rt.session.release === "function") { try { await rt.session.release(); } catch (e) { /* 무시 */ } }
    rt = null;
    const bytes = new Uint8Array(rec.onnx);
    let session = null, ep = "wasm";
    // WebGPU 는 fp32 모델이고 실행 엔진에 WebGPU가 들어 있을 때만 (ort.min.js 1.18 에는 없다 -> 조용히 wasm)
    if (dev.webgpu && meta.variant !== "int8") {
      try { session = await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] }); ep = "webgpu"; } catch (e) { session = null; }
    }
    if (!session) {
      try {
        session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      } catch (e) {
        throw new Error("모델을 이 기기에서 열 수 없습니다: " + ((e && e.message) || e));
      }
      ep = "wasm";
    }
    const itos = Array.isArray(rec.itos) ? rec.itos : (rec.itos && rec.itos.itos) || [];
    rt = { key, session, itos, info: meta.info || {}, ctx: modelCtx(meta.info), ep,
      inputName: (session.inputNames && session.inputNames[0]) || "input_ids",
      outputName: (session.outputNames && session.outputNames[0]) || "logits" };
    return rt;
  }
  function makeRng(seed) {
    if (seed == null) return Math.random;
    let a = (seed >>> 0) || 0x9E3779B9;
    return function () { // mulberry32
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // 상위 k개 중에서 온도를 적용해 하나를 고른다. <pad>/<unk>/<bos>(0~2)는 고르지 않는다.
  function sampleTopK(logits, k, temperature, rng) {
    const V = logits.length;
    const banned = V > N_SPECIAL + 1 ? EOS : 0; // 0..2 금지 (단어가 아주 적은 모델이면 금지 안 함)
    if (!(temperature > 0)) {
      let best = -1, bv = -Infinity;
      for (let i = banned; i < V; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
      return best < 0 ? EOS : best;
    }
    k = Math.max(1, Math.min(k | 0 || V, V - banned));
    const idx = [], val = [];
    for (let i = banned; i < V; i++) {
      const v = logits[i];
      if (idx.length < k) {
        let j = idx.length;
        idx.push(i); val.push(v);
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
        val[j] = v; idx[j] = i;
      } else if (v > val[k - 1]) {
        let j = k - 1;
        while (j > 0 && val[j - 1] < v) { val[j] = val[j - 1]; idx[j] = idx[j - 1]; j--; }
        val[j] = v; idx[j] = i;
      }
    }
    const mx = val[0];
    let sum = 0;
    const p = val.map((v) => { const e = Math.exp((v - mx) / temperature); sum += e; return e; });
    let r = (rng || Math.random)() * sum;
    for (let j = 0; j < p.length; j++) { r -= p[j]; if (r <= 0) return idx[j]; }
    return idx[idx.length - 1];
  }
  function encodePrompt(prompt, itos) {
    const stoi = new Map();
    for (let i = N_SPECIAL; i < itos.length; i++) if (!stoi.has(itos[i])) stoi.set(itos[i], i);
    const ids = [BOS];
    for (const ch of Array.from(String(prompt || ""))) ids.push(stoi.has(ch) ? stoi.get(ch) : UNK);
    return ids;
  }

  // ------------------------------------------------------------------------------------------------
  // 공개 API
  // ------------------------------------------------------------------------------------------------
  let devicePromise = null;

  const Onboard = {
    version: "2",
    ORT_VERSION, ORT_URL, ORT_DIST,
    TIER_LIMITS,
    db: kv,

    detectDevice(opts) {
      if (!devicePromise || (opts && opts.refresh)) devicePromise = detect();
      return devicePromise;
    },

    async saveSnapshot(chunks, opts) {
      opts = opts || {};
      const onProgress = typeof opts.onProgress === "function" ? opts.onProgress : null;
      const list = Array.isArray(chunks) ? chunks : chunks && Array.isArray(chunks.chunks) ? chunks.chunks : null;
      if (!list) throw new Error("저장할 자료가 없습니다.");
      const rows = list.map(normalizeChunk).filter((r) => r.x || r.t);
      const index = await buildIndexAsync(rows, onProgress);
      const docs = new Set(rows.map((r) => r.d)).size;
      const chars = rows.reduce((a, r) => a + r.x.length, 0);
      const saved_at = nowIso();
      const meta = { count: rows.length, saved_at, docs, chars,
        index_bytes: index.off.byteLength + index.docs.byteLength + index.tfs.byteLength + index.dl.byteLength + index.terms.length * 2 };
      if (onProgress) { try { onProgress(0.98, "기기에 저장하는 중"); } catch (e) { /* 무시 */ } }
      await kv.setMany({ [K_SNAPSHOT]: { v: 2, chunks: rows, index, saved_at, count: rows.length }, [K_SNAPSHOT_META]: meta });
      snapGen++;
      const prep = prepare(rows, index);
      prep.saved_at = saved_at;
      snapCache = prep;
      if (onProgress) { try { onProgress(1, "완료"); } catch (e) { /* 무시 */ } }
      return meta;
    },

    async snapshotInfo() {
      try {
        const meta = await kv.get(K_SNAPSHOT_META);
        if (meta && typeof meta.count === "number") return meta;
        const snap = await kv.get(K_SNAPSHOT); // 옛 형식
        if (!snap) return null;
        const count = Array.isArray(snap.chunks) ? snap.chunks.length : Array.isArray(snap.documents) ? snap.documents.length : 0;
        return { count, saved_at: snap.saved_at || null, legacy: !Array.isArray(snap.chunks) };
      } catch (e) {
        return null;
      }
    },

    async deleteSnapshot() {
      snapGen++;
      snapCache = null;
      await kv.setMany({ [K_SNAPSHOT]: undefined, [K_SNAPSHOT_META]: undefined });
    },

    async search(q, opts) {
      const prep = await loadSnapshot();
      if (!prep) throw new Error("이 기기에 저장된 자료가 없습니다. 먼저 [검색] 탭에서 '이 기기에 저장'으로 자료를 받으세요.");
      return bm25Search(prep, q, opts || {});
    },

    extract(q, rows, n) {
      const lookup = (r) => {
        if (!snapCache || r == null || r.chunk_id == null) return null;
        if (!snapCache.byChunkId) {
          snapCache.byChunkId = new Map();
          snapCache.chunks.forEach((c, i) => snapCache.byChunkId.set(c.c, i));
        }
        const i = snapCache.byChunkId.get(r.chunk_id);
        return i == null ? null : snapCache.chunks[i].x;
      };
      return extract(q, rows, n == null ? 3 : n, lookup);
    },

    async saveModel(meta, onnxArrayBuffer, itos, variant) {
      meta = meta || {};
      let buf = onnxArrayBuffer;
      if (buf && typeof buf.arrayBuffer === "function" && !(buf instanceof ArrayBuffer) && !ArrayBuffer.isView(buf)) buf = await buf.arrayBuffer();
      buf = toArrayBuffer(buf);
      if (!buf || !buf.byteLength) throw new Error("모델 파일이 비어 있습니다.");
      const vocab = Array.isArray(itos) ? itos : itos && Array.isArray(itos.itos) ? itos.itos : null;
      if (!vocab || vocab.length < N_SPECIAL + 1) throw new Error("모델 글자 목록(vocab)이 올바르지 않습니다.");
      const v = variant === "int8" || variant === "fp32" ? variant : "fp32";
      const dev = await Onboard.detectDevice();
      const info = meta.info || {};
      const m = { id: meta.id != null ? meta.id : null, name: meta.name || info.name || "모델 #" + meta.id, variant: v,
        bytes: buf.byteLength, info, saved_at: nowIso() };
      const mb = buf.byteLength / 1048576;
      const warning = mb > dev.limits.maxModelMB
        ? "이 기기(" + TIER_NAMES[dev.tier] + ")에 비해 큰 모델입니다 (" + mb.toFixed(1) + "MB > 권장 " + dev.limits.maxModelMB + "MB). 느리거나 메모리가 부족할 수 있습니다."
        : null;
      await kv.setMany({ [K_MODEL]: { meta: m, onnx: buf, itos: vocab }, [K_MODEL_META]: m });
      if (rt) { try { if (rt.session && rt.session.release) await rt.session.release(); } catch (e) { /* 무시 */ } rt = null; }
      return Object.assign({}, m, { warning });
    },

    async modelInfo() {
      try {
        const m = await kv.get(K_MODEL_META);
        if (m) return { id: m.id, name: m.name, variant: m.variant, bytes: m.bytes, info: m.info, saved_at: m.saved_at };
        const old = await kv.get(K_LEGACY_MODEL);
        if (old && old.onnx) {
          return { id: old.id, name: (old.info && old.info.name) || "모델 #" + old.id, variant: "fp32",
            bytes: old.onnx.byteLength, info: old.info || {}, saved_at: old.saved_at || null };
        }
        return null;
      } catch (e) {
        return null;
      }
    },

    async deleteModel() {
      if (rt) { try { if (rt.session && rt.session.release) await rt.session.release(); } catch (e) { /* 무시 */ } rt = null; }
      await kv.setMany({ [K_MODEL]: undefined, [K_MODEL_META]: undefined, [K_LEGACY_MODEL]: undefined });
    },

    async generate(prompt, opts) {
      opts = opts || {};
      const run = async () => {
        const dev = await Onboard.detectDevice();
        const r = await getRuntime(dev);
        const ort = root.ort;
        const maxNew = Math.max(1, Math.min(2000, parseInt(opts.maxNew, 10) || dev.limits.maxNewTokens));
        const temperature = opts.temperature == null ? 0.8 : Number(opts.temperature);
        const topK = opts.topK == null ? 20 : parseInt(opts.topK, 10);
        const onToken = typeof opts.onToken === "function" ? opts.onToken : null;
        const rng = makeRng(opts.seed);
        const ids = encodePrompt(prompt, r.itos);
        let text = "";
        let lastYield = Date.now();
        for (let step = 0; step < maxNew; step++) {
          if (opts.signal && opts.signal.aborted) break;
          const ctx = ids.length > r.ctx ? ids.slice(ids.length - r.ctx) : ids;
          const data = new BigInt64Array(ctx.length);
          for (let i = 0; i < ctx.length; i++) data[i] = BigInt(ctx[i]);
          const feeds = {};
          feeds[r.inputName] = new ort.Tensor("int64", data, [1, ctx.length]);
          const out = await r.session.run(feeds);
          const lg = out[r.outputName] || out[Object.keys(out)[0]];
          const dims = lg.dims || [1, ctx.length, lg.data.length / ctx.length];
          const V = dims[dims.length - 1];
          const L = lg.data.length / V;
          const next = sampleTopK(lg.data.subarray((L - 1) * V, L * V), topK, temperature, rng);
          if (typeof lg.dispose === "function") { try { lg.dispose(); } catch (e) { /* 무시 */ } }
          if (next === EOS) break;
          ids.push(next);
          if (next >= N_SPECIAL && next < r.itos.length) {
            const piece = r.itos[next];
            text += piece;
            if (onToken) { try { onToken(piece, text); } catch (e) { /* 무시 */ } }
          }
          if (Date.now() - lastYield > 30) { await sleep(0); lastYield = Date.now(); } // 화면 갱신
        }
        return text;
      };
      const p = runLock.then(run, run);
      runLock = p.then(() => {}, () => {});
      return p;
    },

    async answer(q, opts) {
      opts = opts || {};
      const limit = parseInt(opts.limit, 10) || 5;
      const rows = await Onboard.search(q, { limit, sources: opts.sources || null });
      const ex = Onboard.extract(q, rows, 3);
      let generated = null, note = null;
      if (opts.generate !== false && ex.length) {
        const info = await Onboard.modelInfo();
        if (info) {
          try {
            const best = ex[0].sentence;
            generated = await Onboard.generate(best, { maxNew: opts.maxNew, temperature: opts.temperature, topK: opts.topK,
              onToken: opts.onToken, signal: opts.signal, seed: opts.seed });
          } catch (e) {
            note = (e && e.message) || String(e);
          }
        } else note = "이 기기에 받은 모델이 없어 핵심 문장만 보여 줍니다.";
      } else if (!rows.length) note = "이 기기에 저장된 자료에서 찾지 못했습니다.";
      return { rows, extract: ex, generated, note };
    },

    supported() {
      const n = nav() || {};
      return { indexedDB: !!idbFactory(), wasm: typeof WebAssembly === "object", webgpu: !!n.gpu,
        serviceWorker: !!n.serviceWorker, crossOriginIsolated: !!(root && root.crossOriginIsolated) };
    },

    async requestPersist() {
      try {
        const n = nav();
        if (n && n.storage && typeof n.storage.persist === "function") return await n.storage.persist();
      } catch (e) { /* 무시 */ }
      return false;
    },

    // 테스트/통합용 (순수 함수와 이음새)
    _loadScript: loadScriptDOM,
    _tokenize: tokenize,
    _buildIndex: buildIndex,
    _buildIndexAsync: buildIndexAsync,
    _prepare: prepare,
    _bm25Search: bm25Search,
    _parseQuery: parseQuery,
    _relaxedTerms: relaxedTerms,
    _snippet: (text, q, width) => {
      const pq = parseQuery(q);
      return makeSnippet(text, needleRegex(pq.terms.concat(pq.phrases, relaxedTerms(pq))), width);
    },
    _sentences: sentences,
    _extract: extract,
    _tierFor: tierFor,
    _deviceLabel: deviceLabel,
    _sampleTopK: sampleTopK,
    _encodePrompt: encodePrompt,
    _modelCtx: modelCtx,
    _reset() { snapGen++; snapCache = null; snapLoading = null; rt = null; devicePromise = null; ortPromise = null; ensureOrt._configured = false; },
  };

  if (root) root.Onboard = Onboard;
  if (typeof module === "object" && module && module.exports) module.exports = Onboard;
})(typeof window !== "undefined" ? window : typeof self !== "undefined" ? self : typeof globalThis !== "undefined" ? globalThis : this);
