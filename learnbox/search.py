"""학습 파일 안에서 찾기 (핵심 기능).

- 키워드 검색: SQLite FTS5 trigram 색인. 3글자 이상 단어/구문은 MATCH, 2글자 이하는 LIKE.
  "구문" 은 붙어 있는 그대로, -단어 는 그 단어가 들어 있는 문서를 뺀다. 어떤 입력에도 예외를 내지 않는다.
- 의미 검색: 임베딩(기본 intfloat/multilingual-e5-small, CPU)으로 조각을 벡터화해 DB에 float32로 저장하고
  질문 벡터와 코사인 유사도로 전부 비교한다(brute force).
- 통합 검색(hybrid): 키워드 순위와 의미 순위를 Reciprocal Rank Fusion(k=60)으로 합친다. 문서당 가장 잘 맞는
  조각 하나만 보여 준다. 엄격(AND) 검색 결과가 없으면 조사를 떼고 OR로 넓혀 다시 찾는다(relaxed).
- 핵심 문장 뽑기(extractive_answer), 문서 보기(doc_view: 일치 위치), 답변 엔진용 문맥 고르기(retrieve).

torch / transformers 는 Embedder 안에서만 import 한다 (없어도 키워드 검색은 동작).

테스트/통합용 이음새(seam): 실제 모델을 받지 않도록 가짜 임베더를 꽂을 수 있다.
  - `set_embedder(obj)` (= 모듈 변수 EMBEDDER_OVERRIDE): get_embedder()가 이 객체를 돌려준다. None으로 해제.
  - 또는 index_embeddings / semantic_search / search / retrieve 에 키워드 인자 `embedder=obj` 를 넘긴다.
  가짜 임베더는 `model_name: str` 속성과 `encode(texts, is_query) -> list[list[float]]`(L2 정규화) 만 있으면 된다.
"""
from __future__ import annotations

import importlib.util
import math
import re
import threading
import time
import unicodedata
from array import array
from collections import OrderedDict

from .db import HL_END, HL_START

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"
RRF_K = 60
CANDIDATE_CAP = 3000          # 키워드 후보 조각 상한 (문서 단위 중복 제거/페이지 나눔 전)
SNIPPET_CHARS = 200
MAX_QUERY_CHARS = 512
MAX_TERMS = 20
MAX_PHRASES = 10
MAX_EXCLUDE = 10
MAX_MATCHES = 500
SOURCE_ALIASES = {"web": "link", "웹": "link", "검색수집": "search", "검색": "search", "파일": "file",
                  "자율": "auto", "all": None, "전체": None}

# FTS 연산자 단어 (따옴표로 감싸므로 오류는 안 나지만, 사용자가 연산자로 쓴 것이므로 버린다)
_OPERATORS = {"AND", "OR", "NOT", "NEAR"}
# 단어 양 끝에서 떼는 기호 (C++, C#, 100% 처럼 뜻 있는 기호는 남긴다)
_EDGE_STRIP = "\"'`*^:()[]{}<>,;.!?~|\\/=“”‘’「」『』《》〈〉【】·…-"
# 단어 가운데 있어도 공백으로 바꾸는 FTS 문법 기호
_INNER_SPLIT = re.compile(r"[()\[\]{}*^\"“”]+")
_PHRASE_RE = re.compile(r'(-?)"([^"]*)"')
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# 넓혀 찾기(relaxed)에서 단어 끝에서 떼는 조사/어미 (긴 것부터). 떼고 남은 길이가 2글자 이상일 때만 뗀다.
_KO_SUFFIXES = sorted({
    "으로부터", "에서부터", "이라는", "이라고", "에게서", "으로써", "으로서", "에서는", "에서도", "이란", "라는",
    "라고", "에서", "에게", "한테", "으로", "부터", "까지", "처럼", "보다", "이나", "이랑", "하고", "과의", "와의",
    "에는", "에도", "로서", "로써", "입니다", "이다", "이에요", "예요", "인가요", "인가", "인지", "이란?", "란",
    "은", "는", "이", "가", "을", "를", "의", "에", "로", "와", "과", "도", "만", "나", "랑", "요",
}, key=len, reverse=True)
_STOPWORDS = {
    "무엇", "무엇인가", "무엇인가요", "무엇인지", "무엇이", "뭐", "뭐야", "뭔가요", "뭔지", "뭘", "어떻게", "어떤",
    "왜", "언제", "어디", "어디서", "누구", "누가", "얼마나", "무슨", "대해", "대해서", "대한", "관해", "관해서",
    "관한", "설명", "설명해", "설명해줘", "설명해주세요", "알려줘", "알려주세요", "알려", "주세요", "해줘", "있나요",
    "있는", "하는", "있다", "이다", "합니다", "하나요", "인가요", "그리고", "또는", "그", "이", "저", "것",
    "what", "is", "are", "was", "were", "the", "a", "an", "of", "to", "in", "how", "why", "who", "when",
    "where", "which", "does", "do", "did", "and", "or", "for", "on", "with", "about", "please", "explain",
    "tell", "me", "can", "you", "i", "it", "this", "that", "be",
}


def _noop_progress(p: float, m: str = "") -> None:
    return None


def _noop_cancel() -> None:
    return None


# ---------------------------------------------------------------------------------------------------------------
# 질의 해석
# ---------------------------------------------------------------------------------------------------------------
def _clean_term(t: str) -> str:
    t = t.strip(_EDGE_STRIP + " \t")
    return t


def _norm_query(q) -> str:
    q = "" if q is None else str(q)
    q = unicodedata.normalize("NFC", q)[:MAX_QUERY_CHARS]
    q = q.replace("“", '"').replace("”", '"').replace("＂", '"')
    q = q.replace(HL_START, " ").replace(HL_END, " ")
    return _CTRL.sub(" ", q)


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        k = x.lower()
        if x and k not in seen:
            seen.add(k)
            out.append(x)
    return out


def parse_query(q: str) -> dict:
    """검색어 해석. "따옴표 구문", -제외어, 나머지는 모두 들어 있어야 하는 단어(AND).

    반환 {"terms": [...], "phrases": [...], "exclude": [...]} — FTS 연산자/기호는 걸러진다.
    """
    q = _norm_query(q)
    phrases: list[str] = []
    exclude: list[str] = []

    def take_phrase(m: re.Match) -> str:
        body = _WS.sub(" ", _INNER_SPLIT.sub(" ", m.group(2))).strip(_EDGE_STRIP + " ")
        if body:
            (exclude if m.group(1) else phrases).append(body[:200])
        return " "

    rest = _PHRASE_RE.sub(take_phrase, q)
    terms: list[str] = []
    for raw in rest.split():
        neg = raw.startswith("-") and len(raw) > 1
        if neg:
            raw = raw[1:]
        for part in _INNER_SPLIT.sub(" ", raw).split():
            t = _clean_term(part)
            if not t or t in _OPERATORS:
                continue
            (exclude if neg else terms).append(t[:100])
    phrases = _dedupe(phrases)[:MAX_PHRASES]
    terms = _dedupe(terms)[:MAX_TERMS]
    exclude = _dedupe(exclude)[:MAX_EXCLUDE]
    return {"terms": terms, "phrases": phrases, "exclude": exclude}


def _strip_suffix(t: str) -> str:
    for s in _KO_SUFFIXES:
        if t.endswith(s) and len(t) - len(s) >= 2:
            return t[: -len(s)]
    return t


def relaxed_terms(pq: dict) -> list[str]:
    """넓혀 찾기용 단어: 조사/어미를 떼고 의문사·불용어를 뺀다. 구문은 그대로 둔다."""
    out: list[str] = []
    for t in pq.get("terms", []):
        low = t.lower()
        if low in _STOPWORDS:
            continue
        s = _strip_suffix(t)
        if s.lower() in _STOPWORDS:
            continue
        out.append(s)
    multi = [t for t in out if len(t) >= 2]
    out = multi or out  # 한 글자 단어는 다른 단어가 없을 때만
    out = _dedupe(out + list(pq.get("phrases", [])))
    if not out:
        out = _dedupe(list(pq.get("terms", [])) + list(pq.get("phrases", [])))
    return out[:MAX_TERMS]


def _needles(pq: dict, relaxed: bool = False) -> list[str]:
    n = list(pq.get("terms", [])) + list(pq.get("phrases", []))
    if relaxed:
        n += relaxed_terms(pq)
    return _dedupe([x for x in n if x])


def _fts_quote(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def _like_pat(s: str) -> str:
    return "%" + s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _norm_sources(sources) -> list[str] | None:
    if sources is None:
        return None
    if isinstance(sources, str):
        sources = sources.split(",")
    out = []
    for s in sources:
        s = str(s or "").strip()
        if not s:
            continue
        s = SOURCE_ALIASES.get(s, s) if s in SOURCE_ALIASES else s
        if s is None:  # "전체" -> 필터 없음
            return None
        out.append(s)
    return _dedupe(out) or None


def _filters(sources, doc_id, exclude) -> tuple[str, list]:
    """공통 WHERE 조각 (chunks + documents d 조인 기준)."""
    sql, params = "", []
    srcs = _norm_sources(sources)
    if srcs:
        sql += f" AND d.source IN ({','.join('?' * len(srcs))})"
        params += srcs
    if doc_id is not None:
        sql += " AND chunks.doc_id = ?"
        params.append(int(doc_id))
    for e in exclude:  # 문서 단위 제외: 제목이나 본문 어디에든 있으면 뺀다
        sql += " AND NOT (d.title LIKE ? ESCAPE '\\' OR d.text LIKE ? ESCAPE '\\')"
        p = _like_pat(e)
        params += [p, p]
    return sql, params


# ---------------------------------------------------------------------------------------------------------------
# 키워드 검색
# ---------------------------------------------------------------------------------------------------------------
_BM25 = "bm25(chunks, 2.0, 1.0)"
_JOIN = "FROM chunks JOIN documents d ON d.id = chunks.doc_id"


def _strict_candidates(db, pq: dict, sources, doc_id, cap: int = CANDIDATE_CAP) -> tuple[list[tuple], int]:
    """모든 단어/구문이 들어 있는 조각. 반환 ([(chunk_id, doc_id, score)...] 좋은 순, 일치 문서 수)."""
    positives = pq["terms"] + pq["phrases"]
    long_ = [t for t in positives if len(t) >= 3]
    short = [t for t in positives if len(t) < 3]
    excl = pq["exclude"]
    fsql, fparams = _filters(sources, doc_id, excl)
    short_sql = "".join(" AND (chunks.content LIKE ? ESCAPE '\\' OR chunks.title LIKE ? ESCAPE '\\')" for _ in short)
    short_params: list = []
    for t in short:
        short_params += [_like_pat(t), _like_pat(t)]

    with db.conn() as c:
        if long_:
            match = "(" + " AND ".join(_fts_quote(t) for t in long_) + ")"
            match += "".join(f" NOT {_fts_quote(e)}" for e in excl if len(e) >= 3)
            where = f"WHERE chunks MATCH ?{short_sql}{fsql}"
            params = [match, *short_params, *fparams]
            rows = c.execute(
                f"SELECT chunks.rowid AS chunk_id, chunks.doc_id AS doc_id, -{_BM25} AS score {_JOIN} {where} "
                f"ORDER BY {_BM25} LIMIT ?", [*params, cap]).fetchall()
            out = [(r["chunk_id"], r["doc_id"], float(r["score"] or 0.0)) for r in rows]
            if len(out) >= cap:
                total = c.execute(f"SELECT count(DISTINCT chunks.doc_id) {_JOIN} {where}", params).fetchone()[0]
            else:
                total = len({o[1] for o in out})
            return out, total
        if short:
            # LIKE만: 등장 횟수(제목 가산)로 점수를 매긴다
            parts, sparams = [], []
            for t in short:
                lt = t.lower()
                parts.append("((length(chunks.content) - length(replace(lower(chunks.content), ?, ''))) / ?"
                             " + 2.0 * (instr(lower(chunks.title), ?) > 0))")
                sparams += [lt, float(len(lt)), lt]
            score = f"(({' + '.join(parts)}) / (1.0 + length(chunks.content) / 800.0))"
            where = f"WHERE 1{short_sql}{fsql}"
            params = [*short_params, *fparams]
            rows = c.execute(
                f"SELECT chunks.rowid AS chunk_id, chunks.doc_id AS doc_id, {score} AS score {_JOIN} {where} "
                "ORDER BY score DESC, chunks.rowid DESC LIMIT ?", [*sparams, *params, cap]).fetchall()
            out = [(r["chunk_id"], r["doc_id"], float(r["score"] or 0.0)) for r in rows]
            if len(out) >= cap:
                total = c.execute(f"SELECT count(DISTINCT chunks.doc_id) {_JOIN} {where}", params).fetchone()[0]
            else:
                total = len({o[1] for o in out})
            return out, total
        if excl:
            # 제외어만 있으면: 제외어가 없는 조각 전부 (최근 문서 먼저)
            where = f"WHERE 1{fsql}"
            rows = c.execute(
                f"SELECT chunks.rowid AS chunk_id, chunks.doc_id AS doc_id {_JOIN} {where} "
                "ORDER BY chunks.doc_id DESC, chunks.rowid ASC LIMIT ?", [*fparams, cap]).fetchall()
            out = [(r["chunk_id"], r["doc_id"], 0.0) for r in rows]
            if len(out) >= cap:
                total = c.execute(f"SELECT count(DISTINCT chunks.doc_id) {_JOIN} {where}", fparams).fetchone()[0]
            else:
                total = len({o[1] for o in out})
            return out, total
    return [], 0


def _relaxed_candidates(db, pq: dict, sources, doc_id, cap: int = CANDIDATE_CAP) -> tuple[list[tuple], int]:
    """넓혀 찾기: 조사를 뗀 단어 중 하나라도 들어 있는 조각. 들어 있는 단어가 많을수록 위."""
    rterms = relaxed_terms(pq)
    if not rterms:
        return [], 0
    long_ = [t for t in rterms if len(t) >= 3]
    short = [t for t in rterms if len(t) < 3]
    fsql, fparams = _filters(sources, doc_id, pq["exclude"])
    found: dict[int, dict] = {}
    with db.conn() as c:
        if long_:
            match = " OR ".join(_fts_quote(t) for t in long_)
            for r in c.execute(
                    f"SELECT chunks.rowid AS chunk_id, chunks.doc_id AS doc_id, chunks.title AS title, "
                    f"chunks.content AS content, -{_BM25} AS bm {_JOIN} WHERE chunks MATCH ?{fsql} "
                    f"ORDER BY {_BM25} LIMIT ?", [match, *fparams, cap]):
                found[r["chunk_id"]] = dict(r)
        if short:
            cond = " OR ".join("chunks.content LIKE ? ESCAPE '\\' OR chunks.title LIKE ? ESCAPE '\\'" for _ in short)
            sp: list = []
            for t in short:
                sp += [_like_pat(t), _like_pat(t)]
            for r in c.execute(
                    f"SELECT chunks.rowid AS chunk_id, chunks.doc_id AS doc_id, chunks.title AS title, "
                    f"chunks.content AS content, 0.0 AS bm {_JOIN} WHERE ({cond}){fsql} "
                    "ORDER BY chunks.rowid DESC LIMIT ?", [*sp, *fparams, cap]):
                found.setdefault(r["chunk_id"], dict(r))
    out = []
    for r in found.values():
        hay = f"{r['title'] or ''}\n{r['content'] or ''}".lower()
        cov = sum(min(len(t), 4) for t in rterms if t.lower() in hay)
        if cov <= 0:
            continue
        bm = max(0.0, float(r["bm"] or 0.0))
        out.append((r["chunk_id"], r["doc_id"], cov + bm / (1.0 + bm)))
    out.sort(key=lambda x: (-x[2], x[0]))
    out = out[:cap]
    return out, len({o[1] for o in out})


def _can_relax(pq: dict) -> bool:
    if not pq["terms"]:
        return False
    rt = relaxed_terms(pq)
    strict = _dedupe(pq["terms"] + pq["phrases"])
    return len(strict) >= 2 or [t.lower() for t in rt] != [t.lower() for t in strict]


def _merge_spans(spans: list[tuple[int, int]]) -> list[list[int]]:
    spans = sorted(spans)
    out: list[list[int]] = []
    for s, e in spans:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _find_spans(text: str, needles: list[str], limit: int = MAX_MATCHES) -> list[list[int]]:
    needles = sorted({n for n in needles if n}, key=len, reverse=True)
    if not text or not needles:
        return []
    pat = re.compile("|".join(re.escape(n) for n in needles), re.IGNORECASE)
    spans = []
    for m in pat.finditer(text):
        if m.end() > m.start():
            spans.append((m.start(), m.end()))
        if len(spans) >= limit * 2:
            break
    return _merge_spans(spans)[:limit]


def make_snippet(text: str, needles: list[str], width: int = SNIPPET_CHARS) -> str:
    """일치가 가장 많이 몰린 곳 주변 약 width자. 일치 부분은 HL_START/HL_END로 감싼다."""
    text = (text or "").replace(HL_START, "").replace(HL_END, "")
    spans = _find_spans(text, needles, 60)
    if not spans:
        s, e = 0, min(len(text), width)
    else:
        best, s = -1, 0
        for a, _ in spans:
            start = max(0, a - width // 4)
            n = sum(1 for x, y in spans if x >= start and y <= start + width)
            if n > best:
                best, s = n, start
        e = min(len(text), s + width)
        s = max(0, min(s, e - width))
    parts, pos = [], s
    for a, b in spans:
        if b <= s or a >= e:
            continue
        a, b = max(a, s), min(b, e)
        parts += [text[pos:a], HL_START, text[a:b], HL_END]
        pos = b
    parts.append(text[pos:e])
    body = _WS.sub(" ", "".join(parts)).strip()
    return ("…" if s > 0 else "") + body + ("…" if e < len(text) else "")


def _hydrate(db, cands: list[tuple], needles: list[str]) -> list[dict]:
    """[(chunk_id, doc_id, score)] -> 검색 결과 행 (content, snippet 포함)."""
    rows = db.get_chunks([c[0] for c in cands])
    by_id = {r["chunk_id"]: r for r in rows}
    out = []
    for cid, _did, score in cands:
        r = by_id.get(cid)
        if not r:
            continue
        out.append({"chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "url": r["url"], "title": r["title"] or "",
                    "source": r["source"], "snippet": make_snippet(r["content"], needles),
                    "content": r["content"], "score": round(float(score), 6)})
    return out


def keyword_search(db, q, limit: int = 20, offset: int = 0, sources=None, doc_id=None, *,
                   match_all: bool = True) -> list[dict]:
    """FTS5 trigram 키워드 검색 (조각 단위, 좋은 순). 이상한 입력에도 예외 없이 [] 를 돌려준다.

    match_all=False 이면 조사를 떼고 단어 중 하나라도 맞으면 찾는다(넓혀 찾기).
    행: {"chunk_id","doc_id","url","title","source","snippet","content","score"(클수록 좋음)}
    """
    pq = parse_query(q)
    limit, offset = max(1, int(limit or 20)), max(0, int(offset or 0))
    cap = min(CANDIDATE_CAP, offset + limit)
    if match_all:
        cands, _ = _strict_candidates(db, pq, sources, doc_id, cap)
    else:
        cands, _ = _relaxed_candidates(db, pq, sources, doc_id, max(cap, 200))
    return _hydrate(db, cands[offset:offset + limit], _needles(pq, relaxed=not match_all))


# ---------------------------------------------------------------------------------------------------------------
# 의미 검색 (임베딩)
# ---------------------------------------------------------------------------------------------------------------
def pack_vector(v) -> bytes:
    return array("f", [float(x) for x in v]).tobytes()


def unpack_vector(blob: bytes) -> array:
    a = array("f")
    a.frombytes(blob)
    return a


class Embedder:
    """문장 임베딩 (transformers AutoModel + 평균 풀링 + L2 정규화). 처음 encode 할 때 모델을 읽는다."""

    RETRY_AFTER_SEC = 300  # 모델 로드 실패 후 다시 시도하기까지 (검색마다 네트워크를 기다리지 않도록)

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, batch_size: int = 32, max_length: int = 512,
                 device: str | None = "cpu"):
        self.model_name = model_name or DEFAULT_EMBED_MODEL
        self.batch_size = max(1, int(batch_size or 32))
        self.max_length = max(16, int(max_length or 512))
        self.device = device or "cpu"
        self._tok = None
        self._model = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._error_at = 0.0

    @property
    def is_e5(self) -> bool:
        return "e5" in self.model_name.lower()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if self._error and time.monotonic() - self._error_at < self.RETRY_AFTER_SEC:
                raise RuntimeError(self._error)
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                dev = self.device
                if dev == "auto":
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                tok = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModel.from_pretrained(self.model_name)
                model.eval()
                model.to(dev)
            except Exception as e:  # 네트워크 없음, 모델 이름 오류 등
                self._error = f"의미 검색 모델({self.model_name})을 불러오지 못했습니다: {e}"
                self._error_at = time.monotonic()
                raise RuntimeError(self._error) from e
            self.device = dev
            self._tok, self._model = tok, model
            self._error = None

    def encode(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """e5 계열은 "query: " / "passage: " 접두어를 붙인다. 반환 벡터는 L2 정규화되어 있다."""
        if not texts:
            return []
        self._load()
        import torch

        prefix = ("query: " if is_query else "passage: ") if self.is_e5 else ""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [prefix + _WS.sub(" ", str(t or "")).strip() for t in texts[i:i + self.batch_size]]
            enc = self._tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                hidden = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            out.extend(pooled.cpu().tolist())
        return out


EMBEDDER_OVERRIDE = None  # 테스트/통합용: 여기에 가짜 임베더를 넣으면 get_embedder가 이것을 돌려준다
_EMBEDDERS: dict[tuple, Embedder] = {}
_EMB_LOCK = threading.Lock()


def set_embedder(embedder) -> None:
    """get_embedder()가 돌려줄 임베더를 강제로 지정한다 (None이면 해제). 테스트에서 가짜 임베더 주입용."""
    global EMBEDDER_OVERRIDE
    EMBEDDER_OVERRIDE = embedder


def _emb_cfg(cfg) -> dict:
    try:
        e = cfg["embeddings"]
    except (KeyError, TypeError):
        e = None
    return e if isinstance(e, dict) else {}


def embeddings_available() -> bool:
    return importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("torch") is not None


def get_embedder(cfg):
    """설정의 임베더 (프로세스 전체에서 모델 이름별로 하나). 꺼져 있거나 transformers/torch가 없으면 None."""
    ec = _emb_cfg(cfg)
    if not ec.get("enabled", True):
        return None
    if EMBEDDER_OVERRIDE is not None:
        return EMBEDDER_OVERRIDE
    if not embeddings_available():
        return None
    name = ec.get("model") or DEFAULT_EMBED_MODEL
    key = (name, ec.get("device") or "cpu")
    with _EMB_LOCK:
        emb = _EMBEDDERS.get(key)
        if emb is None:
            emb = _EMBEDDERS[key] = Embedder(name, batch_size=int(ec.get("batch_size") or 32), device=key[1])
        return emb


def _count_missing(db, model: str) -> int:
    with db.conn() as c:
        return c.execute(
            "SELECT count(*) FROM chunks LEFT JOIN embeddings e ON e.chunk_id = chunks.rowid AND e.model = ? "
            "WHERE e.chunk_id IS NULL", (model,)).fetchone()[0]


def index_embeddings(cfg, db, progress=_noop_progress, check_cancel=_noop_cancel, batch_size=None, *,
                     embedder=None) -> dict:
    """아직 벡터가 없는 조각을 임베딩해 저장한다. 반환 {"indexed","total","model"}."""
    emb = embedder or get_embedder(cfg)
    if emb is None:
        if not _emb_cfg(cfg).get("enabled", True):
            raise ValueError("의미 검색이 꺼져 있습니다. 설정 > 의미 검색에서 켜세요.")
        raise ValueError("의미 검색을 쓰려면 transformers와 torch가 필요합니다: pip install -r requirements-train.txt")
    model = emb.model_name
    bs = max(1, int(batch_size or _emb_cfg(cfg).get("batch_size") or 32))
    total_missing = _count_missing(db, model)
    progress(0.0, f"의미 검색 색인 준비 중 (새 조각 {total_missing}개, 모델 {model})")
    done, last_ids = 0, None
    while True:
        check_cancel()
        rows = db.missing_embedding_chunks(model, limit=bs)
        if not rows:
            break
        ids = [r["chunk_id"] for r in rows]
        if ids == last_ids:  # 저장이 안 되는 이상 상황에서 무한 반복 방지
            raise RuntimeError("임베딩을 저장하지 못했습니다.")
        last_ids = ids
        texts = [f"{r['title']}\n{r['content']}" if r.get("title") else r["content"] for r in rows]
        vecs = emb.encode(texts, is_query=False)
        if len(vecs) != len(rows):
            raise RuntimeError("임베딩 개수가 조각 개수와 다릅니다.")
        db.upsert_embeddings([(r["chunk_id"], r["doc_id"], model, len(v), pack_vector(v)) for r, v in zip(rows, vecs)])
        done += len(rows)
        progress(min(0.99, done / max(1, total_missing)), f"의미 검색 색인 {done}/{total_missing}")
    progress(1.0, f"의미 검색 색인 완료 (새로 {done}개)")
    return {"indexed": done, "total": db.embedding_count(model), "model": model}


# 임베딩 행렬 캐시: (db 경로, 모델) -> (서명, 데이터). 서명이 바뀌면(추가/삭제) 다시 읽는다.
_MATRIX_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_MATRIX_LOCK = threading.Lock()


def _load_matrix(db, model: str) -> dict:
    with db.conn() as c:
        sig = tuple(c.execute("SELECT count(*), coalesce(sum(chunk_id),0), coalesce(max(chunk_id),0) "
                              "FROM embeddings WHERE model=?", (model,)).fetchone())
    key = (str(db.path), model)
    with _MATRIX_LOCK:
        hit = _MATRIX_CACHE.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    chunk_ids, doc_ids, sources, vecs = [], [], [], []
    dim = None
    with db.conn() as c:
        for r in c.execute("SELECT e.chunk_id AS chunk_id, e.doc_id AS doc_id, e.dim AS dim, e.vec AS vec, "
                           "d.source AS source FROM embeddings e JOIN documents d ON d.id = e.doc_id "
                           "WHERE e.model = ?", (model,)):
            v = unpack_vector(r["vec"])
            if dim is None:
                dim = len(v)
            if len(v) != dim:
                continue
            chunk_ids.append(r["chunk_id"])
            doc_ids.append(r["doc_id"])
            sources.append(r["source"])
            vecs.append(v)
    try:
        import numpy as np

        mat = np.frombuffer(b"".join(v.tobytes() for v in vecs), dtype=np.float32).reshape(len(vecs), dim or 0) \
            if vecs else None
        data = {"np": True, "mat": mat}
    except ImportError:
        data = {"np": False, "mat": vecs}
    data.update(chunk_ids=chunk_ids, doc_ids=doc_ids, sources=sources, dim=dim or 0)
    with _MATRIX_LOCK:
        _MATRIX_CACHE[key] = (sig, data)
        _MATRIX_CACHE.move_to_end(key)
        while len(_MATRIX_CACHE) > 2:
            _MATRIX_CACHE.popitem(last=False)
    return data


def _excluded_docs(db, exclude: list[str]) -> set[int]:
    if not exclude:
        return set()
    cond = " OR ".join("title LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\'" for _ in exclude)
    params: list = []
    for e in exclude:
        params += [_like_pat(e), _like_pat(e)]
    with db.conn() as c:
        return {r[0] for r in c.execute(f"SELECT id FROM documents WHERE {cond}", params)}


def _semantic_query_text(pq: dict) -> str:
    return " ".join(pq["phrases"] + pq["terms"]).strip()


def _semantic_candidates(cfg, db, pq: dict, emb, sources, doc_id, top: int) -> list[tuple]:
    """[(chunk_id, doc_id, cosine)] 유사도 높은 순 top개."""
    text = _semantic_query_text(pq)
    if not text:
        return []
    data = _load_matrix(db, emb.model_name)
    n = len(data["chunk_ids"])
    if not n:
        return []
    qv = emb.encode([text], is_query=True)[0]
    if len(qv) != data["dim"]:
        raise RuntimeError("검색어 벡터 차원이 색인과 다릅니다. 의미 검색 색인을 다시 만드세요.")
    srcs = set(_norm_sources(sources) or [])
    excl = _excluded_docs(db, pq["exclude"])
    floor = float(_emb_cfg(cfg).get("min_score") or -1.0)
    did = int(doc_id) if doc_id is not None else None

    def ok(i: int) -> bool:
        return ((not srcs or data["sources"][i] in srcs) and (did is None or data["doc_ids"][i] == did)
                and data["doc_ids"][i] not in excl)

    if data["np"]:
        import numpy as np

        sims = data["mat"] @ np.asarray(qv, dtype=np.float32)
        if srcs or did is not None or excl:
            mask = np.fromiter((ok(i) for i in range(n)), dtype=bool, count=n)
            sims = np.where(mask, sims, -np.inf)
        k = min(top, n)
        idx = np.argpartition(-sims, k - 1)[:k] if k < n else np.arange(n)
        idx = idx[np.argsort(-sims[idx], kind="stable")]
        out = [(data["chunk_ids"][i], data["doc_ids"][i], float(sims[i])) for i in idx.tolist()]
    else:
        scored = []
        for i, v in enumerate(data["mat"]):
            if ok(i):
                scored.append((sum(a * b for a, b in zip(v, qv)), i))
        scored.sort(key=lambda x: -x[0])
        out = [(data["chunk_ids"][i], data["doc_ids"][i], float(s)) for s, i in scored[:top]]
    return [o for o in out if math.isfinite(o[2]) and o[2] >= floor]


def semantic_search(cfg, db, q, limit: int = 20, sources=None, doc_id=None, *, embedder=None) -> list[dict]:
    """임베딩 코사인 유사도 검색 (조각 단위). 임베더나 색인이 없으면 []. 행 모양은 keyword_search와 같다."""
    pq = parse_query(q)
    emb = embedder or get_embedder(cfg)
    if emb is None or not _semantic_query_text(pq) or db.embedding_count(emb.model_name) == 0:
        return []
    cands = _semantic_candidates(cfg, db, pq, emb, sources, doc_id, max(1, int(limit or 20)))
    return _hydrate(db, cands, _needles(pq, relaxed=True))


# ---------------------------------------------------------------------------------------------------------------
# 통합 검색
# ---------------------------------------------------------------------------------------------------------------
def _best_per_doc(cands: list[tuple]) -> "OrderedDict[int, tuple]":
    out: OrderedDict[int, tuple] = OrderedDict()
    for cid, did, score in cands:
        if did not in out:
            out[did] = (cid, score)
    return out


def _search_core(cfg, db, q, sources, doc_id, mode: str, need: int, embedder=None) -> dict:
    pq = parse_query(q)
    res = {"pq": pq, "fused": [], "kw": [], "sem": [], "mode_used": "keyword", "relaxed": False, "note": None,
           "total": 0}
    if not (pq["terms"] or pq["phrases"] or pq["exclude"]):
        return res
    kw, kw_total = _strict_candidates(db, pq, sources, doc_id)
    if not kw and _can_relax(pq):
        kw, kw_total = _relaxed_candidates(db, pq, sources, doc_id)
        res["relaxed"] = bool(kw)
    res["kw"] = kw
    kw_docs = _best_per_doc(kw)

    sem_docs: OrderedDict[int, tuple] = OrderedDict()
    sem_used = False
    if mode in ("hybrid", "semantic") and _semantic_query_text(pq):
        emb = embedder or get_embedder(cfg)
        if emb is not None and db.embedding_count(emb.model_name) > 0:
            try:
                sem = _semantic_candidates(cfg, db, pq, emb, sources, doc_id, max(200, need * 5))
                res["sem"] = sem
                sem_docs = _best_per_doc(sem)
                sem_used = True
            except Exception as e:  # 모델 로드 실패 등: 키워드 검색으로 계속
                res["note"] = f"의미 검색을 건너뛰었습니다: {e}"
        elif mode == "semantic":
            res["note"] = "의미 검색 색인이 없어 키워드 검색으로 대신했습니다. '의미 검색 색인 만들기'를 실행하세요."

    fused: list[tuple] = []  # (doc_id, chunk_id, score, match)
    if sem_used and mode == "semantic":
        for did, (cid, s) in sem_docs.items():
            fused.append((did, cid, s, "both" if did in kw_docs else "semantic"))
        res["mode_used"] = "semantic"
        res["total"] = len(sem_docs)
    elif sem_used:
        kw_rank = {d: i for i, d in enumerate(kw_docs, 1)}
        sem_rank = {d: i for i, d in enumerate(sem_docs, 1)}
        scores = {}
        for d in set(kw_rank) | set(sem_rank):
            s = 0.0
            if d in kw_rank:
                s += 1.0 / (RRF_K + kw_rank[d])
            if d in sem_rank:
                s += 1.0 / (RRF_K + sem_rank[d])
            scores[d] = s
        order = sorted(scores, key=lambda d: (-scores[d], kw_rank.get(d, 10 ** 9), sem_rank.get(d, 10 ** 9)))
        for d in order:
            if d in kw_docs:
                cid = kw_docs[d][0]
                match = "both" if d in sem_docs else "keyword"
            else:
                cid, match = sem_docs[d][0], "semantic"
            fused.append((d, cid, scores[d], match))
        res["mode_used"] = "hybrid"
        res["total"] = kw_total + sum(1 for d in sem_docs if d not in kw_docs)
    else:
        for did, (cid, s) in kw_docs.items():
            fused.append((did, cid, s, "keyword"))
        res["total"] = max(kw_total, len(kw_docs))
    res["fused"] = fused
    return res


def search(cfg, db, q, limit: int = 10, offset: int = 0, sources=None, doc_id=None, mode: str = "hybrid", *,
           embedder=None) -> dict:
    """학습 파일 검색. mode: hybrid(통합) | keyword(키워드) | semantic(의미). 문서당 가장 잘 맞는 조각 하나.

    반환 {"results": [{"doc_id","chunk_id","url","title","source","snippet","score","match"}], "mode_used",
          "total", "offset", "limit", "extract", "relaxed", "note"}
    """
    mode = mode if mode in ("hybrid", "keyword", "semantic") else "hybrid"
    try:
        limit = max(1, min(int(limit or 10), 100))
    except (TypeError, ValueError):
        limit = 10
    try:
        offset = max(0, int(offset or 0))
    except (TypeError, ValueError):
        offset = 0
    core = _search_core(cfg, db, q, sources, doc_id, mode, offset + limit, embedder)
    page = core["fused"][offset:offset + limit]
    needles = _needles(core["pq"], relaxed=True)
    hyd = {r["chunk_id"]: r for r in _hydrate(db, [(c, d, s) for d, c, s, _ in page], needles)}
    results, full = [], []
    for d, c, s, match in page:
        r = hyd.get(c)
        if not r:
            continue
        full.append(r)
        results.append({"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "url": r["url"], "title": r["title"],
                        "source": r["source"], "snippet": r["snippet"], "score": round(float(s), 6),
                        "match": match})
    return {"results": results, "mode_used": core["mode_used"], "total": core["total"], "offset": offset,
            "limit": limit, "extract": extractive_answer(q, full[:5]), "relaxed": core["relaxed"],
            "note": core["note"]}


def retrieve(cfg, db, q, k: int = 6, sources=None, *, embedder=None) -> list[dict]:
    """답변 엔진(LoRA/온라인/소형 AI)에 넘길 문맥 조각 (통합 검색 순서).

    문서마다 가장 좋은 조각을 먼저 고르고, 모자라면 같은 문서의 다음 조각으로 채운다.
    반환 [{"doc_id","chunk_id","title","url","content","source"}]
    """
    k = max(1, int(k or 6))
    core = _search_core(cfg, db, q, sources, None, "hybrid", k, embedder)
    picked: list[int] = [c for _d, c, _s, _m in core["fused"][:k]]
    if len(picked) < k:
        seen = set(picked)
        extra = [c for c, _d, _s in core["kw"]] + [c for c, _d, _s in core["sem"]]
        for c in extra:
            if c not in seen:
                seen.add(c)
                picked.append(c)
                if len(picked) >= k:
                    break
    rows = db.get_chunks(picked)
    return [{"doc_id": r["doc_id"], "chunk_id": r["chunk_id"], "title": r["title"] or "", "url": r["url"],
             "content": r["content"], "source": r["source"]} for r in rows]


# ---------------------------------------------------------------------------------------------------------------
# 핵심 문장 / 문서 보기
# ---------------------------------------------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|(?<=[다요죠음함임됨]\.)(?=\S)")


def split_sentences(text: str) -> list[str]:
    out = []
    for line in re.split(r"\n+", text or ""):
        for s in _SENT_SPLIT.split(line):
            s = s.strip()
            if s:
                out.append(s)
    return out


def extractive_answer(q: str, rows: list[dict], max_sentences: int = 3) -> list[dict]:
    """후보 조각에서 검색어가 많이 들어 있는 문장을 골라 [{"sentence","doc_id","title","url"}] (좋은 순, 중복 없음)."""
    pq = parse_query(q)
    strict = [t.lower() for t in pq["terms"]]
    phrases = [p.lower() for p in pq["phrases"]]
    relaxed = [t.lower() for t in relaxed_terms(pq)] if pq["terms"] or pq["phrases"] else []
    if not (strict or phrases or relaxed) or not rows:
        return []
    cands = []
    seen = set()
    order = 0
    for ri, r in enumerate(rows or []):
        for s in split_sentences(r.get("content") or ""):
            order += 1
            low = s.lower()
            score = 0.0
            for t in set(strict) | set(relaxed):
                if t and t in low:
                    score += min(len(t), 4)
            for p in phrases:
                if p and p in low:
                    score += 2 * min(len(p), 8)
            if score <= 0:
                continue
            key = _WS.sub(" ", low).strip()
            if key in seen:
                continue
            seen.add(key)
            if len(s) < 15:
                score *= 0.5
            if len(s) > 400:  # 너무 긴 문장은 첫 일치 주변만
                s = make_snippet(s, _needles(pq, relaxed=True), 300).replace(HL_START, "").replace(HL_END, "")
            score += 0.5 / (1 + ri)  # 앞 순위 결과의 문장을 조금 더
            cands.append((score, order, {"sentence": s, "doc_id": r.get("doc_id"), "title": r.get("title") or "",
                                         "url": r.get("url")}))
    cands.sort(key=lambda x: (-x[0], x[1]))
    return [c[2] for c in cands[:max(0, int(max_sentences))]]


def doc_view(db, doc_id: int, q: str | None = None) -> dict | None:
    """문서 전체와 검색어 일치 위치. matches = [[시작, 끝], ...] (글자 위치, 대소문자 무시, 최대 500개)."""
    try:
        d = db.get_document(int(doc_id))
    except (TypeError, ValueError):
        return None
    if not d:
        return None
    text = d.get("text") or ""
    matches: list[list[int]] = []
    if q:
        pq = parse_query(q)
        matches = _find_spans(text, _needles(pq))
        if not matches:  # 원래 단어가 없으면 조사를 뗀 단어로
            matches = _find_spans(text, relaxed_terms(pq))
    return {"id": d["id"], "url": d.get("url"), "title": d.get("title") or "", "source": d.get("source"),
            "query": d.get("query"), "fetched_at": d.get("fetched_at"), "chars": d.get("chars", len(text)),
            "text": text, "matches": matches, "matches_utf16": utf16_spans(text, matches),
            "file_id": d.get("file_id")}


def utf16_spans(text: str, spans: list[list[int]]) -> list[list[int]]:
    """파이썬 글자 위치 -> JavaScript 문자열 위치(UTF-16). 이모지 같은 BMP 밖 글자가 있을 때만 달라진다."""
    astral = [i for i, ch in enumerate(text) if ord(ch) > 0xFFFF] if text else []
    if not astral:
        return [list(s) for s in spans]
    from bisect import bisect_left

    return [[s + bisect_left(astral, s), e + bisect_left(astral, e)] for s, e in spans]
