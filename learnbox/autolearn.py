"""자율 학습: 허용 시간대 안에서 조금씩 수집하고, 새 자료가 충분히 모이면 이어서 학습한다.

- AutoLearner.tick()이 20초마다 한 번 판단한다 (수집 작업 / 학습 작업을 작업 관리자에 제출).
- 수집 대상: 예약 목록(watchlist) → 비었거나 이번 세션에서 한 바퀴 다 돌았으면 자료에서 주제를 스스로 찾아 추가
  (discover_topics) → 그래도 없으면 문서 제목에서 자주 나오는 낱말로 시작.
- 학습(autolearn_train 작업): 말뭉치를 만들고 소형 AI 모델(이어 학습) / Q&A 데이터 / LoRA / 의미 검색 색인.
- 학습 종료(finalize_learning 작업): 마지막 학습 → results.export_results → (선택) 외장 저장장치 복사 → 알림.

모델 학습 관련 모듈(tinylm, train_pc, search, online, devices)은 함수 안에서만 import 한다.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import re
import shutil
import threading
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .db import now_iso
from .timegate import describe, is_allowed

COLLECT_KINDS = ("collect_links", "collect_search")
HEAVY_KINDS = ("autolearn_train", "finalize_learning", "train_tiny", "train_lora", "train_mobile", "train_pc",
               "export")
TICK_SEC = 20.0
TRAIN_MIN_GAP_SEC = 300      # 학습 작업 제출 사이 최소 간격 (실패가 반복될 때 20초마다 다시 제출하지 않게)
DISCOVER_PER_CYCLE = 5       # 한 번에 새로 추가하는 자동 주제 수
SEED_WORDS = 5
MAX_JOB_IDS = 200            # stats의 collect_jobs/train_jobs 보관 개수
FINALIZE_RETRIES = 3

TARGET_LABELS = {"tiny": "소형 AI 모델", "lora": "LoRA 모델", "embeddings": "의미 검색 색인", "qa": "Q&A 학습 데이터"}
TRAIN_TARGETS = ("tiny", "lora", "embeddings", "qa")
ARCH_KEYS = ("d_model", "n_layer", "n_head", "seq_len", "vocab_size")

STATS_DEFAULTS = {
    "cycles": 0, "pages_collected": 0, "docs_new": 0, "docs_updated": 0, "chars_new": 0, "discovered_topics": 0,
    "trainings": 0, "last_collect_at": None, "last_train_at": None, "collect_jobs": [], "train_jobs": [],
    "models": [], "train_history": [], "hour_pages": [],
}

# 주제 찾기: 떼어 낼 조사/어미 (긴 것부터 시도)
KO_SUFFIXES = sorted({
    "은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로", "로", "와", "과", "도", "만", "까지", "부터",
    "이다", "있다", "하는", "한", "및", "등",
    "에게", "에는", "에서는", "으로는", "으로서", "으로써", "이나", "이며", "이고", "하고", "하며", "한다", "했다",
    "된다", "되는", "되어", "하여", "해서", "했던", "들", "들의", "들은", "들이", "들을", "이라는", "라는", "입니다",
    "합니다", "있는", "였다", "이었다", "였던", "에도", "보다", "처럼", "만큼",
}, key=len, reverse=True)
STOPWORDS = {
    # 한국어
    "있다", "이다", "하는", "있는", "없는", "하다", "한다", "했다", "된다", "되는", "그리고", "그러나", "하지만", "또한",
    "또는", "그래서", "따라서", "때문", "위해", "위한", "대한", "대해", "통해", "통한", "관련", "경우", "이후", "이전",
    "현재", "당시", "가장", "매우", "모든", "여러", "다른", "같은", "이번", "우리", "그것", "이것", "저것", "어떤",
    "무엇", "누구", "어디", "언제", "어떻게", "것이", "것은", "것을", "것이다", "것으로", "것이며", "것과", "것도",
    "수가", "수는", "수를", "수도", "정도", "사이", "이상", "이하",
    "부분", "내용", "자료", "문서", "제목", "출처", "참고", "편집", "목차", "보기", "바로가기", "이용", "사용",
    "다음", "주요", "기타", "모두", "각각", "더욱", "아래", "위키백과", "우리말", "있습니다", "합니다", "입니다",
    # 영어
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were", "has", "have", "had", "not", "but",
    "you", "your", "they", "their", "its", "his", "her", "which", "who", "what", "when", "where", "how", "can",
    "will", "would", "should", "could", "may", "also", "into", "than", "then", "there", "these", "those", "such",
    "been", "being", "more", "most", "other", "some", "about", "after", "before", "over", "under", "between",
    "only", "very", "all", "any", "each", "one", "two", "use", "used", "using", "new", "http", "https", "www",
    "com", "html", "org", "net", "edit", "page", "pages", "wikipedia", "retrieved", "archived", "original",
}
_TOKEN_SPLIT = re.compile(r"[\W_]+", re.UNICODE)
_HANGUL = re.compile(r"[가-힣]")
_DIGITISH = re.compile(r"^[\d.,]+$")

_STATS_LOCK = threading.RLock()  # 세션 stats 읽기-수정-쓰기 (tick / 작업 스레드 사이)


def _noop_progress(p: float, m: str = "") -> None:
    return None


def _mod(name: str):
    """learnbox 하위 모듈을 필요할 때 불러온다 (무거운 의존성 지연 import; sys.modules 대체도 존중)."""
    return importlib.import_module(f"{__package__}.{name}")


def _noop_cancel() -> None:
    return None


# ---------------------------------------------------------------------------------------------------------------
# 시각 / 설정 도우미
# ---------------------------------------------------------------------------------------------------------------
def _aware(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now.astimezone() if now.tzinfo is None else now


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(iso) -> datetime | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(str(iso))
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on", "y", "예", "켜기")
    return bool(v)


def _coerce(key: str, default, value):
    """사용자 입력(문자열일 수 있음)을 기본값과 같은 형식으로."""
    try:
        if isinstance(default, bool):
            return _bool(value)
        if isinstance(default, int):
            return int(float(value))
        if isinstance(default, float):
            return float(value)
        if isinstance(default, list):
            if isinstance(value, str):
                return [x.strip() for x in value.split(",") if x.strip()]
            return list(value)
        if isinstance(default, dict):
            if not isinstance(value, dict):
                raise TypeError
            out = copy.deepcopy(default)
            for k, v in value.items():
                out[k] = _coerce(f"{key}.{k}", default[k], v) if k in default else v
            return out
        if isinstance(default, str):
            return "" if value is None else str(value)
    except (TypeError, ValueError):
        raise ValueError(f"자율 학습 설정 '{key}' 값이 올바르지 않습니다: {value!r}") from None
    return value


_MINIMUMS = {"collect_interval_min": 0, "pages_per_cycle": 1, "max_pages_per_hour": 1, "max_discovered_topics": 0,
             "train_min_new_chars": 1, "train_interval_min": 0, "steps_per_cycle": 1, "threads": 0}


def _autolearn_defaults(cfg) -> dict:
    from .config import DEFAULTS

    base = copy.deepcopy(DEFAULTS["autolearn"])
    try:
        cur = cfg["autolearn"] or {}
    except (KeyError, TypeError):
        cur = {}
    for k, v in cur.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(copy.deepcopy(v))
        else:
            base[k] = copy.deepcopy(v)
    return base


def _session_conf(cfg, session: dict) -> dict:
    """세션에 저장된 설정 + (예전 세션에 없을 수 있는) 기본값."""
    conf = _autolearn_defaults(cfg)
    for k, v in (session.get("config") or {}).items():
        if isinstance(v, dict) and isinstance(conf.get(k), dict) and k == "targets":
            conf[k].update(v)
        else:
            conf[k] = v
    return conf


def _norm_stats(stats: dict | None) -> dict:
    out = copy.deepcopy(STATS_DEFAULTS)
    out.update(copy.deepcopy(stats or {}))
    for k in ("collect_jobs", "train_jobs", "models", "train_history", "hour_pages"):
        if not isinstance(out.get(k), list):
            out[k] = []
    return out


def _mutate_stats(db, session_id: int, fn) -> dict | None:
    """세션 stats를 잠금 안에서 새로 읽어 고치고 저장한다 (작업 스레드와 tick 사이 경쟁 방지)."""
    with _STATS_LOCK:
        s = db.get_session(session_id)
        if not s:
            return None
        stats = _norm_stats(s.get("stats"))
        fn(stats)
        db.update_session(session_id, stats=stats)
        s["stats"] = stats
        return s


def _push(lst: list, item, cap: int = MAX_JOB_IDS) -> None:
    lst.append(item)
    del lst[:-cap]


def _session_jobs(db, session_id: int, kinds: tuple[str, ...], statuses=("queued", "running")) -> list[dict]:
    """이 세션(params.session_id)의 작업 중 상태가 statuses인 것."""
    qk = ",".join("?" * len(kinds))
    qs = ",".join("?" * len(statuses))
    with db.conn() as c:
        rows = c.execute(f"SELECT id, kind, status, params FROM jobs WHERE kind IN ({qk}) AND status IN ({qs}) "
                         "ORDER BY id", (*kinds, *statuses)).fetchall()
    out = []
    for r in rows:
        try:
            p = json.loads(r["params"] or "{}")
        except ValueError:
            p = {}
        if isinstance(p, dict) and str(p.get("session_id")) == str(session_id):
            out.append({"id": r["id"], "kind": r["kind"], "status": r["status"], "params": p})
    return out


def _hour_pages(stats: dict, now: datetime) -> tuple[int, datetime | None]:
    """최근 1시간 동안 수집한 페이지 수와 그중 가장 오래된 기록 시각."""
    cutoff = now - timedelta(hours=1)
    used, oldest = 0, None
    for h in stats.get("hour_pages") or []:
        ts = _parse(h.get("ts"))
        if ts and ts > cutoff:
            used += int(h.get("n") or 0)
            oldest = ts if oldest is None or ts < oldest else oldest
    return used, oldest


def _train_baseline(db, stats: dict) -> str | None:
    """'새 자료' 기준 시각: 마지막 학습 시각, 없으면 가장 최근 모델이 만들어진 시각, 그것도 없으면 처음부터."""
    if stats.get("last_train_at"):
        return stats["last_train_at"]
    times = [m.get("created_at") for m in db.list_models() if m.get("kind") in ("tiny", "lora") and m.get("created_at")]
    return max(times) if times else None


def _is_cancel(e: BaseException) -> bool:
    if type(e).__name__ == "Cancelled":
        return True
    try:
        from .jobs import Cancelled
    except Exception:  # pragma: no cover
        return False
    return isinstance(e, Cancelled)


def _sub(progress, lo: float, hi: float):
    def f(p, m=""):
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = 0.0
        progress(lo + (hi - lo) * max(0.0, min(1.0, p)), m)
    return f


# ---------------------------------------------------------------------------------------------------------------
# 주제 찾기
# ---------------------------------------------------------------------------------------------------------------
def _strip_suffix(tok: str) -> str:
    for _ in range(2):  # "학교에서는" -> "학교에서" -> "학교"
        for suf in KO_SUFFIXES:
            if tok.endswith(suf) and len(tok) - len(suf) >= 2:
                tok = tok[: -len(suf)]
                break
        else:
            break
    return tok


def _terms(text: str) -> list[str]:
    out = []
    for raw in _TOKEN_SPLIT.split(text or ""):
        if not raw:
            continue
        t = raw.lower()
        if _HANGUL.search(t):
            t = _strip_suffix(t)
        elif len(t) < 3:  # 영어는 3글자 이상
            continue
        if not 2 <= len(t) <= 12 or _DIGITISH.match(t) or t.isdigit() or t in STOPWORDS:
            continue
        if sum(ch.isdigit() for ch in t) * 2 > len(t):
            continue
        out.append(t)
    return out


def discover_topics(db, limit: int = 10, exclude: set[str] | None = None) -> list[str]:
    """최근 문서에서 주제 후보(한국어/영어 낱말)를 찾는다. 점수 = 최근 20개 문서의 빈도 × log((N+1)/df)."""
    limit = max(0, int(limit))
    if not limit:
        return []
    excl = {str(x).strip().lower() for x in (exclude or set())}
    with db.conn() as c:
        rows = c.execute("SELECT id, title, substr(text, 1, 20000) AS text FROM documents "
                         "ORDER BY coalesce(fetched_at,'') DESC, id DESC LIMIT 1000").fetchall()
    if not rows:
        return []
    n = len(rows)
    df: Counter = Counter()
    tf: Counter = Counter()
    for i, r in enumerate(rows):
        toks = _terms(f"{r['title'] or ''}\n{r['text'] or ''}")
        df.update(set(toks))
        if i < 20:
            tf.update(toks)
            tf.update(_terms(r["title"] or ""))  # 제목 낱말에 가중치
    scored = []
    for t, f in tf.items():
        if f < 2 and n > 1:
            continue  # 한 번만 나온 낱말은 주제로 보기 어렵다
        scored.append((f * math.log((n + 1) / df[t]), f, t))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    out = []
    for _, _, t in scored:
        if t in excl or t in out:
            continue
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _seed_words(db, limit: int = SEED_WORDS, exclude: set[str] | None = None) -> list[str]:
    """예약 목록이 비었을 때: 문서 제목에서 가장 자주 나오는 낱말."""
    excl = {str(x).strip().lower() for x in (exclude or set())}
    cnt: Counter = Counter()
    with db.conn() as c:
        for r in c.execute("SELECT title FROM documents ORDER BY id DESC LIMIT 5000"):
            cnt.update(set(_terms(r["title"] or "")))
    return [w for w, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])) if w not in excl][:max(0, int(limit))]


# ---------------------------------------------------------------------------------------------------------------
# AutoLearner
# ---------------------------------------------------------------------------------------------------------------
class AutoLearner:
    def __init__(self, app):
        self.app = app
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._halt = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_tick: dict | None = None

    @property
    def cfg(self):
        return self.app.cfg

    @property
    def db(self):
        return self.app.db

    # ---------------- 백그라운드 ----------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._halt.clear()
        self._thread = threading.Thread(target=self._loop, name="autolearn", daemon=True)
        self._thread.start()

    def stop(self) -> None:  # 테스트/종료용
        self._halt.set()
        self._wake.set()

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        self._wake.wait(3.0)  # 엔진이 막 시작했을 때 조금 기다린다
        while not self._halt.is_set():
            self._wake.clear()
            try:
                self.last_tick = self.tick()
            except Exception:
                traceback.print_exc()
            self._wake.wait(TICK_SEC)

    # ---------------- 세션 시작/종료 ----------------
    def _resolve_config(self, overrides: dict | None) -> dict:
        ov = dict(overrides or {})
        base = _autolearn_defaults(self.cfg)
        conf = copy.deepcopy(base)
        for k, v in ov.items():
            if k in base:
                conf[k] = _coerce(k, base[k], v)
        for k, lo in _MINIMUMS.items():
            if k in conf:
                conf[k] = max(lo, conf[k])
        conf["formats"] = [f for f in conf.get("formats") or [] if f in ("jsonl", "csv", "md", "txt")] or \
            ["jsonl", "csv", "md", "txt"]
        conf["targets"] = {k: _bool(conf["targets"].get(k, False)) for k in TRAIN_TARGETS}

        # 소형 AI 모델 크기: 지금 정해서 세션에 저장한다 (auto -> 이 PC 추천)
        tt = self.cfg.get("tiny_training") or {}
        if ov.get("tiny_preset") not in (None, ""):
            preset = str(ov["tiny_preset"]).strip().lower()
            arch_ov = ov.get("tiny_arch") if isinstance(ov.get("tiny_arch"), dict) else {}
            arch_ov = {**{k: ov[k] for k in ARCH_KEYS if ov.get(k) not in (None, "", 0, "0")}, **arch_ov}
        else:
            preset = str(tt.get("preset") or "auto").strip().lower()
            arch_ov = ov.get("tiny_arch") if isinstance(ov.get("tiny_arch"), dict) else \
                {k: tt.get(k) for k in ARCH_KEYS if tt.get(k)}
        recommended, rec_source, rec = "mini", "fallback", {}
        if preset in ("auto", "", "none", "recommended"):
            try:
                rec = _mod("devices").recommend() or {}
                if rec.get("tiny_preset"):
                    recommended, rec_source = str(rec["tiny_preset"]), "devices"
            except Exception:
                rec = {}
        try:
            resolve_arch = _mod("tinylm").resolve_arch
        except (ImportError, AttributeError):
            resolve_arch = None
        arch = None
        if resolve_arch is not None:
            try:
                arch = resolve_arch(preset, arch_ov or None, recommended=recommended)
            except ValueError:
                if conf["targets"]["tiny"]:
                    raise
        batch = ov.get("tiny_batch_size") or tt.get("batch_size") or 32
        try:
            batch = max(1, int(float(batch)))
        except (TypeError, ValueError):
            raise ValueError(f"자율 학습 설정 'tiny_batch_size' 값이 올바르지 않습니다: {batch!r}") from None
        if rec.get("tiny_batch_size") and not ov.get("tiny_batch_size"):
            batch = min(batch, int(rec["tiny_batch_size"]))
        try:
            lr = float(ov.get("tiny_lr") or tt.get("lr") or 3e-3)
        except (TypeError, ValueError):
            raise ValueError("자율 학습 설정 'tiny_lr' 값이 올바르지 않습니다.") from None
        conf.update({
            "tiny_preset_requested": preset,
            "tiny_preset": arch["preset"] if arch else preset,
            "tiny_arch": arch,
            "tiny_batch_size": batch,
            "tiny_lr": lr,
            "recommended": {"tiny_preset": recommended, "source": rec_source},
        })
        if not conf.get("threads") and rec.get("threads"):
            conf["threads_recommended"] = int(rec["threads"])
        for k in ("lora_base_model", "lora_steps_per_cycle", "qa_chunks_per_cycle", "qa_per_chunk"):
            if ov.get(k) not in (None, ""):
                conf[k] = ov[k] if k == "lora_base_model" else max(1, int(float(ov[k])))
        return conf

    def start_session(self, actor: str, overrides: dict | None = None) -> dict:
        with self._lock:
            if self.db.active_session():
                raise ValueError("이미 자율 학습이 진행 중입니다.")
            conf = self._resolve_config(overrides)
            sid = self.db.create_session(actor, conf)
            self.db.update_session(sid, stats=_norm_stats({}))
            targets = [TARGET_LABELS[k] for k in TRAIN_TARGETS if conf["targets"].get(k)]
            size = ""
            if conf["targets"].get("tiny") and conf.get("tiny_arch"):
                size = f" · 소형 AI 모델 크기: {conf['tiny_preset']}"
            self.db.notify("info", "자율 학습을 시작했습니다",
                           f"허용 시간({describe(self.cfg['allowed_windows'])})에 조금씩 수집하고 새 자료가 모이면 "
                           f"학습합니다. 학습 대상: {', '.join(targets) or '없음'}{size}",
                           {"session_id": sid, "kind": "autolearn_start"})
            self.db.audit(actor, "autolearn_start", {"session_id": sid, "overrides": overrides or {}})
            session = self.db.get_session(sid)
        self._wake.set()
        return session

    def stop_session(self, actor: str) -> dict:
        with self._lock:
            s = self.db.active_session()
            if not s:
                raise ValueError("진행 중인 자율 학습이 없습니다.")
            if s["status"] != "running":
                raise ValueError("이미 학습 종료를 진행하고 있습니다.")
            sid = s["id"]
            self.db.update_session(sid, status="finishing")
            cancelled = []
            for j in _session_jobs(self.db, sid, COLLECT_KINDS):
                self.app.jobs.cancel(j["id"])
                cancelled.append(j["id"])
            # 대기 중인 정기 학습은 취소 (마무리 작업이 마지막 학습을 한다). 실행 중인 학습은 끝까지 둔다.
            for j in _session_jobs(self.db, sid, ("autolearn_train",), ("queued",)):
                self.app.jobs.cancel(j["id"])
                cancelled.append(j["id"])
            try:
                job_id = self.app.jobs.submit("finalize_learning", {"session_id": sid}, actor)
            except Exception:
                self.db.update_session(sid, status="running")
                raise
            now = now_iso()

            def upd(st):
                st["stop_requested_at"] = now
                st["stopped_by"] = actor
                st["finalize_job_id"] = job_id
            _mutate_stats(self.db, sid, upd)
            self.db.notify("info", "학습 종료를 시작했습니다",
                           "마지막 학습과 결과 파일 내보내기를 진행합니다. 끝나면 알려 드립니다.",
                           {"session_id": sid, "job_id": job_id, "kind": "autolearn_finishing"})
            self.db.audit(actor, "autolearn_stop", {"session_id": sid, "job_id": job_id, "cancelled": cancelled})
        self._wake.set()
        return {"session_id": sid, "job_id": job_id}

    # ---------------- 상태 ----------------
    def status(self) -> dict:
        s = self.db.active_session()
        gate = self.app.gate()
        out = {"active": bool(s), "session": s, "phase": "idle", "window": gate, "next_collect_at": None,
               "next_train_check_at": None, "new_chars_since_train": 0,
               "recent_sessions": self.db.list_sessions(5), "jobs": {"collect": None, "train": None, "finalize": None}}
        if not s:
            return out
        sid = s["id"]
        conf = _session_conf(self.cfg, s)
        stats = _norm_stats(s.get("stats"))
        now = datetime.now(timezone.utc)
        collect = _session_jobs(self.db, sid, COLLECT_KINDS)
        train = _session_jobs(self.db, sid, ("autolearn_train",))
        fin = _session_jobs(self.db, sid, ("finalize_learning",))
        out["jobs"] = {"collect": collect[0]["id"] if collect else None, "train": train[0]["id"] if train else None,
                       "finalize": fin[0]["id"] if fin else stats.get("finalize_job_id")}
        if s["status"] == "finishing":
            out["phase"] = "finishing"
        elif any(j["status"] == "running" for j in train) or (train and not collect):
            out["phase"] = "training"
        elif collect:
            out["phase"] = "collecting"
        elif not gate.get("allowed"):
            out["phase"] = "waiting_window"
        else:
            out["phase"] = "cooldown"
        baseline = _train_baseline(self.db, stats)
        out["new_chars_since_train"] = int(self.db.chars_since(baseline))
        if s["status"] == "running":
            last = _parse(stats.get("last_collect_at"))
            t = last + timedelta(minutes=float(conf["collect_interval_min"])) if last else now
            used, oldest = _hour_pages(stats, now)
            if used >= int(conf["max_pages_per_hour"]) and oldest:
                t = max(t, oldest + timedelta(hours=1))
            if not gate.get("allowed"):
                nxt = _parse(gate.get("next_allowed"))
                t = max(t, nxt) if nxt else None
            out["next_collect_at"] = _iso(max(t, now)) if t else None
            lt = _parse(stats.get("last_train_at")) or _parse(s.get("started_at")) or now
            tt = lt + timedelta(minutes=float(conf["train_interval_min"]))
            if not gate.get("allowed") and not _bool(conf.get("train_outside_window")):
                nxt = _parse(gate.get("next_allowed"))
                tt = max(tt, nxt) if nxt else tt
            out["next_train_check_at"] = _iso(max(tt, now))
        out["hour_pages"] = _hour_pages(stats, now)[0]
        return out

    # ---------------- 한 번의 판단 ----------------
    def tick(self, now: datetime | None = None) -> dict:
        now = _aware(now)
        with self._lock:
            return self._tick(now)

    def _tick(self, now: datetime) -> dict:
        s = self.db.active_session()
        if not s:
            return {"actions": [], "session_id": None}
        sid = s["id"]
        actions: list[str] = []
        if s["status"] == "finishing":
            # 엔진 재시작 등으로 마무리 작업이 사라졌으면 다시 제출한다
            if not _session_jobs(self.db, sid, ("finalize_learning",)):
                stats = _norm_stats(s.get("stats"))
                if int(stats.get("finalize_retries") or 0) < FINALIZE_RETRIES:
                    job_id = self.app.jobs.submit("finalize_learning", {"session_id": sid},
                                                  s.get("actor") or "autolearn")

                    def upd(st):
                        st["finalize_retries"] = int(st.get("finalize_retries") or 0) + 1
                        st["finalize_job_id"] = job_id
                    _mutate_stats(self.db, sid, upd)
                    actions.append("finalize_resubmit")
            return {"actions": actions or ["finishing"], "session_id": sid}
        if s["status"] != "running":
            return {"actions": [], "session_id": sid}

        conf = _session_conf(self.cfg, s)
        stats = _norm_stats(s.get("stats"))
        window_open = is_allowed(self.cfg["allowed_windows"], self.cfg["timezone"], now)
        if window_open:
            actions += self._maybe_collect(s, conf, stats, now)
        else:
            actions.append("waiting_window")
        if window_open or _bool(conf.get("train_outside_window")):
            actions += self._maybe_train(s, conf, stats, now)
        return {"actions": actions, "session_id": sid, "window_open": window_open}

    def _maybe_collect(self, s: dict, conf: dict, stats: dict, now: datetime) -> list[str]:
        sid = s["id"]
        if self.db.running_jobs(COLLECT_KINDS):
            return ["collect_busy"]
        last = _parse(stats.get("last_collect_at"))
        if last and (now - last).total_seconds() < float(conf["collect_interval_min"]) * 60:
            return []
        cap = int(conf["max_pages_per_hour"])
        used, _ = _hour_pages(stats, now)
        if used >= cap:
            return ["hour_cap"]
        n = max(1, min(int(conf["pages_per_cycle"]), cap - used))
        actions: list[str] = []
        item, added = self._pick_target(s, conf, actions)
        if not item:
            def upd_none(st):
                st["discovered_topics"] = int(st.get("discovered_topics") or 0) + added
                st["note"] = "수집할 주제가 없습니다. 예약 목록에 링크나 검색어를 추가하거나 파일을 가져오세요."
            _mutate_stats(self.db, sid, upd_none)
            return actions + ["no_topic"]
        if item["kind"] == "search":
            kind, params = "collect_search", {"query": item["value"], "limit": n, "session_id": sid}
        else:
            kind, params = "collect_links", {"urls": [item["value"]], "depth": int(item.get("depth") or 0),
                                             "same_domain": True, "max_pages": n, "session_id": sid}
        job_id = self.app.jobs.submit(kind, params, "autolearn")
        self.db.touch_watch(item["id"])
        ts = _iso(now)

        def upd(st):
            st["cycles"] = int(st.get("cycles") or 0) + 1
            st["last_collect_at"] = ts
            st["discovered_topics"] = int(st.get("discovered_topics") or 0) + added
            _push(st["collect_jobs"], job_id)
            st["last_target"] = {"kind": item["kind"], "value": item["value"], "origin": item.get("origin")}
            st.pop("note", None)
        _mutate_stats(self.db, sid, upd)
        return actions + [f"{kind}:{item['value']}"]

    def _pick_target(self, s: dict, conf: dict, actions: list[str]) -> tuple[dict | None, int]:
        db = self.db
        item = db.next_watch()
        added = 0
        existing = {str(w["value"]).strip().lower() for w in db.list_watch()}
        cycled = bool(item and item.get("last_run") and item["last_run"] >= (s.get("started_at") or ""))
        if _bool(conf.get("discover_topics")) and (item is None or cycled):
            room = int(conf.get("max_discovered_topics") or 0) - db.count_watch("auto")
            if room > 0:
                for t in discover_topics(db, limit=min(room, DISCOVER_PER_CYCLE), exclude=existing):
                    if db.add_watch("search", t, 0, origin="auto"):
                        added += 1
                        existing.add(t)
                if added:
                    actions.append(f"discover:{added}")
                    item = db.next_watch()
        if item is None:
            seeded = 0
            for w in _seed_words(db, SEED_WORDS, existing):
                if db.add_watch("search", w, 0, origin="auto"):
                    seeded += 1
            if seeded:
                actions.append(f"seed:{seeded}")
                added += seeded
                item = db.next_watch()
        return item, added

    def _maybe_train(self, s: dict, conf: dict, stats: dict, now: datetime) -> list[str]:
        sid = s["id"]
        targets = conf.get("targets") or {}
        if not any(_bool(targets.get(k)) for k in TRAIN_TARGETS):
            return []
        if self.db.running_jobs(HEAVY_KINDS):
            return ["train_busy"]
        last_submit = _parse(stats.get("last_train_submit_at"))
        if last_submit and (now - last_submit).total_seconds() < TRAIN_MIN_GAP_SEC:
            return []
        new_chars = int(self.db.chars_since(_train_baseline(self.db, stats)))
        if new_chars < 1:
            return []
        last_train = _parse(stats.get("last_train_at")) or _parse(s.get("started_at")) or now
        interval_due = (now - last_train).total_seconds() >= float(conf["train_interval_min"]) * 60
        if new_chars < int(conf["train_min_new_chars"]) and not interval_due:
            return []
        job_id = self.app.jobs.submit("autolearn_train", {"session_id": sid}, "autolearn")
        ts = _iso(now)

        def upd(st):
            _push(st["train_jobs"], job_id)
            st["last_train_submit_at"] = ts
        _mutate_stats(self.db, sid, upd)
        return [f"train:{new_chars}"]


# ---------------------------------------------------------------------------------------------------------------
# 수집 작업 결과 반영
# ---------------------------------------------------------------------------------------------------------------
def record_collect(app, session_id: int, stats: dict, now: datetime | None = None) -> None:
    """수집 작업(session_id가 있는)이 끝나면 H의 작업 래퍼가 호출: 페이지/문서/글자 수를 세션 통계에 더한다."""
    if session_id is None:
        return
    st_in = stats if isinstance(stats, dict) else {}
    pages = int(st_in.get("visited") or 0)
    new, updated = int(st_in.get("new") or 0), int(st_in.get("updated") or 0)
    ts = _iso(_aware(now))
    db = app.db
    s = db.get_session(int(session_id))
    if not s:
        return
    chars = int(db.chars_since(s.get("started_at")))

    def upd(st):
        st["pages_collected"] = int(st.get("pages_collected") or 0) + pages
        st["docs_new"] = int(st.get("docs_new") or 0) + new
        st["docs_updated"] = int(st.get("docs_updated") or 0) + updated
        st["chars_new"] = max(int(st.get("chars_new") or 0), chars)
        st["collect_errors"] = int(st.get("collect_errors") or 0) + len(st_in.get("errors") or [])
        if pages:
            cutoff = _aware(now) - timedelta(hours=2)
            hp = [h for h in st["hour_pages"] if (_parse(h.get("ts")) or cutoff) > cutoff]
            hp.append({"ts": ts, "n": pages})
            st["hour_pages"] = hp
        st["last_collect_result"] = {"ts": ts, "pages": pages, "new": new, "updated": updated,
                                     "unchanged": int(st_in.get("unchanged") or 0),
                                     "skipped": int(st_in.get("skipped") or 0),
                                     "errors": len(st_in.get("errors") or []), "stopped": st_in.get("stopped")}
    _mutate_stats(db, int(session_id), upd)


# ---------------------------------------------------------------------------------------------------------------
# 학습 작업 (autolearn_train)
# ---------------------------------------------------------------------------------------------------------------
def _new_model_dir(models_dir: Path, prefix: str) -> tuple[str, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for i in range(1, 1000):
        name = f"{prefix}-{stamp}" + (f"-{i}" if i > 1 else "")
        p = models_dir / name
        try:
            p.mkdir(parents=True)
        except FileExistsError:
            continue
        return name, p
    raise RuntimeError("모델 폴더 이름을 만들 수 없습니다.")


def _latest_session_model(db, stats: dict, kind: str, accept=None) -> dict | None:
    by_id = {m["id"]: m for m in db.list_models()}
    for mid in reversed(stats.get("models") or []):
        m = by_id.get(mid)
        if not m or m.get("kind") != kind or not Path(m.get("path") or "").is_dir():
            continue
        if accept is None or accept(m):
            return m
    return None


def _tiny_arch_of(model: dict) -> dict | None:
    arch = (model.get("info") or {}).get("arch")
    if isinstance(arch, dict) and all(k in arch for k in ARCH_KEYS):
        return arch
    try:
        return _mod("tinylm").model_arch(model["path"])
    except Exception:
        return None


def _same_arch(a: dict | None, b: dict | None) -> bool:
    if not a or not b:
        return False
    try:
        return all(int(a[k]) == int(b[k]) for k in ARCH_KEYS)
    except (KeyError, TypeError, ValueError):
        return False


def _warn_once(db, session_id: int, target: str, msg: str) -> None:
    key = f"{target}:{msg[:120]}"
    fresh = {"v": False}

    def upd(st):
        warned = st.get("warned") if isinstance(st.get("warned"), list) else []
        if key not in warned:
            warned.append(key)
            fresh["v"] = True
        st["warned"] = warned[-50:]
    _mutate_stats(db, session_id, upd)
    if fresh["v"]:
        db.notify("warning", f"자율 학습: {TARGET_LABELS.get(target, target)} 학습 실패", msg[:1000],
                  {"session_id": session_id, "target": target, "kind": "autolearn_warning"})


def _threads(conf: dict) -> int | None:
    try:
        t = int(conf.get("threads") or 0)
    except (TypeError, ValueError):
        t = 0
    return t if t > 0 else None


def run_autolearn_train(app, session_id: int, progress=_noop_progress, check_cancel=_noop_cancel) -> dict:
    """autolearn_train 작업 본문. 한 대상의 실패가 다른 대상을 막지 않는다 (오류는 결과와 경고 알림에 기록)."""
    cfg, db = app.cfg, app.db
    s = db.get_session(int(session_id))
    if not s:
        raise ValueError("자율 학습 세션을 찾을 수 없습니다.")
    sid = s["id"]
    result = {"session_id": sid, "models": [], "steps": {}, "eval": {}, "errors": {}, "skipped": {}}
    if s["status"] not in ("running", "finishing"):
        result["skipped"]["all"] = "자율 학습 세션이 이미 끝났습니다."
        return result
    conf = _session_conf(cfg, s)
    targets = {k: _bool((conf.get("targets") or {}).get(k)) for k in TRAIN_TARGETS}
    started = now_iso()
    _mutate_stats(db, sid, lambda st: st.update(last_train_at=started))  # 이 시각 이후 자료가 다음 학습의 '새 자료'

    weights = {"corpus": 0.3 if (targets["tiny"] or targets["lora"]) else 0.0, "tiny": 4.0 * targets["tiny"],
               "qa": 1.0 * targets["qa"], "lora": 4.0 * targets["lora"], "embeddings": 1.0 * targets["embeddings"]}
    total_w = sum(weights.values()) or 1.0
    spans, acc = {}, 0.0
    for k in ("corpus", "tiny", "qa", "lora", "embeddings"):
        spans[k] = (acc / total_w, (acc + weights[k]) / total_w)
        acc += weights[k]
    history: list[dict] = []
    models_added: list[int] = []

    def fail(target: str, e: BaseException) -> None:
        if _is_cancel(e):
            raise e
        msg = f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e)
        result["errors"][target] = msg[:500]
        _warn_once(db, sid, target, msg)

    # ---- 말뭉치 ----
    corpus = None
    if targets["tiny"] or targets["lora"]:
        check_cancel()
        progress(spans["corpus"][0], "학습용 말뭉치 만드는 중")
        try:
            from .dataset import build_corpus

            corpus = build_corpus(db, cfg.datasets_dir)
            result["corpus"] = corpus
        except Exception as e:
            if _is_cancel(e):
                raise
            for t in ("tiny", "lora"):
                if targets[t]:
                    fail(t, e)
    stats0 = _norm_stats((db.get_session(sid) or s).get("stats"))
    train_path = f"{corpus['dir']}/train.jsonl" if corpus else None
    eval_path = f"{corpus['dir']}/eval.jsonl" if corpus else None

    # ---- 소형 AI 모델 ----
    if targets["tiny"] and corpus:
        check_cancel()
        out = None
        try:
            tinylm = _mod("tinylm")
            arch = conf.get("tiny_arch")
            if not isinstance(arch, dict):
                arch = tinylm.resolve_arch(conf.get("tiny_preset") or "auto",
                                           recommended=(conf.get("recommended") or {}).get("tiny_preset") or "mini")
            parent = _latest_session_model(db, stats0, "tiny", accept=lambda m: _same_arch(_tiny_arch_of(m), arch))
            name, out = _new_model_dir(cfg.models_dir, "tiny-auto")
            lo, hi = spans["tiny"]
            progress(lo, "소형 AI 모델 " + ("이어 학습" if parent else "새로 학습") + " 시작")
            info = tinylm.train(train_path, eval_path, out, dict(arch),
                                steps=int(conf.get("steps_per_cycle") or 300),
                                batch_size=int(conf.get("tiny_batch_size") or 32),
                                lr=float(conf.get("tiny_lr") or 3e-3),
                                resume_from=parent["path"] if parent else None, threads=_threads(conf),
                                progress=_sub(progress, lo, hi), check_cancel=check_cancel)
            info = dict(info or {})
            info.update(kind="tiny", session_id=sid, autolearn=True, corpus=corpus)
            mid = db.add_model("tiny", name, str(out), info)
            models_added.append(mid)
            result["steps"]["tiny"] = info.get("steps")
            result["eval"]["tiny"] = {"before": info.get("eval_loss_before"), "after": info.get("eval_loss_after"),
                                      "model_id": mid, "parent": parent["id"] if parent else None}
            history.append({"ts": now_iso(), "model_id": mid, "kind": "tiny",
                            "eval_loss_before": info.get("eval_loss_before"),
                            "eval_loss_after": info.get("eval_loss_after")})
            out = None
        except Exception as e:
            if out is not None:
                shutil.rmtree(out, ignore_errors=True)
            fail("tiny", e)

    # ---- Q&A 학습 데이터 (LoRA 전에) ----
    if targets["qa"]:
        check_cancel()
        lo, hi = spans["qa"]
        try:
            online = _mod("online")
            st = online.status(cfg)
            if not st.get("available"):
                result["skipped"]["qa"] = st.get("reason") or "온라인 모델을 사용할 수 없습니다."
            else:
                progress(lo, "Q&A 학습 데이터 만드는 중 (온라인 모델)")
                result["qa"] = online.build_qa_dataset(
                    cfg, db, cfg.datasets_dir / "qa.jsonl", limit_chunks=int(conf.get("qa_chunks_per_cycle") or 20),
                    per_chunk=int(conf.get("qa_per_chunk") or 2), progress=_sub(progress, lo, hi),
                    check_cancel=check_cancel)
        except Exception as e:
            fail("qa", e)

    # ---- LoRA ----
    if targets["lora"] and corpus:
        check_cancel()
        out = None
        try:
            train_pc = _mod("train_pc")
            lt = dict(cfg.get("lora_training") or {})
            base_model = str(conf.get("lora_base_model") or lt.get("base_model") or "")
            parent = _latest_session_model(
                db, stats0, "lora",
                accept=lambda m: str((m.get("info") or {}).get("base_model") or "") == base_model)
            qa_file = cfg.datasets_dir / "qa.jsonl"
            qa_path = str(qa_file) if _bool(lt.get("use_qa", True)) and qa_file.is_file() else None
            name, out = _new_model_dir(cfg.models_dir, "lora-auto")
            lo, hi = spans["lora"]
            progress(lo, "LoRA 학습 시작")
            info = train_pc.train_lora(
                train_path, eval_path, out, base_model,
                max_steps=int(conf.get("lora_steps_per_cycle") or lt.get("max_steps") or 200),
                lr=float(lt.get("lr") or 2e-4), seq_len=int(lt.get("seq_len") or 512),
                batch_size=int(lt.get("batch_size") or 2), grad_accum=int(lt.get("grad_accum") or 4),
                lora_r=int(lt.get("lora_r") or 8), lora_alpha=int(lt.get("lora_alpha") or 16), qa_path=qa_path,
                resume_from=parent["path"] if parent else None, threads=_threads(conf),
                progress=_sub(progress, lo, hi), check_cancel=check_cancel)
            info = dict(info or {})
            info.update(kind="lora", session_id=sid, autolearn=True, corpus=corpus)
            mid = db.add_model("lora", name, str(out), info)
            models_added.append(mid)
            result["steps"]["lora"] = info.get("steps")
            result["eval"]["lora"] = {"before": info.get("eval_loss_before"), "after": info.get("eval_loss_after"),
                                      "model_id": mid, "parent": parent["id"] if parent else None}
            history.append({"ts": now_iso(), "model_id": mid, "kind": "lora",
                            "eval_loss_before": info.get("eval_loss_before"),
                            "eval_loss_after": info.get("eval_loss_after")})
            out = None
        except Exception as e:
            if out is not None:
                shutil.rmtree(out, ignore_errors=True)
            fail("lora", e)

    # ---- 의미 검색 색인 ----
    if targets["embeddings"]:
        check_cancel()
        lo, hi = spans["embeddings"]
        try:
            progress(lo, "의미 검색 색인 만드는 중")
            result["embeddings"] = _mod("search").index_embeddings(cfg, db, progress=_sub(progress, lo, hi),
                                                           check_cancel=check_cancel)
        except Exception as e:
            fail("embeddings", e)

    result["models"] = models_added
    chars = int(db.chars_since(s.get("started_at")))
    old_corpus = {"v": None}

    def upd(st):
        st["trainings"] = int(st.get("trainings") or 0) + 1
        st["models"] = list(st.get("models") or []) + models_added
        st["train_history"] = (list(st.get("train_history") or []) + history)[-500:]
        st["chars_new"] = max(int(st.get("chars_new") or 0), chars)
        st["last_train_result"] = {"ts": now_iso(), "models": models_added, "errors": result["errors"],
                                   "skipped": result["skipped"]}
        if corpus:
            old_corpus["v"] = st.get("corpus_dir")
            st["corpus_dir"] = corpus["dir"]
    _mutate_stats(db, sid, upd)
    # 이전 주기에 만든 말뭉치는 지운다 (매 주기 전체 사본이 쌓이지 않게). 이 모듈이 만든 datasets 안의 폴더만.
    prev = old_corpus["v"]
    if prev and corpus and prev != corpus["dir"]:
        p = Path(prev)
        try:
            if p.is_dir() and p.resolve().parent == Path(cfg.datasets_dir).resolve() and p.name.startswith("corpus-"):
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass

    if models_added or result.get("embeddings") or result.get("qa"):
        parts = []
        for k in ("tiny", "lora"):
            if k in result["eval"]:
                e = result["eval"][k]
                parts.append(f"{TARGET_LABELS[k]} {result['steps'].get(k)}단계 · 평가 loss "
                             f"{_fmt(e.get('before'))} → {_fmt(e.get('after'))}")
        if result.get("embeddings"):
            parts.append(f"의미 검색 색인 {result['embeddings'].get('indexed', 0)}개 추가")
        if result.get("qa"):
            parts.append(f"Q&A {result['qa'].get('added', 0)}개 추가")
        if result["errors"]:
            parts.append("실패: " + ", ".join(TARGET_LABELS.get(k, k) for k in result["errors"]))
        db.notify("success", "자율 학습: 새 자료로 학습했습니다", "\n".join(parts),
                  {"session_id": sid, "models": models_added, "kind": "autolearn_train"})
    progress(1.0, "자율 학습 완료" + (f" (실패 {len(result['errors'])}개)" if result["errors"] else ""))
    return result


def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "-"


# ---------------------------------------------------------------------------------------------------------------
# 학습 종료 (finalize_learning)
# ---------------------------------------------------------------------------------------------------------------
def _copy_to_target(cfg, res: dict, target: str) -> str:
    from .export import validate_target

    tdir = validate_target(cfg, target)
    src_dir, src_zip = Path(res["dir"]), Path(res["zip"])
    need = int(res.get("total_bytes") or 0) + int(res.get("zip_bytes") or 0)
    free = shutil.disk_usage(tdir).free
    if need * 1.05 > free:
        raise ValueError(f"공간 부족: 필요 약 {need / 1e9:.2f}GB, 남은 공간 {free / 1e9:.2f}GB")
    dst = tdir / src_dir.name
    shutil.copytree(src_dir, dst)
    shutil.copy2(src_zip, tdir / src_zip.name)
    return str(dst)


def run_finalize(app, session_id: int, progress=_noop_progress, check_cancel=_noop_cancel) -> dict:
    """finalize_learning 작업 본문: 마지막 학습 → 결과 파일 내보내기 → (선택) 외장 저장장치 복사 → 세션 종료 + 알림."""
    results = _mod("results")
    cfg, db = app.cfg, app.db
    s = db.get_session(int(session_id))
    if not s:
        raise ValueError("자율 학습 세션을 찾을 수 없습니다.")
    sid = s["id"]
    if s["status"] not in ("running", "finishing"):
        raise ValueError("이미 끝난 자율 학습 세션입니다.")
    if s["status"] == "running":
        db.update_session(sid, status="finishing")
    actor = s.get("actor") or "autolearn"
    try:
        conf = _session_conf(cfg, s)
        stats = _norm_stats(s.get("stats"))
        targets = conf.get("targets") or {}
        final_train = None
        lo = 0.0
        if _bool(conf.get("final_train")) and any(_bool(targets.get(k)) for k in TRAIN_TARGETS):
            new_chars = int(db.chars_since(_train_baseline(db, stats)))
            if new_chars > 0:
                lo = 0.6
                progress(0.0, f"마지막 학습 중 (새 자료 {new_chars:,}자)")
                try:
                    final_train = run_autolearn_train(app, sid, progress=_sub(progress, 0.0, lo),
                                                      check_cancel=check_cancel)
                except Exception as e:
                    if _is_cancel(e):
                        raise
                    final_train = {"error": f"{type(e).__name__}: {e}"[:500]}
                    db.notify("warning", "학습 종료: 마지막 학습 실패", final_train["error"],
                              {"session_id": sid, "kind": "autolearn_warning"})
            else:
                final_train = {"skipped": "새 자료가 없어 마지막 학습을 건너뛰었습니다."}
        check_cancel()
        s = db.get_session(sid) or s
        stats = _norm_stats(s.get("stats"))
        model_ids = list(dict.fromkeys(int(i) for i in stats.get("models") or []))
        note = None
        if not model_ids:  # 이번 세션에 학습한 모델이 없으면 종류별 최신 모델을 담는다
            latest: dict[str, int] = {}
            for m in db.list_models():  # 최신순
                if m.get("kind") in ("tiny", "lora") and m["kind"] not in latest and Path(m.get("path") or "").is_dir():
                    latest[m["kind"]] = m["id"]
            model_ids = list(latest.values())
            if model_ids:
                note = "이번 세션에서 학습한 모델이 없어 가장 최근 모델을 담았습니다."
        sess_for_report = dict(s)
        sess_for_report["ended_at"] = now_iso()
        res = results.export_results(cfg, db, session=sess_for_report, model_ids=model_ids,
                                     formats=conf.get("formats"), progress=_sub(progress, lo, 0.95),
                                     check_cancel=check_cancel)
        if note:
            res["summary"].setdefault("notes", []).append(note)

        copied_to, copy_error = None, None
        target = str(conf.get("results_export_target") or "").strip()
        if target:
            progress(0.96, f"외장 저장장치로 복사 중: {target}")
            try:
                copied_to = _copy_to_target(cfg, res, target)
            except Exception as e:
                copy_error = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
                db.notify("warning", "학습 결과를 외장 저장장치에 복사하지 못했습니다", copy_error[:1000],
                          {"session_id": sid, "result": res["name"], "kind": "autolearn_warning"})

        ended = now_iso()
        files_display = results.display_files(res["files"])
        session_result = {"name": res["name"], "dir": res["dir"], "zip": res["zip"], "zip_bytes": res["zip_bytes"],
                          "total_bytes": res["total_bytes"], "file_count": len(res["files"]),
                          "files": files_display, "summary": res["summary"], "copied_to": copied_to,
                          "copy_error": copy_error, "final_train": final_train}
        with _STATS_LOCK:
            db.update_session(sid, status="ended", ended_at=ended, result=session_result)
        body = (f"결과 폴더: {res['dir']}\n"
                f"ZIP: {res['zip']} ({results.fmt_bytes(res['zip_bytes'])})\n"
                f"파일 {len(res['files'])}개 · 전체 {results.fmt_bytes(res['total_bytes'])}")
        if copied_to:
            body += f"\n외장 저장장치 복사: {copied_to}"
        db.notify("success", "학습이 종료되었습니다", body,
                  {"result": res["name"], "session_id": sid, "kind": "learning_end"})
        db.audit(actor, "autolearn_finalize", {"session_id": sid, "result": res["name"],
                                               "files": len(res["files"]), "bytes": res["total_bytes"],
                                               "copied_to": copied_to})
        progress(1.0, f"학습 종료 · 결과 {res['name']}")
        out = dict(res)
        out.update(session_id=sid, final_train=final_train, copied_to=copied_to, copy_error=copy_error,
                   file_count=len(res["files"]))
        return out
    except BaseException as e:
        cancelled = _is_cancel(e)
        msg = "취소됨" if cancelled else f"{type(e).__name__}: {e}"[:500]
        try:
            with _STATS_LOCK:
                db.update_session(sid, status="cancelled" if cancelled else "failed", ended_at=now_iso(),
                                  result={"error": msg})
            db.notify("warning" if cancelled else "error",
                      "학습 종료가 취소되었습니다" if cancelled else "학습 종료 중 오류가 발생했습니다",
                      "" if cancelled else msg, {"session_id": sid, "kind": "learning_end_failed"})
            db.audit(actor, "autolearn_finalize_failed", {"session_id": sid, "error": msg})
        except Exception:
            traceback.print_exc()
        raise
