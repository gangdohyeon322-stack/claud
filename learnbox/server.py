"""HTTP 서버: JSON API + 공유 웹 UI. PC 브라우저와 휴대폰이 같은 화면/기능/데이터를 쓴다."""
from __future__ import annotations

import copy
import json
import mimetypes
import re
import secrets
import shutil
import socket
import ssl
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlsplit

from . import ingest, results, search, tinylm
from .app import App, LbwInvalid
from .config import MASK, amp_value, device_value
from .db import now_iso
from .export import list_drives, validate_target
from .jobs import missing_package_message
from .online import OnlineError

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/onboard.js": "onboard.js", "/trainer.js": "trainer.js",
          "/style.css": "style.css", "/sw.js": "sw.js", "/manifest.webmanifest": "manifest.webmanifest",
          "/icon.svg": "icon.svg"}
# 휴대폰/브라우저가 받을 수 있는 소형 AI 모델 파일 (model.pt는 PyTorch 체크포인트라 내려주지 않는다)
MOBILE_FILES = ("model.onnx", "model.int8.onnx", "vocab.json", "info.json", "model.lbw")
SESSION_HOURS = 12
MAX_BODY = 1_000_000
MAX_DRAIN = 256 * 1024 * 1024      # 거절한 업로드를 끝까지 읽어 버릴 때의 상한 (연결이 끊겨 오류가 가려지는 것을 막기 위함)
MAX_SNAPSHOT_CHUNKS = 200_000
DEFAULT_SNAPSHOT_CHUNKS = 6000
FILE_TYPES = {".zip": "application/zip", ".onnx": "application/octet-stream", ".lbw": "application/octet-stream", ".json": "application/json",
              ".jsonl": "application/x-ndjson", ".csv": "text/csv", ".md": "text/markdown", ".txt": "text/plain",
              ".db": "application/octet-stream", ".pt": "application/octet-stream",
              ".safetensors": "application/octet-stream", ".webmanifest": "application/manifest+json"}
# 온라인 모델 오류 -> HTTP 상태. (상위 API의 401/403을 그대로 돌려주면 웹 화면이 로그아웃되므로 502로 바꾼다)
ONLINE_STATUS = {"disabled": 503, "no_sdk": 503, "no_key": 503, "credentials": 503, "rate_limit": 429,
                 "bad_request": 400, "too_large": 400}


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Sessions:
    def __init__(self):
        self._s: dict[str, tuple[str, float]] = {}
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def create(self, role: str) -> str:
        tok = secrets.token_urlsafe(32)
        with self._lock:
            self._s[tok] = (role, time.time() + SESSION_HOURS * 3600)
        return tok

    def role(self, tok: str | None) -> str | None:
        if not tok:
            return None
        with self._lock:
            v = self._s.get(tok)
            if not v or v[1] < time.time():
                self._s.pop(tok, None)
                return None
            return v[0]

    def drop(self, tok: str) -> None:
        with self._lock:
            self._s.pop(tok, None)

    def locked(self, ip: str) -> bool:
        now = time.time()
        with self._lock:
            fails = [t for t in self._fails.get(ip, []) if now - t < 300]
            self._fails[ip] = fails
            return len(fails) >= 8

    def fail(self, ip: str) -> None:
        with self._lock:
            self._fails.setdefault(ip, []).append(time.time())


def _scrub_settings(body: dict) -> dict:
    """감사 기록에 남길 설정 변경 내용에서 비밀값(API 키)을 뺀다."""
    d = copy.deepcopy(body)
    for sec, key in (("search", "brave_api_key"), ("online", "api_key")):
        if isinstance(d.get(sec), dict) and key in d[sec]:
            d[sec][key] = MASK if d[sec][key] else ""
    return d


def _content_type(path: Path) -> str:
    ct = FILE_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return ct + ("; charset=utf-8" if ct.startswith("text/") or ct in ("application/json", "application/x-ndjson")
                 else "")


def make_handler(app: App, sessions: Sessions):
    routes: list[tuple[str, re.Pattern, str | None, bool, callable]] = []

    def route(method: str, pattern: str, role: str | None = "user", raw: bool = False):
        """role: None(로그인 불필요) | user | admin. raw=True면 JSON 본문을 읽지 않고 핸들러가 직접 읽는다(파일 업로드)."""
        def deco(fn):
            routes.append((method, re.compile(f"^{pattern}$"), role, raw, fn))
            return fn
        return deco

    def need_admin(role):
        if role != "admin":
            raise HttpError(403, "관리자만 할 수 있습니다.")

    def actor(h, role) -> str:
        return f"{role}@{h.client_address[0]}"

    # ---------- 인증 ----------
    @route("GET", "/api/info", None)
    def info(h, m, body, role):
        return {"name": "LearnBox", "version": 2, "initialized": app.cfg.initialized, "role": role}

    @route("POST", "/api/login", None)
    def login(h, m, body, role):
        ip = h.client_address[0]
        if sessions.locked(ip):
            raise HttpError(429, "로그인 실패가 많습니다. 5분 뒤 다시 시도하세요.")
        r = app.cfg.check_pin(str(body.get("pin", "")))
        if not r:
            sessions.fail(ip)
            time.sleep(0.5)
            raise HttpError(401, "PIN이 올바르지 않습니다.")
        app.db.audit(r, "login", {"ip": ip})
        return {"token": sessions.create(r), "role": r}

    @route("POST", "/api/logout")
    def logout(h, m, body, role):
        sessions.drop(h.token)
        return {"ok": True}

    # ---------- 조회 ----------
    @route("GET", "/api/status")
    def status(h, m, body, role):
        return {**app.status(), "role": role}

    @route("GET", "/api/device")
    def device(h, m, body, role):
        return app.device_info()

    @route("POST", "/api/estimate")
    def estimate(h, m, body, role):
        return app.estimate(body)

    @route("GET", "/api/search")
    def search_route(h, m, body, role):
        q = h.qstr("q").strip()
        limit = h.qint("limit", 10, 1, 50)
        offset = h.qint("offset", 0, 0, 100000)
        sources = [s.strip() for s in h.qstr("sources").split(",") if s.strip()] or None
        mode = h.qstr("mode", "hybrid")
        if mode not in ("hybrid", "keyword", "semantic"):
            raise HttpError(400, "mode는 hybrid, keyword, semantic 중 하나여야 합니다.")
        doc_id = h.qint("doc_id", 0, 0, 2**31) or None
        if not q:
            return {"results": [], "mode_used": mode, "total": 0, "offset": offset, "limit": limit, "extract": []}
        return search.search(app.cfg, app.db, q, limit=limit, offset=offset, sources=sources, doc_id=doc_id,
                             mode=mode)

    @route("POST", "/api/ask")
    def ask(h, m, body, role):
        q = str(body.get("q", "")).strip()
        if not q:
            raise HttpError(400, "질문을 입력하세요.")
        sources = body.get("sources")
        if sources is not None and not (isinstance(sources, list) and all(isinstance(s, str) for s in sources)):
            raise HttpError(400, "sources는 문자열 목록이어야 합니다.")
        history = body.get("history")
        if history is not None:
            if not isinstance(history, list):
                raise HttpError(400, "history는 목록이어야 합니다.")
            history = [{"role": h.get("role"), "content": str(h.get("content", ""))[:4000]}
                       for h in history[-12:] if isinstance(h, dict)]
        try:
            return app.ask(q, body.get("engine") or "search", body.get("model_id"), sources, history=history)
        except OnlineError as e:
            raise HttpError(ONLINE_STATUS.get(e.kind, 502), str(e))

    @route("POST", "/api/generate")
    def generate(h, m, body, role):
        prompt = str(body.get("prompt", ""))
        if not prompt.strip():
            raise HttpError(400, "시작 문장을 입력하세요.")
        try:
            mdl = app.db.get_model(int(body.get("model_id")))
        except (TypeError, ValueError):
            mdl = None
        if not mdl:
            raise HttpError(404, "모델을 찾을 수 없습니다.")
        if mdl["kind"] not in ("tiny", "mobile"):
            raise HttpError(400, "소형 AI 모델만 글을 이어 쓸 수 있습니다. LoRA 모델은 [검색] 탭에서 질문하세요.")
        if not Path(mdl["path"]).is_dir():
            raise HttpError(404, "모델 폴더가 없습니다.")
        try:
            max_new = max(1, min(int(body.get("max_new") or 120), 500))
            temperature = float(body.get("temperature", 0.8))
            top_k = max(0, int(body.get("top_k", 20)))
        except (TypeError, ValueError):
            raise HttpError(400, "max_new, temperature, top_k는 숫자여야 합니다.")
        return {"text": tinylm.generate(mdl["path"], prompt[:2000], max_new=max_new, temperature=temperature,
                                        top_k=top_k), "model": mdl["name"]}

    @route("GET", "/api/documents")
    def documents(h, m, body, role):
        off = h.qint("offset", 0, 0, 10**9)
        limit = h.qint("limit", 50, 1, 200)
        source = h.qstr("source") or None
        return {"documents": app.db.list_documents(limit, off, source), "counts": app.db.counts()}

    @route("GET", r"/api/documents/(\d+)")
    def document(h, m, body, role):
        d = search.doc_view(app.db, int(m.group(1)), h.qstr("q") or None)
        if not d:
            raise HttpError(404, "문서를 찾을 수 없습니다.")
        return d

    @route("DELETE", r"/api/documents/(\d+)", "admin")
    def delete_document(h, m, body, role):
        app.db.delete_document(int(m.group(1)))
        app.db.audit(role, "delete_document", {"id": int(m.group(1))})
        return {"ok": True}

    @route("GET", "/api/snapshot")
    def snapshot(h, m, body, role):
        n = h.qint("max_chunks", DEFAULT_SNAPSHOT_CHUNKS, 1, MAX_SNAPSHOT_CHUNKS)
        return {"chunks": app.db.chunk_snapshot(n), "created_at": now_iso()}

    # ---------- 파일 가져오기 ----------
    @route("POST", "/api/files", "user", raw=True)
    def upload(h, m, body, role):
        name = re.sub(r"[\x00-\x1f]", "", unquote(h.headers.get("X-File-Name", "") or ""))
        name = name.replace("\\", "/").rsplit("/", 1)[-1].strip()   # 경로는 버리고 파일 이름만
        if not name:
            raise HttpError(400, "파일 이름이 없습니다 (X-File-Name 헤더).")
        ext = Path(name).suffix.lower()
        if ext not in ingest.SUPPORTED:
            raise HttpError(400, f"지원하지 않는 파일 형식입니다: {ext or '(확장자 없음)'} "
                                 f"(지원 형식: {', '.join(sorted(ingest.SUPPORTED))})")
        max_mb = float(app.cfg["ingest"].get("max_file_mb", 50) or 50)
        raw_len = h.headers.get("Content-Length")
        if raw_len is None:
            raise HttpError(411, "Content-Length 헤더가 필요합니다.")
        try:
            n = int(raw_len)
        except ValueError:
            raise HttpError(400, "Content-Length가 올바르지 않습니다.")
        if n <= 0:
            raise HttpError(400, "빈 파일입니다.")
        if n > max_mb * 1024 * 1024:    # 본문을 읽기 전에 거절한다
            raise HttpError(413, f"파일이 너무 큽니다 ({n / 1024 / 1024:.1f}MB). 최대 {max_mb:g}MB까지 가져올 수 있습니다.")
        data = h.read_exact(n)
        if len(data) < n:
            raise HttpError(400, "파일을 끝까지 받지 못했습니다.")
        return ingest.ingest_bytes(app.cfg, app.db, name, data, actor=role)

    @route("GET", "/api/files")
    def files(h, m, body, role):
        return {"files": app.db.list_files()}

    @route("POST", "/api/ingest/path", "admin")
    def ingest_path_route(h, m, body, role):
        path = str(body.get("path", "")).strip()
        if not path:
            raise HttpError(400, "가져올 파일이나 폴더 경로를 입력하세요.")
        p = Path(path).expanduser()
        if not p.exists():
            raise HttpError(400, f"경로를 찾을 수 없습니다: {path}")
        resolved = p.resolve()
        if resolved == app.cfg.data_dir or app.cfg.data_dir in resolved.parents:
            raise HttpError(400, "LearnBox 데이터 폴더 안의 파일은 가져올 수 없습니다. 다른 폴더의 파일을 고르세요.")
        return {"job_id": app.jobs.submit("ingest_path", {"path": str(p), "actor": actor(h, role)}, role)}

    # ---------- 수집 ----------
    def gate_or_403():
        g = app.gate()
        if not g["allowed"]:
            raise HttpError(403, f"지금은 수집 허용 시간이 아닙니다 ({g['windows']}). 다음 허용: {g['next_allowed']}")

    @route("POST", "/api/collect/links")
    def collect_links(h, m, body, role):
        gate_or_403()
        urls = [u.strip() for u in body.get("urls", []) if isinstance(u, str) and u.strip()][:50]
        if not urls:
            raise HttpError(400, "링크를 입력하세요.")
        for u in urls:
            if not u.startswith(("http://", "https://")):
                raise HttpError(400, f"http(s) 주소가 아닙니다: {u}")
        p = {"urls": urls, "depth": min(int(body.get("depth", 0)), 2), "same_domain": bool(body.get("same_domain", True))}
        return {"job_id": app.jobs.submit("collect_links", p, role)}

    @route("POST", "/api/collect/search")
    def collect_search(h, m, body, role):
        gate_or_403()
        q = str(body.get("query", "")).strip()
        if not q:
            raise HttpError(400, "검색어를 입력하세요.")
        return {"job_id": app.jobs.submit("collect_search", {"query": q, "limit": body.get("limit")}, role)}

    @route("GET", "/api/watch")
    def watch_list(h, m, body, role):
        return {"items": app.db.list_watch(), "auto_collect": app.cfg["auto_collect"]}

    @route("POST", "/api/watch")
    def watch_add(h, m, body, role):
        kind = body.get("kind")
        value = str(body.get("value", "")).strip()
        if kind not in ("link", "search") or not value:
            raise HttpError(400, "종류(link/search)와 값을 입력하세요.")
        if kind == "link" and not value.startswith(("http://", "https://")):
            raise HttpError(400, "http(s) 주소가 아닙니다.")
        app.db.add_watch(kind, value, min(int(body.get("depth", 0)), 2))
        return {"ok": True}

    @route("DELETE", r"/api/watch/(\d+)")
    def watch_del(h, m, body, role):
        app.db.remove_watch(int(m.group(1)))
        return {"ok": True}

    # ---------- 작업 ----------
    @route("GET", "/api/jobs")
    def jobs(h, m, body, role):
        # 학습 작업에는 checkpoint 요약과 can_pause/can_resume/can_finish/can_restart가 붙는다 (화면이 버튼을 고름)
        return {"jobs": app.list_jobs(40)}

    def job_or_404(m) -> dict:
        job = app.db.get_job(int(m.group(1)))
        if not job:
            raise HttpError(404, "작업이 없습니다.")
        return job

    @route("POST", r"/api/jobs/(\d+)/cancel")
    def cancel(h, m, body, role):
        job = job_or_404(m)
        if role != "admin" and not job["kind"].startswith("collect"):
            raise HttpError(403, "수집이 아닌 작업은 관리자만 취소할 수 있습니다.")
        app.cancel_job(job["id"])         # 일시정지 중인 학습도 깨워서 취소한다
        return {"ok": True}

    # ----- 학습 일시정지 · 재개 · 지금까지 저장하고 끝내기 · 이어서 재개 · 체크포인트 삭제 (관리자) -----
    def job_reply(job_id: int) -> dict:
        return {"ok": True, "job": app.annotate_job(app.db.get_job(job_id))}

    @route("POST", r"/api/jobs/(\d+)/pause", "admin")
    def job_pause(h, m, body, role):
        job = job_or_404(m)
        app.jobs.pause(job["id"])         # 학습 작업이 아니거나 실행 중이 아니면 한국어 ValueError -> 400
        app.db.audit(actor(h, role), "job_pause", {"job_id": job["id"]})
        return job_reply(job["id"])

    @route("POST", r"/api/jobs/(\d+)/resume", "admin")
    def job_resume(h, m, body, role):
        job = job_or_404(m)
        app.jobs.resume(job["id"])
        app.db.audit(actor(h, role), "job_resume", {"job_id": job["id"]})
        return job_reply(job["id"])

    @route("POST", r"/api/jobs/(\d+)/finish", "admin")
    def job_finish(h, m, body, role):
        job = job_or_404(m)
        app.jobs.finish(job["id"])
        app.db.audit(actor(h, role), "job_finish", {"job_id": job["id"]})
        return job_reply(job["id"])

    @route("POST", r"/api/jobs/(\d+)/restart", "admin")
    def job_restart(h, m, body, role):
        job = job_or_404(m)
        if job["kind"] in ("train_tiny", "train_mobile"):
            need_pkgs("torch")
        elif job["kind"] in ("train_lora", "train_pc"):
            need_pkgs("torch", "transformers", "peft")
        # device(선택): 이어서 할 장치. 없으면 지금 장치에서 재개 (원래 작업의 device는 이어받지 않음, PRD 6.9)
        dev = body.get("device") if isinstance(body, dict) else None
        amp = body.get("amp") if isinstance(body, dict) else None   # (선택) LoRA 정밀도를 바꿔 이어서
        return {"job_id": app.restart_job(job["id"], actor(h, role), device=dev, amp=amp)}

    @route("POST", r"/api/jobs/(\d+)/discard-checkpoint", "admin")
    def job_discard_checkpoint(h, m, body, role):
        job = job_or_404(m)
        return app.discard_checkpoint(job["id"], actor(h, role))

    # ---------- 학습 (관리자) ----------
    def need_pkgs(*mods: str):
        caps = app.capabilities()
        if not all(caps.get(x) for x in mods):
            raise HttpError(400, "학습 패키지가 없습니다: pip install -r requirements-train.txt "
                                 f"(필요: {', '.join(mods)})")

    def train_params(body: dict) -> dict:
        """학습 요청 본문 -> 작업 params. 체크포인트 경로(resume_checkpoint)는 '이어서 재개'(restart)만 정한다."""
        p = {k: v for k, v in body.items() if k not in ("resume_checkpoint", "resumed_from_job")}
        if p.get("checkpoint_minutes") not in (None, ""):
            try:
                v = float(p["checkpoint_minutes"])
            except (TypeError, ValueError):
                raise HttpError(400, "중간 저장 간격(checkpoint_minutes)은 숫자여야 합니다.")
            if not v >= 0:
                raise HttpError(400, "중간 저장 간격(checkpoint_minutes)은 0 이상이어야 합니다.")
        return p

    def store_normalized(p: dict, keys: tuple[str, ...]) -> None:
        """요청에 들어 있던 device/amp만 정식 이름으로 바꿔 작업 params에 남긴다 (주지 않았으면 params 그대로)."""
        if "device" in keys and "device" in p:
            p["device"] = device_value(p["device"], "device")
        if "amp" in keys and "amp" in p:
            p["amp"] = amp_value(p["amp"], "amp")

    @route("POST", "/api/train/mobile", "admin")   # 이전 이름
    @route("POST", "/api/train/tiny", "admin")
    def train_tiny(h, m, body, role):
        need_pkgs("torch")
        p = train_params(body)
        app.plan_tiny(p)            # 잘못된 값은 작업을 만들기 전에 바로 알려 준다
        store_normalized(p, ("device",))
        return {"job_id": app.jobs.submit("train_tiny", p, role)}

    @route("POST", "/api/train/pc", "admin")       # 이전 이름
    @route("POST", "/api/train/lora", "admin")
    def train_lora(h, m, body, role):
        need_pkgs("torch", "transformers", "peft")
        p = train_params(body)
        app.plan_lora(p)
        store_normalized(p, ("device", "amp"))
        return {"job_id": app.jobs.submit("train_lora", p, role)}

    @route("POST", "/api/index/embeddings", "admin")
    def index_embeddings(h, m, body, role):
        if not app.cfg["embeddings"].get("enabled", True):
            raise HttpError(400, "의미 검색이 꺼져 있습니다. 관리 > 설정 > 의미 검색에서 켜세요.")
        if not search.embeddings_available():
            raise HttpError(400, "의미 검색을 쓰려면 transformers와 torch가 필요합니다: pip install -r requirements-train.txt")
        return {"job_id": app.jobs.submit("index_embeddings", {}, role)}

    @route("POST", "/api/qa/build", "admin")
    def build_qa(h, m, body, role):
        from . import online

        st = online.status(app.cfg)
        if not st["available"]:
            raise HttpError(400, st["reason"] or "온라인 모델을 사용할 수 없습니다.")
        p = {"limit_chunks": body.get("limit_chunks"), "per_chunk": body.get("per_chunk")}
        return {"job_id": app.jobs.submit("build_qa", {k: v for k, v in p.items() if v not in (None, "")}, role)}

    # ---------- 자율 학습 ----------
    @route("GET", "/api/autolearn")
    def autolearn_status(h, m, body, role):
        return app.autolearn.status()

    @route("POST", "/api/autolearn/start", "admin")
    def autolearn_start(h, m, body, role):
        base = app.cfg["autolearn"].get("targets") or {}
        given = body.get("targets") if isinstance(body.get("targets"), dict) else {}
        targets = {k: bool(v) for k, v in {**base, **given}.items()}
        if targets.get("tiny") or targets.get("lora"):
            need_pkgs("torch")
        if targets.get("lora"):
            need_pkgs("transformers", "peft")
        return {"session": app.autolearn.start_session(actor(h, role), body)}

    @route("POST", "/api/autolearn/stop", "admin")
    def autolearn_stop(h, m, body, role):
        return app.autolearn.stop_session(actor(h, role))

    # ---------- 알림 ----------
    @route("GET", "/api/notifications")
    def notifications(h, m, body, role):
        unread = h.qstr("unread").lower() in ("1", "true", "yes")
        return {"notifications": app.db.list_notifications(h.qint("limit", 50, 1, 200), unread),
                "unread": app.db.unread_notifications()}

    @route("POST", "/api/notifications/read")
    def notifications_read(h, m, body, role):
        ids = body.get("ids")
        if ids is None:
            app.db.mark_notifications_read(None)
        else:
            try:
                ids = [int(i) for i in ids]
            except (TypeError, ValueError):
                raise HttpError(400, "ids는 알림 번호 목록이어야 합니다.")
            if ids:   # 빈 목록은 '아무것도 안 함' (db는 빈 목록을 '전체'로 읽는다)
                app.db.mark_notifications_read(ids)
        return {"ok": True}

    # ---------- 학습 결과 파일 (관리자) ----------
    @route("GET", "/api/results", "admin")
    def results_list(h, m, body, role):
        return {"results": results.list_results(app.cfg)}

    def result_path(name: str, rel: str | None) -> Path:
        try:
            return results.result_file_path(app.cfg, unquote(name), unquote(rel) if rel is not None else None)
        except ValueError as e:
            raise HttpError(404, str(e))

    @route("GET", r"/api/results/([^/]+)/download", "admin")
    def results_zip(h, m, body, role):
        p = result_path(m.group(1), None)
        app.db.audit(role, "results_download", {"name": p.stem})
        h.send_file(p, download_name=p.name)

    @route("GET", r"/api/results/([^/]+)/files/(.+)", "admin")
    def results_file(h, m, body, role):
        p = result_path(m.group(1), m.group(2))
        h.send_file(p, download_name=p.name)

    # ---------- 모델 ----------
    @route("GET", "/api/models")
    def models(h, m, body, role):
        return {"models": [{k: v for k, v in x.items() if k != "path"} for x in app.db.list_models()]}

    @route("GET", r"/api/models/(\d+)/files/([\w.\-]+)")
    def model_file(h, m, body, role):
        mdl = app.db.get_model(int(m.group(1)))
        name = m.group(2)
        if not mdl or mdl["kind"] not in ("tiny", "mobile") or name not in MOBILE_FILES:
            raise HttpError(404, "파일이 없습니다.")
        path = Path(mdl["path"]) / name
        if name == "model.lbw" and not path.is_file() and (Path(mdl["path"]) / "model.pt").is_file():
            path = app.ensure_lbw(mdl)     # model.pt에서 만들어 폴더에 캐시 (torch 없으면 ImportError -> 400)
        if not path.is_file():
            raise HttpError(404, "파일이 없습니다.")
        h.send_file(path)

    @route("POST", "/api/models/upload", "admin", raw=True)
    def model_upload(h, m, body, role):
        """휴대폰에서 학습한 가중치(.lbw)를 PC 모델로 등록한다."""
        max_mb = float((app.cfg["mobile_training"] or {}).get("max_upload_mb", 64) or 64)
        raw_len = h.headers.get("Content-Length")
        if raw_len is None:
            raise HttpError(411, "Content-Length 헤더가 필요합니다.")
        try:
            n = int(raw_len)
        except ValueError:
            raise HttpError(400, "Content-Length가 올바르지 않습니다.")
        if n <= 0:
            raise HttpError(400, "빈 파일입니다.")
        if n > max_mb * 1024 * 1024:    # 본문을 읽기 전에 거절한다
            raise HttpError(413, f"파일이 너무 큽니다 ({n / 1024 / 1024:.1f}MB). 최대 {max_mb:g}MB까지 올릴 수 있습니다.")
        data = h.read_exact(n)
        if len(data) < n:
            raise HttpError(400, "파일을 끝까지 받지 못했습니다.")
        hint = re.sub(r"[\x00-\x1f]", "", unquote(h.headers.get("X-Model-Name", "") or "")).strip() or None
        try:
            return app.register_device_model(data, hint, actor(h, role))
        except LbwInvalid as e:
            h._send(400, {"error": str(e), "errors": e.errors})

    @route("POST", "/api/mobile-training/report")
    def mobile_report(h, m, body, role):
        keep = {k: body.get(k) for k in ("device", "arch", "batch_size", "seq_len", "ms_per_step", "steps")}
        app.db.audit(actor(h, role), "mobile_benchmark", keep)
        return {"ok": True}

    # ---------- 설정/내보내기 (관리자) ----------
    def checkpoint_settings(sec) -> dict:
        """설정 '학습 중간 저장'(training_checkpoint) 검사·정리. 잘못된 값은 저장 전에 400."""
        if not isinstance(sec, dict):
            raise HttpError(400, "학습 중간 저장 설정이 올바르지 않습니다.")
        out = {}
        if "enabled" in sec:
            v = sec["enabled"]
            out["enabled"] = v.strip().lower() in ("1", "true", "yes", "on", "y", "예") if isinstance(v, str) else bool(v)
        for key, label, as_int in (("every_minutes", "저장 간격(분)", False), ("every_steps", "단계 간격", True)):
            if key not in sec:
                continue
            raw = sec[key]
            try:
                v = 0.0 if raw in (None, "") else float(raw)
            except (TypeError, ValueError):
                raise HttpError(400, f"학습 중간 저장의 {label}은(는) 숫자여야 합니다.")
            if not v >= 0 or v == float("inf"):
                raise HttpError(400, f"학습 중간 저장의 {label}은(는) 0 이상이어야 합니다.")
            out[key] = int(v) if as_int else (int(v) if v == int(v) else v)
        if "dir" in sec:
            d = str(sec["dir"] or "").strip()
            if d:
                p = Path(d).expanduser()
                if not p.is_absolute():
                    raise HttpError(400, "저장 위치는 전체 경로로 입력하세요 (예: D:\\LearnBox\\checkpoints, /home/me/checkpoints).")
                if p.exists() and not p.is_dir():
                    raise HttpError(400, f"저장 위치가 폴더가 아닙니다: {d}")
            out["dir"] = d
        return out

    @route("GET", "/api/settings", "admin")
    def settings_get(h, m, body, role):
        return app.cfg.public()

    @route("POST", "/api/settings", "admin")
    def settings_set(h, m, body, role):
        if body.get("search", {}).get("brave_api_key") == MASK:
            body["search"].pop("brave_api_key")
        from .timegate import _minutes
        for w in body.get("allowed_windows", []):
            _minutes(w["start"]), _minutes(w["end"])
        target = (body.get("autolearn") or {}).get("results_export_target") if isinstance(body.get("autolearn"), dict) else None
        if target:
            validate_target(app.cfg, str(target))    # 없는 경로/데이터 폴더 안은 저장 전에 알려 준다
        if "training_checkpoint" in body:
            body["training_checkpoint"] = checkpoint_settings(body["training_checkpoint"])
        # 학습 장치·LoRA 정밀도: 정식 이름으로 저장, 잘못된 값은 저장 전에 400 (PRD 6.1, 6.4)
        if "training_device" in body:
            body["training_device"] = device_value(body["training_device"], "학습 장치(training_device)")
        lt = body.get("lora_training")
        if isinstance(lt, dict) and "amp" in lt:
            lt["amp"] = amp_value(lt["amp"], "LoRA 정밀도(lora_training.amp)")
        app.cfg.update(body)
        app.db.audit(role, "settings", _scrub_settings(body))
        return app.cfg.public()

    @route("GET", "/api/drives", "admin")
    def drives(h, m, body, role):
        return {"drives": list_drives()}

    @route("POST", "/api/export", "admin")
    def do_export(h, m, body, role):
        target = str(body.get("target", "")).strip()
        if not target:
            raise HttpError(400, "저장할 위치를 입력하세요.")
        validate_target(app.cfg, target)
        p = {"target": target, "include": body.get("include", []), "actor": f"admin@{h.client_address[0]}"}
        return {"job_id": app.jobs.submit("export", p, role)}

    @route("GET", "/api/audit", "admin")
    def audit(h, m, body, role):
        return {"audit": app.db.list_audit(100)}

    class Handler(BaseHTTPRequestHandler):
        server_version = "LearnBox/2.0"
        token = None
        query: dict = {}
        _body_read = False

        def log_message(self, fmt, *args):
            if "/api/" in (args[0] if args else "") and "status" not in str(args[0]):
                super().log_message(fmt, *args)

        # ----- 질의 문자열 -----
        def qstr(self, key: str, default: str = "") -> str:
            return self.query.get(key, [default])[0]

        def qint(self, key: str, default: int, lo: int, hi: int) -> int:
            raw = self.qstr(key, "")
            if raw == "":
                return default
            try:
                v = int(raw)
            except ValueError:
                raise HttpError(400, f"{key}은(는) 정수여야 합니다.")
            return max(lo, min(hi, v))

        # ----- 응답 -----
        def _send(self, status: int, payload, ctype="application/json; charset=utf-8"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def send_file(self, path: Path, download_name: str | None = None):
            """파일을 조각으로 나눠 보낸다 (ZIP/모델처럼 큰 파일도 메모리에 다 올리지 않는다)."""
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", _content_type(path))
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download_name:
                ascii_name = re.sub(r'[^\w.\-]+', "_", download_name.encode("ascii", "replace").decode()) or "download"
                self.send_header("Content-Disposition",
                                 f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(download_name)}")
            self.end_headers()
            try:
                with open(path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile, 1 << 20)
            except (BrokenPipeError, ConnectionResetError):
                pass

        # ----- 요청 본문 -----
        def read_exact(self, n: int) -> bytes:
            self._body_read = True
            chunks, left = [], n
            while left > 0:
                b = self.rfile.read(min(left, 1 << 20))
                if not b:
                    break
                chunks.append(b)
                left -= len(b)
            return b"".join(chunks)

        def drain(self) -> None:
            """읽지 않은 본문을 버린다. 거절 응답(401/413 등)을 보낸 뒤 연결을 닫을 때, 아직 올리는 중인 클라이언트가
            '연결 끊김'으로만 보지 않고 거절 이유를 읽을 수 있게 한다."""
            if self._body_read:
                return
            self._body_read = True
            try:
                left = min(int(self.headers.get("Content-Length") or 0), MAX_DRAIN)
                while left > 0:
                    b = self.rfile.read(min(left, 1 << 20))
                    if not b:
                        break
                    left -= len(b)
            except (OSError, ValueError):
                pass

        def _dispatch(self, method: str):
            parts = urlsplit(self.path)
            self.query = parse_qs(parts.query)
            path = parts.path
            self._body_read = False
            if method == "GET" and path in STATIC:
                f = WEB_DIR / STATIC[path]
                ctype = mimetypes.guess_type(f.name)[0] or "text/plain"
                if f.suffix == ".js":
                    ctype = "application/javascript"
                if f.suffix == ".webmanifest":
                    ctype = "application/manifest+json"
                return self._send(200, f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") or "javascript" in ctype else ""))
            auth = self.headers.get("Authorization", "")
            self.token = auth[7:] if auth.startswith("Bearer ") else None
            role = sessions.role(self.token)
            for meth, pat, need, raw, fn in routes:
                mt = pat.match(path)
                if not mt or meth != method:
                    continue
                try:
                    if need and not role:
                        raise HttpError(401, "로그인이 필요합니다.")
                    if need == "admin":
                        need_admin(role)
                    body = {}
                    if method in ("POST", "DELETE") and not raw:
                        n = int(self.headers.get("Content-Length") or 0)
                        if n > MAX_BODY:
                            raise HttpError(413, "요청이 너무 큽니다.")
                        raw_body = self.read_exact(n) if n else b""
                        body = json.loads(raw_body.decode() or "{}") if raw_body else {}
                        if not isinstance(body, dict):
                            raise HttpError(400, "잘못된 요청입니다.")
                    res = fn(self, mt, body, role)
                    if res is not None:
                        self._send(200, res)
                except HttpError as e:
                    self.drain()
                    self._send(e.status, {"error": str(e)})
                except OnlineError as e:
                    self.drain()
                    self._send(ONLINE_STATUS.get(e.kind, 502), {"error": str(e)})
                except ImportError as e:   # 선택 패키지(torch 등)가 없을 때
                    self.drain()
                    self._send(400, {"error": missing_package_message(e)})
                except (ValueError, KeyError, TypeError) as e:
                    self.drain()
                    self._send(400, {"error": str(e) or type(e).__name__})
                except Exception as e:
                    traceback.print_exc()
                    self.drain()
                    self._send(500, {"error": f"{type(e).__name__}: {e}"})
                return
            self.drain()
            self._send(404, {"error": "없는 경로입니다."})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

    return Handler


def lan_addresses() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 80))  # 실제로 패킷을 보내지 않음: 기본 경로의 내 IP 확인용
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips)


def serve(app: App, host: str = "127.0.0.1", port: int = 8750, certfile: str | None = None,
          keyfile: str | None = None) -> None:
    if not app.cfg.initialized:
        raise SystemExit("먼저 PIN을 설정하세요: python -m learnbox init")
    app.start_background()
    httpd = ThreadingHTTPServer((host, port), make_handler(app, Sessions()))
    scheme = "http"
    if certfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
    print(f"LearnBox 실행 중: {scheme}://127.0.0.1:{port}")
    if host in ("0.0.0.0", "::"):
        for ip in lan_addresses():
            print(f"  휴대폰(같은 Wi-Fi)에서 접속: {scheme}://{ip}:{port}")
    print(f"데이터 폴더: {app.cfg.data_dir}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        if app.jobs.active_training():   # 학습 중이면 지금 상태를 저장하고 '중단됨'으로 남긴다 (나중에 이어서 재개)
            print("\n학습 중인 작업을 저장하는 중입니다… (한 번 더 Ctrl+C를 누르면 저장하지 않고 바로 끝냅니다)", flush=True)
            try:
                app.suspend_training()
            except KeyboardInterrupt:
                pass
        print("\n종료합니다.")
    finally:
        app.autolearn.stop()
