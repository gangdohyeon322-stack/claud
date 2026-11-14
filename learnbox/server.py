"""HTTP 서버: JSON API + 공유 웹 UI. PC 브라우저와 휴대폰이 같은 화면/기능/데이터를 쓴다."""
from __future__ import annotations

import json
import mimetypes
import re
import secrets
import socket
import ssl
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .app import App
from .export import list_drives, validate_target

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css",
          "/sw.js": "sw.js", "/manifest.webmanifest": "manifest.webmanifest", "/icon.svg": "icon.svg"}
MOBILE_FILES = {"model.onnx", "vocab.json", "info.json"}
SESSION_HOURS = 12
MAX_BODY = 1_000_000


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


def make_handler(app: App, sessions: Sessions):
    routes: list[tuple[str, re.Pattern, str | None, callable]] = []

    def route(method: str, pattern: str, role: str | None = "user"):
        def deco(fn):
            routes.append((method, re.compile(f"^{pattern}$"), role, fn))
            return fn
        return deco

    def need_admin(role):
        if role != "admin":
            raise HttpError(403, "관리자만 할 수 있습니다.")

    # ---------- 인증 ----------
    @route("GET", "/api/info", None)
    def info(h, m, body, role):
        return {"name": "LearnBox", "initialized": app.cfg.initialized, "role": role}

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

    @route("GET", "/api/search")
    def search(h, m, body, role):
        q = h.query.get("q", [""])[0].strip()
        limit = min(int(h.query.get("limit", ["10"])[0]), 50)
        rows = app.db.search(q, limit) if q else []
        return {"results": [{k: r[k] for k in ("doc_id", "url", "title", "snippet")} for r in rows]}

    @route("POST", "/api/ask")
    def ask(h, m, body, role):
        q = str(body.get("q", "")).strip()
        if not q:
            raise HttpError(400, "질문을 입력하세요.")
        return app.ask(q, body.get("model_id"))

    @route("GET", "/api/documents")
    def documents(h, m, body, role):
        off = int(h.query.get("offset", ["0"])[0])
        return {"documents": app.db.list_documents(50, off), "counts": app.db.counts()}

    @route("DELETE", r"/api/documents/(\d+)", "admin")
    def delete_document(h, m, body, role):
        app.db.delete_document(int(m.group(1)))
        app.db.audit(role, "delete_document", {"id": int(m.group(1))})
        return {"ok": True}

    @route("GET", "/api/snapshot")
    def snapshot(h, m, body, role):
        return {"documents": app.db.snapshot(), "created_at": time.time()}

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
        return {"jobs": app.db.list_jobs(40)}

    @route("POST", r"/api/jobs/(\d+)/cancel")
    def cancel(h, m, body, role):
        job = app.db.get_job(int(m.group(1)))
        if not job:
            raise HttpError(404, "작업이 없습니다.")
        if role != "admin" and not job["kind"].startswith("collect"):
            raise HttpError(403, "학습/내보내기 작업은 관리자만 취소할 수 있습니다.")
        app.jobs.cancel(job["id"])
        return {"ok": True}

    # ---------- 학습 (관리자) ----------
    @route("POST", "/api/train/pc", "admin")
    def train_pc(h, m, body, role):
        caps = app.capabilities()
        if not (caps["torch"] and caps["transformers"] and caps["peft"]):
            raise HttpError(400, "학습 패키지가 없습니다: pip install -r requirements-train.txt")
        return {"job_id": app.jobs.submit("train_pc", body, role)}

    @route("POST", "/api/train/mobile", "admin")
    def train_mobile(h, m, body, role):
        if not app.capabilities()["torch"]:
            raise HttpError(400, "PyTorch가 없습니다: pip install -r requirements-train.txt")
        return {"job_id": app.jobs.submit("train_mobile", body, role)}

    @route("GET", "/api/models")
    def models(h, m, body, role):
        return {"models": [{k: v for k, v in x.items() if k != "path"} for x in app.db.list_models()]}

    @route("GET", r"/api/models/(\d+)/files/([\w.]+)")
    def model_file(h, m, body, role):
        mdl = app.db.get_model(int(m.group(1)))
        name = m.group(2)
        if not mdl or mdl["kind"] != "mobile" or name not in MOBILE_FILES:
            raise HttpError(404, "파일이 없습니다.")
        path = Path(mdl["path"]) / name
        if not path.exists():
            raise HttpError(404, "파일이 없습니다.")
        h.send_file(path)
        return None

    # ---------- 설정/내보내기 (관리자) ----------
    @route("GET", "/api/settings", "admin")
    def settings_get(h, m, body, role):
        return app.cfg.public()

    @route("POST", "/api/settings", "admin")
    def settings_set(h, m, body, role):
        if body.get("search", {}).get("brave_api_key") == "********":
            body["search"].pop("brave_api_key")
        from .timegate import _minutes
        for w in body.get("allowed_windows", []):
            _minutes(w["start"]), _minutes(w["end"])
        app.cfg.update(body)
        app.db.audit(role, "settings", {k: v for k, v in body.items() if k != "search"})
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
        p ={"target": target, "include": body.get("include", []), "actor": f"admin@{h.client_address[0]}"}
        return {"job_id": app.jobs.submit("export", p, role)}

    @route("GET", "/api/audit", "admin")
    def audit(h, m, body, role):
        return {"audit": app.db.list_audit(100)}

    class Handler(BaseHTTPRequestHandler):
        server_version = "LearnBox/0.1"
        token = None
        query: dict = {}

        def log_message(self, fmt, *args):
            if "/api/" in (args[0] if args else "") and "status" not in str(args[0]):
                super().log_message(fmt, *args)

        def _send(self, status: int, payload, ctype="application/json; charset=utf-8"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def send_file(self, path: Path):
            data = path.read_bytes()
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(200, data, ctype)

        def _dispatch(self, method: str):
            parts = urlsplit(self.path)
            self.query = parse_qs(parts.query)
            path = parts.path
            if method == "GET" and path in STATIC:
                f = WEB_DIR / STATIC[path]
                ctype = mimetypes.guess_type(f.name)[0] or "text/plain"
                if f.suffix == ".webmanifest":
                    ctype = "application/manifest+json"
                return self._send(200, f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") or "javascript" in ctype else ""))
            auth = self.headers.get("Authorization", "")
            self.token = auth[7:] if auth.startswith("Bearer ") else None
            role = sessions.role(self.token)
            for meth, pat, need, fn in routes:
                mt = pat.match(path)
                if not mt or meth != method:
                    continue
                try:
                    if need and not role:
                        raise HttpError(401, "로그인이 필요합니다.")
                    if need == "admin":
                        need_admin(role)
                    body = {}
                    if method in ("POST", "DELETE"):
                        n = int(self.headers.get("Content-Length") or 0)
                        if n > MAX_BODY:
                            raise HttpError(413, "요청이 너무 큽니다.")
                        raw = self.rfile.read(n) if n else b""
                        body = json.loads(raw.decode() or "{}") if raw else {}
                        if not isinstance(body, dict):
                            raise HttpError(400, "잘못된 요청입니다.")
                    res = fn(self, mt, body, role)
                    if res is not None:
                        self._send(200, res)
                except HttpError as e:
                    self._send(e.status, {"error": str(e)})
                except (ValueError, KeyError, TypeError) as e:
                    self._send(400, {"error": str(e) or type(e).__name__})
                except Exception as e:
                    traceback.print_exc()
                    self._send(500, {"error": f"{type(e).__name__}: {e}"})
                return
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
        print("\n종료합니다.")
