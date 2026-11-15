"""네트워크 없이 도는 테스트: python -m unittest discover -s tests"""
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path

from learnbox.app import App
from learnbox.config import Config
from learnbox.db import DB, HL_START, chunk_text
from learnbox.export import export, verify
from learnbox.extract import decode, html_to_text
from learnbox.fetcher import BlockedURL, Fetcher, host_matches
from learnbox.timegate import OutsideWindow, ensure_allowed, is_allowed

KO_TEXT = "딥 러닝은 여러 층의 인공 신경망을 사용하는 기계 학습 방법이다.\n" * 20


class TimeGateTest(unittest.TestCase):
    def test_daytime_window(self):
        w = [{"days": [0, 1, 2, 3, 4], "start": "09:00", "end": "18:00"}]
        self.assertTrue(is_allowed(w, "Asia/Seoul", datetime(2026, 11, 13, 10, 0)))   # 금 10시
        self.assertFalse(is_allowed(w, "Asia/Seoul", datetime(2026, 11, 13, 18, 0)))  # 끝 시각은 제외
        self.assertFalse(is_allowed(w, "Asia/Seoul", datetime(2026, 11, 14, 10, 0)))  # 토

    def test_overnight_window(self):
        w = [{"days": [4], "start": "22:00", "end": "06:00"}]  # 금요일 밤 ~ 토요일 새벽
        self.assertTrue(is_allowed(w, "Asia/Seoul", datetime(2026, 11, 13, 23, 30)))
        self.assertTrue(is_allowed(w, "Asia/Seoul", datetime(2026, 11, 14, 5, 59)))
        self.assertFalse(is_allowed(w, "Asia/Seoul", datetime(2026, 11, 14, 6, 0)))
        self.assertFalse(is_allowed(w, "Asia/Seoul", datetime(2026, 11, 12, 23, 30)))  # 목요일 밤

    def test_fetch_refused_outside_window(self):
        cfg = Config(tempfile.mkdtemp())
        cfg.data["allowed_windows"] = []
        with self.assertRaises(OutsideWindow):
            ensure_allowed(cfg)
        with self.assertRaises(OutsideWindow):
            Fetcher(cfg).fetch_page("https://example.com/")


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = DB(Path(self.dir) / "t.db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_chunking_keeps_all_text(self):
        chunks = chunk_text(KO_TEXT, 200)
        self.assertTrue(all(len(c) <= 200 for c in chunks))
        self.assertEqual("".join(chunks).replace("\n", ""), KO_TEXT.replace("\n", ""))

    def test_add_update_unchanged(self):
        self.assertEqual(self.db.add_document("https://a/1", "딥 러닝", KO_TEXT, "link"), "new")
        self.assertEqual(self.db.add_document("https://a/1", "딥 러닝", KO_TEXT, "link"), "unchanged")
        self.assertEqual(self.db.add_document("https://a/1", "딥 러닝", KO_TEXT + "추가", "link"), "updated")
        self.assertEqual(self.db.counts()["documents"], 1)

    def test_korean_search_long_and_short_terms(self):
        self.db.add_document("https://a/1", "딥 러닝", KO_TEXT, "link")
        self.db.add_document("https://a/2", "요리", "김치찌개는 한국 음식이다.\n" * 20, "link")
        r = self.db.search("신경망")
        self.assertEqual([x["url"] for x in r], ["https://a/1"])
        self.assertIn(HL_START, r[0]["snippet"])
        self.assertEqual([x["url"] for x in self.db.search("딥")], ["https://a/1"])  # 2글자 이하 -> LIKE
        self.assertEqual([x["url"] for x in self.db.search("신경망 딥")], ["https://a/1"])
        self.assertEqual(self.db.search("존재하지않는단어"), [])
        self.assertEqual(self.db.search('"따옴표" 신경망'), [])  # 특수문자도 오류 없이


class ExtractTest(unittest.TestCase):
    def test_html_to_text(self):
        html = """<html><head><title>제목</title><style>.x{}</style></head><body>
        <nav>메뉴 메뉴</nav><article><h1>본문 제목</h1><p>첫 문단입니다.</p><script>alert(1)</script>
        <p>둘째 <a href="/next">링크</a> 문단</p></article><footer>저작권</footer></body></html>"""
        title, text, links = html_to_text(html, "https://ex.com/a/b")
        self.assertEqual(title, "제목")
        self.assertIn("첫 문단입니다.", text)
        self.assertNotIn("alert", text)
        self.assertNotIn("메뉴", text)
        self.assertNotIn("저작권", text)
        self.assertEqual(links, ["https://ex.com/next"])

    def test_decode_euc_kr(self):
        body = "한글 문서".encode("cp949")
        self.assertEqual(decode(body, "text/html; charset=EUC-KR"), "한글 문서")


class FetcherRulesTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(tempfile.mkdtemp())

    def test_domain_rules(self):
        self.assertTrue(host_matches("ko.wikipedia.org", ["wikipedia.org"]))
        self.assertFalse(host_matches("evilwikipedia.org", ["wikipedia.org"]))
        self.cfg.data["allowed_domains"] = ["example.org"]
        self.cfg.data["allow_private_network"] = True
        f = Fetcher(self.cfg)
        f.check_url("https://www.example.org/x")
        with self.assertRaises(BlockedURL):
            f.check_url("https://other.com/")
        self.cfg.data["allowed_domains"] = []
        self.cfg.data["blocked_domains"] = ["bad.com"]
        with self.assertRaises(BlockedURL):
            f.check_url("https://x.bad.com/")
        with self.assertRaises(BlockedURL):
            f.check_url("file:///etc/passwd")

    def test_private_network_blocked(self):
        f = Fetcher(self.cfg)
        for url in ("http://127.0.0.1/", "http://localhost:8750/", "http://192.168.0.1/", "http://169.254.169.254/"):
            with self.assertRaises(BlockedURL, msg=url):
                f.check_url(url)


class FetcherNetworkTest(unittest.TestCase):
    """로컬 서버로 확인: 한글 주소(IRI) 요청, 서버가 알려 준 시간(Retry-After) 뒤 한 번만 다시 시도."""

    @classmethod
    def setUpClass(cls):
        seen, state = [], {"limited": 0}

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(self.path)
                if self.path == "/robots.txt":
                    return self._reply(404, "text/plain", b"")
                if self.path.startswith("/always429"):
                    return self._reply(429, "text/plain", b"slow down")
                if self.path.startswith("/wiki/") and state["limited"] == 0:
                    state["limited"] = 1
                    return self._reply(429, "text/plain", b"slow down", {"Retry-After": "0"})
                html = "<html><head><title>딥러닝 문서</title></head><body><article><p>" + "딥러닝은 인공 신경망이다. " * 20 + "</p></article></body></html>"
                self._reply(200, "text/html; charset=utf-8", html.encode())

            def _reply(self, code, ctype, body, extra=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        cls.seen = seen
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def fetcher(self):
        cfg = Config(tempfile.mkdtemp())
        cfg.data.update(allow_private_network=True, request_interval_sec=0)
        return Fetcher(cfg)

    def test_korean_url_is_percent_encoded_and_retry_after_is_honored(self):
        from learnbox.fetcher import to_ascii_url
        self.assertEqual(to_ascii_url("https://ko.wikipedia.org/wiki/딥러닝"),
                         "https://ko.wikipedia.org/wiki/%EB%94%A5%EB%9F%AC%EB%8B%9D")
        self.assertEqual(to_ascii_url("https://a.test/wiki/%EB%94%A5#x"), "https://a.test/wiki/%EB%94%A5")  # 이중 인코딩 안 함
        page = self.fetcher().fetch_page(self.base + "/wiki/딥러닝")
        self.assertEqual(page.title, "딥러닝 문서")
        self.assertIn("신경망", page.text)
        wiki = [p for p in self.seen if p.startswith("/wiki/")]
        self.assertEqual(wiki, ["/wiki/%EB%94%A5%EB%9F%AC%EB%8B%9D"] * 2)   # 429(Retry-After: 0) 뒤 한 번 다시 시도

    def test_429_without_retry_after_explains_user_agent(self):
        from learnbox.fetcher import FetchError
        with self.assertRaises(FetchError) as cm:
            self.fetcher().fetch_page(self.base + "/always429")
        self.assertIn("HTTP 429", str(cm.exception))
        self.assertIn("user_agent", str(cm.exception))
        self.assertEqual(len([p for p in self.seen if p.startswith("/always429")]), 1)   # Retry-After가 없으면 다시 시도하지 않음


class ConfigMigrationTest(unittest.TestCase):
    def test_v1_training_settings_are_carried_over(self):
        d = tempfile.mkdtemp()
        (Path(d) / "config.json").write_text(json.dumps({
            "pc_training": {"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "max_steps": 77, "bogus": 1},
            "mobile_training": {"preset": "small", "steps": 555, "batch_size": 16, "lr": 0.001}}), encoding="utf-8")
        cfg = Config(d)
        self.assertEqual((cfg["lora_training"]["base_model"], cfg["lora_training"]["max_steps"], cfg["lora_training"]["lora_r"]),
                         ("Qwen/Qwen2.5-1.5B-Instruct", 77, 8))
        self.assertNotIn("bogus", cfg["lora_training"])
        self.assertEqual((cfg["tiny_training"]["steps"], cfg["tiny_training"]["batch_size"], cfg["tiny_training"]["preset"]),
                         (555, 16, "auto"))
        # 새 이름이 이미 있으면 덮어쓰지 않는다
        (Path(d) / "config.json").write_text(json.dumps({"pc_training": {"max_steps": 77},
                                                         "lora_training": {"max_steps": 5}}), encoding="utf-8")
        self.assertEqual(Config(d)["lora_training"]["max_steps"], 5)


class ExportTest(unittest.TestCase):
    def test_export_verify_and_secrets_removed(self):
        data, usb = tempfile.mkdtemp(), tempfile.mkdtemp()
        cfg = Config(data)
        cfg.set_pins("1234", "987654")
        cfg.data["search"]["brave_api_key"] = "secret-key"
        db = DB(cfg.db_path)
        db.add_document("https://a/1", "딥 러닝", KO_TEXT, "link")
        (cfg.models_dir / "mobile-x").mkdir()
        (cfg.models_dir / "mobile-x" / "model.onnx").write_bytes(b"\x00" * 1000)
        r = export(cfg, db, usb, [], "admin")
        out = Path(r["path"])
        self.assertTrue(verify(str(out))["ok"])
        pub = (out / "config.public.json").read_text(encoding="utf-8")
        self.assertNotIn("pin_hash", pub)
        self.assertNotIn("secret-key", pub)
        (out / "models" / "mobile-x" / "model.onnx").write_bytes(b"\x01" * 1000)
        self.assertEqual(verify(str(out))["corrupted"], ["models/mobile-x/model.onnx"])
        with self.assertRaises(ValueError):
            export(cfg, db, str(cfg.data_dir / "models"), [], "admin")


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from learnbox.server import Sessions, make_handler

        cls.dir = tempfile.mkdtemp()
        cls.app = App(cls.dir)
        cls.app.cfg.set_pins("1234", "987654")
        cls.app.db.add_document("https://a/1", "딥 러닝", KO_TEXT, "link")
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.app, Sessions()))
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def call(self, method, path, body=None, token=None):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def login(self, pin):
        return self.call("POST", "/api/login", {"pin": pin})[1]["token"]

    def test_roles(self):
        self.assertEqual(self.call("GET", "/api/status")[0], 401)
        self.assertEqual(self.call("POST", "/api/login", {"pin": "0000"})[0], 401)
        user, admin = self.login("1234"), self.login("987654")
        self.assertEqual(self.call("GET", "/api/status", token=user)[1]["role"], "user")
        self.assertEqual(self.call("POST", "/api/train/tiny", {}, token=user)[0], 403)
        self.assertEqual(self.call("POST", "/api/train/mobile", {}, token=user)[0], 403)  # 이전 이름도 관리자만
        self.assertEqual(self.call("POST", "/api/export", {"target": "/tmp"}, token=user)[0], 403)
        self.assertEqual(self.call("GET", "/api/settings", token=user)[0], 403)
        self.assertEqual(self.call("GET", "/api/settings", token=admin)[0], 200)

    def test_search_and_snapshot(self):
        user = self.login("1234")
        s, r = self.call("GET", "/api/search?q=%EC%8B%A0%EA%B2%BD%EB%A7%9D", token=user)
        self.assertEqual(s, 200)
        self.assertEqual(r["results"][0]["url"], "https://a/1")
        snap = self.call("GET", "/api/snapshot?max_chunks=50", token=user)[1]
        self.assertGreaterEqual(len(snap["chunks"]), 1)
        self.assertEqual(set(snap["chunks"][0]), {"c", "d", "t", "x", "u", "s"})
        self.assertTrue(snap["created_at"])

    def test_collect_refused_outside_window(self):
        admin = self.login("987654")
        self.app.cfg.data["allowed_windows"] = []
        try:
            s, r = self.call("POST", "/api/collect/links", {"urls": ["https://example.com"]}, token=admin)
            self.assertEqual(s, 403)
            self.assertIn("허용 시간", r["error"])
        finally:
            self.app.cfg.data["allowed_windows"] = [{"days": list(range(7)), "start": "00:00", "end": "24:00"}]

    def test_settings_cannot_change_pins(self):
        admin = self.login("987654")
        before = dict(self.app.cfg["auth"])
        self.call("POST", "/api/settings", {"auth": {"admin_pin_hash": "x"}, "max_pages_per_job": 7}, token=admin)
        self.assertEqual(self.app.cfg["auth"], before)
        self.assertEqual(self.app.cfg["max_pages_per_job"], 7)




# ======================================================================================================
# v2: 작업 줄 / 수집 작업 / HTTP 경로 / 명령줄 (모델 학습은 아주 작은 구조로 실제로 돌린다)
# ======================================================================================================
import contextlib
import io
import time
import zipfile
from http.server import BaseHTTPRequestHandler
from unittest import mock
from urllib.parse import quote

from learnbox.__main__ import build_parser, main as cli_main
from learnbox.jobs import ALIASES, LANES, JobManager

TINY_ARCH = {"d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 16, "vocab_size": 100}
PHOTO_MD = ("# 광합성 노트\n\n광합성은 식물의 엽록체에서 빛 에너지를 이용해 이산화탄소와 물로 포도당을 만드는 과정이다.\n"
            "엽록체 안의 틸라코이드에서 빛 반응이 일어나고 스트로마에서 캘빈 회로가 돈다.\n").encode()


def wait_job(app, job_id, timeout=240):
    """작업이 끝날 때까지 기다려 최종 작업 행을 돌려준다."""
    end = time.time() + timeout
    while time.time() < end:
        j = app.db.get_job(job_id)
        if j and j["status"] not in ("queued", "running"):
            return j
        time.sleep(0.1)
    raise AssertionError(f"작업 #{job_id}이(가) {timeout}초 안에 끝나지 않았습니다: {app.db.get_job(job_id)}")


def quiet(handler_cls):
    handler_cls.log_message = lambda *a, **k: None
    return handler_cls


class JobLanesTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.db = DB(Path(self.dir) / "j.db")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_lanes_cover_every_spec_kind(self):
        expect = {"collect_links": "collect", "collect_search": "collect", "ingest_path": "light",
                  "index_embeddings": "light", "build_qa": "light", "train_tiny": "heavy", "train_lora": "heavy",
                  "autolearn_train": "heavy", "finalize_learning": "heavy", "export": "heavy"}
        for kind, lane in expect.items():
            self.assertEqual(LANES[kind], lane, kind)
        self.assertEqual(ALIASES, {"train_mobile": "train_tiny", "train_pc": "train_lora"})
        app = App(self.dir)
        self.assertTrue(set(expect) <= set(app.jobs.handlers), "모든 작업 종류에 핸들러가 있어야 합니다")
        self.assertEqual(set(app.jobs.queues), {"collect", "light", "heavy"})

    def test_aliases_light_lane_and_messages(self):
        threads = {}

        def handler(name):
            def fn(ctx, p):
                threads[name] = threading.current_thread().name
                if p.get("fail"):
                    raise ValueError("한국어 오류 메시지")
                return {"ok": name}
            return fn

        jm = JobManager(self.db, {k: handler(k) for k in ("train_tiny", "train_lora", "ingest_path")})
        jm.start()
        ids = [jm.submit("train_mobile", {}, "t"), jm.submit("train_pc", {}, "t"), jm.submit("ingest_path", {}, "t"),
               jm.submit("train_tiny", {"fail": 1}, "t")]
        deadline = time.time() + 10
        while time.time() < deadline and any(self.db.get_job(i)["status"] in ("queued", "running") for i in ids):
            time.sleep(0.05)
        kinds = [self.db.get_job(i)["kind"] for i in ids]
        self.assertEqual(kinds, ["train_tiny", "train_lora", "ingest_path", "train_tiny"])   # 이전 이름은 새 이름으로 저장
        self.assertEqual(threads["ingest_path"], "lane-light")
        self.assertEqual(threads["train_tiny"], "lane-heavy")
        bad = self.db.get_job(ids[3])
        self.assertEqual((bad["status"], bad["message"]), ("failed", "한국어 오류 메시지"))
        with self.assertRaises(ValueError):
            jm.submit("nope", {}, "t")

    def test_auto_collector_yields_to_autolearn(self):
        app = App(self.dir)
        app.cfg.data["auto_collect"] = {"enabled": True, "interval_min": 0}
        app.db.add_watch("search", "테스트", 0)
        app.autolearn.start_session("t", {"targets": {"tiny": False, "lora": False, "embeddings": False, "qa": False}})
        self.assertIsNone(app.auto.tick())
        self.assertEqual(app.db.running_jobs(("collect_search", "collect_links")), 0)


class FakeFetcher:
    class Page:
        def __init__(self, url, title, text, links=()):
            self.url, self.title, self.text, self.links = url, title, text, list(links)

    def fetch_page(self, url):
        return self.Page(url, "수집한 페이지", "인공 신경망은 뇌의 뉴런을 본뜬 계산 모델이다. " * 15)


class StubCtx:
    def progress(self, p, m=""):
        pass

    def check_cancel(self):
        pass


class CollectHandlerTest(unittest.TestCase):
    """자율 학습 세션이 낸 수집 작업은 통계에 반영되고 source가 'auto'가 된다."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.app = App(self.dir)
        self.app.fetcher = FakeFetcher()
        self.sid = self.app.autolearn.start_session(
            "t", {"targets": {"tiny": False, "lora": False, "embeddings": False, "qa": False}})["id"]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def stats(self):
        return self.app.db.get_session(self.sid)["stats"]

    def test_collect_links_records_session_stats(self):
        r = self.app._collect_links(StubCtx(), {"urls": ["https://a.test/x"], "depth": 0, "max_pages": 3,
                                                "session_id": self.sid})
        self.assertEqual(r["new"], 1)
        st = self.stats()
        self.assertEqual((st["pages_collected"], st["docs_new"]), (1, 1))
        self.assertGreater(st["chars_new"], 0)
        self.assertEqual(self.app.db.list_documents(5)[0]["source"], "auto")

    def test_collect_search_records_session_stats(self):
        with mock.patch("learnbox.collector.web_search", return_value=[{"url": "https://b.test/y", "title": "y"}]):
            self.app._collect_search(StubCtx(), {"query": "신경망", "limit": 2, "session_id": self.sid})
        self.assertEqual(self.stats()["docs_new"], 1)
        self.assertEqual(self.app.db.list_documents(5)[0]["source"], "auto")

    def test_manual_collect_is_not_auto_and_not_recorded(self):
        self.app._collect_links(StubCtx(), {"urls": ["https://a.test/m"], "depth": 0})
        self.assertEqual(self.app.db.list_documents(5)[0]["source"], "link")
        self.assertEqual(self.stats()["pages_collected"], 0)


class FakeClaude:
    """POST /v1/messages 를 흉내 내는 로컬 서버 (온라인 모델 엔진 확인용)."""

    def __init__(self):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("content-length") or 0)
                outer.requests.append({"path": self.path, "body": json.loads(self.rfile.read(n) or b"{}")})
                cited = "광합성은 식물의 엽록체에서"
                content = [{"type": "text", "text": "광합성은 엽록체에서 일어납니다",
                            "citations": [{"type": "char_location", "cited_text": cited, "document_index": 0,
                                           "document_title": "광합성 노트", "start_char_index": 0,
                                           "end_char_index": len(cited)}]}]
                if "format" in (outer.requests[-1]["body"].get("output_config") or {}):   # Q&A 만들기: 구조화된 출력
                    n = len(outer.requests)
                    pairs = {"pairs": [{"question": f"질문 {n}-{i}은 무엇인가요?", "answer": f"답변 {n}-{i}입니다."} for i in (1, 2)]}
                    content = [{"type": "text", "text": json.dumps(pairs, ensure_ascii=False)}]
                body = {"id": "msg_01", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                        "content": content,
                        "stop_reason": "end_turn", "stop_sequence": None, "stop_details": None,
                        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0,
                                  "cache_read_input_tokens": 0}}
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class ServerV2Test(unittest.TestCase):
    """실제 서버(스레드)와 실제 작업 관리자로 새 경로를 모두 확인한다. 모델은 아주 작은 구조로 실제 학습한다."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyTorch 없음")
        from learnbox.server import Sessions, make_handler

        cls.dir = tempfile.mkdtemp()
        cls.app = App(cls.dir)
        cls.app.cfg.set_pins("1234", "987654")
        cls.app.cfg.data["allowed_windows"] = []        # 수집은 막아 둔다 (네트워크를 쓰지 않음)
        cls.app.cfg.data["embeddings"]["enabled"] = False
        for i in range(12):
            cls.app.db.add_document(f"https://a/{i}", f"딥 러닝 {i}", KO_TEXT + f"번호 {i}\n" * 4, "link")
        cls.app.jobs.start()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), quiet(make_handler(cls.app, Sessions())))
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.user, cls.admin = cls.login("1234"), cls.login("987654")
        s, r = cls.call("POST", "/api/train/tiny", {"preset": "custom", **TINY_ARCH, "steps": 8, "batch_size": 4},
                        token=cls.admin)
        assert s == 200, r
        cls.train_job = r["job_id"]
        j = wait_job(cls.app, cls.train_job)
        assert j["status"] == "done", j
        cls.train_result = j["result"]
        cls.model_id = j["result"]["model_id"]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    @classmethod
    def call(cls, method, path, body=None, token=None, raw=None, headers=None):
        """JSON 요청 (또는 raw 바이트 본문). 반환 (status, json)"""
        s, h, data = cls.call_raw(method, path, body, token, raw, headers)
        return s, json.loads(data) if data else None

    @classmethod
    def call_raw(cls, method, path, body=None, token=None, raw=None, headers=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(cls.base + path, method=method, data=data)
        req.add_header("Content-Type", "application/octet-stream" if raw is not None else "application/json")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    @classmethod
    def login(cls, pin):
        return cls.call("POST", "/api/login", {"pin": pin})[1]["token"]

    def upload(self, name, data, token=None, **kw):
        return self.call("POST", "/api/files", raw=data, token=token or self.user,
                         headers={"X-File-Name": quote(name), **kw})

    # ---------- 상태 / 장치 / 예상 ----------
    def test_status_has_v2_fields(self):
        s, st = self.call("GET", "/api/status", token=self.user)
        self.assertEqual(s, 200)
        self.assertIn("available", st["online"])
        self.assertIn("active", st["autolearn"])
        self.assertIn("phase", st["autolearn"])
        self.assertIsInstance(st["unread"], int)
        self.assertTrue(st["capabilities"]["torch"])
        self.assertIn("documents", st["counts"])
        self.assertEqual(st["role"], "user")

    def test_device_and_estimate(self):
        s, d = self.call("GET", "/api/device", token=self.user)
        self.assertEqual(s, 200)
        self.assertEqual(set(d["presets"]), {"nano", "mini", "small", "base", "large"})
        self.assertGreater(d["presets"]["nano"]["estimate"]["params"], 0)
        self.assertEqual(len(d["base_models"]), 4)
        self.assertIn("d_model", d["limits"])
        self.assertIn(d["recommend"]["tiny_preset"], d["presets"])
        self.assertIn("cpu_count", d["pc"])

        s, r = self.call("POST", "/api/estimate", {"kind": "tiny", "preset": "mini", "batch_size": 16}, token=self.user)
        self.assertEqual((s, r["arch"]["preset"], r["arch"]["d_model"]), (200, "mini", 256))
        self.assertEqual(r["estimate"]["batch_size"], 16)
        s, r = self.call("POST", "/api/estimate", {"kind": "tiny", "preset": "auto"}, token=self.user)
        self.assertEqual(r["arch"]["preset"], d["recommend"]["tiny_preset"])
        s, r = self.call("POST", "/api/estimate", {"kind": "tiny", "preset": "custom", **TINY_ARCH}, token=self.user)
        self.assertEqual(r["arch"]["preset"], "custom")
        s, r = self.call("POST", "/api/estimate", {"kind": "tiny", "preset": "custom", "d_model": 100, "n_head": 3},
                         token=self.user)
        self.assertEqual(s, 400)
        self.assertIn("나누어떨어", r["error"])
        s, r = self.call("POST", "/api/estimate", {"kind": "tiny", "preset": "custom"}, token=self.user)
        self.assertEqual(s, 400)

        s, r = self.call("POST", "/api/estimate", {"kind": "lora", "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                                                   "lora_r": 8, "seq_len": 512, "batch_size": 2}, token=self.user)
        self.assertEqual(s, 200)
        self.assertEqual(r["base_params_b"], 0.5)
        self.assertGreater(r["trainable_params_est"], 1_000_000)
        self.assertIn("fits", r)
        self.assertEqual(self.call("POST", "/api/estimate", {"kind": "x"}, token=self.user)[0], 400)
        self.assertEqual(self.call("POST", "/api/estimate", {"kind": "lora", "lora_r": "abc"}, token=self.user)[0], 400)
        self.assertEqual(self.call("POST", "/api/estimate", {"kind": "tiny"})[0], 401)

    # ---------- 파일 가져오기 → 검색 → 문서 보기 → 질문 ----------
    def test_upload_search_view_ask(self):
        s, r = self.upload("광합성 노트.md", PHOTO_MD)
        self.assertEqual((s, r["status"], r["docs"]), (200, "ok", 1))
        s, dup = self.upload("다른 이름.md", PHOTO_MD)
        self.assertEqual((s, dup["status"], dup["file_id"]), (200, "duplicate", r["file_id"]))
        files = self.call("GET", "/api/files", token=self.user)[1]["files"]
        self.assertIn("광합성 노트.md", [f["name"] for f in files])

        q = quote("광합성")
        s, res = self.call("GET", f"/api/search?q={q}&sources=file", token=self.user)
        self.assertEqual(s, 200)
        self.assertEqual(res["results"][0]["source"], "file")
        self.assertIn("광합성", res["extract"][0]["sentence"])
        self.assertIn(res["mode_used"], ("keyword", "hybrid", "semantic"))
        self.assertEqual(self.call("GET", f"/api/search?q={q}&sources=link", token=self.user)[1]["results"], [])
        s, empty = self.call("GET", "/api/search?q=", token=self.user)
        self.assertEqual((s, empty["results"]), (200, []))
        self.assertEqual(self.call("GET", "/api/search?q=a&mode=bogus", token=self.user)[0], 400)
        self.assertEqual(self.call("GET", "/api/search?q=a&limit=x", token=self.user)[0], 400)
        s, page2 = self.call("GET", f"/api/search?q={q}&offset=5", token=self.user)
        self.assertEqual(page2["offset"], 5)

        doc_id = res["results"][0]["doc_id"]
        s, d = self.call("GET", f"/api/documents/{doc_id}?q={q}", token=self.user)
        self.assertEqual(s, 200)
        self.assertIn("광합성은 식물의", d["text"])
        self.assertTrue(d["matches"])
        a, b = d["matches"][0]
        self.assertEqual(d["text"][a:b], "광합성")
        self.assertEqual(d["matches_utf16"][0], d["matches"][0])
        self.assertEqual(self.call("GET", "/api/documents/999999", token=self.user)[0], 404)

        s, ans = self.call("POST", "/api/ask", {"q": "광합성은 어디서 일어나?", "engine": "search"}, token=self.user)
        self.assertEqual(s, 200)
        self.assertEqual(ans["engine"], "search")
        self.assertIn("광합성", ans["answer"])
        self.assertIn("[1]", ans["answer"])
        self.assertEqual(ans["citations"][0]["index"], 1)
        self.assertEqual(ans["sources"][0]["doc_id"], doc_id)
        self.assertTrue(ans["extract"])
        self.assertEqual(self.call("POST", "/api/ask", {"q": "  "}, token=self.user)[0], 400)

        s, lst = self.call("GET", "/api/documents?source=file", token=self.user)
        self.assertEqual({x["source"] for x in lst["documents"]}, {"file"})

    def test_upload_errors(self):
        s, r = self.upload("run.exe", b"MZ" * 30)
        self.assertEqual(s, 400)
        self.assertIn("지원하지 않는", r["error"])
        self.assertEqual(self.call("POST", "/api/files", raw=b"x" * 40, token=self.user)[0], 400)   # 이름 없음
        self.assertEqual(self.upload("empty.txt", b"")[0], 400)
        self.assertEqual(self.call("POST", "/api/files", raw=b"x" * 40, headers={"X-File-Name": "a.txt"})[0], 401)
        # 경로 부분은 버리고 파일 이름만 쓴다
        s, r = self.upload("../../etc/passwd.txt", "사용자 목록 파일입니다. 내용이 충분히 길어야 저장됩니다.".encode())
        self.assertEqual((s, r["name"]), (200, "passwd.txt"))
        saved = [f for f in self.app.db.list_files() if f["name"] == "passwd.txt"][0]
        self.assertEqual(Path(saved["path"]).parent, self.app.cfg.data_dir / "files")
        # 크기 제한은 본문을 읽기 전에 거절하되, 올리는 중인 클라이언트가 이유를 읽을 수 있어야 한다
        old = self.app.cfg.data["ingest"]["max_file_mb"]
        self.app.cfg.data["ingest"]["max_file_mb"] = 0.001
        try:
            s, r = self.upload("big.txt", b"a" * (3 * 1024 * 1024))
        finally:
            self.app.cfg.data["ingest"]["max_file_mb"] = old
        self.assertEqual(s, 413)
        self.assertIn("너무 큽니다", r["error"])

    def test_ingest_path_job(self):
        src = tempfile.mkdtemp()
        try:
            (Path(src) / "a.txt").write_text("경로로 가져온 첫 번째 파일입니다. 충분히 긴 한국어 문장이 들어 있습니다.", encoding="utf-8")
            (Path(src) / "b.md").write_text("# 둘째\n\n경로로 가져온 두 번째 파일입니다. 이것도 충분히 긴 문장입니다.", encoding="utf-8")
            self.assertEqual(self.call("POST", "/api/ingest/path", {"path": src}, token=self.user)[0], 403)
            s, r = self.call("POST", "/api/ingest/path", {"path": src}, token=self.admin)
            self.assertEqual(s, 200)
            j = wait_job(self.app, r["job_id"])
            self.assertEqual((j["status"], j["result"]["files"], j["result"]["docs"]), ("done", 2, 2))
            self.assertEqual(self.call("POST", "/api/ingest/path", {"path": src + "/nope"}, token=self.admin)[0], 400)
            self.assertEqual(self.call("POST", "/api/ingest/path", {"path": str(self.app.cfg.data_dir)}, token=self.admin)[0], 400)
        finally:
            shutil.rmtree(src, ignore_errors=True)

    def test_ingest_never_reads_the_data_folder(self):
        """데이터 폴더를 품은 상위 폴더를 가져와도 config.json(PIN 해시/API 키)과 DB는 들어가지 않는다."""
        from learnbox import ingest
        parent = tempfile.mkdtemp()
        try:
            app2 = App(Path(parent) / "data")
            app2.cfg.set_pins("1234", "987654")
            (Path(parent) / "note.txt").write_text("데이터 폴더 밖에 있는 평범한 메모입니다. 충분히 긴 문장입니다.", encoding="utf-8")
            r = ingest.ingest_path(app2.cfg, app2.db, parent)
            self.assertEqual((r["files"], r["errors"]), (1, []))
            self.assertEqual(app2.db.search("pin_hash"), [])
            with self.assertRaises(ValueError):
                ingest.ingest_path(app2.cfg, app2.db, str(Path(parent) / "data" / "config.json"))
        finally:
            shutil.rmtree(parent, ignore_errors=True)

    # ---------- 모델 / 생성 / 질문 엔진 ----------
    def test_train_result_and_model_files(self):
        info = self.train_result
        self.assertEqual((info["kind"], info["preset"]), ("tiny", "custom"))
        self.assertLess(info["eval_loss_after"], info["eval_loss_before"] + 1.0)
        s, ms = self.call("GET", "/api/models", token=self.user)
        m = [x for x in ms["models"] if x["id"] == self.model_id][0]
        self.assertEqual(m["kind"], "tiny")
        self.assertNotIn("path", m)
        self.assertTrue(m["name"].startswith("tiny-"))
        for name in ("model.int8.onnx", "model.onnx"):
            s, h, data = self.call_raw("GET", f"/api/models/{self.model_id}/files/{name}", token=self.user)
            self.assertEqual(s, 200, name)
            self.assertEqual(int(h["Content-Length"]), len(data))
            self.assertGreater(len(data), 1000)
        s, vocab = self.call("GET", f"/api/models/{self.model_id}/files/vocab.json", token=self.user)
        self.assertIn("itos", vocab)
        self.assertEqual(self.call("GET", f"/api/models/{self.model_id}/files/info.json", token=self.user)[0], 200)
        for bad in ("model.pt", "..", "config.json"):
            self.assertEqual(self.call("GET", f"/api/models/{self.model_id}/files/{bad}", token=self.user)[0], 404, bad)
        self.assertEqual(self.call("GET", f"/api/models/{self.model_id}/files/model.onnx")[0], 401)
        self.assertEqual(self.call("GET", "/api/models/999/files/model.onnx", token=self.user)[0], 404)

    def test_generate_and_ask_tiny(self):
        s, r = self.call("POST", "/api/generate", {"model_id": self.model_id, "prompt": "딥 러닝은", "max_new": 20},
                         token=self.user)
        self.assertEqual(s, 200)
        self.assertIsInstance(r["text"], str)
        self.assertEqual(self.call("POST", "/api/generate", {"model_id": 9999, "prompt": "x"}, token=self.user)[0], 404)
        self.assertEqual(self.call("POST", "/api/generate", {"model_id": self.model_id, "prompt": " "}, token=self.user)[0], 400)

        s, a = self.call("POST", "/api/ask", {"q": "딥 러닝이란?", "engine": "tiny", "model_id": self.model_id},
                         token=self.user)
        self.assertEqual(s, 200)
        self.assertEqual(a["engine"], "tiny")
        self.assertEqual(a["model"], self.app.db.get_model(self.model_id)["name"])
        self.assertTrue(a["answer"])
        self.assertTrue(a["note"])
        self.assertTrue(a["sources"])
        # model_id를 생략하면 가장 최근 소형 AI 모델
        self.assertEqual(self.call("POST", "/api/ask", {"q": "딥 러닝이란?", "engine": "tiny"}, token=self.user)[0], 200)

    def test_lora_train_resume_and_ask(self):
        from learnbox import train_pc
        tiny = "hf-internal-testing/tiny-random-gpt2"
        try:
            train_pc._load_tokenizer(tiny)
        except Exception:
            self.skipTest("테스트용 작은 공개 모델(hf-internal-testing/tiny-random-gpt2)을 불러올 수 없습니다")
        s, r = self.call("POST", "/api/train/lora", {"base_model": tiny, "max_steps": 2, "seq_len": 32, "batch_size": 2,
                                                     "lora_r": 4, "use_qa": False}, token=self.admin)
        self.assertEqual(s, 200, r)
        j = wait_job(self.app, r["job_id"])
        self.assertEqual(j["status"], "done", j)
        first = self.app.db.get_model(j["result"]["model_id"])
        self.assertEqual((first["kind"], first["info"]["base_model"], first["info"]["lora_r"]), ("lora", tiny, 4))
        self.assertTrue(first["name"].startswith("lora-"))
        # 이어 학습: 기반 모델을 비워 두면 어댑터의 기반 모델을 쓰고 단계가 누적된다
        s, r = self.call("POST", "/api/train/lora", {"max_steps": 1, "seq_len": 32, "batch_size": 2,
                                                     "resume_from_model_id": first["id"], "use_qa": False}, token=self.admin)
        self.assertEqual(s, 200, r)
        j2 = wait_job(self.app, r["job_id"])
        self.assertEqual(j2["status"], "done", j2)
        self.assertEqual((j2["result"]["total_steps"], j2["result"]["parent"]), (3, first["name"]))
        # 다른 기반 모델로 이어 학습하려 하면 작업이 한국어 메시지로 실패하고 모델 폴더는 남지 않는다
        s, r = self.call("POST", "/api/train/lora", {"base_model": "Qwen/Qwen2.5-0.5B-Instruct", "max_steps": 1,
                                                     "resume_from_model_id": first["id"]}, token=self.admin)
        self.assertEqual(s, 200)
        bad = wait_job(self.app, r["job_id"])
        self.assertEqual(bad["status"], "failed")
        self.assertIn("기반 모델", bad["message"])
        self.assertEqual(self.app.db.list_models()[0]["id"], j2["result"]["model_id"])    # 실패한 학습은 모델 폴더를 남기지 않는다
        # 질문: 검색된 자료를 근거로 LoRA 모델이 답한다
        s, a = self.call("POST", "/api/ask", {"q": "딥 러닝이란?", "engine": "lora", "model_id": j2["result"]["model_id"],
                                              "sources": ["link"]}, token=self.user)
        self.assertEqual(s, 200, a)
        self.assertEqual(a["engine"], "lora")
        self.assertEqual(a["model"], self.app.db.get_model(j2["result"]["model_id"])["name"])
        self.assertIsInstance(a["answer"], str)
        self.assertTrue(a["sources"])
        self.assertTrue(all(x["source"] == "link" for x in a["sources"]))
        self.assertEqual(self.call("POST", "/api/generate", {"model_id": first["id"], "prompt": "x"}, token=self.user)[0], 400)
        train_pc.unload()   # 올려 둔 LoRA 모델을 내려 메모리를 돌려준다

    def test_missing_optional_package_is_a_clear_error(self):
        """torch가 없는 PC에서 소형 AI 질문/생성은 500이 아니라 설치 안내가 나와야 한다."""
        with mock.patch.dict("sys.modules", {"torch": None}):
            s, r = self.call("POST", "/api/generate", {"model_id": self.model_id, "prompt": "딥 러닝은"}, token=self.user)
            self.assertEqual(s, 400)
            self.assertIn("pip install -r requirements-train.txt", r["error"])
            self.assertIn("torch", r["error"])
            s, r = self.call("POST", "/api/ask", {"q": "딥 러닝이란?", "engine": "tiny"}, token=self.user)
            self.assertEqual(s, 400)
            self.assertIn("torch", r["error"])
            j = self.app.jobs.run_inline("train_tiny", {"preset": "nano", "steps": 1}, "t")   # 작업 오류 메시지도 같은 안내
        self.assertEqual(j["status"], "failed")
        self.assertIn("pip install -r requirements-train.txt", j["message"])

    def test_ask_engine_errors(self):
        post = lambda body: self.call("POST", "/api/ask", body, token=self.user)
        s, r = post({"q": "딥 러닝", "engine": "lora"})
        self.assertEqual(s, 400)
        self.assertIn("LoRA 모델이 없습니다", r["error"])
        s, r = post({"q": "딥 러닝", "engine": "lora", "model_id": self.model_id})
        self.assertEqual(s, 400)
        self.assertIn("LoRA 모델이 아닙니다", r["error"])
        s, r = post({"q": "딥 러닝", "engine": "onboard"})
        self.assertEqual(s, 400)
        self.assertIn("브라우저", r["error"])
        self.assertEqual(post({"q": "딥 러닝", "engine": "bogus"})[0], 400)
        self.assertEqual(post({"q": "딥 러닝", "engine": "tiny", "model_id": 9999})[0], 400)
        self.assertEqual(post({"q": "딥 러닝", "sources": "link"})[0], 400)
        old = self.app.cfg.data["online"]["enabled"]
        self.app.cfg.data["online"]["enabled"] = False
        try:
            s, r = post({"q": "딥 러닝", "engine": "online"})
        finally:
            self.app.cfg.data["online"]["enabled"] = old
        self.assertEqual(s, 503)
        self.assertIn("꺼져", r["error"])

    def test_ask_online_with_fake_claude(self):
        self.upload("광합성 노트2.md", PHOTO_MD + "\n온라인 모델 시험용 추가 문장입니다. 광합성 설명이 이어집니다.".encode())
        fake = FakeClaude()
        oc = self.app.cfg.data["online"]
        saved = dict(oc)
        oc.update(enabled=True, api_key="sk-ant-test", base_url=fake.url, fallbacks=False, max_retries=0, timeout_sec=15)
        try:
            self.assertTrue(self.call("GET", "/api/status", token=self.user)[1]["online"]["available"])
            s, a = self.call("POST", "/api/ask", {"q": "광합성은 어디서 일어나?", "engine": "online", "sources": ["file"]},
                             token=self.user)
        finally:
            oc.clear()
            oc.update(saved)
            fake.close()
        self.assertEqual(s, 200, a)
        self.assertEqual(a["engine"], "online")
        self.assertIn("엽록체", a["answer"])
        self.assertTrue(a["answer"].rstrip().endswith("[1]"))
        self.assertEqual(a["citations"][0]["index"], 1)
        self.assertEqual(a["model"], "claude-opus-5-5")
        self.assertEqual(a["sources"][0]["source"], "file")
        # 요청에는 검색된 조각이 citations가 켜진 문서로, 마지막에 질문이 들어간다
        blocks = fake.requests[0]["body"]["messages"][-1]["content"]
        self.assertEqual(blocks[0]["type"], "document")
        self.assertTrue(blocks[0]["citations"]["enabled"])
        self.assertIn("광합성은 어디서 일어나?", blocks[-1]["text"])

    def test_build_qa_job_with_fake_claude(self):
        fake = FakeClaude()
        oc = self.app.cfg.data["online"]
        saved = dict(oc)
        oc.update(enabled=True, api_key="sk-ant-test", base_url=fake.url, fallbacks=False, max_retries=0, timeout_sec=15)
        qa = self.app.cfg.datasets_dir / "qa.jsonl"
        qa.unlink(missing_ok=True)
        try:
            s, r = self.call("POST", "/api/qa/build", {"limit_chunks": 2, "per_chunk": 2}, token=self.admin)
            self.assertEqual(s, 200, r)
            j = wait_job(self.app, r["job_id"])
        finally:
            oc.clear()
            oc.update(saved)
            fake.close()
        self.assertEqual(j["status"], "done", j)
        self.assertEqual((j["result"]["added"], j["result"]["total"]), (4, 4))      # 조각 2개 x 질문 2개
        rows = [json.loads(line) for line in qa.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue({"question", "answer", "source_url"} <= set(rows[0]))
        # 요청에는 조각이 문서로 들어가고 구조화된 출력(JSON 스키마)을 요구한다
        body = fake.requests[0]["body"]
        self.assertEqual(body["output_config"]["format"]["type"], "json_schema")
        # LoRA 학습은 이 파일을 기본으로 섞고, 끌 수 있다
        self.assertTrue(self.app.plan_lora({"base_model": "x/y"})["qa_path"].endswith("qa.jsonl"))
        self.assertIsNone(self.app.plan_lora({"base_model": "x/y", "use_qa": False})["qa_path"])
        qa.unlink()
        self.assertIsNone(self.app.plan_lora({"base_model": "x/y"})["qa_path"])

    # ---------- 알림 / 작업 / 설정 / 자료 ----------
    def test_notifications(self):
        db = self.app.db
        db.mark_notifications_read(None)
        a = db.notify("info", "알림 A", "내용 A")
        b = db.notify("success", "알림 B", "내용 B")
        s, r = self.call("GET", "/api/notifications?unread=1&limit=20", token=self.user)
        self.assertEqual(s, 200)
        self.assertEqual({n["id"] for n in r["notifications"]}, {a, b})
        self.assertEqual(r["unread"], 2)
        self.assertEqual(self.call("POST", "/api/notifications/read", {"ids": []}, token=self.user)[0], 200)
        self.assertEqual(self.call("GET", "/api/notifications", token=self.user)[1]["unread"], 2)   # 빈 목록 = 아무것도 안 함
        self.assertEqual(self.call("POST", "/api/notifications/read", {"ids": ["x"]}, token=self.user)[0], 400)
        self.call("POST", "/api/notifications/read", {"ids": [a]}, token=self.user)
        self.assertEqual(self.call("GET", "/api/notifications?unread=1", token=self.user)[1]["unread"], 1)
        self.call("POST", "/api/notifications/read", {}, token=self.user)
        r = self.call("GET", "/api/notifications?unread=1", token=self.user)[1]
        self.assertEqual((r["unread"], r["notifications"]), (0, []))
        self.assertEqual(self.call("GET", "/api/notifications")[0], 401)

    def test_jobs_cancel_permissions_and_legacy_train_route(self):
        self.assertEqual(self.call("POST", f"/api/jobs/{self.train_job}/cancel", token=self.user)[0], 403)
        self.assertEqual(self.call("POST", "/api/jobs/99999/cancel", token=self.user)[0], 404)
        self.assertEqual(self.call("POST", f"/api/jobs/{self.train_job}/cancel", token=self.admin)[0], 200)
        s, r = self.call("POST", "/api/train/tiny", {"preset": "custom", "d_model": 100, "n_head": 3}, token=self.admin)
        self.assertEqual(s, 400)                              # 잘못된 구조는 작업을 만들기 전에 거절
        s, r = self.call("POST", "/api/train/tiny", {"preset": "tiny", "resume_from_model_id": 9999}, token=self.admin)
        self.assertEqual(s, 400)
        self.assertIn("찾을 수 없습니다", r["error"])
        self.assertEqual(self.call("POST", "/api/train/lora", {"resume_from_model_id": self.model_id}, token=self.admin)[0], 400)
        self.assertEqual(self.call("POST", "/api/qa/build", {}, token=self.admin)[0] in (200, 400), True)
        self.assertEqual(self.call("POST", "/api/qa/build", {}, token=self.user)[0], 403)
        self.assertEqual(self.call("POST", "/api/index/embeddings", {}, token=self.admin)[0], 400)   # 의미 검색 꺼짐

    def test_documents_delete_and_settings_audit(self):
        self.assertEqual(self.call("DELETE", "/api/documents/1", token=self.user)[0], 403)
        self.app.db.add_document("https://del/1", "지울 문서", "지울 문서의 본문입니다. " * 10, "link")
        did = [d for d in self.app.db.list_documents(50) if d["url"] == "https://del/1"][0]["id"]
        self.assertEqual(self.call("DELETE", f"/api/documents/{did}", token=self.admin)[0], 200)
        self.assertIsNone(self.app.db.get_document(did))

        s, r = self.call("POST", "/api/settings", {"online": {"api_key": "sk-secret-xyz", "effort": "high"},
                                                   "autolearn": {"max_pages_per_hour": 33}}, token=self.admin)
        self.assertEqual(s, 200)
        self.assertEqual(r["online"]["api_key"], "********")
        self.assertEqual(self.app.cfg["online"]["api_key"], "sk-secret-xyz")
        self.call("POST", "/api/settings", {"online": {"api_key": "********", "effort": "low"}}, token=self.admin)
        self.assertEqual(self.app.cfg["online"]["api_key"], "sk-secret-xyz")        # 가려진 값은 기존 값을 지우지 않는다
        audit = json.dumps(self.call("GET", "/api/audit", token=self.admin)[1], ensure_ascii=False)
        self.assertNotIn("sk-secret-xyz", audit)                                      # 감사 기록에 API 키를 남기지 않는다
        self.app.cfg.data["online"]["api_key"] = ""
        s, r = self.call("POST", "/api/settings", {"autolearn": {"results_export_target": "/no/such/dir"}}, token=self.admin)
        self.assertEqual(s, 400)

    def test_static_files(self):
        for path, ctype in (("/onboard.js", "javascript"), ("/app.js", "javascript"), ("/sw.js", "javascript"),
                            ("/style.css", "css"), ("/", "html")):
            s, h, data = self.call_raw("GET", path)
            self.assertEqual(s, 200, path)
            self.assertIn(ctype, h["Content-Type"], path)
            self.assertGreater(len(data), 100)

    # ---------- 자율 학습 → 학습 종료 → 결과 파일 → 알림 ----------
    def test_zz_autolearn_end_to_end(self):
        call, app = self.call, self.app
        self.assertEqual(call("POST", "/api/autolearn/start", {}, token=self.user)[0], 403)
        self.assertEqual(call("POST", "/api/autolearn/stop", {}, token=self.user)[0], 403)
        s, r = call("POST", "/api/autolearn/stop", {}, token=self.admin)
        self.assertEqual(s, 400)
        self.assertIn("진행 중인 자율 학습이 없습니다", r["error"])
        self.assertFalse(call("GET", "/api/autolearn", token=self.user)[1]["active"])

        body = {"targets": {"tiny": True, "lora": False, "embeddings": False, "qa": False}, "tiny_preset": "custom",
                "tiny_arch": TINY_ARCH, "tiny_batch_size": 4, "steps_per_cycle": 5, "train_min_new_chars": 1,
                "train_outside_window": True}
        s, r = call("POST", "/api/autolearn/start", body, token=self.admin)
        self.assertEqual(s, 200, r)
        sid = r["session"]["id"]
        self.assertEqual(r["session"]["config"]["tiny_arch"]["d_model"], 32)
        self.assertEqual(call("POST", "/api/autolearn/start", body, token=self.admin)[0], 400)   # 이미 진행 중
        st = call("GET", "/api/autolearn", token=self.user)[1]
        self.assertTrue(st["active"])
        self.assertTrue(call("GET", "/api/status", token=self.user)[1]["autolearn"]["active"])

        # 허용 시간이 아니라 수집은 하지 않고, 새 글자가 있으면 학습만 한다 (train_outside_window)
        app.db.add_document("https://new/1", "새 문서", "자율 학습 중에 들어온 새 문서의 본문입니다. " * 10, "link")
        tick = app.autolearn.tick()
        self.assertIn("waiting_window", tick["actions"])
        self.assertTrue(any(a.startswith("train:") for a in tick["actions"]), tick)
        job = [j for j in app.db.list_jobs(10) if j["kind"] == "autolearn_train"][0]
        j = wait_job(app, job["id"])
        self.assertEqual(j["status"], "done", j)
        self.assertEqual(len(j["result"]["models"]), 1)
        self.assertIn("tiny", j["result"]["eval"])
        self.assertEqual(app.db.get_session(sid)["stats"]["trainings"], 1)

        s, r = call("POST", "/api/autolearn/stop", {}, token=self.admin)
        self.assertEqual((s, r["session_id"]), (200, sid))
        self.assertEqual(call("POST", "/api/autolearn/stop", {}, token=self.admin)[0], 400)      # 이미 종료 진행 중
        fin = wait_job(app, r["job_id"])
        self.assertEqual(fin["status"], "done", fin)
        name = fin["result"]["name"]
        self.assertGreater(fin["result"]["file_count"], 5)
        self.assertLess(len(json.dumps(fin["result"])), 20000)          # 작업 기록에는 줄여 둔 파일 목록만

        st = call("GET", "/api/autolearn", token=self.user)[1]
        self.assertFalse(st["active"])
        sess = st["recent_sessions"][0]
        self.assertEqual((sess["id"], sess["status"], sess["result"]["name"]), (sid, "ended", name))
        self.assertTrue(sess["ended_at"])
        self.assertEqual(sess["result"]["summary"]["documents"], app.db.counts()["documents"])

        # 알림: 학습 종료
        notes = call("GET", "/api/notifications?limit=30", token=self.user)[1]
        end = [n for n in notes["notifications"] if n["data"].get("kind") == "learning_end"]
        self.assertEqual(len(end), 1)
        self.assertEqual((end[0]["level"], end[0]["title"], end[0]["data"]["result"]), ("success", "학습이 종료되었습니다", name))
        self.assertIn(name, end[0]["body"])
        self.assertGreaterEqual(notes["unread"], 1)

        # 결과 파일: 관리자만 목록/다운로드
        self.assertEqual(call("GET", "/api/results", token=self.user)[0], 403)
        self.assertEqual(call("GET", f"/api/results/{name}/download", token=self.user)[0], 403)
        listing = call("GET", "/api/results", token=self.admin)[1]["results"]
        row = [x for x in listing if x["name"] == name][0]
        self.assertGreater(row["zip_bytes"], 1000)
        self.assertTrue(any(f["path"] == "documents/markdown/" and f.get("dir") for f in row["files"]))

        s, h, data = self.call_raw("GET", f"/api/results/{name}/download", token=self.admin)
        self.assertEqual(s, 200)
        self.assertEqual(h["Content-Type"], "application/zip")
        self.assertIn("attachment", h["Content-Disposition"])
        self.assertIn(name, h["Content-Disposition"])
        self.assertEqual(int(h["Content-Length"]), len(data))
        zf = zipfile.ZipFile(io.BytesIO(data))
        names = zf.namelist()
        for must in ("manifest.json", "REPORT.md", "documents/documents.jsonl", "documents/documents.csv",
                     "documents/all.txt", "index/learnbox.db"):
            self.assertIn(f"{name}/{must}", names)
        self.assertTrue(any(n.endswith("/model.int8.onnx") and "/models/" in n for n in names))
        self.assertTrue(any("/documents/markdown/" in n and n.endswith(".md") for n in names))
        self.assertIsNone(zf.testzip())

        s, h, one = self.call_raw("GET", f"/api/results/{name}/files/documents/documents.jsonl", token=self.admin)
        self.assertEqual(s, 200)
        self.assertIn("application/x-ndjson", h["Content-Type"])
        first = json.loads(one.decode("utf-8").splitlines()[0])
        self.assertTrue({"id", "url", "title", "text"} <= set(first))
        md = [n[len(name) + 1:] for n in names if "/documents/markdown/" in n and n.endswith(".md")][0]
        s, h, mdata = self.call_raw("GET", f"/api/results/{name}/files/{quote(md)}", token=self.admin)   # 한글 파일 이름
        self.assertEqual(s, 200)
        self.assertTrue(mdata.decode("utf-8").startswith("---"))
        for bad in ("..%2F..%2Fconfig.json", "documents%2F..%2F..%2Fconfig.json", "%2Fetc%2Fpasswd", "documents/",
                    "nope.txt"):
            self.assertEqual(call("GET", f"/api/results/{name}/files/{bad}", token=self.admin)[0], 404, bad)
        for bad in ("learnbox-results-19990101-000000", "..%2F", "x"):
            self.assertEqual(call("GET", f"/api/results/{bad}/download", token=self.admin)[0], 404, bad)

        # 매니페스트가 실제 파일과 맞는다 (python -m learnbox verify 와 같은 검사)
        from learnbox.export import verify
        self.assertTrue(verify(row["dir"])["ok"])
        self.assertEqual(call("GET", "/api/status", token=self.user)[1]["autolearn"]["active"], False)


class CliTest(unittest.TestCase):
    """명령줄: 새 기능 명령이 서버 없이 동작한다."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("PyTorch 없음")
        cls.dir = tempfile.mkdtemp()
        cls.src = tempfile.mkdtemp()
        cfg = Config(cls.dir)
        cfg.data["allowed_windows"] = []           # 수집(인터넷)은 막는다
        cfg.data["embeddings"]["enabled"] = False
        cfg.save()
        (Path(cls.src) / "광합성.md").write_bytes(PHOTO_MD)
        for i in range(8):
            (Path(cls.src) / f"문서{i}.txt").write_text(KO_TEXT + f"번호 {i}\n" * 5, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)
        shutil.rmtree(cls.src, ignore_errors=True)

    def run_cli(self, *argv, code=0):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cli_main(["--data", self.dir, *argv])
        self.assertEqual(rc, code, out.getvalue())
        return out.getvalue()

    def test_01_init_ingest_search_ask(self):
        self.assertIn("PIN을 설정했습니다", self.run_cli("init", "--user-pin", "1234", "--admin-pin", "987654"))
        out = self.run_cli("ingest", self.src)
        self.assertIn("[done]", out)
        self.assertIn('"files": 9', out)
        out = self.run_cli("search", "광합성")
        self.assertIn("검색 방식", out)
        self.assertIn("[핵심 문장]", out)
        self.assertIn("[광합성]", out)                      # 강조 표시
        self.assertIn("광합성 노트", self.run_cli("search", '"엽록체" -신경망', "--sources", "file", "--mode", "keyword"))
        self.assertIn("결과가 없습니다", self.run_cli("search", "광합성", "--sources", "link"))
        out = self.run_cli("ask", "광합성은 어디서 일어나?", "--engine", "search")
        self.assertIn("[search]", out)
        self.assertIn("엽록체", out)
        self.assertIn("[인용]", out)
        self.assertIn("학습한 소형 AI 모델이 없습니다", self.run_cli("ask", "x", "--engine", "tiny", code=1))
        self.assertGreaterEqual(json.loads(self.run_cli("status"))["counts"]["documents"], 9)

    def test_02_estimate_and_device(self):
        out = self.run_cli("estimate", "--preset", "mini")
        self.assertIn("mini", out)
        self.assertIn("파라미터", out)
        self.assertIn("휴대폰 고사양 예", out)
        out = self.run_cli("estimate", "--preset", "custom", "--d-model", "64", "--n-layer", "2", "--n-head", "4",
                           "--batch-size", "8")
        self.assertIn("custom", out)
        self.assertEqual(json.loads(self.run_cli("estimate", "--preset", "nano", "--json"))["arch"]["preset"], "nano")
        self.assertIn("나누어떨어", self.run_cli("estimate", "--preset", "custom", "--d-model", "100", "--n-head", "3", code=1))
        out = self.run_cli("estimate", "--kind", "lora", "--base-model", "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", out)
        out = self.run_cli("device")
        self.assertIn("추천", out)
        self.assertIn("CPU", out)
        self.assertIn("tiny_preset", self.run_cli("device", "--json"))

    def test_03_train_alias_resume_and_ask_tiny(self):
        arch = ["--preset", "custom", "--d-model", "32", "--n-layer", "1", "--n-head", "2", "--seq-len", "16",
                "--vocab-size", "100"]
        out = self.run_cli("train-mobile", *arch, "--steps", "6", "--batch-size", "4")      # 이전 이름
        self.assertIn("[done]", out)
        self.assertIn('"kind": "tiny"', out)
        models = self.run_cli("models")
        self.assertIn("[소형 AI] tiny-", models)
        mid = int(models.split("#")[1].split()[0])
        out = self.run_cli("train-tiny", "--resume", str(mid), "--steps", "4", "--batch-size", "4")   # 구조는 부모를 따른다
        self.assertIn("[done]", out)
        self.assertIn('"total_steps": 10', out)
        out = self.run_cli("ask", "딥 러닝이란?", "--engine", "tiny")
        self.assertIn("[tiny] 모델: tiny-", out)
        self.assertIn("[근거 자료]", out)
        out = self.run_cli("train-tiny", "--resume", "9999", code=1)
        self.assertIn("[failed]", out)
        self.assertIn("찾을 수 없습니다", out)

    def test_04_autolearn_run_until_end_and_results(self):
        self.assertIn("진행 중인 자율 학습이 없습니다", self.run_cli("autolearn", "stop", code=1))
        self.assertIn("결과 파일이 아직 없습니다", self.run_cli("results"))
        out = self.run_cli("autolearn", "start", "--targets", "", "--tiny-preset", "auto")
        self.assertIn("세션 #", out)
        st = json.loads(self.run_cli("autolearn", "status"))
        self.assertTrue(st["active"])
        self.assertIn("이미 자율 학습이 진행 중입니다", self.run_cli("autolearn", "start", code=1))
        out = self.run_cli("autolearn", "run", "--until-end", "--stop-after", "0.3", "--interval", "0.1")
        self.assertIn("학습 종료를 요청했습니다", out)
        self.assertIn("ended", out)
        st = json.loads(self.run_cli("autolearn", "status"))
        self.assertFalse(st["active"])
        self.assertEqual(st["recent_sessions"][0]["status"], "ended")
        out = self.run_cli("results")
        self.assertIn("learnbox-results-", out)
        self.assertIn(".zip", out)
        # 새 세션을 만들어 `autolearn stop`으로 바로 끝내기 (끝날 때까지 기다림)
        self.run_cli("autolearn", "start", "--targets", "")
        out = self.run_cli("autolearn", "stop")
        self.assertIn("[done]", out)
        self.assertEqual(len([l for l in self.run_cli("results").splitlines() if l.startswith("learnbox-results-")]), 2)

    def test_05_parser(self):
        p = build_parser()
        self.assertEqual(p.parse_args(["train-mobile", "--preset", "tiny"]).cmd, "train-mobile")
        self.assertEqual(p.parse_args(["train-pc", "--max-steps", "3"]).max_steps, 3)
        a = p.parse_args(["autolearn", "run", "--start", "--until-end", "--stop-after", "5"])
        self.assertEqual((a.cmd, a.action, a.until_end, a.stop_after), ("autolearn", "run", True, 5.0))
        with self.assertRaises(SystemExit):
            p.parse_args(["ask", "q", "--engine", "bogus"])


if __name__ == "__main__":
    unittest.main()
