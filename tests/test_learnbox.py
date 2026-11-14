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
        self.assertEqual(self.call("POST", "/api/train/mobile", {}, token=user)[0], 403)
        self.assertEqual(self.call("POST", "/api/export", {"target": "/tmp"}, token=user)[0], 403)
        self.assertEqual(self.call("GET", "/api/settings", token=user)[0], 403)
        self.assertEqual(self.call("GET", "/api/settings", token=admin)[0], 200)

    def test_search_and_snapshot(self):
        user = self.login("1234")
        s, r = self.call("GET", "/api/search?q=%EC%8B%A0%EA%B2%BD%EB%A7%9D", token=user)
        self.assertEqual(s, 200)
        self.assertEqual(r["results"][0]["url"], "https://a/1")
        self.assertEqual(len(self.call("GET", "/api/snapshot", token=user)[1]["documents"]), 1)

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


class MobileTrainingTest(unittest.TestCase):
    def test_tiny_training_and_onnx(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch 없음")
        from learnbox import train_mobile
        from learnbox.dataset import build_corpus

        d = tempfile.mkdtemp()
        db = DB(Path(d) / "t.db")
        for i in range(30):
            db.add_document(f"https://a/{i}", f"문서{i}", KO_TEXT + f"번호 {i}\n" * 5, "link")
        c = build_corpus(db, Path(d) / "ds")
        info = train_mobile.train_mobile(f"{c['dir']}/train.jsonl", f"{c['dir']}/eval.jsonl", Path(d) / "m",
                                         preset="tiny", steps=60, batch_size=8)
        self.assertLess(info["eval_loss_after"], info["eval_loss_before"])
        self.assertTrue((Path(d) / "m" / "vocab.json").exists())
        if info["onnx"]:
            self.assertGreater((Path(d) / "m" / "model.onnx").stat().st_size, 1000)
        self.assertIsInstance(train_mobile.generate(Path(d) / "m", "딥 러닝", max_new=10), str)


if __name__ == "__main__":
    unittest.main()
