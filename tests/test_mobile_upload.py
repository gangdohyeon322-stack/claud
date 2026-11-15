"""휴대폰 학습 결과 올리기/내려받기 서버 경로 테스트. 실행: python -m unittest tests.test_mobile_upload -v"""
import json
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from learnbox import lbw
from learnbox.app import App
from tests.test_learnbox import KO_TEXT, TINY_ARCH, quiet, wait_job


class MobileUploadTest(unittest.TestCase):
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
        cls.app.cfg.data["allowed_windows"] = []
        cls.app.cfg.data["embeddings"]["enabled"] = False
        for i in range(12):
            cls.app.db.add_document(f"https://a/{i}", f"딥 러닝 {i}", KO_TEXT + f"번호 {i}\n" * 4, "link")
        cls.app.jobs.start()
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), quiet(make_handler(cls.app, Sessions())))
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.user, cls.admin = cls.login("1234"), cls.login("987654")
        s, r = cls.call("POST", "/api/train/tiny", {"preset": "custom", **TINY_ARCH, "steps": 6, "batch_size": 4},
                        token=cls.admin)
        assert s == 200, r
        j = wait_job(cls.app, r["job_id"])
        assert j["status"] == "done", j
        cls.model_id = j["result"]["model_id"]

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.dir, ignore_errors=True)

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
    def call(cls, *a, **k):
        s, h, d = cls.call_raw(*a, **k)
        return s, json.loads(d) if d else None

    @classmethod
    def login(cls, pin):
        s, r = cls.call("POST", "/api/login", {"pin": pin})
        assert s == 200, r
        return r["token"]

    def get_lbw(self):
        s, h, data = self.call_raw("GET", f"/api/models/{self.model_id}/files/model.lbw", token=self.user)
        self.assertEqual(s, 200)
        return data

    def test_1_download_creates_cache(self):
        mdl = self.app.db.get_model(self.model_id)
        cache = Path(mdl["path"]) / "model.lbw"
        self.assertFalse(cache.exists())
        data = self.get_lbw()
        self.assertTrue(cache.is_file())
        self.assertEqual(cache.read_bytes(), data)
        h, t = lbw.read_lbw(data)
        self.assertEqual(h["arch"]["d_model"], TINY_ARCH["d_model"])
        self.assertEqual(h["parent_model_id"], self.model_id)
        self.assertEqual(lbw.validate_lbw(h, t, 2_000_000), [])
        s, _, _ = self.call_raw("GET", f"/api/models/{self.model_id}/files/model.lbw")
        self.assertEqual(s, 401)

    def make_upload(self):
        h, t = lbw.read_lbw(self.get_lbw())
        meta = {k: v for k, v in h.items() if k not in ("format", "arch", "itos", "tensors")}
        meta.update({"steps": 30, "total_steps": 36, "trained_on": "device", "train_loss": 2.5,
                     "eval_loss_before": 3.2, "eval_loss_after": 2.9, "train_chars": 999, "seconds": 12.5,
                     "device": {"label": "테스트 태블릿", "ua": "x"}, "parent": "tiny-x",
                     "parent_model_id": self.model_id})
        return lbw.build_lbw(t, h["arch"], h["itos"], meta)

    def test_2_upload_ok_and_permissions(self):
        data = self.make_upload()
        s, r = self.call("POST", "/api/models/upload", raw=data, token=self.user)
        self.assertEqual(s, 403)
        s, r = self.call("POST", "/api/models/upload", raw=data)
        self.assertEqual(s, 401)
        before = len(self.app.db.list_models())
        s, r = self.call("POST", "/api/models/upload", raw=data, token=self.admin,
                         headers={"X-Model-Name": "%ED%83%9C%EB%B8%94%EB%A6%BF"})
        self.assertEqual(s, 200, r)
        self.assertTrue(r["name"].startswith("tiny-") and r["name"].endswith("-device"), r["name"])
        info = r["info"]
        self.assertEqual(info["trained_on"], "device")
        self.assertTrue(info["onnx"])
        self.assertEqual((info["steps"], info["total_steps"]), (30, 36))
        self.assertEqual(info["display_name"], "태블릿")
        s, lst = self.call("GET", "/api/models", token=self.user)
        row = next(x for x in lst["models"] if x["id"] == r["model_id"])
        self.assertEqual(row["kind"], "tiny")
        self.assertEqual(row["info"]["trained_on"], "device")
        self.assertTrue(row["info"]["onnx"])
        self.assertEqual(len(lst["models"]), before + 1)
        d = Path(self.app.db.get_model(r["model_id"])["path"])
        for f in ("model.pt", "vocab.json", "model.lbw", "model.onnx", "info.json"):
            self.assertTrue((d / f).is_file(), f)
        self.assertEqual((d / "model.lbw").read_bytes(), data)
        self.assertEqual(json.loads((d / "info.json").read_text(encoding="utf-8"))["trained_on"], "device")
        # 새 모델은 다시 내려받을 수 있다
        s, _, got = self.call_raw("GET", f"/api/models/{r['model_id']}/files/model.lbw", token=self.user)
        self.assertEqual((s, got), (200, data))
        notes = self.app.db.list_notifications() if hasattr(self.app.db, "list_notifications") else []
        if notes:
            self.assertTrue(any("휴대폰에서 학습한 모델" in n["title"] for n in notes))
        with self.app.db.conn() as c:
            self.assertTrue(c.execute("SELECT 1 FROM notifications WHERE title LIKE '%휴대폰에서 학습한 모델%'").fetchone())
            self.assertTrue(c.execute("SELECT 1 FROM audit WHERE action='model_upload'").fetchone())

    def test_3_corrupt(self):
        data = self.make_upload()
        n = len(list(self.app.cfg.models_dir.iterdir()))
        s, r = self.call("POST", "/api/models/upload", raw=b"garbage" * 10, token=self.admin)
        self.assertEqual(s, 400)
        self.assertTrue(r["errors"] and r["error"])
        # 마지막 텐서에 NaN
        bad = bytearray(data)
        bad[-4:] = b"\x00\x00\xc0\x7f"
        s, r = self.call("POST", "/api/models/upload", raw=bytes(bad), token=self.admin)
        self.assertEqual(s, 400)
        self.assertTrue(any("NaN" in e for e in r["errors"]), r)
        s, r = self.call("POST", "/api/models/upload", raw=data[:-8], token=self.admin)
        self.assertEqual(s, 400)
        self.assertEqual(len(list(self.app.cfg.models_dir.iterdir())), n)   # 폴더가 남지 않는다

    def test_4_params_limit(self):
        old = self.app.cfg.data["mobile_training"]["max_params"]
        self.app.cfg.data["mobile_training"]["max_params"] = 100
        try:
            s, r = self.call("POST", "/api/models/upload", raw=self.make_upload(), token=self.admin)
        finally:
            self.app.cfg.data["mobile_training"]["max_params"] = old
        self.assertEqual(s, 400)
        self.assertTrue(any("파라미터" in e for e in r["errors"]))

    def test_5_oversize_413_and_length(self):
        import socket
        mt = self.app.cfg.data["mobile_training"]
        old = mt["max_upload_mb"]
        mt["max_upload_mb"] = 0.001          # 약 1KB
        n = len(self.app.db.list_models())
        try:
            s, r = self.call("POST", "/api/models/upload", raw=self.make_upload(), token=self.admin)
        finally:
            mt["max_upload_mb"] = old
        self.assertEqual(s, 413, r)
        self.assertIn("너무 큽니다", r["error"])
        self.assertEqual(len(self.app.db.list_models()), n)
        # 헤더만 보고 거절: 본문을 보내지 않아도 큰 Content-Length는 (drain 한도 내) 즉시 거절 응답을 준다
        host, port = self.httpd.server_address
        sock = socket.create_connection((host, port))
        sock.sendall((f"POST /api/models/upload HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {self.admin}\r\n"
                      f"Connection: close\r\n\r\n").encode())      # Content-Length 없음 -> 411
        sock.settimeout(5)
        resp = sock.recv(4096).decode("utf-8", "replace")
        sock.close()
        self.assertIn(" 411 ", resp.split("\r\n")[0])

    def test_6_report(self):
        s, r = self.call("POST", "/api/mobile-training/report", {"device": {"label": "폰"}, "ms_per_step": 123,
                                                                  "batch_size": 8, "seq_len": 64, "steps": 10},
                         token=self.user)
        self.assertEqual((s, r), (200, {"ok": True}))
        s, _ = self.call("POST", "/api/mobile-training/report", {})
        self.assertEqual(s, 401)
        with self.app.db.conn() as c:
            row = c.execute("SELECT detail FROM audit WHERE action='mobile_benchmark'").fetchone()
        self.assertIn("123", row[0])


if __name__ == "__main__":
    unittest.main()
