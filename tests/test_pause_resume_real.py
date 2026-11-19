"""실제 학습 코드로 끝까지: 일시정지 → 재개, 끊김(장치 오류) → /restart → 목표 단계까지, 가중치 동일성.

docs/SPEC-pause-resume.md §7 통합 확인. tinylm.train을 가짜로 바꾸지 않는다 (끊김은 진행 콜백에서 예외를 내는
얇은 감싸개로만 흉내 낸다). 기준은 같은 seed·같은 자료·같은 설정으로 '끊김 없이' 학습한 결과와의 가중치 최대 차이 ≤ 1e-5.
"""
import json
import re
import shutil
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from learnbox import checkpoint
from learnbox.app import App

KO_TEXT = "딥 러닝은 여러 층의 인공 신경망을 사용하는 기계 학습 방법이다.\n" * 20
TINY_ARCH = {"preset": "custom", "d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 16, "vocab_size": 100}
STEPS = 1500
JOB = {**TINY_ARCH, "steps": STEPS, "batch_size": 4, "threads": 2}
LIVE = ("queued", "running", "paused")


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def wait_for(pred, timeout=60.0, what="조건"):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError(f"{timeout}초 안에 {what}이(가) 맞지 않았습니다")


@unittest.skipUnless(_has_torch(), "PyTorch 없음")
class RealTinyPauseResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from learnbox.server import Sessions, make_handler

        cls.dir = tempfile.mkdtemp(prefix="lb-real-pause-")
        cls.app = App(cls.dir)
        cls.app.cfg.set_pins("1234", "987654")
        cls.app.cfg.data["embeddings"]["enabled"] = False
        for i in range(10):
            cls.app.db.add_document(f"https://r/{i}", f"문서 {i}", KO_TEXT + f"번호 {i}\n" * 3, "link")
        cls.app.jobs.start()
        handler = make_handler(cls.app, Sessions())
        handler.log_message = lambda *a, **k: None
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.admin = cls.call("POST", "/api/login", {"pin": "987654"})[1]["token"]

    @classmethod
    def tearDownClass(cls):
        for j in cls.app.db.list_jobs(50):
            if j["status"] in LIVE:
                cls.app.jobs.cancel(j["id"])
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    @classmethod
    def call(cls, method, path, body=None, token=None):
        data = json.dumps(body).encode() if body is not None else (b"{}" if method == "POST" else None)
        req = urllib.request.Request(cls.base + path, method=method, data=data)
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"null")

    # ----- 도우미 -----
    def job(self, jid):
        return self.app.db.get_job(jid)

    def wait_done(self, jid, timeout=120):
        wait_for(lambda: self.job(jid)["status"] not in LIVE, timeout, f"작업 #{jid} 끝남")
        return self.job(jid)

    def listed(self, jid):
        s, r = self.call("GET", "/api/jobs", token=self.admin)
        self.assertEqual(s, 200, r)
        return next(j for j in r["jobs"] if j["id"] == jid)

    def saved_step(self, jid) -> int:
        res = self.job(jid)["result"] or {}
        return int(((res.get("checkpoint") or {}).get("step")) or 0)

    def reference_weights(self, meta: dict, tag: str):
        """체크포인트에 기록된 자료·설정으로 끊김 없이 STEPS단계 학습한 가중치 (같은 seed, 같은 스레드 수)."""
        from learnbox import tinylm

        out = Path(self.dir) / f"reference-{tag}"
        info = tinylm.train(meta["train_path"], meta["eval_path"], out, meta["arch"], steps=meta["steps"],
                            batch_size=meta["batch_size"], lr=meta["lr"], seed=meta["seed"], threads=2)
        self.assertEqual(info["steps"], STEPS)
        return self.weights(out)

    @staticmethod
    def weights(model_dir):
        import torch

        return torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=True)["state_dict"]

    def assert_same_weights(self, a, b):
        self.assertEqual(set(a), set(b))
        diff = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
        self.assertLessEqual(diff, 1e-5, f"최대 가중치 차이 {diff}")

    # ----- 테스트 -----
    def test_pause_then_resume_via_jobmanager(self):
        """실제 tinylm.train: 실행 중 일시정지 → 저장·대기(진행 없음) → 재개 → 1500단계 완료, 체크포인트 삭제,
        일시정지 없이 학습한 것과 가중치가 같다."""
        self.app.cfg.data["training_checkpoint"] = {"enabled": True, "every_minutes": 0, "every_steps": 50, "dir": ""}
        jid = self.app.jobs.submit("train_tiny", dict(JOB), "admin")
        wait_for(lambda: self.saved_step(jid) >= 100, 60, "100단계 이상 정기 저장")
        self.app.jobs.pause(jid)
        wait_for(lambda: self.job(jid)["status"] == "paused", 30, "일시정지")
        a = self.app.annotate_job(self.job(jid))
        ck = a["checkpoint"]
        k = ck["step"]
        self.assertEqual((ck["kind"], ck["steps"]), ("tiny", STEPS))
        self.assertTrue(100 <= k < STEPS, ck)
        self.assertEqual(Path(ck["path"]), self.app.checkpoint_root(jid))
        self.assertRegex(a["message"], rf"^일시정지됨 · {k}/{STEPS}단계 · \d\d:\d\d 저장$")
        self.assertEqual((a["can_pause"], a["can_resume"], a["can_finish"], a["can_restart"]), (False, True, True, False))
        self.assertEqual(a["result"]["checkpoint"]["step"], k)
        with self.assertRaises(ValueError):
            self.app.jobs.pause(jid)                         # 이미 일시정지됨

        time.sleep(1.0)                                       # 일시정지 중에는 진행하지 않는다
        self.assertEqual(self.job(jid)["status"], "paused")
        self.assertEqual(checkpoint.summary(self.app.checkpoint_root(jid))["step"], k)
        meta = checkpoint.load(self.app.checkpoint_root(jid))[0]
        self.assertEqual(meta["kind"], "tiny")
        self.assertTrue(Path(meta["train_path"]).is_file())

        self.app.jobs.resume(jid)
        done = self.wait_done(jid)
        self.assertEqual(done["status"], "done", done["message"])
        res = done["result"]
        self.assertEqual((res["steps"], res["total_steps"], res["stopped_early"]), (STEPS, STEPS, False))
        self.assertNotIn("resumed_from_step", res)           # 같은 작업 안의 재개는 '이어서 재개'가 아니다
        self.assertNotIn("checkpoint", res)
        self.assertLess(res["train_seconds"], res["seconds"] + 1e-6)
        self.assertFalse(self.app.checkpoint_root(jid).exists())
        model_dir = self.app.db.get_model(res["model_id"])["path"]
        self.assertTrue((Path(model_dir) / "model.onnx").is_file())
        self.assert_same_weights(self.weights(model_dir), self.reference_weights(meta, "pause"))

    def test_device_error_then_http_restart_matches_uninterrupted(self):
        """정기 저장(100단계마다) 뒤 252단계에서 장치 오류로 실패 → result.checkpoint(200단계) 보존 →
        POST /restart → 새 작업이 200단계 다음부터 1500단계까지 → 끊김 없는 학습과 가중치 동일."""
        s, r = self.call("POST", "/api/settings", {"training_checkpoint": {
            "enabled": True, "every_minutes": 0, "every_steps": 100, "dir": ""}}, token=self.admin)
        self.assertEqual(s, 200, r)
        from learnbox import tinylm

        real_train = tinylm.train

        def unplugged(*a, **kw):          # 처음 실행만: 250단계를 넘기면 저장장치가 빠진 것처럼 OSError
            if kw.get("resume_checkpoint"):
                return real_train(*a, **kw)
            inner = kw["progress"]

            def progress(p, m=""):
                hit = re.match(r"소형 AI 모델 학습 (\d+)/", str(m))
                if hit and int(hit.group(1)) >= 250:
                    raise OSError(5, "입출력 오류: 저장장치 연결이 끊겼습니다 (테스트)")
                inner(p, m)
            return real_train(*a, **{**kw, "progress": progress})

        with mock.patch("learnbox.tinylm.train", unplugged):
            s, r = self.call("POST", "/api/train/tiny", dict(JOB), token=self.admin)
            self.assertEqual(s, 200, r)
            jid = r["job_id"]
            failed = self.wait_done(jid)
        self.assertEqual(failed["status"], "failed", failed["message"])
        self.assertIn("저장장치 연결이 끊겼습니다", failed["message"])
        self.assertEqual(failed["result"]["checkpoint"]["step"], 200)
        self.assertTrue(failed["result"]["resumable"])
        root = self.app.checkpoint_root(jid)
        meta = checkpoint.load(root)[0]
        self.assertEqual((meta["step"], meta["steps"], meta["resumes"]), (200, STEPS, 0))
        a = self.listed(jid)
        self.assertEqual((a["checkpoint"]["step"], a["checkpoint"]["steps"]), (200, STEPS))
        self.assertEqual((a["can_restart"], a["can_pause"], a["can_resume"], a["can_finish"]), (True, False, False, False))

        s, r = self.call("POST", f"/api/jobs/{jid}/restart", {}, token=self.admin)
        self.assertEqual(s, 200, r)
        new = r["job_id"]
        done = self.wait_done(new)
        self.assertEqual(done["status"], "done", done["message"])
        res = done["result"]
        self.assertEqual((res["total_steps"], res["steps"], res["resumed_from_step"], res["resumes"]),
                         (STEPS, STEPS, 200, 1))
        self.assertEqual(res["resumed_from_job"], jid)
        self.assertTrue(res["corpus"]["resumed"])
        self.assertFalse(root.exists())                       # 성공하면 체크포인트 삭제
        orig = self.listed(jid)
        self.assertEqual((orig["result"]["resumed_by"], orig["can_restart"], orig["checkpoint"]), (new, False, None))
        self.assertIn(f"작업 #{new}(으)로 이어서 재개함", orig["message"])
        model_dir = self.app.db.get_model(res["model_id"])["path"]
        self.assert_same_weights(self.weights(model_dir), self.reference_weights(meta, "restart"))


class OrphanModelDirTest(unittest.TestCase):
    """강제 종료(kill -9)로 정리하지 못한 빈 모델 폴더는 다음 시작 때 지운다 (등록된 것·내용 있는 것·다른 이름은 그대로)."""

    def test_start_removes_only_empty_unregistered_training_dirs(self):
        d = tempfile.mkdtemp(prefix="lb-orphan-")
        self.addCleanup(shutil.rmtree, d, True)
        app = App(d)
        md = app.cfg.models_dir
        for name in ("tiny-20261119-130044", "lora-20261119-130206-2", "tiny-20261119-130107", "tiny-20261119-130500",
                     "my-folder", "tiny-notes"):
            (md / name).mkdir()
        (md / "tiny-20261119-130107" / "model.pt").write_bytes(b"x")   # 내용 있음 -> 그대로
        app.db.add_model("tiny", "tiny-20261119-130500", str(md / "tiny-20261119-130500"), {})   # 등록됨 -> 그대로
        self.assertEqual(sorted(app.remove_orphan_model_dirs()), ["lora-20261119-130206-2", "tiny-20261119-130044"])
        self.assertEqual(sorted(p.name for p in md.iterdir()),
                         ["my-folder", "tiny-20261119-130107", "tiny-20261119-130500", "tiny-notes"])
        self.assertEqual(app.remove_orphan_model_dirs(), [])


if __name__ == "__main__":
    unittest.main()
