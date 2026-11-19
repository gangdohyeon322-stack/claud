"""PC 학습 일시정지 · 지금까지 저장하고 끝내기 · 끊긴 학습 이어서 재개 (작업 계층 · HTTP · 명령줄).

docs/SPEC-pause-resume.md §2–§5, §7 셋째 묶음. 실제 JobManager와 스레드 서버(in-process)로 확인한다.
대부분은 체크포인트 규약(§1)을 지키는 가짜 학습 함수(FakeTrain)로 빠르게 돌리고, 실제 tinylm.train과
train_pc.train_lora로도 '일시정지 → 저장 후 중단 → 이어서 재개 → 완료'를 끝까지 한 번씩 확인한다.
"""
import _thread
import contextlib
import inspect
import io
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
from learnbox.db import DB

KO_TEXT = "딥 러닝은 여러 층의 인공 신경망을 사용하는 기계 학습 방법이다.\n" * 20
TINY_ARCH = {"preset": "custom", "d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 16, "vocab_size": 100}
LIVE = ("queued", "running", "paused")
CK_OFF_TIMERS = {"enabled": True, "every_minutes": 0, "every_steps": 0, "dir": ""}


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


class FakeTrain:
    """tinylm.train 대역. SPEC §1 규약(pause_point(save) / stop_requested / on_checkpoint / resume_checkpoint)을 지키며
    가중치 대신 단계 번호를 state.json에 저장한다 (learnbox.checkpoint로 실제 체크포인트를 쓴다)."""

    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.fail_at = None        # 이 단계에서 예외 (저장장치 끊김 흉내)
        self.on_step = None        # callable(step): 테스트가 특정 단계에서 할 일
        self.step = 0
        self.steps_by_root: dict[str, int] = {}   # 체크포인트 root(없으면 'None') -> 지금 단계
        self.calls: list[dict] = []

    def __call__(self, train_path, eval_path, out_dir, arch, steps=1500, batch_size=32, lr=3e-3, resume_from=None,
                 device=None, threads=None, progress=lambda p, m: None, check_cancel=lambda: None, onnx_int8=True,
                 seed=0, max_seconds=None, checkpoint_dir=None, checkpoint_every_steps=0, checkpoint_every_sec=600.0,
                 resume_checkpoint=None, pause_point=None, stop_requested=None, on_checkpoint=None):
        self.calls.append({"checkpoint_dir": checkpoint_dir, "every_steps": checkpoint_every_steps,
                           "every_sec": checkpoint_every_sec, "resume_checkpoint": resume_checkpoint,
                           "train_path": train_path, "arch": arch, "steps": steps})
        start, resumes = 0, 0
        if resume_checkpoint:
            got = checkpoint.load(resume_checkpoint)
            if not got:
                raise ValueError(f"이어서 할 체크포인트를 찾을 수 없습니다: {resume_checkpoint}")
            meta, d = got
            if meta.get("kind") != "tiny":
                raise ValueError("소형 AI 모델의 체크포인트가 아닙니다.")
            assert json.loads((d / "state.json").read_text())["w"] == meta["step"]
            steps, start, train_path = int(meta["steps"]), int(meta["step"]), meta["train_path"]
            resumes = int(meta.get("resumes") or 0) + 1
            if not Path(train_path).is_file():
                raise ValueError(f"체크포인트에 기록된 학습 자료가 없습니다: {train_path}. 이어서 할 수 없습니다.")
        step = start
        last = {"step": None, "summary": None}

        def save():
            if not checkpoint_dir:
                return None
            if last["step"] == step:
                return last["summary"]
            tmp = checkpoint.begin(checkpoint_dir)
            (tmp / "state.json").write_text(json.dumps({"w": step}))
            res = checkpoint.commit(checkpoint_dir, tmp, {"kind": "tiny", "step": step, "steps": steps,
                                                          "train_path": str(train_path), "resumes": resumes})
            s = {**res, "steps": steps}
            last.update(step=step, summary=s)
            if on_checkpoint:
                on_checkpoint(s)
            return s

        stopped, notes = False, []
        while step < steps:
            step += 1
            self.step = self.steps_by_root[str(checkpoint_dir)] = step
            check_cancel()
            time.sleep(self.delay)
            if self.fail_at == step:
                raise RuntimeError("저장장치 연결이 끊겼습니다 (테스트)")
            if checkpoint_dir and checkpoint_every_steps and step % checkpoint_every_steps == 0:
                save()
            progress(step / steps, f"소형 AI 모델 학습 {step}/{steps}")
            if self.on_step:
                self.on_step(step)
            if pause_point is not None:
                pause_point(save)
            if stop_requested is not None and step < steps and stop_requested():
                save()
                stopped = True
                notes.append(f"사용자가 {step}/{steps}단계에서 학습을 끝냈습니다.")
                break
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.pt").write_bytes(b"fake")
        info = {"kind": "tiny", "name": out.name, "steps": step, "total_steps": step, "stopped_early": stopped,
                "notes": notes, "eval_loss_before": 4.0, "eval_loss_after": 3.0}
        if resume_checkpoint:
            info.update(resumed_from_step=start, resumes=resumes)
        return info


class FakeFetcher:
    class Page:
        def __init__(self, url):
            self.url, self.title, self.links = url, "수집한 페이지", []
            self.text = "인공 신경망은 뇌의 뉴런을 본뜬 계산 모델이다. " * 15

    def fetch_page(self, url):
        return self.Page(url)


def wait_for(pred, timeout=30.0, what="조건"):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError(f"{timeout}초 안에 {what}이(가) 맞지 않았습니다")


def wait_job(app, job_id, timeout=60.0):
    """작업이 대기/실행/일시정지를 벗어날 때까지 기다린다."""
    wait_for(lambda: app.db.get_job(job_id)["status"] not in LIVE, timeout, f"작업 #{job_id} 끝남")
    return app.db.get_job(job_id)


def wait_status(app, job_id, status, timeout=30.0):
    wait_for(lambda: app.db.get_job(job_id)["status"] == status, timeout, f"작업 #{job_id} 상태 {status}")
    return app.db.get_job(job_id)


def make_tiny_checkpoint(root: Path, step: int, steps: int, kind: str = "tiny", train_path: str = "/x/train.jsonl"):
    tmp = checkpoint.begin(root)
    (tmp / "state.json").write_text(json.dumps({"w": step}))
    return {**checkpoint.commit(root, tmp, {"kind": kind, "step": step, "steps": steps, "train_path": train_path}),
            "steps": steps}


@unittest.skipUnless(_has_torch(), "PyTorch 없음")
class PauseResumeServerTest(unittest.TestCase):
    """실제 서버(스레드) + 실제 작업 관리자. tinylm.train은 FakeTrain으로 바꿔 끼운다 (실제 학습 테스트는 따로)."""

    @classmethod
    def setUpClass(cls):
        from learnbox.server import Sessions, make_handler

        cls.dir = tempfile.mkdtemp(prefix="lb-pause-")
        cls.src = tempfile.mkdtemp(prefix="lb-pause-src-")
        (Path(cls.src) / "노트.txt").write_text(KO_TEXT, encoding="utf-8")
        cls.app = App(cls.dir)
        cls.app.cfg.set_pins("1234", "987654")
        cls.app.cfg.data["embeddings"]["enabled"] = False
        cls.app.fetcher = FakeFetcher()                      # 수집 줄은 네트워크 없이 돈다
        for i in range(12):
            cls.app.db.add_document(f"https://a/{i}", f"딥 러닝 {i}", KO_TEXT + f"번호 {i}\n" * 4, "link")
        cls.app.jobs.start()
        handler = make_handler(cls.app, Sessions())
        handler.log_message = lambda *a, **k: None
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.user, cls.admin = cls.login("1234"), cls.login("987654")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        shutil.rmtree(cls.dir, ignore_errors=True)
        shutil.rmtree(cls.src, ignore_errors=True)

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

    @classmethod
    def login(cls, pin):
        return cls.call("POST", "/api/login", {"pin": pin})[1]["token"]

    def setUp(self):
        self.app.cfg.data["training_checkpoint"] = dict(CK_OFF_TIMERS)
        self.fake = FakeTrain()
        if "_real_" not in self._testMethodName:          # 실제 학습 테스트는 바꿔 끼우지 않는다
            patcher = mock.patch("learnbox.tinylm.train", self.fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        for j in self.app.db.list_jobs(200):
            if j["status"] in LIVE:
                self.app.jobs.cancel(j["id"])
        wait_for(lambda: not any(j["status"] in LIVE for j in self.app.db.list_jobs(200)), 30, "작업 정리")

    # ----- 도우미 -----
    def start_tiny(self, steps, **extra):
        s, r = self.call("POST", "/api/train/tiny", {**TINY_ARCH, "steps": steps, "batch_size": 4, **extra},
                         token=self.admin)
        self.assertEqual(s, 200, r)
        return r["job_id"]

    def listed(self, job_id) -> dict:
        s, r = self.call("GET", "/api/jobs", token=self.user)
        self.assertEqual(s, 200)
        return next(j for j in r["jobs"] if j["id"] == job_id)

    def post(self, job_id, action, token=None):
        return self.call("POST", f"/api/jobs/{job_id}/{action}", {}, token=token or self.admin)

    def reached(self, job_id, min_step) -> bool:
        root = self.app.checkpoint_root(job_id)
        return self.app.jobs.is_active(job_id) and self.fake.steps_by_root.get(str(root), 0) >= min_step

    def pause_now(self, job_id, min_step=3) -> dict:
        wait_for(lambda: self.reached(job_id, min_step), 30, f"{min_step}단계 도달")
        s, r = self.post(job_id, "pause")
        self.assertEqual(s, 200, r)
        return wait_status(self.app, job_id, "paused")

    # ----- 테스트 -----
    def test_pause_keeps_heavy_lane_then_resume_completes_and_removes_checkpoint(self):
        self.fake.delay = 0.03
        jid = self.start_tiny(40)
        wait_for(lambda: self.reached(jid, 3), 30, "학습 시작")
        j = self.listed(jid)
        self.assertEqual(j["status"], "running")
        self.assertEqual((j["can_pause"], j["can_resume"], j["can_finish"], j["can_restart"]), (True, False, True, False))
        self.assertIsNone(j["checkpoint"])                                    # 정기 저장을 꺼 둠 -> 아직 없음

        s, r = self.post(jid, "pause")
        self.assertEqual(s, 200, r)
        self.assertTrue(r["job"]["pause_requested"] or r["job"]["status"] == "paused")
        wait_status(self.app, jid, "paused")
        j = self.listed(jid)
        ck = j["checkpoint"]
        self.assertIsNotNone(ck, j)
        self.assertEqual(ck["steps"], 40)
        self.assertEqual(Path(ck["path"]), self.app.checkpoint_root(jid))
        self.assertRegex(j["message"], rf"^일시정지됨 · {ck['step']}/40단계 · \d\d:\d\d 저장$")
        self.assertEqual(j["result"]["checkpoint"]["step"], ck["step"])     # on_checkpoint가 result에 남김
        self.assertEqual((j["can_pause"], j["can_resume"], j["can_finish"], j["can_restart"]), (False, True, True, False))
        self.assertEqual(self.post(jid, "pause")[0], 400)                     # 이미 일시정지됨

        # 일시정지 중에는 진행하지 않고, 무거운 줄은 막혀 있다. 수집/가벼운 줄은 계속 돈다.
        paused_step = self.fake.step
        heavy2 = self.start_tiny(3)
        light = self.app.jobs.submit("ingest_path", {"path": self.src, "actor": "t"}, "admin")
        coll = self.app.jobs.submit("collect_links", {"urls": ["https://x.test/a"], "depth": 0}, "admin")
        self.assertEqual(wait_job(self.app, light)["status"], "done")
        self.assertEqual(wait_job(self.app, coll)["status"], "done")
        time.sleep(0.6)
        self.assertEqual(self.fake.step, paused_step)
        self.assertEqual(self.app.db.get_job(heavy2)["status"], "queued")
        self.assertEqual(self.app.db.running_jobs(("train_tiny",)), 2)        # 일시정지도 '실행 중'으로 센다
        s, r = self.post(heavy2, "pause")
        self.assertEqual(s, 400)
        self.assertIn("아직 시작하지 않은", r["error"])

        s, r = self.post(jid, "resume")
        self.assertEqual(s, 200, r)
        done = wait_job(self.app, jid)
        self.assertEqual(done["status"], "done", done)
        self.assertEqual(done["result"]["steps"], 40)
        self.assertFalse(done["result"]["stopped_early"])
        self.assertNotIn("checkpoint", done["result"])                       # 최종 result가 덮어씀
        self.assertIsNotNone(self.app.db.get_model(done["result"]["model_id"]))
        self.assertFalse(Path(ck["path"]).exists())                           # 성공하면 체크포인트 삭제
        self.assertEqual(wait_job(self.app, heavy2)["status"], "done")
        self.assertEqual(self.post(jid, "resume")[0], 400)                    # 끝난 작업

    def test_finish_while_running_registers_stopped_early_model(self):
        jid = self.start_tiny(500)
        wait_for(lambda: self.reached(jid, 4), 30, "4단계 도달")
        s, r = self.post(jid, "finish")
        self.assertEqual(s, 200, r)
        self.assertEqual(self.post(jid, "finish")[0], 400)                   # 이미 끝내는 중
        done = wait_job(self.app, jid)
        self.assertEqual(done["status"], "done", done)
        res = done["result"]
        self.assertTrue(res["stopped_early"])
        self.assertLess(res["steps"], 500)
        self.assertIn(f"사용자가 {res['steps']}/500단계에서 학습을 끝냈습니다.", res["notes"])
        m = self.app.db.get_model(res["model_id"])
        self.assertTrue(m and m["info"]["stopped_early"])
        self.assertFalse(self.app.checkpoint_root(jid).exists())

    def test_finish_from_paused(self):
        jid = self.start_tiny(500)
        self.pause_now(jid)
        step = self.listed(jid)["checkpoint"]["step"]
        s, r = self.post(jid, "finish")
        self.assertEqual(s, 200, r)
        done = wait_job(self.app, jid)
        self.assertEqual(done["status"], "done", done)
        self.assertTrue(done["result"]["stopped_early"])
        self.assertEqual(done["result"]["steps"], step)
        self.assertFalse(self.app.checkpoint_root(jid).exists())

    def test_cancel_paused_job_removes_checkpoint_and_frees_lane(self):
        jid = self.start_tiny(500)
        self.pause_now(jid)
        root = self.app.checkpoint_root(jid)
        self.assertTrue(root.is_dir())
        self.assertEqual(self.post(jid, "cancel", token=self.user)[0], 403)   # 학습은 관리자만 취소
        self.assertEqual(self.post(jid, "cancel")[0], 200)
        j = wait_job(self.app, jid, 10)
        self.assertEqual((j["status"], j["message"]), ("cancelled", "취소됨"))
        self.assertIsNone(j["result"]["checkpoint"])
        self.assertFalse(root.exists())                                       # 취소 = 사용자가 버림
        self.assertIsNone(self.listed(jid)["checkpoint"])
        self.assertEqual(wait_job(self.app, self.start_tiny(2))["status"], "done")   # 무거운 줄이 풀렸다

    def test_crash_keeps_checkpoint_and_restart_continues_to_target(self):
        self.app.cfg.data["training_checkpoint"]["every_steps"] = 5
        self.fake.fail_at = 12
        models_before = len(self.app.db.list_models())
        jid = self.start_tiny(30)
        j = wait_job(self.app, jid)
        self.assertEqual(j["status"], "failed")
        self.assertIn("저장장치 연결이 끊겼습니다", j["message"])
        self.assertEqual(j["result"]["checkpoint"]["step"], 10)
        self.assertTrue(j["result"]["resumable"])
        self.assertIn("traceback", j["result"])
        self.assertEqual(len(self.app.db.list_models()), models_before)       # 실패한 모델 폴더는 등록 안 함
        root = self.app.checkpoint_root(jid)
        a = self.listed(jid)
        self.assertEqual((a["checkpoint"]["step"], a["checkpoint"]["steps"]), (10, 30))
        self.assertEqual((a["can_restart"], a["can_pause"], a["can_resume"], a["can_finish"]), (True, False, False, False))

        self.fake.fail_at = None
        s, r = self.post(jid, "restart")
        self.assertEqual(s, 200, r)
        new = r["job_id"]
        self.assertEqual(self.post(jid, "restart")[0], 400)                  # 같은 체크포인트로 두 번은 안 됨
        b = wait_job(self.app, new)
        self.assertEqual(b["status"], "done", b)
        self.assertEqual(b["params"]["resume_checkpoint"], str(root))         # 원래 root를 그대로 사용
        self.assertEqual(b["params"]["resumed_from_job"], jid)
        call = self.fake.calls[-1]
        self.assertEqual((call["resume_checkpoint"], call["checkpoint_dir"]), (str(root), str(root)))
        self.assertIsNone(call["train_path"])                                 # 말뭉치를 새로 만들지 않음
        res = b["result"]
        self.assertEqual((res["resumed_from_step"], res["total_steps"], res["resumes"]), (10, 30, 1))
        self.assertEqual(res["resumed_from_job"], jid)
        self.assertTrue(res["corpus"]["resumed"])
        orig = self.app.db.get_job(jid)
        self.assertEqual(orig["result"]["resumed_by"], new)
        self.assertEqual(orig["message"], f"RuntimeError: 저장장치 연결이 끊겼습니다 (테스트) · 작업 #{new}(으)로 이어서 재개함")
        self.assertFalse(root.exists())
        a = self.listed(jid)
        self.assertEqual((a["can_restart"], a["checkpoint"]), (False, None))
        s, r = self.post(jid, "restart")
        self.assertEqual(s, 400)
        self.assertIn(f"이미 작업 #{new}", r["error"])

    def test_cancelled_queued_restart_gives_restart_back(self):
        self.app.cfg.data["training_checkpoint"]["every_steps"] = 2
        self.fake.fail_at = 5
        jid = self.start_tiny(10)
        self.assertEqual(wait_job(self.app, jid)["status"], "failed")
        self.fake.fail_at = None
        blocker = self.start_tiny(500)                                        # 무거운 줄을 막아 새 작업을 대기시킨다
        self.pause_now(blocker, 1)
        s, r = self.post(jid, "restart")
        self.assertEqual(s, 200, r)
        new = r["job_id"]
        self.assertEqual(self.app.db.get_job(new)["status"], "queued")
        self.assertFalse(self.listed(jid)["can_restart"])
        self.assertEqual(self.post(new, "cancel")[0], 200)
        self.assertEqual(self.app.db.get_job(new)["status"], "cancelled")
        a = self.listed(jid)
        self.assertTrue(a["can_restart"], a)                                  # 체크포인트는 그대로, 다시 재개 가능
        self.assertEqual(a["checkpoint"]["step"], 4)
        self.assertEqual(a["message"], "RuntimeError: 저장장치 연결이 끊겼습니다 (테스트) · 이어서 재개할 수 있습니다 (4/10단계)")

    def test_server_shutdown_saves_and_marks_interrupted(self):
        jid = self.start_tiny(400)
        wait_for(lambda: self.reached(jid, 3), 30, "3단계 도달")
        ids = self.app.suspend_training(timeout=10)
        self.assertEqual(ids, [jid])
        j = self.app.db.get_job(jid)
        self.assertEqual(j["status"], "interrupted")
        step = j["result"]["checkpoint"]["step"]
        self.assertEqual(j["message"], f"서버를 종료해 중단됨 · 이어서 재개할 수 있습니다 ({step}/400단계)")
        self.assertTrue(j["result"]["resumable"])
        self.assertTrue(self.listed(jid)["can_restart"])
        s, r = self.post(jid, "restart")
        self.assertEqual(s, 200, r)
        self.assertEqual(self.app.db.get_job(r["job_id"])["actor"], "admin")  # 다른 작업처럼 역할만 (IP는 감사 기록에)
        b = wait_job(self.app, r["job_id"])
        self.assertEqual((b["status"], b["result"]["resumed_from_step"], b["result"]["total_steps"]), ("done", step, 400))

    def test_permissions_bad_states_and_discard(self):
        self.app.cfg.data["training_checkpoint"]["every_steps"] = 2
        self.fake.fail_at = 4
        jid = self.start_tiny(10)
        self.assertEqual(wait_job(self.app, jid)["status"], "failed")
        root = self.app.checkpoint_root(jid)
        self.assertEqual(checkpoint.summary(root)["step"], 2)
        for action in ("pause", "resume", "finish", "restart", "discard-checkpoint"):
            self.assertEqual(self.post(jid, action, token=self.user)[0], 403, action)
            self.assertEqual(self.call("POST", f"/api/jobs/{jid}/{action}", {})[0], 401, action)
            self.assertEqual(self.post(99999, action)[0], 404, action)
        for action in ("pause", "resume", "finish"):                          # 실패한 작업은 실행 중이 아님
            s, r = self.post(jid, action)
            self.assertEqual(s, 400, action)
            self.assertRegex(r["error"], r"[가-힣]")
        self.fake.fail_at = None
        ok = self.start_tiny(2)
        self.assertEqual(wait_job(self.app, ok)["status"], "done")
        s, r = self.post(ok, "restart")
        self.assertEqual(s, 400)
        self.assertIn("중단되었거나 실패한", r["error"])
        s, r = self.post(ok, "discard-checkpoint")
        self.assertEqual(s, 400)
        self.assertIn("지울 체크포인트가 없습니다", r["error"])
        ing = self.app.jobs.submit("ingest_path", {"path": self.src, "actor": "t"}, "admin")
        wait_job(self.app, ing)
        s, r = self.post(ing, "pause")
        self.assertEqual(s, 400)
        self.assertIn("학습 작업만", r["error"])
        self.assertEqual(self.post(ing, "restart")[0], 400)
        self.assertFalse(self.listed(ing)["can_pause"])

        s, r = self.post(jid, "discard-checkpoint")
        self.assertEqual(s, 200, r)
        self.assertGreater(r["bytes"], 0)
        self.assertFalse(root.exists())
        j = self.app.db.get_job(jid)
        self.assertEqual((j["result"]["checkpoint"], j["result"]["checkpoint_discarded"]), (None, True))
        self.assertNotIn("resumable", j["result"])
        # '이어서 재개할 수 있습니다' 안내는 지우고 사유는 남긴다
        self.assertEqual(j["message"], "RuntimeError: 저장장치 연결이 끊겼습니다 (테스트) · 체크포인트 삭제함")
        self.assertFalse(self.listed(jid)["can_restart"])
        s, r = self.post(jid, "restart")
        self.assertEqual(s, 400)
        self.assertIn("체크포인트가 없습니다", r["error"])
        self.assertEqual(self.post(jid, "discard-checkpoint")[0], 400)

    def test_train_params_checkpoint_settings_and_validation(self):
        jid = self.start_tiny(3, checkpoint_minutes=0.5, resume_checkpoint="/etc", resumed_from_job=1)
        j = wait_job(self.app, jid)
        self.assertEqual(j["status"], "done", j)
        self.assertNotIn("resume_checkpoint", j["params"])                   # 체크포인트 경로는 restart만 정한다
        self.assertNotIn("resumed_from_job", j["params"])
        call = self.fake.calls[-1]
        self.assertEqual((call["every_sec"], call["every_steps"], call["resume_checkpoint"]), (30.0, 0, None))
        self.assertEqual(call["checkpoint_dir"], str(self.app.checkpoint_root(jid)))
        self.assertEqual(self.call("POST", "/api/train/tiny", {**TINY_ARCH, "checkpoint_minutes": -1},
                                   token=self.admin)[0], 400)

        for bad in ({"every_minutes": -1}, {"every_minutes": "abc"}, {"every_steps": -5}, {"dir": "상대/경로"}, "x"):
            s, r = self.call("POST", "/api/settings", {"training_checkpoint": bad}, token=self.admin)
            self.assertEqual(s, 400, bad)
            self.assertRegex(r["error"], r"[가-힣]")
        ck_dir = tempfile.mkdtemp(prefix="lb-ck-")
        try:
            s, r = self.call("POST", "/api/settings", {"training_checkpoint": {
                "enabled": True, "every_minutes": "5", "every_steps": 100, "dir": ck_dir}}, token=self.admin)
            self.assertEqual(s, 200, r)
            self.assertEqual(r["training_checkpoint"], {"enabled": True, "every_minutes": 5, "every_steps": 100,
                                                        "dir": ck_dir})
            self.assertEqual(self.app.checkpoint_root(7), Path(ck_dir).resolve() / "job-7")
            jid = self.start_tiny(2)
            self.assertEqual(wait_job(self.app, jid)["status"], "done")
            call = self.fake.calls[-1]
            self.assertEqual((call["every_sec"], call["every_steps"]), (300.0, 100))
            self.assertTrue(call["checkpoint_dir"].startswith(str(Path(ck_dir).resolve())))
            s, r = self.call("POST", "/api/settings", {"training_checkpoint": {"enabled": False}}, token=self.admin)
            self.assertEqual(s, 200)
            self.assertIsNone(self.app.checkpoint_root(7))
            jid = self.start_tiny(2)
            self.assertEqual(wait_job(self.app, jid)["status"], "done")
            self.assertIsNone(self.fake.calls[-1]["checkpoint_dir"])          # 꺼져 있으면 저장하지 않음
            self.assertEqual(self.call("POST", "/api/settings", {"training_checkpoint": {"every_minutes": 1}},
                                       token=self.user)[0], 403)
        finally:
            shutil.rmtree(ck_dir, ignore_errors=True)

    def test_pause_without_checkpoint_dir(self):
        self.app.cfg.data["training_checkpoint"]["enabled"] = False
        jid = self.start_tiny(500)
        j = self.pause_now(jid)
        self.assertEqual(j["message"], "일시정지됨 · 체크포인트 없음")
        self.assertIsNone(self.listed(jid)["checkpoint"])
        self.assertEqual(self.post(jid, "resume")[0], 200)
        wait_status(self.app, jid, "running")
        self.assertEqual(self.post(jid, "finish")[0], 200)
        self.assertEqual(wait_job(self.app, jid)["status"], "done")

    def test_real_tinylm_pause_interrupt_restart(self):
        """실제 tinylm.train: 일시정지(저장) → 저장 후 중단 → /restart → 목표 단계까지 완료."""
        from learnbox import tinylm

        self.assertNotIsInstance(tinylm.train, FakeTrain)
        params = inspect.signature(tinylm.train).parameters
        if not {"checkpoint_dir", "pause_point", "stop_requested", "on_checkpoint", "resume_checkpoint"} <= set(params):
            self.skipTest("tinylm.train에 중간 저장 인자가 아직 없습니다")
        steps = 1500
        jid = self.start_tiny(steps)
        wait_for(lambda: self.app.jobs.is_active(jid), 30, "학습 시작")
        s, r = self.post(jid, "pause")                                        # 학습 고리 전이라도 요청은 받는다
        self.assertEqual(s, 200, r)
        wait_status(self.app, jid, "paused", 60)
        ck = self.listed(jid)["checkpoint"]
        self.assertEqual(ck["kind"], "tiny")
        self.assertEqual(ck["steps"], steps)
        k = ck["step"]
        self.assertTrue(1 <= k < steps, ck)
        self.assertEqual(self.app.suspend_training(timeout=30), [jid])        # 서버 종료처럼: 저장 후 중단
        a = self.app.db.get_job(jid)
        self.assertEqual(a["status"], "interrupted", a)
        self.assertTrue(self.listed(jid)["can_restart"])
        s, r = self.post(jid, "restart")
        self.assertEqual(s, 200, r)
        b = wait_job(self.app, r["job_id"], 120)
        self.assertEqual(b["status"], "done", b)
        res = b["result"]
        self.assertEqual((res["total_steps"], res["steps"], res["resumed_from_step"], res["resumes"]), (steps, steps, k, 1))
        self.assertEqual(res["resumed_from_job"], jid)
        self.assertTrue((Path(self.app.db.get_model(res["model_id"])["path"]) / "model.onnx").is_file())
        self.assertFalse(self.app.checkpoint_root(jid).exists())


class StartupInterruptedTest(unittest.TestCase):
    """엔진 재시작: 대기/실행/일시정지 작업을 '중단됨'으로 바꾸고(result 보존), 체크포인트가 있는 학습은 알린다."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lb-restart-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_mark_interrupted_keeps_result_and_notifies(self):
        old = App(self.dir)
        db = old.db
        run = db.create_job("train_tiny", {"steps": 6000}, "admin")
        db.update_job(run, status="running")
        make_tiny_checkpoint(old.checkpoint_root(run), 4580, 6000)            # result 기록 전에 끊긴 경우: 기본 위치로 찾음
        paused = db.create_job("train_lora", {"max_steps": 8}, "admin")
        s = make_tiny_checkpoint(old.checkpoint_root(paused), 3, 8, kind="lora")
        db.update_job(paused, status="paused", result={"checkpoint": s})
        bare = db.create_job("train_tiny", {}, "admin")
        db.update_job(bare, status="running")
        queued = db.create_job("train_tiny", {"resume_checkpoint": str(old.checkpoint_root(run))}, "admin")
        coll = db.create_job("collect_links", {"urls": ["https://a"]}, "admin")
        db.update_job(coll, status="running", result={"checkpoint": {"step": 1, "steps": 2}})
        done = db.create_job("train_tiny", {}, "admin")
        db.update_job(done, status="done", message="완료")

        app = App(self.dir)                                                   # 새 프로세스처럼
        app.jobs.start()
        j = {i: app.db.get_job(i) for i in (run, paused, bare, queued, coll, done)}
        for i in (run, paused, bare, queued, coll):
            self.assertEqual(j[i]["status"], "interrupted", i)
        self.assertEqual(j[done]["status"], "done")
        self.assertEqual(j[run]["message"], "엔진이 재시작되어 중단됨 · 이어서 재개할 수 있습니다 (4580/6000단계)")
        self.assertEqual(j[paused]["message"], "엔진이 재시작되어 중단됨 · 이어서 재개할 수 있습니다 (3/8단계)")
        self.assertEqual(j[queued]["message"], "엔진이 재시작되어 중단됨 · 이어서 재개할 수 있습니다 (4580/6000단계)")
        self.assertEqual(j[bare]["message"], "엔진이 재시작되어 중단됨")
        self.assertEqual(j[paused]["result"], {"checkpoint": s})               # result는 그대로
        self.assertIsNotNone(j[coll]["result"])
        notes = [n for n in app.db.list_notifications() if n["data"].get("kind") == "train_interrupted"]
        self.assertEqual(sorted(n["data"]["job_id"] for n in notes), sorted([run, paused, queued]))
        n = next(n for n in notes if n["data"]["job_id"] == run)
        self.assertEqual((n["level"], n["title"]), ("warning", "학습이 중단되었습니다"))
        self.assertEqual(n["body"], "작업 탭에서 '이어서 재개'를 누르면 4580/6000단계부터 계속합니다.")
        a = app.annotate_job(app.db.get_job(run))
        self.assertTrue(a["can_restart"])
        self.assertEqual(a["checkpoint"]["step"], 4580)
        self.assertFalse(app.annotate_job(app.db.get_job(bare))["can_restart"])

    def test_db_mark_interrupted_default_uses_result(self):
        db = DB(Path(self.dir) / "x.db")
        a = db.create_job("train_tiny", {}, "t")
        db.update_job(a, status="paused", result={"checkpoint": {"step": 5, "steps": 9, "path": "/nope"}})
        b = db.create_job("ingest_path", {}, "t")
        db.update_job(b, status="queued")
        out = {j["id"]: j for j in db.mark_interrupted()}
        self.assertEqual(set(out), {a, b})
        self.assertEqual(out[a]["checkpoint"]["step"], 5)
        self.assertEqual(db.get_job(a)["message"], "엔진이 재시작되어 중단됨 · 이어서 재개할 수 있습니다 (5/9단계)")
        self.assertEqual(db.get_job(a)["result"]["checkpoint"]["step"], 5)
        self.assertEqual(db.get_job(b)["message"], "엔진이 재시작되어 중단됨")
        self.assertEqual(db.mark_interrupted(), [])


@unittest.skipUnless(_has_torch(), "PyTorch 없음")
class CliPauseResumeTest(unittest.TestCase):
    """명령줄: Ctrl+C면 저장하고 멈춘 뒤 resume-job 안내, jobs 목록, resume-job으로 이어서 완료."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="lb-cli-pause-")
        app = App(cls.dir)
        for i in range(10):
            app.db.add_document(f"https://c/{i}", f"문서 {i}", KO_TEXT + f"번호 {i}\n" * 3, "link")
        app.cfg.data["training_checkpoint"] = dict(CK_OFF_TIMERS)
        app.cfg.save()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def run_cli(self, *argv):
        from learnbox.__main__ import main as cli_main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = cli_main(["--data", self.dir, *argv])
        return rc, out.getvalue()

    def test_ctrl_c_saves_then_jobs_and_resume_job(self):
        steps = 300
        fake = FakeTrain(delay=0.005)
        fake.on_step = lambda s: _thread.interrupt_main() if s == 5 else None   # 5단계에서 Ctrl+C
        arch = ["--preset", "custom", "--d-model", "32", "--n-layer", "1", "--n-head", "2", "--seq-len", "16",
                "--vocab-size", "100"]
        with mock.patch("learnbox.tinylm.train", fake):
            rc, out = self.run_cli("train-tiny", *arch, "--steps", str(steps), "--checkpoint-minutes", "0")
        self.assertEqual(rc, 130, out)
        jid = int(re.search(r"학습 작업 #(\d+)", out).group(1))
        self.assertIn(f"python -m learnbox --data {self.dir} resume-job {jid}로 이어서 할 수 있습니다.", out)
        db = DB(Path(self.dir) / "learnbox.db")
        j = db.get_job(jid)
        self.assertEqual(j["status"], "interrupted")
        self.assertEqual(j["params"]["checkpoint_minutes"], 0)
        self.assertEqual(fake.calls[0]["every_sec"], 0)
        k = j["result"]["checkpoint"]["step"]
        self.assertTrue(5 <= k < steps, k)
        self.assertEqual(j["message"], f"명령줄에서 중지됨 · 이어서 재개할 수 있습니다 ({k}/{steps}단계)")

        rc, out = self.run_cli("jobs")
        self.assertEqual(rc, 0)
        self.assertIn(f"#{jid} 소형 AI 학습 · 중단 · 체크포인트 {k}/{steps}단계", out)
        self.assertIn(f"resume-job {jid}", out)

        fake2 = FakeTrain(delay=0.001)
        with mock.patch("learnbox.tinylm.train", fake2):
            rc, out = self.run_cli("resume-job", str(jid))
        self.assertEqual(rc, 0, out)
        self.assertIn("[done]", out)
        self.assertIn(f'"resumed_from_step": {k}', out)
        self.assertIn(f'"total_steps": {steps}', out)
        new = db.get_job(jid)["result"]["resumed_by"]
        self.assertEqual(db.get_job(new)["status"], "done")
        self.assertEqual(db.get_job(jid)["message"], f"명령줄에서 중지됨 · 작업 #{new}(으)로 이어서 재개함")
        self.assertFalse(Path(j["result"]["checkpoint"]["path"]).exists())
        rc, out = self.run_cli("resume-job", str(jid))
        self.assertEqual(rc, 1)
        self.assertIn(f"이미 작업 #{new}", out)
        rc, out = self.run_cli("resume-job", "99999")
        self.assertEqual((rc, out.strip()), (1, "작업이 없습니다."))

    def test_parser(self):
        from learnbox.__main__ import build_parser

        p = build_parser()
        self.assertEqual(p.parse_args(["train-tiny", "--checkpoint-minutes", "5"]).checkpoint_minutes, 5.0)
        self.assertEqual(p.parse_args(["train-lora", "--checkpoint-minutes", "0"]).checkpoint_minutes, 0.0)
        self.assertIsNone(p.parse_args(["train-tiny"]).checkpoint_minutes)
        self.assertEqual(p.parse_args(["resume-job", "12"]).job_id, 12)
        self.assertEqual(p.parse_args(["jobs", "--limit", "5"]).limit, 5)


class RealLoraJobTest(unittest.TestCase):
    """실제 train_pc.train_lora로 작업 계층의 '일시정지 → 저장 후 중단 → 이어서 재개'를 확인한다 (모델 캐시 없으면 건너뜀)."""

    TINY = "hf-internal-testing/tiny-random-gpt2"

    @classmethod
    def setUpClass(cls):
        try:
            import peft  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401

            from learnbox import train_pc
        except ImportError as e:
            raise unittest.SkipTest(f"학습 라이브러리 없음: {e}")
        if "pause_point" not in inspect.signature(train_pc.train_lora).parameters:
            raise unittest.SkipTest("train_lora에 중간 저장 인자가 아직 없습니다")
        try:
            train_pc._load_tokenizer(cls.TINY)
        except Exception as e:
            raise unittest.SkipTest(f"테스트 모델을 불러올 수 없음: {e}")
        cls.dir = tempfile.mkdtemp(prefix="lb-lora-job-")
        cls.app = App(cls.dir)
        for i in range(10):
            cls.app.db.add_document(f"https://l/{i}", f"문서 {i}", KO_TEXT + f"번호 {i}\n" * 3, "link")
        cls.app.cfg.data["training_checkpoint"] = dict(CK_OFF_TIMERS)
        cls.app.jobs.start()

    @classmethod
    def tearDownClass(cls):
        for j in cls.app.db.list_jobs(50):
            if j["status"] in LIVE:
                cls.app.jobs.cancel(j["id"])
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_pause_interrupt_restart(self):
        app = self.app
        jid = app.jobs.submit("train_lora", {"base_model": self.TINY, "max_steps": 6, "seq_len": 32, "batch_size": 1,
                                             "grad_accum": 1, "lora_r": 2, "lora_alpha": 4, "use_qa": False}, "t")
        wait_for(lambda: app.jobs.is_active(jid), 30, "학습 시작")
        app.jobs.pause(jid)
        j = wait_status(app, jid, "paused", 120)
        ck = app.annotate_job(j)["checkpoint"]
        self.assertEqual((ck["kind"], ck["steps"]), ("lora", 6))
        k = ck["step"]
        self.assertTrue(1 <= k < 6, ck)
        self.assertTrue(app.annotate_job(j)["can_resume"])
        self.assertEqual(app.suspend_training(timeout=60), [jid])
        self.assertEqual(app.db.get_job(jid)["status"], "interrupted")
        new = app.restart_job(jid, "t")
        b = wait_job(app, new, 180)
        self.assertEqual(b["status"], "done", b)
        res = b["result"]
        self.assertEqual((res["resumed_from_step"], res["resumes"], res["resumed_from_job"]), (k, 1, jid))
        self.assertEqual((res["steps"], res["total_steps"]), (6, 6))      # 목표 단계까지 (소형 AI와 같은 뜻)
        self.assertEqual(res["base_model"], self.TINY)
        self.assertFalse(app.checkpoint_root(jid).exists())


if __name__ == "__main__":
    unittest.main()
