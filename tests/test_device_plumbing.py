"""학습 장치(device)·LoRA 정밀도(amp) 값 전달 테스트 (docs/PRD-tpu-training.md 6.1, 6.4, 6.8, 6.9, 6.11).

설정 → 요청(웹·명령줄·자율 학습) → 학습 함수까지 device/amp가 정해진 우선순위로 전달되는지 확인한다.
TPU·GPU 없이 돈다: 학습 함수는 가짜로 바꿔 끼우고 받은 인자만 본다.

실행: python -m unittest tests.test_device_plumbing -v
"""
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from learnbox import accel, checkpoint, devices
from learnbox.config import DEFAULTS, Config, amp_value, device_value

HAS_TORCH = importlib.util.find_spec("torch") is not None
KO_TEXT = "딥 러닝은 여러 층의 인공 신경망을 사용하는 기계 학습 방법이다.\n" * 20
TINY_ARCH = {"preset": "custom", "d_model": 32, "n_layer": 1, "n_head": 2, "seq_len": 16, "vocab_size": 100}


@contextlib.contextmanager
def env(**values):
    """LEARNBOX_DEVICE / LEARNBOX_AMP를 지운 뒤 주어진 값만 넣는다 (테스트가 끝나면 원래대로)."""
    with mock.patch.dict(os.environ):
        os.environ.pop("LEARNBOX_DEVICE", None)
        os.environ.pop("LEARNBOX_AMP", None)
        for k, v in values.items():
            os.environ[k] = v
        yield


class RecordingTinyTrain:
    """tinylm.train 대역. 시그니처는 tests/test_pause_resume.FakeTrain과 같다(amp도 **kw도 없음):
    소형 AI 학습에 amp를 넘기면 TypeError로 실패한다."""

    def __init__(self):
        self.calls = []

    def __call__(self, train_path, eval_path, out_dir, arch, steps=1500, batch_size=32, lr=3e-3, resume_from=None,
                 device=None, threads=None, progress=lambda p, m: None, check_cancel=lambda: None, onnx_int8=True,
                 seed=0, max_seconds=None, checkpoint_dir=None, checkpoint_every_steps=0, checkpoint_every_sec=600.0,
                 resume_checkpoint=None, pause_point=None, stop_requested=None, on_checkpoint=None):
        self.calls.append({"device": device, "resume_checkpoint": resume_checkpoint, "arch": arch, "steps": steps})
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.pt").write_bytes(b"fake")
        return {"kind": "tiny", "name": out.name, "steps": steps or 1, "total_steps": steps or 1,
                "eval_loss_before": 4.0, "eval_loss_after": 3.0, "notes": []}


class RecordingLoraTrain:
    """train_pc.train_lora 대역: device/amp를 실제 키워드로 받는다 (**ignored로 들어가면 안 됨)."""

    def __init__(self):
        self.calls = []

    def __call__(self, train_path, eval_path, out_dir, base_model, max_steps=200, lr=2e-4, seq_len=512, batch_size=2,
                 grad_accum=4, lora_r=8, lora_alpha=16, qa_path=None, resume_from=None, threads=None,
                 progress=lambda p, m: None, check_cancel=lambda: None, *, device=None, dtype="auto", amp=None,
                 checkpoint_dir=None, checkpoint_every_steps=0, checkpoint_every_sec=600.0, resume_checkpoint=None,
                 pause_point=None, stop_requested=None, on_checkpoint=None):
        self.calls.append({"device": device, "amp": amp, "resume_checkpoint": resume_checkpoint,
                           "base_model": base_model})
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_config.json").write_text("{}", encoding="utf-8")
        return {"kind": "lora", "base_model": base_model, "steps": max_steps or 1, "total_steps": max_steps or 1,
                "eval_loss_before": 2.5, "eval_loss_after": 2.0, "notes": []}


def make_checkpoint(root: Path, kind: str, device: str | None = "cpu", train_path: str = "/x/train.jsonl"):
    tmp = checkpoint.begin(root)
    (tmp / "state.json").write_text("{}")
    meta = {"kind": kind, "step": 1, "steps": 3, "train_path": train_path}
    if device:
        meta["device"] = device
    return checkpoint.commit(root, tmp, meta)


# ---------------------------------------------------------------------------------------------------------------
class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lb-devcfg-")
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_defaults(self):
        self.assertEqual(DEFAULTS["training_device"], "auto")
        self.assertEqual(DEFAULTS["lora_training"]["amp"], "auto")
        cfg = Config(self.dir)
        self.assertEqual((cfg["training_device"], cfg["lora_training"]["amp"]), ("auto", "auto"))

    def test_old_config_file_gets_defaults_and_update_merges(self):
        Path(self.dir, "config.json").write_text(json.dumps({"lora_training": {"max_steps": 5}}), encoding="utf-8")
        cfg = Config(self.dir)
        self.assertEqual((cfg["training_device"], cfg["lora_training"]["amp"]), ("auto", "auto"))
        cfg.update({"training_device": "tpu", "lora_training": {"amp": "off"}})
        cfg = Config(self.dir)
        self.assertEqual(cfg["training_device"], "tpu")                   # 최상위 키라 update가 버리지 않는다
        self.assertEqual((cfg["lora_training"]["amp"], cfg["lora_training"]["max_steps"]), ("off", 5))

    def test_value_checks(self):
        for raw, want in (("auto", "auto"), ("", "auto"), (None, "auto"), ("GPU", "cuda"), ("rocm", "cuda"),
                          ("TPU", "tpu"), ("xla", "tpu"), ("arc", "xpu"), ("mac", "mps"), ("cpu", "cpu"),
                          ("cuda:1", "cuda:1"), ("xpu:0", "xpu:0")):
            self.assertEqual(device_value(raw), want, raw)
        for bad in ("bogus", "tpu:1", "cpu:0", "cuda:x", 3, ["cpu"]):
            with self.assertRaises(ValueError) as cm:
                device_value(bad, "학습 장치")
            self.assertIn("학습 장치", str(cm.exception))
            self.assertIn("auto, cuda, tpu, xpu, mps, cpu", str(cm.exception))
        for raw, want in (("auto", "auto"), (None, "auto"), ("", "auto"), ("FP16", "fp16"), ("float16", "fp16"),
                          ("off", "off"), ("0", "off"), ("fp32", "fp32"), ("float32", "fp32")):
            self.assertEqual(amp_value(raw), want, raw)
        with self.assertRaisesRegex(ValueError, "정밀도"):
            amp_value("bf8", "정밀도")


# ---------------------------------------------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH, "PyTorch 없음")
class AppPlumbingTest(unittest.TestCase):
    def setUp(self):
        from learnbox.app import App

        self.dir = tempfile.mkdtemp(prefix="lb-devapp-")
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.app = App(self.dir)
        self.app.cfg.data["embeddings"]["enabled"] = False
        for i in range(6):
            self.app.db.add_document(f"https://a/{i}", f"딥 러닝 {i}", KO_TEXT + f"번호 {i}\n" * 4, "link")
        self.tiny, self.lora = RecordingTinyTrain(), RecordingLoraTrain()
        for target, fake in (("learnbox.tinylm.train", self.tiny), ("learnbox.train_pc.train_lora", self.lora)):
            p = mock.patch(target, fake)
            p.start()
            self.addCleanup(p.stop)

    def run_job(self, kind, params):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.app.jobs.run_inline(kind, params)

    # --- plan_*: 요청 > LEARNBOX_DEVICE > 설정 training_device > auto ---
    def test_plan_device_precedence(self):
        tiny = {"preset": "nano"}
        with env():
            self.assertEqual(self.app.plan_tiny(tiny)["device"], "auto")
            self.app.cfg.data["training_device"] = "cpu"
            self.assertEqual(self.app.plan_tiny(tiny)["device"], "cpu")
        with env(LEARNBOX_DEVICE="mps"):
            self.assertEqual(self.app.plan_tiny(tiny)["device"], "mps")                  # 환경변수 > 설정
            self.assertEqual(self.app.plan_tiny({**tiny, "device": "TPU"})["device"], "tpu")  # 요청 > 환경변수
            self.assertEqual(self.app.plan_tiny({**tiny, "device": "auto"})["device"], "mps")  # auto = 지정 안 함
            self.assertEqual(self.app.plan_lora({"base_model": "x/y", "device": "gpu"})["device"], "cuda")
            self.assertEqual(self.app.plan_lora({"base_model": "x/y"})["device"], "mps")
        for plan, p in ((self.app.plan_tiny, tiny), (self.app.plan_lora, {"base_model": "x/y"})):
            with self.assertRaisesRegex(ValueError, "auto, cuda, tpu, xpu, mps, cpu"):
                plan({**p, "device": "quantum"})

    def test_plan_lora_amp_precedence(self):
        p = {"base_model": "x/y"}
        with env():
            self.assertEqual(self.app.plan_lora(p)["amp"], "auto")
            self.app.cfg.data["lora_training"]["amp"] = "off"
            self.assertEqual(self.app.plan_lora(p)["amp"], "off")
        with env(LEARNBOX_AMP="fp16"):
            self.assertEqual(self.app.plan_lora(p)["amp"], "fp16")                        # 환경변수 > 설정
            self.assertEqual(self.app.plan_lora({**p, "amp": "fp32"})["amp"], "fp32")     # 요청 > 환경변수
            self.assertEqual(self.app.plan_lora({**p, "amp": "float16"})["amp"], "fp16")
        with self.assertRaisesRegex(ValueError, "amp"):
            self.app.plan_lora({**p, "amp": "int4"})
        self.assertNotIn("amp", self.app.plan_tiny({"preset": "nano"}))                  # 소형 AI는 amp 없음

    # --- 작업 핸들러가 학습 함수에 넘기는 값 ---
    def test_train_handlers_pass_device_and_amp(self):
        with env():
            j = self.run_job("train_tiny", {**TINY_ARCH, "steps": 2})
            self.assertEqual(j["status"], "done", j["message"])
            self.assertEqual(self.tiny.calls[-1]["device"], "auto")
            self.app.cfg.data["training_device"] = "cpu"
            j = self.run_job("train_tiny", {**TINY_ARCH, "steps": 2})
            self.assertEqual((j["status"], self.tiny.calls[-1]["device"]), ("done", "cpu"))
            j = self.run_job("train_tiny", {**TINY_ARCH, "steps": 2, "device": "tpu"})
            self.assertEqual((j["status"], self.tiny.calls[-1]["device"]), ("done", "tpu"))

            self.app.cfg.data["lora_training"]["amp"] = "off"
            j = self.run_job("train_lora", {"base_model": "x/y", "max_steps": 2})
            self.assertEqual(j["status"], "done", j["message"])
            self.assertEqual((self.lora.calls[-1]["device"], self.lora.calls[-1]["amp"]), ("cpu", "off"))
            j = self.run_job("train_lora", {"base_model": "x/y", "max_steps": 2, "device": "xpu", "amp": "fp16"})
            self.assertEqual((self.lora.calls[-1]["device"], self.lora.calls[-1]["amp"]), ("xpu", "fp16"))

    def test_resume_branches_pass_device_and_amp(self):
        root = Path(self.dir) / "ck-tiny"
        make_checkpoint(root, "tiny")
        with env(LEARNBOX_DEVICE="cpu"):
            j = self.run_job("train_tiny", {"resume_checkpoint": str(root)})
            self.assertEqual(j["status"], "done", j["message"])
            self.assertEqual((self.tiny.calls[-1]["device"], self.tiny.calls[-1]["resume_checkpoint"]),
                             ("cpu", str(root)))
            root = Path(self.dir) / "ck-lora"
            make_checkpoint(root, "lora")
            j = self.run_job("train_lora", {"resume_checkpoint": str(root), "amp": "fp32", "device": "mps"})
            self.assertEqual(j["status"], "done", j["message"])
            self.assertEqual((self.lora.calls[-1]["device"], self.lora.calls[-1]["amp"]), ("mps", "fp32"))

    # --- 이어서 재개: 기본은 지금 장치(device를 이어받지 않음), amp는 이어받음 ---
    def failed_job(self, kind="train_tiny", **params):
        jid = self.app.db.create_job(kind, {"steps": 3, **params}, "admin")
        self.app.db.update_job(jid, status="failed", message="장치 오류")
        make_checkpoint(self.app.checkpoint_root(jid), {"train_tiny": "tiny", "train_lora": "lora"}[kind], "tpu")
        return jid

    def test_restart_job_drops_or_overrides_device(self):
        jid = self.failed_job(device="tpu", amp="fp16")
        new = self.app.restart_job(jid, "t", enqueue=False)
        p = self.app.db.get_job(new)["params"]
        self.assertNotIn("device", p)
        self.assertEqual((p["amp"], p["steps"], p["resumed_from_job"]), ("fp16", 3, jid))

        jid = self.failed_job("train_lora", device="tpu")
        new = self.app.restart_job(jid, "t", False, device="GPU")
        self.assertEqual(self.app.db.get_job(new)["params"]["device"], "cuda")

        jid = self.failed_job(device="cpu")
        n_jobs = len(self.app.db.list_jobs(200))
        with self.assertRaisesRegex(ValueError, "auto, cuda"):
            self.app.restart_job(jid, "t", enqueue=False, device="warp")
        self.assertEqual(len(self.app.db.list_jobs(200)), n_jobs)             # 잘못된 값이면 새 작업을 만들지 않음
        new = self.app.restart_job(jid, "t", enqueue=False, device="auto")
        self.assertNotIn("device", self.app.db.get_job(new)["params"])

    def test_restart_job_amp_override(self):
        """fp16 nan으로 멈춘 작업을 정밀도를 바꿔 이어서 (원래 작업의 amp보다 우선)."""
        jid = self.failed_job("train_lora", amp="fp16")
        new = self.app.restart_job(jid, "t", enqueue=False, amp="OFF")
        self.assertEqual(self.app.db.get_job(new)["params"]["amp"], "off")
        jid = self.failed_job("train_lora", amp="fp16")
        n_jobs = len(self.app.db.list_jobs(200))
        with self.assertRaises(ValueError):
            self.app.restart_job(jid, "t", enqueue=False, amp="bf8")
        self.assertEqual(len(self.app.db.list_jobs(200)), n_jobs)

    def test_checkpoint_summary_has_device(self):
        jid = self.failed_job()
        s = checkpoint.summary(self.app.checkpoint_root(jid))
        self.assertEqual(s["device"], "tpu")
        self.assertEqual(self.app.job_checkpoint(self.app.db.get_job(jid))["device"], "tpu")
        root = Path(self.dir) / "old"
        make_checkpoint(root, "tiny", device=None)                               # 예전 체크포인트
        self.assertIsNone(checkpoint.summary(root)["device"])

    # --- 장치 정보 ---
    def test_device_info_accel_and_defaults(self):
        with mock.patch.object(accel, "_probe_tpu", side_effect=AssertionError("TPU를 열면 안 됨")):
            devices.pc_profile(refresh=True)
            d = self.app.device_info()
        self.assertEqual(d["defaults"], {"training_device": "auto", "amp": "auto"})
        a = d["accel"]
        self.assertTrue({"auto", "devices", "cuda_bf16", "rocm", "torch_xla_version"} <= set(a))
        self.assertEqual(set(a["devices"]), {"auto", "cuda", "tpu", "xpu", "mps", "cpu"})
        for k, v in a["devices"].items():
            self.assertIsInstance(v["available"], bool, k)
            self.assertTrue(v["label"], k)
        self.assertTrue(a["devices"]["cpu"]["available"])
        self.assertIn(a["auto"], ("cuda", "tpu", "xpu", "mps", "cpu"))
        if not a["devices"]["cuda"]["available"]:
            self.assertIsNone(a["cuda_bf16"])
        self.assertIn("accel", d["pc"])
        self.assertIn("torch_xla", d["pc"])
        self.app.cfg.data["training_device"] = "tpu"
        self.app.cfg.data["lora_training"]["amp"] = "off"
        self.assertEqual(self.app.device_info()["defaults"], {"training_device": "tpu", "amp": "off"})

    def test_training_defaults_follow_env_before_config(self):
        """웹의 '자동'은 요청에서 빠지므로 서버가 환경변수 → 설정 순서로 정한다. 기본값 표시도 같은 순서."""
        self.app.cfg.data["training_device"] = "cpu"
        self.app.cfg.data["lora_training"]["amp"] = "off"
        with mock.patch.dict(os.environ, {"LEARNBOX_DEVICE": "tpu", "LEARNBOX_AMP": "fp16"}):
            self.assertEqual(self.app.training_defaults(), {"training_device": "tpu", "amp": "fp16"})
        env = {k: v for k, v in os.environ.items() if k not in ("LEARNBOX_DEVICE", "LEARNBOX_AMP")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(self.app.training_defaults(), {"training_device": "cpu", "amp": "off"})
        self.app.cfg.data["training_device"] = "warp-drive"  # 잘못 저장된 값
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(self.app.training_defaults()["training_device"], "auto")

    def test_capabilities_device_strings(self):
        self.app._profile_warm.set()
        base = {"tier": "pc-cpu", "gpu": {"type": None, "name": None, "vram_gb": None}}
        cases = [
            ({"gpu": {"type": "cuda", "name": "RTX 4090", "vram_gb": 24.0}}, "cuda (RTX 4090)"),
            ({"gpu": {"type": "cuda", "name": "Radeon RX 7900", "vram_gb": 24.0, "vendor": "amd", "rocm": "6.2"}},
             "cuda (Radeon RX 7900 · AMD ROCm)"),
            ({"gpu": {"type": "xpu", "name": "Intel Arc A770", "vram_gb": 16.0}}, "xpu (Intel Arc A770)"),
            ({"gpu": {"type": "mps", "name": "Apple Silicon GPU", "vram_gb": 9.6}}, "mps (Apple)"),
            ({"accel": {"available": {"tpu": True}, "tpu": {"installed": True, "likely": True, "ok": False,
                                                            "accelerator_type": "v5litepod-1"}}}, "tpu (v5litepod-1)"),
            ({}, "cpu"),
        ]
        for extra, want in cases:
            with mock.patch.object(devices, "pc_profile", return_value={**base, **extra}):
                self.assertEqual(self.app.capabilities()["device"], want)

    def test_xla_version_mismatch_is_not_tpu(self):
        """torch와 torch_xla의 major.minor가 다르면 pick('auto')가 CPU를 고르므로 기기 카드·자동 선택도 TPU가 아니다."""
        self.app._profile_warm.set()
        acc = {"available": {"auto": True, "tpu": True, "cpu": True}, "torch_version": "2.15.0+cpu",
               "torch_xla_version": "2.9.0",
               "tpu": {"installed": True, "packages": ["torch_xla"], "likely": True, "ok": False,
                       "backend": "torch_xla", "accelerator_type": "v5litepod-1", "hbm_gb": 16.0}}
        pc = {"tier": "pc-cpu", "gpu": {"type": None, "name": None, "vram_gb": None}, "accel": acc}
        with mock.patch.object(devices, "pc_profile", return_value=pc):
            self.assertEqual(self.app.capabilities()["device"], "cpu")
        a = self.app._accel_devices(pc)
        self.assertEqual(a["auto"], "cpu")
        self.assertFalse(a["devices"]["tpu"]["available"])
        self.assertIn("setup_accel", a["devices"]["tpu"]["note"])
        self.assertIn("setup_accel", a["tpu"]["mismatch"])
        from learnbox.__main__ import _accel_line

        self.assertIn("버전 불일치", _accel_line(a, pc))
        ok = {**acc, "torch_xla_version": "2.15.1"}                              # 짝이 맞으면 예전처럼 TPU
        a = self.app._accel_devices({**pc, "accel": ok})
        self.assertEqual((a["auto"], a["tpu"]["mismatch"]), ("tpu", None))
        self.assertNotIn("note", a["devices"]["tpu"])
        with mock.patch.object(devices, "pc_profile", return_value={**pc, "accel": ok}):
            self.assertEqual(self.app.capabilities()["device"], "tpu (v5litepod-1)")

    def test_estimate_device(self):
        lora = {"kind": "lora", "base_model": "Qwen/Qwen2.5-0.5B-Instruct"}
        plain = self.app.estimate(dict(lora))
        self.assertEqual(self.app.estimate({**lora, "device": "auto"}), plain)   # auto면 예전과 같은 결과
        r = self.app.estimate({**lora, "device": "tpu"})
        self.assertEqual(r["base_model"], lora["base_model"])
        import inspect

        from learnbox import train_pc
        if "device" in inspect.signature(train_pc.estimate_lora).parameters:
            self.assertIn("tpu", r["fits"])
        with self.assertRaisesRegex(ValueError, "auto, cuda"):
            self.app.estimate({**lora, "device": "abacus"})

        r = self.app.estimate({"kind": "tiny", "preset": "nano", "batch_size": 8})
        self.assertEqual(set(r), {"arch", "estimate"})                         # 예전 키 그대로
        r = self.app.estimate({"kind": "tiny", "preset": "nano", "batch_size": 8, "device": "tpu"})
        self.assertEqual(r["device_fit"]["device"], "tpu")
        self.assertIn(r["device_fit"]["fits"], (True, False, None))


# ---------------------------------------------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH, "PyTorch 없음")
class ServerPlumbingTest(unittest.TestCase):
    """실제 HTTP 서버. 작업 스레드는 켜지 않는다(제출한 작업은 대기열에 남아 params만 확인)."""

    @classmethod
    def setUpClass(cls):
        from learnbox.app import App
        from learnbox.server import Sessions, make_handler

        cls.dir = tempfile.mkdtemp(prefix="lb-devsrv-")
        cls.app = App(cls.dir)
        cls.app.cfg.set_pins("1234", "987654")
        handler = make_handler(cls.app, Sessions())
        handler.log_message = lambda *a, **k: None
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.admin = cls.call("POST", "/api/login", {"pin": "987654"})[1]["token"]

    @classmethod
    def tearDownClass(cls):
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

    def setUp(self):
        self.env = env()
        self.env.__enter__()
        self.addCleanup(self.env.__exit__, None, None, None)

    def test_settings_validation(self):
        for body, word in (({"training_device": "bogus"}, "학습 장치"), ({"training_device": 7}, "학습 장치"),
                           ({"lora_training": {"amp": "int8"}}, "정밀도")):
            s, r = self.call("POST", "/api/settings", body, token=self.admin)
            self.assertEqual(s, 400, r)
            self.assertIn(word, r["error"])
        self.assertEqual(self.app.cfg["training_device"], "auto")                 # 잘못된 값은 저장하지 않음
        s, r = self.call("POST", "/api/settings", {"training_device": "GPU", "lora_training": {"amp": "float16"}},
                         token=self.admin)
        self.assertEqual(s, 200, r)
        self.assertEqual((r["training_device"], r["lora_training"]["amp"]), ("cuda", "fp16"))
        self.assertEqual(r["lora_training"]["base_model"], DEFAULTS["lora_training"]["base_model"])
        s, r = self.call("POST", "/api/settings", {"training_device": "auto", "lora_training": {"amp": "auto"}},
                         token=self.admin)
        self.assertEqual((s, r["training_device"], r["lora_training"]["amp"]), (200, "auto", "auto"))

    def test_train_routes_store_normalized_values_only_when_given(self):
        s, r = self.call("POST", "/api/train/tiny", {**TINY_ARCH, "steps": 2}, token=self.admin)
        self.assertEqual(s, 200, r)
        self.assertNotIn("device", self.app.db.get_job(r["job_id"])["params"])  # 주지 않으면 params 그대로
        s, r = self.call("POST", "/api/train/tiny", {**TINY_ARCH, "steps": 2, "device": "TPU"}, token=self.admin)
        self.assertEqual(s, 200, r)
        self.assertEqual(self.app.db.get_job(r["job_id"])["params"]["device"], "tpu")
        s, r = self.call("POST", "/api/train/tiny", {**TINY_ARCH, "device": "warp"}, token=self.admin)
        self.assertEqual(s, 400, r)
        self.assertIn("auto, cuda", r["error"])

        need = all(importlib.util.find_spec(m) for m in ("transformers", "peft"))
        if not need:
            return
        s, r = self.call("POST", "/api/train/lora", {"base_model": "x/y", "max_steps": 2}, token=self.admin)
        self.assertEqual(s, 200, r)
        p = self.app.db.get_job(r["job_id"])["params"]
        self.assertNotIn("device", p)
        self.assertNotIn("amp", p)
        s, r = self.call("POST", "/api/train/lora", {"base_model": "x/y", "device": "Arc", "amp": "FLOAT16"},
                         token=self.admin)
        self.assertEqual(s, 200, r)
        p = self.app.db.get_job(r["job_id"])["params"]
        self.assertEqual((p["device"], p["amp"]), ("xpu", "fp16"))
        s, r = self.call("POST", "/api/train/lora", {"base_model": "x/y", "amp": "half-ish"}, token=self.admin)
        self.assertEqual(s, 400, r)

    def test_restart_route_passes_device(self):
        jids = []
        for _ in range(2):
            jid = self.app.db.create_job("train_tiny", {"steps": 3, "device": "tpu"}, "admin")
            self.app.db.update_job(jid, status="failed")
            make_checkpoint(self.app.checkpoint_root(jid), "tiny", "tpu")
            jids.append(jid)
        s, r = self.call("POST", f"/api/jobs/{jids[0]}/restart", {}, token=self.admin)
        self.assertEqual(s, 200, r)
        self.assertNotIn("device", self.app.db.get_job(r["job_id"])["params"])
        s, r = self.call("POST", f"/api/jobs/{jids[1]}/restart", {"device": "bogus"}, token=self.admin)
        self.assertEqual(s, 400, r)
        s, r = self.call("POST", f"/api/jobs/{jids[1]}/restart", {"device": "cpu"}, token=self.admin)
        self.assertEqual(s, 200, r)
        self.assertEqual(self.app.db.get_job(r["job_id"])["params"]["device"], "cpu")

    def test_device_route(self):
        s, d = self.call("GET", "/api/device", token=self.admin)
        self.assertEqual(s, 200)
        self.assertIn("accel", d)
        self.assertIn("defaults", d)
        self.assertIn("tpu", d["accel"]["devices"])


# ---------------------------------------------------------------------------------------------------------------
class CliTest(unittest.TestCase):
    def test_parser_flags(self):
        from learnbox.__main__ import _start_overrides, build_parser

        p = build_parser()
        self.assertEqual(p.parse_args(["train-tiny", "--device", "tpu"]).device, "tpu")
        self.assertEqual(p.parse_args(["train-mobile", "--device", "GPU"]).device, "cuda")
        self.assertIsNone(p.parse_args(["train-tiny"]).device)
        a = p.parse_args(["train-lora", "--device", "xpu", "--amp", "off"])
        self.assertEqual((a.device, a.amp), ("xpu", "off"))
        self.assertIsNone(p.parse_args(["train-pc"]).amp)
        for amp in ("auto", "fp16", "off", "fp32"):
            self.assertEqual(p.parse_args(["train-lora", "--amp", amp]).amp, amp)
        self.assertEqual(p.parse_args(["resume-job", "3", "--device", "cpu"]).device, "cpu")
        self.assertIsNone(p.parse_args(["resume-job", "3"]).device)
        self.assertEqual(p.parse_args(["resume-job", "3", "--amp", "off"]).amp, "off")
        self.assertIsNone(p.parse_args(["resume-job", "3"]).amp)
        self.assertEqual(p.parse_args(["estimate", "--kind", "lora", "--device", "tpu"]).device, "tpu")
        a = p.parse_args(["autolearn", "start", "--device", "cuda"])
        self.assertEqual(_start_overrides(a)["device"], "cuda")
        self.assertNotIn("device", _start_overrides(p.parse_args(["autolearn", "start"])))
        self.assertEqual(p.parse_args(["autolearn", "run", "--device", "mps"]).device, "mps")
        with contextlib.redirect_stderr(io.StringIO()):
            for bad in (["train-tiny", "--device", "warp"], ["train-lora", "--amp", "bf8"],
                        ["train-tiny", "--amp", "fp16"]):
                with self.assertRaises(SystemExit):
                    p.parse_args(bad)

    @unittest.skipUnless(HAS_TORCH, "PyTorch 없음")
    def test_cli_train_and_resume_pass_device(self):
        from learnbox.__main__ import main

        d = tempfile.mkdtemp(prefix="lb-devcli-")
        self.addCleanup(shutil.rmtree, d, True)
        from learnbox.app import App

        app = App(d)
        for i in range(6):
            app.db.add_document(f"https://a/{i}", f"딥 러닝 {i}", KO_TEXT + f"번호 {i}\n" * 4, "link")
        fake = RecordingTinyTrain()
        arch = ["--preset", "custom", "--d-model", "32", "--n-layer", "1", "--n-head", "2", "--seq-len", "16",
                "--vocab-size", "100", "--steps", "2"]
        with env(), mock.patch("learnbox.tinylm.train", fake), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data", d, "train-tiny", *arch]), 0)
            self.assertEqual(main(["--data", d, "train-tiny", *arch, "--device", "cpu"]), 0)
        jobs = sorted(App(d).db.list_jobs(10), key=lambda j: j["id"])
        self.assertNotIn("device", jobs[0]["params"])                       # 주지 않으면 예전 params 그대로
        self.assertEqual(jobs[1]["params"]["device"], "cpu")
        self.assertEqual([c["device"] for c in fake.calls], ["auto", "cpu"])

        jid = app.db.create_job("train_tiny", {"steps": 3, "device": "tpu"}, "cli")
        app.db.update_job(jid, status="failed")
        make_checkpoint(app.checkpoint_root(jid), "tiny", "tpu")
        out = io.StringIO()
        with env(), mock.patch("learnbox.tinylm.train", fake), contextlib.redirect_stdout(out):
            self.assertEqual(main(["--data", d, "resume-job", str(jid), "--device", "cpu"]), 0)
        self.assertEqual(fake.calls[-1]["device"], "cpu")
        self.assertIn(f"작업 #{jid}의 체크포인트에서 이어서 학습합니다", out.getvalue())

    @unittest.skipUnless(HAS_TORCH, "PyTorch 없음")
    def test_device_command_adds_accel_line(self):
        from learnbox.__main__ import main

        d = tempfile.mkdtemp(prefix="lb-devcmd-")
        self.addCleanup(shutil.rmtree, d, True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--data", d, "device"]), 0)
        lines = out.getvalue().splitlines()
        heads = [ln.split(":", 1)[0] for ln in lines[:5]]
        self.assertEqual(heads, ["운영체제", "GPU", "패키지", "가속", "등급"])     # 기존 줄 순서는 그대로, 새 줄만 추가
        acc = lines[3]
        for word in ("torch", "torch_xla", "ROCm", "xpu", "TPU"):
            self.assertIn(word, acc)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(["--data", d, "device", "--json"])
        self.assertIn("accel", json.loads(out.getvalue())["pc"])


# ---------------------------------------------------------------------------------------------------------------
class DevicesTest(unittest.TestCase):
    def test_constants(self):
        self.assertIn("torch_xla", devices.MODULES)
        self.assertEqual(devices.LORA_MODELS["3B"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(devices.tier_of({"type": "xpu", "vram_gb": 16}), "pc-gpu-high")
        self.assertEqual(devices.tier_of({"type": "xpu", "vram_gb": 8}), "pc-gpu")
        self.assertEqual(devices.tier_of({"type": "tpu", "vram_gb": 16}), "tpu")

    def test_profile_has_accel_without_opening_tpu(self):
        with mock.patch.object(accel, "_probe_tpu", side_effect=AssertionError("TPU를 열면 안 됨")):
            p = devices.pc_profile(refresh=True)
        self.assertIsInstance(p["torch_xla"], bool)
        self.assertTrue({"available", "torch_version", "torch_xla_version", "rocm", "tpu"} <= set(p["accel"]))
        self.assertIn(p["tier"], ("pc-cpu", "pc-gpu", "pc-gpu-high"))         # pc 등급의 뜻은 그대로

    @unittest.skipUnless(HAS_TORCH, "PyTorch 없음")
    def test_rocm_vendor_keys(self):
        import torch

        props = mock.Mock(total_memory=24 * 1024 ** 3)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "get_device_properties", return_value=props), \
                mock.patch.object(torch.cuda, "get_device_name", return_value="AMD Radeon RX 7900 XTX"), \
                mock.patch.object(torch.cuda, "device_count", return_value=1), \
                mock.patch.object(torch.version, "hip", "6.2.41133", create=True):
            gpu, _ = devices._gpu_info(32.0)
        self.assertEqual((gpu["type"], gpu["vram_gb"], gpu["vendor"], gpu["rocm"]),
                         ("cuda", 24.0, "amd", "6.2.41133"))
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "get_device_properties", return_value=props), \
                mock.patch.object(torch.cuda, "get_device_name", return_value="RTX"), \
                mock.patch.object(torch.cuda, "device_count", return_value=1), \
                mock.patch.object(torch.version, "hip", None, create=True):
            gpu, _ = devices._gpu_info(32.0)
        self.assertEqual(set(gpu), {"type", "name", "vram_gb", "count"})           # NVIDIA는 예전 키 그대로

    def tpu_profile(self, installed=True, hbm=16.0, gpu=None):
        return {"os": "Linux", "cpu_count": 8, "ram_gb": 48.0, "gpu": gpu or {"type": None, "name": None,
                                                                              "vram_gb": None},
                "tier": "pc-cpu", "torch": True, "transformers": True, "peft": True, "onnxruntime": True,
                "anthropic": True,
                "accel": {"available": {"tpu": installed}, "tpu": {"installed": installed, "likely": True,
                                                                   "ok": False, "hbm_gb": hbm,
                                                                   "accelerator_type": "v5litepod-1"}}}

    def test_recommend_tpu_branch(self):
        r = devices.recommend(self.tpu_profile())
        self.assertEqual(r["tier"], "tpu")
        self.assertEqual(r["lora_base_model"], devices.LORA_MODELS["1.5B"])
        self.assertEqual(r["lora_max_steps"], 300)
        self.assertTrue(all(isinstance(n, str) and any("가" <= c <= "힣" for c in n) for n in r["notes"]))
        self.assertTrue(any("TPU" in n for n in r["notes"]))
        from learnbox import tinylm

        est = tinylm.estimate(tinylm.resolve_arch(r["tiny_preset"]), r["tiny_batch_size"])
        self.assertLessEqual(est["train_ram_mb"], 16 * 1024 * 0.75)
        r8 = devices.recommend(self.tpu_profile(hbm=None))                     # HBM을 모르면 16GB로 본다
        self.assertEqual((r8["tier"], r8["tiny_preset"]), ("tpu", r["tiny_preset"]))

    def test_recommend_tpu_needs_torch_xla_and_no_gpu(self):
        r = devices.recommend(self.tpu_profile(installed=False))
        self.assertEqual(r["tier"], "pc-cpu")
        self.assertTrue(any("setup_accel" in n for n in r["notes"]))
        g = {"type": "cuda", "name": "T4", "vram_gb": 15.0}
        r = devices.recommend({**self.tpu_profile(gpu=g), "tier": devices.tier_of(g)})
        self.assertEqual(r["tier"], "pc-gpu")                                   # GPU가 있으면 예전 추천 그대로
        plain = {k: v for k, v in self.tpu_profile().items() if k != "accel"}
        self.assertEqual(devices.recommend(plain)["tier"], "pc-cpu")

    def test_recommend_tpu_xla_version_mismatch(self):
        def prof(tv, xv, packages=("torch_xla",)):
            p = self.tpu_profile()
            p["accel"].update(torch_version=tv, torch_xla_version=xv)
            p["accel"]["tpu"].update(packages=list(packages), backend=packages[0])
            return p

        p = prof("2.15.0", "2.9.0")
        self.assertEqual(devices.tpu_usable(p)[0], False)
        r = devices.recommend(p)
        self.assertEqual(r["tier"], "pc-cpu")                                   # pick('auto')처럼 CPU 기준
        self.assertNotEqual(r["tiny_preset"], "large")
        msg = devices.xla_mismatch(p["accel"])
        self.assertIn("python -m learnbox.setup_accel", msg)
        self.assertTrue(any(msg in n for n in r["notes"]))
        self.assertFalse(any("TPU(" in n for n in r["notes"]))
        for ok in (prof("2.9.0+cpu", "2.9.0"), prof("2.15.0", "2.15.1"),        # major.minor가 같으면 TPU
                   prof("2.15.0", "2.9.0", ("torch_tpu", "torch_xla")),          # torch_tpu로 열 수 있음
                   prof(None, "2.9.0")):                                         # 버전을 모르면 확인 생략
            self.assertEqual(devices.recommend(ok)["tier"], "tpu")
        p = prof("2.15.0", "2.9.0")
        p["accel"]["tpu"]["ok"] = True                                          # 이미 열린 TPU는 그대로
        self.assertEqual(devices.recommend(p)["tier"], "tpu")

    def test_profile_marks_xla_version_mismatch(self):
        hint = {"likely": True, "signals": ["TPU_ACCELERATOR_TYPE"], "accelerator_type": "v5litepod-1",
                "chips": 0, "hbm_gb": 16.0}
        vers = {"torch": "2.15.0", "torch_xla": "2.9.0", "torch_tpu": None}
        with mock.patch.dict(devices._CACHE, {"t": 0.0, "v": None}), \
                mock.patch.dict(accel._TPU, {"probed": False, "device": None, "reason": None, "backend": None}), \
                mock.patch.object(accel, "_probe_tpu", side_effect=AssertionError("TPU를 열면 안 됨")), \
                mock.patch.object(accel, "tpu_packages", return_value=["torch_xla"]), \
                mock.patch.object(accel, "tpu_hint", return_value=hint), \
                mock.patch.object(accel, "_version", side_effect=vers.get):
            p = devices.pc_profile(refresh=True)
            self.assertIn("setup_accel", p["accel"]["tpu"]["mismatch"])
            self.assertFalse(p["accel"]["available"]["tpu"])
            self.assertFalse(devices.tpu_usable(p)[0])
            self.assertEqual(devices.recommend(p)["tier"], p["tier"])
            vers["torch_xla"] = "2.15.0"                                         # 짝이 맞으면 표시하지 않는다
            p = devices.pc_profile(refresh=True)
            self.assertNotIn("mismatch", p["accel"]["tpu"])
            self.assertTrue(p["accel"]["available"]["tpu"])


# ---------------------------------------------------------------------------------------------------------------
class AutolearnDeviceTest(unittest.TestCase):
    """tests/test_autolearn.py와 같은 방식: 무거운 모듈은 가짜, App 대신 cfg/db/jobs만 있는 FakeApp."""

    def setUp(self):
        from learnbox.db import DB

        self.tmp = Path(tempfile.mkdtemp(prefix="lb-devauto-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        cfg = Config(self.tmp / "data")
        cfg.data["autolearn"]["targets"] = {"tiny": True, "lora": True, "embeddings": False, "qa": False}
        db = DB(cfg.db_path)

        class Jobs:
            def submit(self, kind, params, actor):
                return db.create_job(kind, params, actor)

            def cancel(self, job_id):
                db.update_job(job_id, status="cancelled")

        self.app = types.SimpleNamespace(cfg=cfg, db=db, jobs=Jobs(), gate=lambda: {"allowed": True})
        for i, (t, x) in enumerate((("인공지능", "인공지능은 사람의 학습 능력을 컴퓨터로 구현한 기술이다. " * 20),
                                    ("기계학습", "기계학습은 데이터에서 규칙을 배우는 방법이다. " * 20))):
            db.add_document(f"https://ex.com/{i}", t, x, "link")
        self.calls = {"tiny": [], "lora": []}
        calls = self.calls

        fake_devices = types.ModuleType("learnbox.devices")
        fake_devices.recommend = lambda profile=None: {"tiny_preset": "nano", "tiny_batch_size": 8, "threads": 3,
                                                       "notes": []}
        tinylm = types.ModuleType("learnbox.tinylm")
        tinylm.resolve_arch = lambda preset=None, overrides=None, recommended="mini": {
            "preset": "nano", "d_model": 128, "n_layer": 2, "n_head": 4, "seq_len": 128, "vocab_size": 2000}

        def train(train_path, eval_path, out_dir, arch, steps=1500, batch_size=32, lr=3e-3, resume_from=None,
                  device=None, threads=None, progress=lambda p, m: None, check_cancel=lambda: None):
            calls["tiny"].append({"device": device})                      # amp를 넘기면 TypeError
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            return {"kind": "tiny", "steps": steps, "total_steps": steps, "eval_loss_before": 5.0,
                    "eval_loss_after": 4.0}
        tinylm.train = train
        train_pc = types.ModuleType("learnbox.train_pc")

        def train_lora(train_path, eval_path, out_dir, base_model, **kw):
            calls["lora"].append({"device": kw.get("device", "<없음>"), "amp": kw.get("amp", "<없음>")})
            return {"kind": "lora", "base_model": base_model, "steps": 1, "total_steps": 1,
                    "eval_loss_before": 2.5, "eval_loss_after": 2.0}
        train_pc.train_lora = train_lora
        mods = mock.patch.dict(sys.modules, {"learnbox.devices": fake_devices, "learnbox.tinylm": tinylm,
                                             "learnbox.train_pc": train_pc})
        mods.start()
        self.addCleanup(mods.stop)
        self.env = env()
        self.env.__enter__()
        self.addCleanup(self.env.__exit__, None, None, None)

    def learner(self):
        from learnbox.autolearn import AutoLearner

        return AutoLearner(self.app)

    def test_session_device_is_validated_and_stored(self):
        al = self.learner()
        with self.assertRaisesRegex(ValueError, "device"):
            al.start_session("admin", {"device": "warp"})
        self.assertIsNone(self.app.db.active_session())
        s = al.start_session("admin", {"device": "GPU"})
        self.assertEqual(s["config"]["device"], "cuda")
        self.app.db.update_session(s["id"], status="ended")
        s = al.start_session("admin")
        self.assertNotIn("device", s["config"])                       # 주지 않으면 세션 설정은 예전 그대로

    def test_run_passes_device_and_amp(self):
        from learnbox.autolearn import run_autolearn_train

        al = self.learner()
        self.app.cfg.data["training_device"] = "cpu"
        self.app.cfg.data["lora_training"]["amp"] = "off"
        s = al.start_session("admin")
        r = run_autolearn_train(self.app, s["id"])
        self.assertEqual(r["errors"], {}, r)
        self.assertEqual(self.calls["tiny"][-1], {"device": "cpu"})          # 설정 training_device
        self.assertEqual(self.calls["lora"][-1], {"device": "cpu", "amp": "off"})
        os.environ["LEARNBOX_DEVICE"] = "xpu"
        os.environ["LEARNBOX_AMP"] = "fp16"
        run_autolearn_train(self.app, s["id"])
        self.assertEqual(self.calls["tiny"][-1], {"device": "xpu"})          # 환경변수 > 설정
        self.assertEqual(self.calls["lora"][-1], {"device": "xpu", "amp": "fp16"})
        self.app.db.update_session(s["id"], status="ended")
        s = al.start_session("admin", {"device": "mps"})
        run_autolearn_train(self.app, s["id"])
        self.assertEqual(self.calls["tiny"][-1], {"device": "mps"})          # 세션 설정 > 환경변수
        self.assertEqual(self.calls["lora"][-1]["device"], "mps")
        jobs = [j for j in self.app.db.list_jobs(50) if j["kind"] in ("autolearn_train", "finalize_learning")]
        self.assertTrue(all(j["params"] == {"session_id": j["params"]["session_id"]} for j in jobs))


if __name__ == "__main__":
    unittest.main()
