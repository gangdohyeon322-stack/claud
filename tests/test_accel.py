"""장치 계층(learnbox/accel.py) 단위 테스트. TPU가 없는 환경에서도 돌도록 torch_xla를 가짜 모듈로 흉내 낸다."""
import importlib.util
import os
import subprocess
import sys
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learnbox import accel  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _fake_xla(kind="TPU", calls=None, fail=None):
    """가짜 torch_xla 패키지 (sys.modules에 넣을 dict)."""
    import torch

    calls = calls if calls is not None else []
    txla = types.ModuleType("torch_xla")
    xm = types.ModuleType("torch_xla.core.xla_model")
    xr = types.ModuleType("torch_xla.runtime")
    core = types.ModuleType("torch_xla.core")
    dbg = types.ModuleType("torch_xla.debug")
    met = types.ModuleType("torch_xla.debug.metrics")
    dev = torch.device("xla:0")

    def device():
        calls.append("device")
        if fail:
            raise RuntimeError(fail)
        return dev

    txla.device = device
    txla.sync = lambda: calls.append("sync")
    xm.xla_device_hw = lambda d: kind
    xm.set_rng_state = lambda seed, device=None: calls.append(("seed", seed))
    xm.get_memory_info = lambda d: {"bytes_limit": 16 * 1024 ** 3, "bytes_used": 1024 ** 3}
    xr.device_type = lambda: kind
    met.metric_data = lambda name: (7, 1.0, []) if name == "CompileTime" else None
    txla.core, core.xla_model, txla.runtime, txla.debug, dbg.metrics = core, xm, xr, dbg, met
    return {"torch_xla": txla, "torch_xla.core": core, "torch_xla.core.xla_model": xm,
            "torch_xla.runtime": xr, "torch_xla.debug": dbg, "torch_xla.debug.metrics": met}, dev


class NamesTest(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        code = "import sys; import learnbox.accel; print('torch' in sys.modules, 'torch_xla' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "False False", out.stderr)

    def test_normalize(self):
        self.assertEqual(accel.normalize(None), "auto")
        self.assertEqual(accel.normalize(""), "auto")
        self.assertEqual(accel.normalize("GPU"), "cuda")
        self.assertEqual(accel.normalize("xla"), "tpu")
        self.assertEqual(accel.normalize("rocm"), "cuda")
        self.assertEqual(accel.normalize("cuda:1"), "cuda:1")
        self.assertEqual(accel.normalize("xla:0"), "tpu")
        self.assertEqual(accel.normalize("TPU:1"), "tpu")
        self.assertEqual(accel.name("xla:0"), "tpu")
        self.assertEqual(accel.label("tpu"), "TPU")

    def test_requested_device_precedence(self):
        with mock.patch.dict(os.environ, {"LEARNBOX_DEVICE": "tpu"}):
            self.assertEqual(accel.requested_device("cpu", "cuda"), "cpu")
            self.assertEqual(accel.requested_device("auto", "cuda"), "tpu")
            self.assertEqual(accel.requested_device(None, "cuda"), "tpu")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LEARNBOX_DEVICE", None)
            self.assertEqual(accel.requested_device(None, "cuda"), "cuda")
            self.assertEqual(accel.requested_device("", "auto"), "auto")

    def test_requested_amp_precedence(self):
        with mock.patch.dict(os.environ, {"LEARNBOX_AMP": "1"}):
            self.assertEqual(accel.requested_amp(None, "off"), "fp16")
            self.assertEqual(accel.requested_amp("off", None), "off")
            self.assertEqual(accel.requested_amp("auto", "off"), "fp16")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LEARNBOX_AMP", None)
            self.assertEqual(accel.requested_amp(None, "off"), "off")
            self.assertEqual(accel.requested_amp(None, None), "auto")
            self.assertEqual(accel.requested_amp("garbage", None), "auto")
        self.assertEqual(accel.normalize_amp("float32"), "fp32")
        self.assertIsNone(accel.normalize_amp("bogus"))


class PrecisionTest(unittest.TestCase):
    def test_cpu_and_mps_keep_plan(self):
        for d in ("cpu", "mps"):
            self.assertEqual(accel.lora_precision(d, "fp16", plan_dtype="float32"), ("float32", False, None))
            self.assertEqual(accel.lora_precision(d, "auto", plan_dtype="bfloat16"), ("bfloat16", False, None))

    def test_cuda_policy(self):
        with mock.patch.object(accel, "native_bf16", return_value=False):
            dt, amp, note = accel.lora_precision("cuda", "auto", plan_dtype="bfloat16")
            self.assertEqual((dt, amp), ("float16", True))
            self.assertIn("LEARNBOX_AMP=off", note)
            self.assertEqual(accel.lora_precision("cuda", "off", plan_dtype="bfloat16"), ("bfloat16", False, None))
            self.assertEqual(accel.lora_precision("cuda", "off", plan_dtype="float32"), ("float32", False, None))
            self.assertEqual(accel.lora_precision("cuda", "fp32", plan_dtype="bfloat16"), ("float32", False, None))
            self.assertEqual(accel.lora_precision("cuda", "fp16")[:2], ("float16", True))
            # 직접 준(또는 저장된) dtype이 우선. float16이면 혼합 정밀도를 켠다.
            self.assertEqual(accel.lora_precision("cuda", "auto", requested_dtype="float16"), ("float16", True, None))
            self.assertEqual(accel.lora_precision("cuda", "fp16", requested_dtype="float32"), ("float32", False, None))
        with mock.patch.object(accel, "native_bf16", return_value=True):
            self.assertEqual(accel.lora_precision("cuda", "auto", plan_dtype="bfloat16"), ("bfloat16", False, None))

    def test_tpu_policy(self):
        self.assertEqual(accel.lora_precision("tpu", "auto"), ("bfloat16", False, None))
        dt, amp, note = accel.lora_precision("tpu", "fp16")
        self.assertEqual((dt, amp), ("bfloat16", False))
        self.assertTrue(note)
        self.assertEqual(accel.lora_precision("tpu", "auto", requested_dtype="float16")[:2], ("bfloat16", False))
        self.assertEqual(accel.lora_precision("tpu", "fp32"), ("float32", False, None))

    def test_env_used_when_mode_none(self):
        with mock.patch.dict(os.environ, {"LEARNBOX_AMP": "off"}), \
                mock.patch.object(accel, "native_bf16", return_value=False):
            self.assertEqual(accel.lora_precision("cuda", None, plan_dtype="float32"), ("float32", False, None))


@unittest.skipUnless(HAS_TORCH, "torch 필요")
class TorchPathsTest(unittest.TestCase):
    def setUp(self):
        accel._reset_tpu_probe()
        self.addCleanup(accel._reset_tpu_probe)

    def test_pick_cpu_paths_match_old_notes(self):
        import torch

        if torch.cuda.is_available():
            self.skipTest("CUDA가 있는 환경")
        self.assertEqual(accel.pick(None)[:2], ("cpu", torch.device("cpu")))
        self.assertEqual(accel.pick("cuda")[2], "CUDA GPU를 찾지 못해 CPU로 학습했습니다.")
        self.assertIn("알 수 없는 장치 'weird'", accel.pick("weird")[2])
        with mock.patch.object(accel, "_has", side_effect=lambda m: m not in accel.TPU_BACKENDS):
            n, d, note = accel.pick("tpu")
            self.assertEqual(n, "cpu")
            self.assertIn("torch_xla", note)

    def test_pick_tpu_with_fake_xla(self):
        import torch

        if torch.cuda.is_available():
            self.skipTest("CUDA가 있는 환경")
        calls = []
        mods, dev = _fake_xla(calls=calls)
        # TPU 신호가 없으면 자동 선택은 TPU 패키지를 불러오지 않는다
        with mock.patch.dict(sys.modules, mods), mock.patch.object(accel, "_has", return_value=True), \
                mock.patch.object(accel, "tpu_hint", return_value={"likely": False}):
            self.assertEqual(accel.pick("auto")[0], "cpu")
            self.assertEqual(calls, [])
        hint = {"likely": True, "signals": ["pci:v5ex1"], "accelerator_type": "v5e", "chips": 1, "hbm_gb": 16.0}
        with mock.patch.dict(sys.modules, mods), mock.patch.object(accel, "_has", return_value=True), \
                mock.patch.object(accel, "tpu_hint", return_value=hint):
            self.assertEqual(accel.pick("auto"), ("tpu", dev, None))
            self.assertEqual(accel.pick("tpu"), ("tpu", dev, None))
            self.assertEqual(calls.count("device"), 1)  # 한 번만 연다
            accel.step_end(dev)
            self.assertIn("sync", calls)
            accel.manual_seed(dev, 3)
            self.assertIn(("seed", 3), calls)
            self.assertEqual(accel.compile_count(), 7)
            self.assertAlmostEqual(accel.mem_total_gb(dev), 16.0)
            self.assertTrue(accel.available()["tpu"])
            # 다른 장치는 아무것도 하지 않는다
            n = len(calls)
            accel.step_end("cpu")
            accel.manual_seed("cpu", 1)
            self.assertEqual(len(calls), n)
            self.assertEqual(accel.pick("tpu", allow_tpu=False)[0], "cpu")
            self.assertEqual(accel.pick("auto", allow_tpu=False)[0], "cpu")

    def test_pick_rejects_xla_cpu(self):
        mods, _ = _fake_xla(kind="CPU")
        with mock.patch.dict(sys.modules, mods), mock.patch.object(accel, "_has", return_value=True):
            n, d, note = accel.pick("tpu")
            self.assertEqual(n, "cpu")
            self.assertIn("TPU가 아닙니다", note)

    def test_torch_tpu_backend_preferred(self):
        """torch_tpu(즉시 실행)가 있으면 그것을 쓰고, 단계 마무리·장치 seed는 아무것도 하지 않는다."""
        import torch

        if torch.cuda.is_available():
            self.skipTest("CUDA가 있는 환경")
        tdev = types.SimpleNamespace(type="tpu")
        mods, _ = _fake_xla(calls=[])
        hint = {"likely": True, "signals": ["pci:v5ex1"], "accelerator_type": "v5e", "chips": 1, "hbm_gb": 16.0}
        with mock.patch.dict(sys.modules, mods), \
                mock.patch.object(accel, "_has", side_effect=lambda m: m == "torch_tpu"), \
                mock.patch.object(accel, "_open_torch_tpu", return_value=tdev), \
                mock.patch.object(accel, "tpu_hint", return_value=hint):
            self.assertEqual(accel.pick("auto"), ("tpu", tdev, None))
            self.assertEqual(accel.tpu_backend(tdev), "torch_tpu")
            self.assertEqual(accel.tpu_backend("tpu"), "torch_tpu")
            with mock.patch.dict(sys.modules, {"torch_xla": None}):  # torch_xla를 부르면 ImportError
                accel.step_end(tdev)
                accel.manual_seed(tdev, 1)
            self.assertIsNone(accel.compile_count())
            self.assertTrue(accel.info()["tpu"]["ok"])
            self.assertEqual(accel.info()["tpu"]["backend"], "torch_tpu")

    def test_torch_tpu_failure_falls_back_to_xla(self):
        import torch

        if torch.cuda.is_available():
            self.skipTest("CUDA가 있는 환경")
        mods, dev = _fake_xla(calls=[])
        with mock.patch.dict(sys.modules, mods), mock.patch.object(accel, "_has", return_value=True), \
                mock.patch.object(accel, "_open_torch_tpu", side_effect=RuntimeError("no chip")):
            self.assertEqual(accel.pick("tpu"), ("tpu", dev, None))
            self.assertEqual(accel.tpu_backend(dev), "torch_xla")

    def test_auto_explains_unusable_tpu_on_tpu_host(self):
        """TPU 런타임 신호가 있는데 TPU를 못 열면 자동 선택도 이유를 남긴다(다른 호스트는 안내 없음)."""
        import torch

        if torch.cuda.is_available():
            self.skipTest("CUDA가 있는 환경")
        msg = "Failed to acquire a TPU device node '/dev/vfio/0' because it is already opened by another process"
        hint = {"likely": True, "signals": ["pci:v5ex1"], "accelerator_type": "v5e", "chips": 1, "hbm_gb": 16.0}
        with mock.patch.object(accel, "_has", side_effect=lambda m: m == "torch_tpu"), \
                mock.patch.object(accel, "_open_torch_tpu", side_effect=RuntimeError(msg)), \
                mock.patch.object(accel, "tpu_hint", return_value=hint):
            n, _, note = accel.pick("auto")
        self.assertEqual(n, "cpu")
        self.assertIn("다른 프로세스가 TPU를 쓰고 있습니다", note)
        accel._reset_tpu_probe()
        with mock.patch.object(accel, "_has", side_effect=lambda m: m == "torch_tpu"), \
                mock.patch.object(accel, "tpu_hint", return_value={**hint, "likely": False}):
            self.assertEqual(accel.pick("auto")[2], None)

    def test_torch_tpu_busy_note(self):
        msg = "[TorchTPU] Failed to acquire a TPU device node '/dev/vfio/0' because it is already opened by another process"
        with mock.patch.object(accel, "_has", side_effect=lambda m: m == "torch_tpu"), \
                mock.patch.object(accel, "_open_torch_tpu", side_effect=RuntimeError(msg)):
            n, _, note = accel.pick("tpu")
        self.assertEqual(n, "cpu")
        self.assertIn("다른 프로세스가 TPU를 쓰고 있습니다", note)

    def test_version_mismatch_blocks_import(self):
        mods, _ = _fake_xla()
        calls = []
        mods["torch_xla"].device = lambda: calls.append("opened")
        vers = {"torch": "2.14.1+cpu", "torch_xla": "2.9.0"}
        with mock.patch.dict(sys.modules, mods), mock.patch.object(accel, "_has", return_value=True), \
                mock.patch.object(accel, "_version", side_effect=lambda p: vers.get(p)):
            n, _, note = accel.pick("tpu")
        self.assertEqual(n, "cpu")
        self.assertIn("버전이 맞지 않습니다", note)
        self.assertEqual(calls, [])

    def test_undo_xla_patches(self):
        import torch.nn as nn

        orig_clip, orig_gru = nn.utils.clip_grad_norm_, nn.GRU

        def patched(*a, **k):
            return None

        patched._orig = orig_clip

        class PatchedGRU(nn.GRU):
            _orig = orig_gru

        try:
            nn.utils.clip_grad_norm_, nn.GRU = patched, PatchedGRU
            accel._undo_xla_patches()
            self.assertIs(nn.utils.clip_grad_norm_, orig_clip)
            self.assertIs(nn.GRU, orig_gru)
        finally:
            nn.utils.clip_grad_norm_, nn.GRU = orig_clip, orig_gru

    def test_pick_survives_broken_xla(self):
        mods, _ = _fake_xla(fail="version mismatch")
        with mock.patch.dict(sys.modules, mods), mock.patch.object(accel, "_has", return_value=True):
            self.assertEqual(accel.pick("auto")[0], "cpu")
            n, _, note = accel.pick("tpu")
            self.assertEqual(n, "cpu")
            self.assertIn("version mismatch", note)

    def test_autocast_and_scaler(self):
        import contextlib

        import torch

        self.assertIsInstance(accel.autocast("cpu", None), contextlib.nullcontext)
        self.assertIsInstance(accel.autocast("cpu", torch.float32), contextlib.nullcontext)
        s = accel.make_scaler("cpu", True)
        self.assertFalse(s.is_enabled())
        self.assertFalse(accel.make_scaler("tpu", True).is_enabled())
        self.assertIsNone(accel.tiny_amp_dtype("cpu"))
        self.assertIsNone(accel.tiny_amp_dtype("mps"))
        self.assertEqual(accel.tiny_amp_dtype("tpu"), torch.bfloat16)

    def test_optimizer_step_disabled_scaler_is_plain_step(self):
        import torch

        torch.manual_seed(0)
        p1 = torch.nn.Parameter(torch.randn(3))
        p2 = torch.nn.Parameter(p1.detach().clone())
        o1, o2 = torch.optim.AdamW([p1], lr=0.1), torch.optim.AdamW([p2], lr=0.1)
        for p, o in ((p1, o1), (p2, o2)):
            (p ** 2).sum().backward()
        o1.step()
        accel.optimizer_step(o2, accel.make_scaler("cpu", False), "cpu")
        self.assertTrue(torch.equal(p1, p2))

    def test_to_cpu(self):
        import torch

        sd = OrderedDict(a=torch.ones(2), b=[torch.zeros(1), (torch.ones(1), 3)])
        sd._metadata = {"": {"version": 1}}
        self.assertIs(accel.to_cpu(sd, "cpu"), sd)
        self.assertIs(accel.to_cpu(sd, "cuda"), sd)
        out = accel.to_cpu(sd, "tpu")
        self.assertIsNot(out, sd)
        self.assertIsInstance(out, OrderedDict)
        self.assertEqual(out._metadata, sd._metadata)
        self.assertTrue(torch.equal(out["a"], sd["a"]))
        self.assertEqual(out["b"][1][1], 3)
        # 묶인 가중치(같은 텐서)는 CPU에서도 같은 텐서를 공유한다
        w = torch.nn.Parameter(torch.ones(2))
        tied = accel.to_cpu({"tok": w, "head": w}, "tpu")
        self.assertIs(tied["tok"], tied["head"])

    def test_rng_and_cache_noops(self):
        self.assertIsNone(accel.rng_state("cpu"))
        self.assertIsNone(accel.rng_state("tpu"))
        accel.empty_cache("tpu")
        accel.empty_cache("cpu")
        accel.empty_cache()
        self.assertTrue(accel.is_oom(RuntimeError("CUDA out of memory. Tried")))
        self.assertTrue(accel.is_oom(RuntimeError("RESOURCE_EXHAUSTED: Ran out of memory in memory space hbm")))
        self.assertFalse(accel.is_oom(RuntimeError("shape mismatch")))

    def test_info_without_probe(self):
        with mock.patch.object(accel, "_probe_tpu", side_effect=AssertionError("열면 안 됨")):
            info = accel.info()
        self.assertIn("available", info)
        self.assertTrue(info["available"]["cpu"])
        self.assertIn("tpu", info)


class HintTest(unittest.TestCase):
    def test_pci_scan(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            for i, (ven, dev) in enumerate([("0x1ae0", "0x0063"), ("0x10de", "0x1eb8"), ("0x1ae0", "0x9999")]):
                d = Path(td) / f"0000:00:0{i}.0"
                d.mkdir()
                (d / "vendor").write_text(ven + "\n")
                (d / "device").write_text(dev + "\n")
            real_glob = accel.glob.glob
            with mock.patch.object(accel.glob, "glob",
                                   side_effect=lambda p: real_glob(f"{td}/*/vendor") if p.startswith("/sys/bus/pci")
                                   else []):
                self.assertEqual(accel.pci_tpu_chips(), (1, "v5e"))
                env = {k: v for k, v in os.environ.items() if not k.startswith(("TPU_", "COLAB_TPU", "PJRT"))}
                with mock.patch.dict(os.environ, env, clear=True):
                    h = accel.tpu_hint()
        self.assertTrue(h["likely"])
        self.assertEqual(h["hbm_gb"], 16.0)
        self.assertEqual(h["accelerator_type"], "v5e")

    def test_hint_env(self):
        with mock.patch.dict(os.environ, {"TPU_ACCELERATOR_TYPE": "v5litepod-1"}):
            h = accel.tpu_hint()
        self.assertTrue(h["likely"])
        self.assertEqual(h["hbm_gb"], 16.0)
        self.assertEqual(h["accelerator_type"], "v5litepod-1")

    def test_hint_vfio_alone_is_not_enough(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("TPU_ACCELERATOR_TYPE", "TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID", "PJRT_DEVICE")}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(accel.glob, "glob", side_effect=lambda p: ["/dev/vfio/0"] if "vfio" in p else []):
            h = accel.tpu_hint()
        self.assertFalse(h["likely"])
        self.assertIn("/dev/vfio/*", h["signals"])


if __name__ == "__main__":
    unittest.main()
