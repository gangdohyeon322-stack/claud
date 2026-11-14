"""PC 성능 프로필(devices) 테스트. 실행: python -m unittest tests.test_devices -v"""
import re
import subprocess
import time
import unittest
from unittest import mock

from learnbox import devices, tinylm

HANGUL = re.compile(r"[가-힣]")
SPEC_KEYS = {"os", "cpu_count", "ram_gb", "gpu", "torch", "transformers", "peft", "onnxruntime", "anthropic",
             "pypdf", "tier"}
REC_KEYS = {"tiny_preset", "tiny_steps_per_cycle", "tiny_batch_size", "threads", "lora_base_model",
            "lora_max_steps", "notes"}


def prof(cpu=4, ram=8.0, gpu_type=None, vram=None, name=None, **mods):
    p = {"os": "Linux", "cpu_count": cpu, "ram_gb": ram,
         "gpu": {"type": gpu_type, "name": name, "vram_gb": vram}}
    p.update(mods)
    return p


class ProfileTest(unittest.TestCase):
    def test_keys_and_types(self):
        p = devices.pc_profile(refresh=True)
        self.assertTrue(SPEC_KEYS <= set(p), SPEC_KEYS - set(p))
        self.assertIsInstance(p["os"], str)
        self.assertIsInstance(p["cpu_count"], int)
        self.assertGreaterEqual(p["cpu_count"], 1)
        self.assertTrue(p["ram_gb"] is None or (isinstance(p["ram_gb"], float) and p["ram_gb"] > 0))
        self.assertEqual(set(p["gpu"]) >= {"type", "name", "vram_gb"}, True)
        self.assertIn(p["gpu"]["type"], ("cuda", "mps", None))
        for m in ("torch", "transformers", "peft", "onnxruntime", "anthropic", "pypdf"):
            self.assertIsInstance(p[m], bool)
        self.assertIn(p["tier"], ("pc-cpu", "pc-gpu", "pc-gpu-high"))
        self.assertEqual(p["tier"], devices.tier_of(p["gpu"]))

    def test_cache_60s_and_copy(self):
        a = devices.pc_profile(refresh=True)
        t = time.perf_counter()
        b = devices.pc_profile()
        self.assertLess(time.perf_counter() - t, 0.05)  # 캐시에서 바로
        self.assertEqual(a, b)
        b["cpu_count"] = -1
        b["gpu"]["type"] = "changed"
        c = devices.pc_profile()
        self.assertEqual(c["cpu_count"], a["cpu_count"])  # 돌려준 dict를 바꿔도 캐시는 그대로
        self.assertEqual(c["gpu"]["type"], a["gpu"]["type"])
        devices._CACHE["t"] -= devices.CACHE_SEC + 1  # 60초가 지난 것처럼
        with mock.patch.object(devices, "_cpu_count", return_value=77):
            self.assertEqual(devices.pc_profile()["cpu_count"], 77)  # 60초가 지나면 다시 측정
        devices.pc_profile(refresh=True)

    def test_without_torch(self):
        real_has = devices._has
        with mock.patch.object(devices, "_has", side_effect=lambda m: False if m == "torch" else real_has(m)):
            p = devices.pc_profile(refresh=True)
        self.assertFalse(p["torch"])
        self.assertEqual(p["gpu"], {"type": None, "name": None, "vram_gb": None})
        self.assertEqual(p["tier"], "pc-cpu")
        devices.pc_profile(refresh=True)

    def test_gpu_errors_are_guarded(self):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch 없음")
        import torch

        with mock.patch.object(torch.cuda, "is_available", side_effect=RuntimeError("driver broken")):
            gpu, _ = devices._gpu_info(8.0)
        self.assertIsNone(gpu["type"])
        self.assertIn("driver broken", gpu["error"])
        fake_props = mock.Mock(total_memory=8 * 1024 ** 3)
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.cuda, "get_device_properties", return_value=fake_props), \
                mock.patch.object(torch.cuda, "get_device_name", return_value="Fake RTX"), \
                mock.patch.object(torch.cuda, "device_count", return_value=1):
            gpu, _ = devices._gpu_info(16.0)
        self.assertEqual((gpu["type"], gpu["name"], gpu["vram_gb"]), ("cuda", "Fake RTX", 8.0))

    def test_ram_paths(self):
        total, _ = devices._ram_bytes()
        if devices.platform.system() == "Linux":
            self.assertGreater(total, 0)
        # macOS: sysctl -n hw.memsize
        done = subprocess.CompletedProcess(["sysctl"], 0, stdout="17179869184\n", stderr="")
        with mock.patch.object(devices.platform, "system", return_value="Darwin"), \
                mock.patch.object(devices.subprocess, "run", return_value=done) as run:
            self.assertEqual(devices._ram_bytes()[0], 17179869184)
        self.assertEqual(run.call_args[0][0], ["sysctl", "-n", "hw.memsize"])
        with mock.patch.object(devices.platform, "system", return_value="Darwin"), \
                mock.patch.object(devices.subprocess, "run", side_effect=FileNotFoundError()):
            self.assertEqual(devices._ram_bytes(), (None, None))
        # Windows API가 없는 환경에서도 오류 없이 None
        if devices.platform.system() != "Windows":
            self.assertEqual(devices._ram_windows(), (None, None))

    def test_tier_rules(self):
        self.assertEqual(devices.tier_of({"type": None, "vram_gb": None}), "pc-cpu")
        self.assertEqual(devices.tier_of({"type": "cuda", "vram_gb": 4}), "pc-cpu")
        self.assertEqual(devices.tier_of({"type": "cuda", "vram_gb": 6}), "pc-gpu")
        self.assertEqual(devices.tier_of({"type": "cuda", "vram_gb": 16}), "pc-gpu-high")
        self.assertEqual(devices.tier_of({"type": "mps", "vram_gb": 9.6}), "pc-gpu")
        self.assertEqual(devices.tier_of(None), "pc-cpu")


class RecommendTest(unittest.TestCase):
    def check_common(self, r):
        self.assertTrue(REC_KEYS <= set(r), REC_KEYS - set(r))
        self.assertIn(r["tiny_preset"], tinylm.PRESETS)
        self.assertIsInstance(r["tiny_steps_per_cycle"], int)
        self.assertIsInstance(r["tiny_batch_size"], int)
        self.assertTrue(r["notes"])
        self.assertTrue(all(isinstance(n, str) and HANGUL.search(n) for n in r["notes"]))

    def test_cpu_low_ram(self):
        r = devices.recommend(prof(cpu=4, ram=4.0))
        self.check_common(r)
        self.assertEqual(r["tiny_preset"], "mini")
        self.assertIsNone(r["lora_base_model"])
        self.assertEqual(r["threads"], 3)
        self.assertTrue(any("LoRA" in n for n in r["notes"]))

    def test_cpu_normal_ram(self):
        r = devices.recommend(prof(cpu=8, ram=16.0))
        self.check_common(r)
        self.assertEqual(r["tiny_preset"], "small")
        self.assertEqual(r["lora_base_model"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(r["threads"], 7)
        r = devices.recommend(prof(cpu=8, ram=6.5))
        self.assertEqual(r["tiny_preset"], "small")
        self.assertIsNone(r["lora_base_model"])  # 8GB 미만

    def test_gpu_8gb(self):
        r = devices.recommend(prof(cpu=8, ram=32.0, gpu_type="cuda", vram=8.0, name="RTX 3070"))
        self.check_common(r)
        self.assertEqual(r["tiny_preset"], "base")
        self.assertEqual(r["lora_base_model"], "Qwen/Qwen2.5-1.5B-Instruct")
        est = tinylm.estimate(tinylm.resolve_arch("base"), r["tiny_batch_size"])
        self.assertLessEqual(est["train_ram_mb"], 8 * 1024)

    def test_gpu_24gb(self):
        r = devices.recommend(prof(cpu=16, ram=64.0, gpu_type="cuda", vram=24.0, name="RTX 4090"))
        self.check_common(r)
        self.assertEqual(r["tiny_preset"], "large")
        self.assertEqual(r["lora_base_model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(r["threads"], 15)
        self.assertEqual(r["tiny_batch_size"], 32)
        self.assertGreater(r["tiny_steps_per_cycle"], devices.recommend(prof(cpu=4, ram=16.0))["tiny_steps_per_cycle"])

    def test_gpu_16gb_reduces_batch(self):
        r = devices.recommend(prof(cpu=8, ram=32.0, gpu_type="cuda", vram=16.0))
        self.assertEqual(r["tiny_preset"], "large")
        self.assertLess(r["tiny_batch_size"], 32)
        self.assertLessEqual(tinylm.estimate(tinylm.resolve_arch("large"), r["tiny_batch_size"])["train_ram_mb"],
                             16 * 1024 * 0.75)

    def test_small_gpu_falls_back_to_cpu_plan(self):
        r = devices.recommend(prof(cpu=1, ram=8.0, gpu_type="cuda", vram=2.0))
        self.assertEqual(r["tiny_preset"], "small")
        self.assertEqual(r["threads"], 1)

    def test_missing_libraries(self):
        r = devices.recommend(prof(cpu=8, ram=32.0, gpu_type="cuda", vram=8.0, torch=True, transformers=False,
                                   peft=True, onnxruntime=False, anthropic=False))
        self.assertIsNone(r["lora_base_model"])
        joined = " ".join(r["notes"])
        self.assertIn("transformers", joined)
        self.assertIn("onnxruntime", joined)
        self.assertIn("anthropic", joined)

    def test_default_profile(self):
        r = devices.recommend()
        self.check_common(r)
        self.assertEqual(r["threads"], max(1, devices.pc_profile()["cpu_count"] - 1))


if __name__ == "__main__":
    unittest.main()
