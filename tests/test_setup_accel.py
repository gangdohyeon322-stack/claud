"""TPU·GPU 가속 설치 도우미(learnbox/setup_accel.py) 테스트. 실제로 pip을 실행하지 않는다."""
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learnbox import setup_accel  # noqa: E402

TPU = {"likely": True, "signals": ["pci:v5ex1"], "accelerator_type": "v5e", "chips": 1, "hbm_gb": 16.0}
NO_TPU = {"likely": False, "signals": [], "accelerator_type": None, "chips": 0, "hbm_gb": None}


def st(**kw):
    base = {"os": "Linux", "machine": "x86_64", "python": "3.13", "hint": TPU,
            "torch": "2.14.1+cpu", "torch_tpu": None, "torch_xla": None}
    base.update(kw)
    return base


class PlanTest(unittest.TestCase):
    def test_not_tpu_is_skipped(self):
        self.assertEqual(setup_accel.plan(st(hint=NO_TPU))["action"], "skip")
        self.assertEqual(setup_accel.plan(st(os="Windows"))["action"], "skip")
        self.assertEqual(setup_accel.plan(st(os="Darwin", machine="arm64"))["action"], "skip")

    def test_colab_with_torch_tpu_is_ready(self):
        p = setup_accel.plan(st(torch_tpu="0.2.0"))
        self.assertEqual((p["action"], p["backend"]), ("ready", "torch_tpu"))

    def test_torch_tpu_never_replaced_by_xla(self):
        # torch_tpu가 있으면 torch를 내리는 torch_xla를 설치하지 않는다
        p = setup_accel.plan(st(torch_tpu="0.2.0", python="3.12"))
        self.assertEqual(p["action"], "ready")

    def test_install_torch_tpu_on_new_python(self):
        p = setup_accel.plan(st(python="3.13"))
        self.assertEqual((p["action"], p["backend"]), ("install", "torch_tpu"))
        self.assertTrue(p["requirements"].endswith("requirements-tpu.txt"))

    def test_python_310_uses_xla(self):
        p = setup_accel.plan(st(python="3.10", torch="2.4.0"))
        self.assertEqual((p["action"], p["backend"]), ("install", "torch_xla"))
        self.assertTrue(p["requirements"].endswith("requirements-tpu-xla.txt"))

    def test_xla_ready_and_mismatch(self):
        self.assertEqual(setup_accel.plan(st(torch="2.9.0", torch_xla="2.9.0"))["action"], "ready")
        p = setup_accel.plan(st(torch="2.14.1", torch_xla="2.9.0"))
        self.assertEqual(p["action"], "unsupported")
        self.assertIn("버전이 맞지 않습니다", p["message"])

    def test_old_torch_with_torch_tpu(self):
        p = setup_accel.plan(st(torch="2.9.0", torch_tpu="0.2.0"))
        self.assertEqual(p["action"], "unsupported")

    def test_force_and_backend(self):
        self.assertEqual(setup_accel.plan(st(hint=NO_TPU), force=True)["action"], "install")
        p = setup_accel.plan(st(python="3.12"), backend="torch_xla")
        self.assertEqual(p["backend"], "torch_xla")

    def test_unsupported_python(self):
        self.assertEqual(setup_accel.plan(st(python="3.9"))["action"], "unsupported")


class MainTest(unittest.TestCase):
    def run_main(self, argv, status):
        buf = io.StringIO()
        with mock.patch.object(setup_accel, "status", return_value=status), \
                mock.patch.object(setup_accel.subprocess, "run") as run, redirect_stdout(buf):
            run.return_value = subprocess.CompletedProcess([], 0)
            code = setup_accel.main(argv)
        return code, buf.getvalue(), run

    def test_skip_does_not_run_pip(self):
        code, out, run = self.run_main([], st(hint=NO_TPU))
        self.assertEqual(code, 0)
        self.assertIn("건너뜁니다", out)
        run.assert_not_called()

    def test_install_runs_pip_with_requirements(self):
        code, out, run = self.run_main([], st())
        self.assertEqual(code, 0)
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[:4], [sys.executable, "-m", "pip", "install"])
        self.assertTrue(cmd[cmd.index("-r") + 1].endswith("requirements-tpu.txt"))

    def test_dry_run(self):
        code, out, run = self.run_main(["--dry-run"], st())
        self.assertEqual(code, 0)
        self.assertIn("실행:", out)
        run.assert_not_called()

    def test_check_exit_codes(self):
        for status, want in ((st(torch_tpu="0.2.0"), 0), (st(), 2), (st(hint=NO_TPU), 1)):
            code, out, run = self.run_main(["--check"], status)
            self.assertEqual(code, want)
            self.assertIn("plan", json.loads(out))
            run.assert_not_called()

    def test_install_failure(self):
        buf = io.StringIO()
        with mock.patch.object(setup_accel, "status", return_value=st()), \
                mock.patch.object(setup_accel.subprocess, "run",
                                  return_value=subprocess.CompletedProcess([], 1)), redirect_stdout(buf):
            self.assertEqual(setup_accel.main([]), 1)
        self.assertIn("실패", buf.getvalue())


class FilesTest(unittest.TestCase):
    def test_requirement_files(self):
        tpu = (ROOT / "requirements-tpu.txt").read_text(encoding="utf-8")
        self.assertIn("torch_tpu", tpu)
        self.assertIn("download.pytorch.org/whl/cpu", tpu)
        xla = (ROOT / "requirements-tpu-xla.txt").read_text(encoding="utf-8")
        self.assertIn("torch==2.9.0", xla)
        self.assertIn("torch_xla[tpu]==2.9.0", xla)
        train = (ROOT / "requirements-train.txt").read_text(encoding="utf-8")
        # 기본 학습 패키지 파일에는 TPU 패키지를 설치하는 줄이 없다(주석만)
        lines = [ln for ln in train.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        self.assertFalse(any("torch_xla" in ln or "torch_tpu" in ln for ln in lines))

    def test_status_does_not_import_torch(self):
        code = "import sys; from learnbox import setup_accel; setup_accel.status(); print('torch' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "False", out.stderr)

    def test_torch_build_advice(self):
        self.assertIn("cu126", setup_accel.torch_build_advice("nvidia")["index"])
        with mock.patch.object(setup_accel.platform, "system", return_value="Linux"):
            self.assertIn("xpu", setup_accel.torch_build_advice("intel")["index"])
            self.assertIn("rocm", setup_accel.torch_build_advice("amd")["index"])
        with mock.patch.object(setup_accel.platform, "system", return_value="Windows"):
            self.assertIn("CPU용 torch", setup_accel.torch_build_advice("amd")["message"])


if __name__ == "__main__":
    unittest.main()
