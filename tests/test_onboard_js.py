"""web/onboard.js(Onboard.lbw / Onboard.training)의 node 테스트를 실행하고, 웹 스크립트 문법을 점검한다. node 가 없으면 건너뛴다."""
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "node 가 없어 건너뜀")
class OnboardJsTest(unittest.TestCase):
    def test_node_training_tests(self):
        r = subprocess.run([NODE, str(ROOT / "tests" / "js" / "onboard_training.test.js")],
                           capture_output=True, text=True, timeout=120)
        out = r.stdout + r.stderr
        self.assertEqual(r.returncode, 0, out)
        self.assertNotIn("FAIL", out)
        m = re.search(r"(\d+)/(\d+) passed", out)
        self.assertTrue(m and m.group(1) == m.group(2), out)

    def test_syntax_check(self):
        for name in ("app.js", "onboard.js", "sw.js", "trainer.js"):
            p = ROOT / "web" / name
            if not p.exists():
                continue
            r = subprocess.run([NODE, "--check", str(p)], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, f"{name}: {r.stderr}")


if __name__ == "__main__":
    unittest.main()
