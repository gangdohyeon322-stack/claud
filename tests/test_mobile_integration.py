"""Node(onboard.js/trainer.js) <-> Python(lbw.py/서버) 통합 시험. 실행: python -m unittest tests.test_mobile_integration -v"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from learnbox import lbw
from tests import test_mobile_upload as _mu

MobileUploadTest = _mu.MobileUploadTest
del _mu

ROOT = Path(__file__).resolve().parent.parent
DRIVER = str(ROOT / "tests" / "js" / "integration_driver.js")
NODE = shutil.which("node")


def node(*args, timeout=120):
    r = subprocess.run([NODE, DRIVER, *map(str, args)], capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@unittest.skipUnless(NODE, "node 가 없어 건너뜀")
class MobileIntegrationTest(MobileUploadTest):
    """MobileUploadTest 의 서버 준비 코드만 재사용한다 (상속된 test_* 는 아래에서 제거)."""

    def test_int_1_node_built_lbw_valid_in_python(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "n.lbw"
            node("build", p)
            h, t = lbw.read_lbw(p.read_bytes())
            self.assertEqual(lbw.validate_lbw(h, t, 2_000_000), [])
            self.assertEqual(h["total_steps"], 7)
            self.assertEqual(len(h["itos"]), 12)
            lbw.load_torch_model(h, t)
            # 서버가 받아들여 모델로 등록한다
            s, r = self.call("POST", "/api/models/upload", raw=p.read_bytes(), token=self.admin)
            self.assertEqual(s, 200, r)
            self.assertEqual(r["info"]["trained_on"], "device")
            self.assertTrue(r["info"]["onnx"])

    def test_int_2_server_lbw_parsed_in_node(self):
        data = self.get_lbw()
        h, t = lbw.read_lbw(data)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.lbw"
            p.write_bytes(data)
            out = node("parse", p)
        self.assertEqual(out["arch"], h["arch"])
        self.assertEqual(out["itos_len"], len(h["itos"]))
        self.assertEqual(out["n_tensors"], len(t))
        py_sum = sum(float(v.astype("float64").sum()) for v in t.values())
        self.assertAlmostEqual(out["sum"], py_sum, delta=1e-3 * max(1.0, abs(py_sum)))

    def _train_upload(self, kind, steps):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "o.lbw"
            r = node("train", kind, self.base, self.user, self.model_id, steps, out, timeout=170)
            self.assertTrue(r["ok"])
            data = out.read_bytes()
        h, t = lbw.read_lbw(data)
        self.assertEqual(lbw.validate_lbw(h, t, 2_000_000), [])
        s, resp = self.call("POST", "/api/models/upload", raw=data, token=self.admin)
        self.assertEqual(s, 200, resp)
        self.assertEqual(resp["info"]["trained_on"], "device")
        self.assertTrue(resp["info"]["onnx"])
        self.assertTrue(resp["info"]["onnx_int8"])
        with self.app.db.conn() as c:
            self.assertTrue(c.execute("SELECT 1 FROM notifications WHERE title LIKE '%휴대폰에서 학습한 모델%'").fetchone())
        s, g = self.call("POST", "/api/generate", {"model_id": resp["model_id"], "prompt": "딥", "max_new": 20}, token=self.user)
        self.assertEqual(s, 200, g)
        return r, resp

    def test_int_3_continue_training_roundtrip(self):
        r, resp = self._train_upload("resume", 12)
        base_total = lbw.read_lbw(self.get_lbw())[0].get("total_steps") or 0
        self.assertEqual(r["total_steps"], base_total + 12)
        self.assertEqual(resp["info"]["total_steps"], base_total + 12)

    def test_int_4_scratch_training_roundtrip(self):
        r, resp = self._train_upload("scratch", 12)
        self.assertEqual(r["total_steps"], 12)


for _n in [n for n in dir(MobileUploadTest) if n.startswith("test_")]:
    setattr(MobileIntegrationTest, _n, None)
del MobileUploadTest   # 수집기가 부모 클래스를 다시 실행하지 않도록 모듈 이름에서 제거


if __name__ == "__main__":
    unittest.main()
