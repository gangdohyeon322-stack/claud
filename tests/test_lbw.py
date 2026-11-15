"""learnbox.lbw 테스트: 실제 작은 모델 폴더로 왕복/검증. 실행: python -m unittest tests.test_lbw -v"""
import copy
import importlib.util
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from learnbox import lbw, tinylm
from tests.test_tinylm import write_corpus

HAS_TORCH = importlib.util.find_spec("torch") is not None
ARCH = {"d_model": 32, "n_layer": 2, "n_head": 4, "seq_len": 16, "vocab_size": 100}


@unittest.skipUnless(HAS_TORCH, "PyTorch 없음")
class LbwTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp())
        tr, ev = write_corpus(cls.dir / "c", n_train=20, n_eval=3)
        cls.model_dir = cls.dir / "m"
        arch = {"preset": "custom", **ARCH}
        cls.info = tinylm.train(tr, ev, cls.model_dir, arch, steps=4, batch_size=4, onnx_int8=False)
        cls.arch, cls.state, cls.itos = lbw.state_from_torch_dir(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def load(self, **meta):
        data = lbw.build_lbw(self.state, self.arch, self.itos, meta)
        return lbw.read_lbw(data)

    def test_state_from_dir(self):
        self.assertEqual([n for n, _ in lbw.tensor_spec(self.arch)], list(self.state))
        self.assertNotIn("head.weight", self.state)
        self.assertEqual(self.itos[:4], tinylm.SPECIALS)
        self.assertEqual(len(self.itos), self.arch["vocab_size"])

    def test_roundtrip_file_and_bytes(self):
        p = self.dir / "a.lbw"
        lbw.write_lbw(p, self.state, self.arch, self.itos, {"steps": 4, "trained_on": "device"})
        data = p.read_bytes()
        self.assertEqual(data[:8], b"LBWEIGHT")
        self.assertEqual(struct.unpack_from("<I", data, 8)[0], 1)
        h, t = lbw.read_lbw(data)
        self.assertEqual(h["steps"], 4)
        self.assertEqual(h["itos"], self.itos)
        self.assertEqual(h["arch"], self.arch)
        self.assertEqual(list(t), list(self.state))
        for n in self.state:
            np.testing.assert_array_equal(t[n], self.state[n])
        self.assertEqual(lbw.validate_lbw(h, t, 2_000_000), [])
        self.assertEqual(lbw.lbw_from_model_dir(self.model_dir)[:8], b"LBWEIGHT")

    def test_torch_model_matches_original(self):
        import torch
        h, t = self.load()
        m = lbw.load_torch_model(h, t)
        ref, _ = tinylm.load(self.model_dir)
        x = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            self.assertLess((m(x) - ref(x)).abs().max().item(), 1e-5)
        ts = lbw.torch_state_from_lbw(t)
        self.assertNotIn("head.weight", ts)

    def test_validate_errors(self):
        h, t = self.load()
        t = {k: np.array(v) for k, v in t.items()}
        v = lambda hh, tt, mp=2_000_000: lbw.validate_lbw(hh, tt, mp)  # noqa: E731
        bad = copy.deepcopy(t); bad["ln.bias"][0] = np.nan
        e = v(h, bad); self.assertTrue(any("NaN" in x and "ln.bias" in x for x in e), e)
        bad = copy.deepcopy(t); bad["ln.weight"][1] = np.inf
        self.assertTrue(v(h, bad))
        bad = copy.deepcopy(t); bad["pos.weight"] = bad["pos.weight"][:-1]
        e = v(h, bad); self.assertTrue(any("pos.weight" in x and "모양" in x for x in e), e)
        bad = copy.deepcopy(t); del bad["ln.bias"]
        self.assertTrue(any("없습니다" in x for x in v(h, bad)))
        bad = copy.deepcopy(t); bad["extra"] = np.zeros(2, "f4")
        self.assertTrue(any("알 수 없는 텐서" in x for x in v(h, bad)))
        hh = copy.deepcopy(h); hh["itos"] = hh["itos"][:-1]
        self.assertTrue(any("어휘 수" in x for x in v(hh, t)))
        hh = copy.deepcopy(h); hh["itos"][0] = "x"
        self.assertTrue(any("<pad>" in x for x in v(hh, t)))
        e = v(h, t, 1000); self.assertTrue(any("너무 많" in x for x in e), e)
        hh = copy.deepcopy(h); hh["arch"]["n_head"] = 5
        self.assertTrue(any("나누어" in x for x in v(hh, t)))
        hh = copy.deepcopy(h); hh["arch"]["d_model"] = 8
        self.assertTrue(any("범위" in x for x in v(hh, t)))
        self.assertTrue(v({}, t))

    def test_read_errors(self):
        data = lbw.build_lbw(self.state, self.arch, self.itos, {})
        for bad in (b"", b"NOTMAGIC" + data[8:], data[:8] + struct.pack("<I", 9) + data[12:],
                    data[:-4], data + b"\0\0\0\0"):
            with self.assertRaises(ValueError):
                lbw.read_lbw(bad)
        with self.assertRaises(ValueError):
            lbw.build_lbw({k: v for k, v in self.state.items() if k != "ln.bias"}, self.arch, self.itos)


if __name__ == "__main__":
    unittest.main()
