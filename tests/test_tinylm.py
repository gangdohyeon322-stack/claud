"""소형 AI 모델(tinylm) 테스트: 네트워크 없이, 작은 구조로 빠르게 돈다.

실행: python -m unittest tests.test_tinylm -v
"""
import importlib.util
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from learnbox import tinylm

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_ORT = importlib.util.find_spec("onnxruntime") is not None and importlib.util.find_spec("onnx") is not None

SENTS = [
    "딥 러닝은 여러 층의 인공 신경망을 사용하는 기계 학습 방법이다.",
    "김치찌개는 한국의 대표적인 음식이며 돼지고기와 두부를 넣어 끓인다.",
    "서울은 대한민국의 수도이며 인구가 가장 많은 도시이다.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "검색 기능은 학습한 파일 안에서 원하는 문장을 빠르게 찾아 준다.",
]
TINY = {"d_model": 64, "n_layer": 2, "n_head": 4, "seq_len": 48, "vocab_size": 200}
SPEC_INFO_KEYS = {"kind", "preset", "arch", "params", "steps", "total_steps", "parent", "train_loss",
                  "eval_loss_before", "eval_loss_after", "perplexity_after", "seconds", "device", "threads",
                  "onnx", "onnx_int8", "onnx_verified", "onnx_error", "files", "vocab_size", "seq_len",
                  "train_chars", "sample"}


def write_corpus(d: Path, n_train: int = 80, n_eval: int = 6, extra: str = "") -> tuple[str, str]:
    d.mkdir(parents=True, exist_ok=True)
    tr, ev = d / "train.jsonl", d / "eval.jsonl"
    with open(tr, "w", encoding="utf-8") as f:
        for i in range(n_train):
            text = " ".join(SENTS[(i + j) % len(SENTS)] for j in range(3)) + extra
            f.write(json.dumps({"text": text, "title": f"t{i}", "url": f"https://a/{i}"}, ensure_ascii=False) + "\n")
    with open(ev, "w", encoding="utf-8") as f:
        for i in range(n_eval):
            text = " ".join(SENTS[(i + j + 2) % len(SENTS)] for j in range(3)) + extra
            f.write(json.dumps({"text": text, "title": f"e{i}", "url": f"https://b/{i}"}, ensure_ascii=False) + "\n")
    return str(tr), str(ev)


class ArchTest(unittest.TestCase):
    def test_presets_metadata(self):
        expect = {"nano": (128, 2, 4, 128, 2000, "phone-low", "나노 (저사양 휴대폰)"),
                  "mini": (256, 4, 4, 192, 3000, "phone-high", "미니 (고사양 휴대폰)"),
                  "small": (384, 6, 6, 256, 4000, "pc-cpu", "스몰 (PC CPU)"),
                  "base": (512, 8, 8, 384, 6000, "pc-gpu", "베이스 (PC GPU)"),
                  "large": (768, 12, 12, 512, 8000, "pc-gpu-high", "라지 (고성능 GPU)")}
        self.assertEqual(list(tinylm.PRESETS), list(expect))
        for name, (d, l, h, t, v, target, label) in expect.items():
            p = tinylm.PRESETS[name]
            self.assertEqual((p["d_model"], p["n_layer"], p["n_head"], p["seq_len"], p["vocab_size"]), (d, l, h, t, v))
            self.assertEqual(p["target"], target)
            self.assertEqual(p["label"], label)
            self.assertTrue(p["desc"])
            a = tinylm.resolve_arch(name)
            self.assertEqual(a, {"preset": name, "d_model": d, "n_layer": l, "n_head": h, "seq_len": t,
                                 "vocab_size": v})
        self.assertEqual(tinylm.LIMITS["d_model"], (32, 2048))

    def test_auto_and_recommended(self):
        self.assertEqual(tinylm.resolve_arch()["preset"], "mini")
        self.assertEqual(tinylm.resolve_arch("auto", recommended="small")["preset"], "small")
        self.assertEqual(tinylm.resolve_arch(None, recommended="base")["d_model"], 512)
        self.assertEqual(tinylm.resolve_arch("auto", recommended="nonsense")["preset"], "mini")

    def test_overrides(self):
        a = tinylm.resolve_arch("nano", {"d_model": "256", "n_head": 8, "seq_len": 0, "vocab_size": None,
                                         "batch_size": 64})
        self.assertEqual((a["d_model"], a["n_head"], a["seq_len"], a["vocab_size"]), (256, 8, 128, 2000))
        self.assertEqual(a["preset"], "custom")
        # 프리셋과 같은 값이면 그 이름
        self.assertEqual(tinylm.resolve_arch("custom", {"d_model": 256})["preset"], "mini")
        c = tinylm.resolve_arch("custom", TINY)
        self.assertEqual(c, {"preset": "custom", **TINY})

    def test_validation_errors(self):
        with self.assertRaises(ValueError) as cm:
            tinylm.resolve_arch("custom", {"d_model": 100, "n_head": 3})
        self.assertIn("d_model은 n_head로 나누어떨어져야 합니다", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            tinylm.resolve_arch("custom")
        self.assertIn("하나 이상", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            tinylm.resolve_arch("mini", {"n_layer": 49})
        self.assertIn("n_layer", str(cm.exception))
        self.assertIn("1~48", str(cm.exception))
        for bad in ({"vocab_size": 50}, {"seq_len": 4096}, {"d_model": 16}, {"n_head": 64}, {"d_model": -64}):
            with self.assertRaises(ValueError):
                tinylm.resolve_arch("custom", bad)
        with self.assertRaises(ValueError) as cm:
            tinylm.resolve_arch("custom", {"seq_len": "abc"})
        self.assertIn("정수", str(cm.exception))
        with self.assertRaises(ValueError):
            tinylm.resolve_arch("custom", {"seq_len": True})
        with self.assertRaises(ValueError) as cm:
            tinylm.resolve_arch("huge")
        self.assertIn("알 수 없는", str(cm.exception))


class EstimateTest(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "torch 필요")
    def test_params_match_real_module(self):
        import torch

        archs = [tinylm.resolve_arch(n) for n in tinylm.PRESETS]
        archs.append(tinylm.resolve_arch("custom", {"d_model": 96, "n_layer": 3, "n_head": 3, "seq_len": 70,
                                                    "vocab_size": 333}))
        for a in archs:
            with torch.device("meta"):  # 메모리를 쓰지 않고 실제 모듈 구성
                m = tinylm.build_model(a)
            real = sum(p.numel() for p in m.parameters())
            e = tinylm.estimate(a)
            self.assertEqual(e["params"], real, a)
            self.assertEqual(tinylm.count_params(a["d_model"], a["n_layer"], a["seq_len"], a["vocab_size"]), real)

    def test_shape_and_fits(self):
        e = tinylm.estimate(tinylm.resolve_arch("nano"))
        for k in ("params", "fp32_mb", "int8_mb", "train_ram_mb", "infer_ram_mb", "fits", "label"):
            self.assertIn(k, e)
        for k in ("phone-low", "phone-high", "pc-cpu", "pc-gpu"):
            self.assertIsInstance(e["fits"][k], bool)
        self.assertLess(e["int8_mb"], e["fp32_mb"])
        self.assertAlmostEqual(e["fp32_mb"], e["params"] * 4 / 2 ** 20, delta=0.1)
        self.assertIn("파라미터", e["label"])
        fits = {n: tinylm.estimate(tinylm.resolve_arch(n))["fits"] for n in tinylm.PRESETS}
        self.assertTrue(fits["nano"]["phone-low"])
        self.assertTrue(fits["mini"]["phone-high"])
        self.assertFalse(fits["mini"]["phone-low"])
        self.assertTrue(fits["small"]["pc-cpu"])
        self.assertFalse(fits["small"]["phone-high"])
        self.assertTrue(fits["base"]["pc-gpu"])
        self.assertFalse(fits["base"]["pc-cpu"])
        self.assertFalse(fits["large"]["pc-gpu"])
        self.assertTrue(fits["large"]["pc-gpu-high"])

    def test_batch_scales_memory_and_validates(self):
        a = tinylm.resolve_arch("small")
        self.assertLess(tinylm.estimate(a, 8)["train_ram_mb"], tinylm.estimate(a, 32)["train_ram_mb"])
        self.assertEqual(tinylm.estimate(a, 8)["params"], tinylm.estimate(a, 32)["params"])
        # 학습 메모리는 최소한 가중치+기울기+AdamW(16바이트/파라미터)
        self.assertGreater(tinylm.estimate(a, 1)["train_ram_mb"], tinylm.estimate(a)["params"] * 16 / 2 ** 20)
        with self.assertRaises(ValueError):
            tinylm.estimate({"d_model": 100, "n_layer": 2, "n_head": 3, "seq_len": 64, "vocab_size": 200})
        with self.assertRaises(ValueError):
            tinylm.estimate({"d_model": 64})
        # 문자열 숫자도 허용
        self.assertEqual(tinylm.estimate({k: str(v) for k, v in TINY.items()})["params"],
                         tinylm.count_params(64, 2, 48, 200))


@unittest.skipUnless(HAS_TORCH, "torch 필요")
class TrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.train_path, cls.eval_path = write_corpus(cls.tmp / "corpus")
        cls.arch = tinylm.resolve_arch("custom", TINY)
        cls.msgs = []
        cls.m1 = cls.tmp / "models" / "tiny-1"
        cls.info = tinylm.train(cls.train_path, cls.eval_path, cls.m1, cls.arch, steps=60, batch_size=16, lr=3e-3,
                                threads=2, progress=lambda p, m: cls.msgs.append((p, m)))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_info_and_files(self):
        info = self.info
        self.assertTrue(SPEC_INFO_KEYS <= set(info), SPEC_INFO_KEYS - set(info))
        self.assertEqual(info["kind"], "tiny")
        self.assertEqual(info["preset"], "custom")
        self.assertEqual(info["arch"], self.arch)
        self.assertEqual(info["steps"], 60)
        self.assertEqual(info["total_steps"], 60)
        self.assertIsNone(info["parent"])
        self.assertEqual(info["device"], "cpu")
        self.assertEqual(info["threads"], 2)
        self.assertEqual(info["seq_len"], 48)
        self.assertLessEqual(info["vocab_size"], 200)
        self.assertGreater(info["train_chars"], 1000)
        self.assertLess(info["eval_loss_after"], info["eval_loss_before"])
        self.assertAlmostEqual(info["perplexity_after"], math.exp(info["eval_loss_after"]), delta=0.05)
        self.assertTrue(math.isfinite(info["train_loss"]))
        for name in ("model.pt", "vocab.json", "model.onnx", "model.int8.onnx", "info.json"):
            self.assertTrue((self.m1 / name).exists(), name)
        for name in ("model.pt", "model.onnx", "model.int8.onnx"):
            self.assertEqual(info["files"][name], (self.m1 / name).stat().st_size)
        self.assertFalse([p.name for p in self.m1.iterdir() if p.name.endswith(".tmp")])
        saved = json.loads((self.m1 / "info.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, json.loads(json.dumps(info)))
        itos = json.loads((self.m1 / "vocab.json").read_text(encoding="utf-8"))["itos"]
        self.assertEqual(itos[:4], ["<pad>", "<unk>", "<bos>", "<eos>"])
        self.assertEqual(len(itos), info["vocab_size"])
        self.assertIsInstance(info["sample"], str)
        self.assertTrue(info["sample"].startswith(info["sample_prompt"]))
        # 진행률: 0 -> 1, 줄지 않음, 한국어 메시지
        ps = [p for p, _ in self.msgs]
        self.assertEqual(ps, sorted(ps))
        self.assertEqual(ps[-1], 1.0)
        self.assertTrue(any("학습" in m for _, m in self.msgs))

    def test_checkpoint_keys_compatible(self):
        import torch

        ckpt = torch.load(self.m1 / "model.pt", map_location="cpu", weights_only=True)
        for k in ("vocab_size", "d_model", "n_layer", "n_head", "seq_len"):
            self.assertIn(k, ckpt["config"])
        keys = set(ckpt["state_dict"])
        expect = {"tok.weight", "pos.weight", "ln.weight", "ln.bias", "head.weight"}
        for i in range(2):
            for sub in ("ln1", "qkv", "proj", "ln2", "mlp.0", "mlp.2"):
                expect.add(f"blocks.{i}.{sub}.weight")
                expect.add(f"blocks.{i}.{sub}.bias")
        self.assertEqual(keys, expect)

    @unittest.skipUnless(HAS_ORT, "onnxruntime 필요")
    def test_onnx_matches_torch(self):
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch

        self.assertTrue(self.info["onnx"])
        self.assertTrue(self.info["onnx_verified"])
        self.assertIsNone(self.info["onnx_error"])
        m = onnx.load(str(self.m1 / "model.onnx"))
        self.assertEqual([o.version for o in m.opset_import if o.domain in ("", "ai.onnx")], [17])
        sess = ort.InferenceSession(str(self.m1 / "model.onnx"), providers=["CPUExecutionProvider"])
        inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
        self.assertEqual(inp.name, "input_ids")
        self.assertEqual(inp.type, "tensor(int64)")
        self.assertEqual(inp.shape[0], 1)
        self.assertIsInstance(inp.shape[1], str)  # 동적 길이 축
        self.assertEqual(outp.name, "logits")
        model, itos = tinylm.load(self.m1)
        V = len(itos)
        g = torch.Generator().manual_seed(3)
        for T in (1, 7, 48):
            x = torch.randint(0, V, (1, T), generator=g)
            with torch.no_grad():
                ref = model(x).numpy()
            got = sess.run(["logits"], {"input_ids": x.numpy().astype(np.int64)})[0]
            self.assertEqual(got.shape, (1, T, V))
            self.assertLess(float(np.abs(got - ref).max()), 1e-3, T)
        # 출력층 가중치를 공유해도(전치 사본 제거) 파일이 모델보다 크게 부풀지 않는다
        self.assertLess(self.info["files"]["model.onnx"], self.info["params"] * 4 * 1.1 + 64 * 1024)

    @unittest.skipUnless(HAS_ORT, "onnxruntime 필요")
    def test_int8_loads(self):
        import numpy as np
        import onnxruntime as ort

        self.assertTrue(self.info["onnx_int8"])
        sess = ort.InferenceSession(str(self.m1 / "model.int8.onnx"), providers=["CPUExecutionProvider"])
        x = np.array([[2, 5, 6, 7, 8, 9, 10]], dtype=np.int64)
        out = sess.run(["logits"], {"input_ids": x})[0]
        self.assertEqual(out.shape, (1, 7, self.info["vocab_size"]))
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(self.info["files"]["model.int8.onnx"], self.info["files"]["model.onnx"])

    def test_generate_and_cache(self):
        out = tinylm.generate(self.m1, "서울은", max_new=30, seed=5)
        self.assertIsInstance(out, str)
        self.assertLessEqual(len(out), 30)
        self.assertEqual(out, tinylm.generate(self.m1, "서울은", max_new=30, seed=5))
        self.assertIsInstance(tinylm.generate(self.m1, "", max_new=5, temperature=0), str)
        self.assertIsInstance(tinylm.generate(self.m1, "알수없는글자🙂" * 30, max_new=60), str)  # 문맥보다 긴 입력
        a = tinylm.load(self.m1)
        self.assertIs(a, tinylm.load(str(self.m1)))
        pt = self.m1 / "model.pt"
        st = pt.stat()
        os.utime(pt, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        b = tinylm.load(self.m1)
        self.assertIsNot(a[0], b[0])  # 수정 시각이 바뀌면 다시 불러온다

    def test_kv_cache_matches_full_forward(self):
        import torch

        model, itos = tinylm.load(self.m1)
        x = torch.randint(0, len(itos), (1, 30), generator=torch.Generator().manual_seed(9))
        with torch.no_grad():
            full = model(x)
            lg, past = model.forward_cached(x[:, :10], None, 0)
            parts = [lg]
            for i in range(10, 30):
                lg, past = model.forward_cached(x[:, i:i + 1], past, i)
                parts.append(lg)
        self.assertLess(float((full - torch.cat(parts, dim=1)).abs().max()), 1e-4)

    def test_resume_continues(self):
        m2 = self.tmp / "models" / "tiny-2"
        info2 = tinylm.train(self.train_path, self.eval_path, m2, dict(self.arch), steps=20, batch_size=16,
                             resume_from=str(self.m1), threads=1)
        self.assertEqual(info2["steps"], 20)
        self.assertEqual(info2["total_steps"], 80)
        self.assertEqual(info2["parent"], "tiny-1")
        # 같은 평가 자료 -> 부모의 학습 후 평가값에서 시작
        self.assertAlmostEqual(info2["eval_loss_before"], self.info["eval_loss_after"], places=3)
        self.assertEqual(info2["vocab_size"], self.info["vocab_size"])
        self.assertEqual(json.loads((m2 / "vocab.json").read_text(encoding="utf-8")),
                         json.loads((self.m1 / "vocab.json").read_text(encoding="utf-8")))
        self.assertTrue((m2 / "model.onnx").exists())
        # arch=None -> 부모 구조를 이어받는다, total_steps 누적
        m3 = self.tmp / "models" / "tiny-3"
        info3 = tinylm.train(self.train_path, self.eval_path, m3, None, steps=5, batch_size=8, resume_from=str(m2),
                             onnx_int8=False)
        self.assertEqual(info3["total_steps"], 85)
        self.assertEqual(info3["arch"], self.arch)
        self.assertTrue(tinylm.same_arch(tinylm.model_arch(m3), self.arch))

    def test_resume_arch_mismatch(self):
        other = tinylm.resolve_arch("custom", {**TINY, "d_model": 128})
        with self.assertRaises(ValueError) as cm:
            tinylm.train(self.train_path, self.eval_path, self.tmp / "models" / "bad", other, steps=5,
                         resume_from=str(self.m1))
        self.assertIn("구조가 다릅니다", str(cm.exception))
        self.assertIn("d_model", str(cm.exception))
        self.assertFalse((self.tmp / "models" / "bad").exists())
        with self.assertRaises(ValueError):
            tinylm.train(self.train_path, self.eval_path, self.tmp / "models" / "bad", {**TINY, "vocab_size": 300},
                         steps=5, resume_from=str(self.m1))
        with self.assertRaises(ValueError):
            tinylm.train(self.train_path, self.eval_path, self.tmp / "models" / "bad", self.arch, steps=5,
                         resume_from=str(self.tmp / "nope"))

    def test_resume_new_chars_map_to_unk(self):
        tr, ev = write_corpus(self.tmp / "corpus-new", n_train=20, n_eval=3, extra=" ★☆♥♦ 새로운글자들춃")
        info = tinylm.train(tr, ev, self.tmp / "models" / "tiny-unk", None, steps=3, batch_size=4,
                            resume_from=str(self.m1), onnx_int8=False)
        self.assertGreater(info["unk_rate"], 0)
        self.assertEqual(info["vocab_size"], self.info["vocab_size"])

    def test_old_format_dir(self):
        """예전 train_mobile.py 형식(config 5개 키, info.json에 arch 없음)도 불러오고 이어 학습한다."""
        import torch

        old = self.tmp / "models" / "old-mobile"
        old.mkdir(parents=True)
        itos = json.loads((self.m1 / "vocab.json").read_text(encoding="utf-8"))["itos"]
        torch.manual_seed(1)
        m = tinylm.build_model(TINY, vocab=len(itos))
        cfg = {"vocab_size": len(itos), "d_model": 64, "n_layer": 2, "n_head": 4, "seq_len": 48}
        torch.save({"config": cfg, "state_dict": m.state_dict()}, old / "model.pt")
        (old / "vocab.json").write_text(json.dumps({"itos": itos}, ensure_ascii=False), encoding="utf-8")
        (old / "info.json").write_text(json.dumps({"kind": "mobile", "preset": "tiny", "steps": 7}), encoding="utf-8")
        self.assertIsInstance(tinylm.generate(old, "딥 러닝", max_new=10, seed=0), str)
        arch = tinylm.model_arch(old)
        self.assertEqual((arch["d_model"], arch["n_layer"], arch["n_head"], arch["seq_len"]), (64, 2, 4, 48))
        info = tinylm.train(self.train_path, self.eval_path, self.tmp / "models" / "from-old", None, steps=4,
                            batch_size=8, resume_from=str(old), onnx_int8=False)
        self.assertEqual(info["total_steps"], 11)
        self.assertEqual(info["parent"], "old-mobile")

    def test_out_of_memory_halves_batch(self):
        from unittest import mock

        orig = tinylm.build_model

        def fake_build(arch, vocab=None):
            m = orig(arch, vocab)
            real_forward = m.forward

            def fwd(idx):
                if m.training and idx.shape[0] > 4:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
                return real_forward(idx)

            m.forward = fwd
            return m

        with mock.patch.object(tinylm, "build_model", side_effect=fake_build):
            info = tinylm.train(self.train_path, self.eval_path, self.tmp / "models" / "oom", self.arch, steps=3,
                                batch_size=16, onnx_int8=False)
        self.assertEqual(info["batch_size"], 4)
        self.assertEqual(info["steps"], 3)
        self.assertTrue(any("메모리가 부족" in n for n in info["notes"]))
        self.assertTrue(info["onnx_verified"] or not HAS_ORT)

    def test_small_corpus_and_limits(self):
        d = self.tmp / "small-corpus"
        d.mkdir()
        tr = d / "train.jsonl"
        tr.write_text(json.dumps({"text": "짧은 자료입니다. " * 4, "title": "s", "url": "u"}, ensure_ascii=False) + "\n",
                      encoding="utf-8")
        # 평가 파일이 없으면 학습 자료 끝부분으로 평가, 문맥보다 짧은 자료 -> 문맥/배치 자동 축소
        info = tinylm.train(str(tr), str(d / "missing.jsonl"), d / "m", self.arch, steps=3, batch_size=64,
                            onnx_int8=False, device="cuda", max_seconds=1e-6)
        self.assertLess(info["batch_size"], 64)
        self.assertLess(info["train_seq_len"], 48)
        self.assertEqual(info["eval_source"], "train_tail")
        self.assertTrue(info["notes"])
        self.assertEqual(info["device"], "cpu")  # CUDA가 없으면 CPU로
        self.assertTrue(info["stopped_early"])
        self.assertEqual(info["steps"], 1)
        self.assertFalse(info["onnx_int8"])
        self.assertFalse((d / "m" / "model.int8.onnx").exists())
        self.assertIsInstance(tinylm.generate(d / "m", "짧은", max_new=10), str)
        tr.write_text(json.dumps({"text": "너무 짧음", "title": "s", "url": "u"}, ensure_ascii=False) + "\n",
                      encoding="utf-8")
        with self.assertRaises(ValueError):
            tinylm.train(str(tr), None, d / "m2", self.arch, steps=3)
        tr.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            tinylm.train(str(tr), None, d / "m3", self.arch, steps=3)


if __name__ == "__main__":
    unittest.main()
