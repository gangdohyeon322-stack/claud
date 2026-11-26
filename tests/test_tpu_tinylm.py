"""소형 AI(tinylm) 학습의 장치 계층(accel) 연동 테스트 (docs/PRD-tpu-training.md 6.2, 6.6, 6.7, 7장).

- CPU 동일성: 예전 학습 루프를 그대로 옮긴 기준 구현과 손실 기록·가중치가 비트 단위로 같다.
- 장치 고르기: 쓸 수 없는 장치를 고르면 CPU로 학습하고 예전과 같은 안내 문구를 남긴다.
- 가중치 공유(head.weight = tok.weight)가 장치 변환으로 풀리면 다시 묶는다.
- 가짜 TPU: 실제 torch_xla 없이 accel.pick 등을 바꿔 끼워 TPU 경로(단계마다 step_end, 저장 전 to_cpu,
  진행 문구, info/meta 장치 이름)를 확인한다. 연산은 CPU에서 한다.

실행: python -m unittest tests.test_tpu_tinylm -v
"""
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learnbox import accel, tinylm  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None

SENTS = [
    "딥 러닝은 여러 층의 인공 신경망을 사용하는 기계 학습 방법이다.",
    "김치찌개는 한국의 대표적인 음식이며 돼지고기와 두부를 넣어 끓인다.",
    "서울은 대한민국의 수도이며 인구가 가장 많은 도시이다.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
]
ARCH = {"d_model": 32, "n_layer": 2, "n_head": 4, "seq_len": 16, "vocab_size": 100}
KW = dict(batch_size=8, lr=3e-3, threads=1, seed=0, onnx_int8=False)
STATE_KEYS = {"model", "opt", "scaler", "gen", "torch_rng", "cuda_rng", "recent", "B", "T"}
COMPILE_MSG = "TPU 컴파일 중(처음 한 번, 몇 분 걸릴 수 있음)"


def write_corpus(d: Path, n_train: int = 40, n_eval: int = 4) -> tuple[str, str]:
    d.mkdir(parents=True, exist_ok=True)
    tr, ev = d / "train.jsonl", d / "eval.jsonl"
    with open(tr, "w", encoding="utf-8") as f:
        for i in range(n_train):
            text = " ".join(SENTS[(i + j) % len(SENTS)] for j in range(3))
            f.write(json.dumps({"text": text, "title": f"t{i}", "url": f"https://a/{i}"}, ensure_ascii=False) + "\n")
    with open(ev, "w", encoding="utf-8") as f:
        for i in range(n_eval):
            text = " ".join(SENTS[(i + j + 1) % len(SENTS)] for j in range(3))
            f.write(json.dumps({"text": text, "title": f"e{i}", "url": f"https://b/{i}"}, ensure_ascii=False) + "\n")
    return str(tr), str(ev)


def model_state(model_dir):
    import torch

    return torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=True)["state_dict"]


def reference_cpu_run(train_path, arch, steps, batch_size, lr, seed, threads=1):
    """예전(장치 계층 도입 전) tinylm.train의 CPU 학습 경로를 그대로 옮긴 기준 구현.

    난수 호출 순서(manual_seed → 모델 생성 → CPU 생성기), 연산 순서, 단계마다 loss.item()이 같다.
    꺼진 GradScaler의 scale/unscale_/step/update는 loss.backward()/opt.step()과 같다.
    -> (단계별 손실 목록, 학습 후 state_dict)
    """
    import torch
    import torch.nn.functional as F

    from learnbox.dataset import load_texts

    texts = [t for t in load_texts(train_path) if t]
    itos = tinylm.build_vocab(texts, arch["vocab_size"])
    stoi = {c: i for i, c in enumerate(itos)}
    V = len(itos)
    ids, _ = tinylm._encode_corpus(texts, stoi)
    n_tok = int(ids.numel())
    T = min(arch["seq_len"], n_tok - 2)
    B = max(1, min(batch_size, n_tok - T, max(1, (n_tok - 1) // T)))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        torch.manual_seed(seed)
        model = tinylm.build_model(arch, vocab=V)
        model.fast = True
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=lr, betas=(0.9, 0.95))
        gen = torch.Generator().manual_seed(seed + 1)
        warmup = min(100, max(1, steps // 10))
        min_lr = lr * 0.1
        losses = []
        model.train()
        for step in range(1, steps + 1):
            if step <= warmup:
                cur_lr = lr * step / warmup
            else:
                frac = (step - warmup) / max(1, steps - warmup)
                cur_lr = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * frac))
            for grp in opt.param_groups:
                grp["lr"] = cur_lr
            ix = torch.randint(0, n_tok - T, (B,), generator=gen)
            chunk = ids[ix.unsqueeze(1) + torch.arange(T + 1)].long()
            x, y = chunk[:, :-1], chunk[:, 1:]
            loss = F.cross_entropy(model(x).float().reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        return losses, model.state_dict()
    finally:
        torch.set_num_threads(old_threads)


class _EnvMixin:
    def clean_env(self):
        """LEARNBOX_DEVICE / LEARNBOX_XLA_DEBUG가 바깥 환경에서 새어 들어오지 않게 한다."""
        p = mock.patch.dict(os.environ)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("LEARNBOX_DEVICE", None)
        os.environ.pop("LEARNBOX_XLA_DEBUG", None)


@unittest.skipUnless(HAS_TORCH, "torch 필요")
class CpuIdentityTest(_EnvMixin, unittest.TestCase):
    """CPU에서는 장치 계층 도입 전과 비트 단위로 같아야 한다 (PRD 7장 5)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="lb-tputiny-cpu-"))
        cls.train_path, cls.eval_path = write_corpus(cls.tmp / "data")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.clean_env()

    def _run(self, name, steps, **kw):
        root = self.tmp / f"{name}-ck"
        msgs = []
        info = tinylm.train(self.train_path, self.eval_path, self.tmp / name, ARCH, steps=steps,
                            checkpoint_dir=str(root), checkpoint_every_steps=steps, checkpoint_every_sec=0,
                            progress=lambda p, m: msgs.append((p, m)), **{**KW, **kw})
        return info, tinylm._read_tiny_checkpoint(root), msgs

    def test_matches_reference_loop_bit_for_bit(self):
        import torch

        steps = 24  # recent(최근 50개)에 모든 단계의 손실이 남도록 50 이하
        info, ck, msgs = self._run("ref", steps, device="cpu")
        ref_losses, ref_sd = reference_cpu_run(self.train_path, tinylm._clean_arch(ARCH), steps, KW["batch_size"],
                                               KW["lr"], KW["seed"])
        self.assertEqual(ck["state"]["recent"], ref_losses)  # float 그대로 비교 (비트 단위)
        sd = model_state(self.tmp / "ref")
        self.assertEqual(set(sd), set(ref_sd))
        for k in sd:
            self.assertTrue(torch.equal(sd[k], ref_sd[k]), k)
        # CPU에서는 새 안내 문구·장치 표시가 없다
        self.assertEqual(info["notes"], [])
        self.assertEqual(info["device"], "cpu")
        self.assertEqual(info["dtype"], "float32")
        self.assertEqual(ck["meta"]["device"], "cpu")
        self.assertEqual(set(ck["state"]), STATE_KEYS)
        self.assertIsNone(ck["state"]["cuda_rng"])
        self.assertIsNone(ck["state"]["scaler"])
        self.assertFalse(any("TPU" in m or "Intel GPU" in m for _, m in msgs), msgs)
        step_msgs = [m for _, m in msgs if re.match(r"소형 AI 모델 학습 \d+/", m)]
        self.assertTrue(step_msgs)
        self.assertTrue(all(re.match(r"소형 AI 모델 학습 \d+/\d+ · loss ", m) for m in step_msgs), step_msgs)
        # 가중치 공유 유지 (CPU에서는 다시 묶을 일이 없다)
        self.assertEqual(info["params"], sum(p.numel() for p in tinylm.build_model(ARCH, vocab=info["vocab_size"])
                                             .parameters()))

    def test_same_seed_same_history_and_auto_device(self):
        import torch

        if torch.cuda.is_available() or accel.available()["xpu"] or accel.available()["mps"]:
            self.skipTest("가속기가 있는 환경에서는 auto가 CPU가 아니다")
        a, cka, _ = self._run("a", 12, device="cpu")
        b, ckb, _ = self._run("b", 12)  # device 없음 -> auto -> cpu
        self.assertEqual(cka["state"]["recent"], ckb["state"]["recent"])
        self.assertEqual(a["eval_loss_after"], b["eval_loss_after"])
        self.assertEqual(b["device"], "cpu")
        self.assertEqual(b["notes"], [])
        sa, sb = model_state(self.tmp / "a"), model_state(self.tmp / "b")
        self.assertTrue(all(torch.equal(sa[k], sb[k]) for k in sa))


@unittest.skipUnless(HAS_TORCH, "torch 필요")
class DevicePickTest(_EnvMixin, unittest.TestCase):
    def setUp(self):
        self.clean_env()

    def test_unavailable_devices_fall_back_with_old_notes(self):
        import torch

        name, dev, note = tinylm._pick_device("cpu")
        self.assertEqual((name, dev.type, note), ("cpu", "cpu", None))
        if not torch.cuda.is_available():
            self.assertEqual(tinylm._pick_device("cuda"),
                             ("cpu", torch.device("cpu"), "CUDA GPU를 찾지 못해 CPU로 학습했습니다."))
            self.assertEqual(tinylm._pick_device("cuda:1")[2], "CUDA GPU를 찾지 못해 CPU로 학습했습니다.")
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            self.assertEqual(tinylm._pick_device("mps")[2], "Apple GPU(MPS)를 찾지 못해 CPU로 학습했습니다.")
        # 사용자가 쓴 값 그대로 안내한다 (예전과 같은 문구)
        self.assertEqual(tinylm._pick_device("Weird9"),
                         ("cpu", torch.device("cpu"), "알 수 없는 장치 'Weird9' 대신 CPU로 학습했습니다."))

    def test_env_used_only_when_not_requested(self):
        import torch

        with mock.patch.object(accel, "_probe_tpu", return_value=(None, "not_installed")):
            os.environ["LEARNBOX_DEVICE"] = "tpu"
            name, dev, note = tinylm._pick_device(None)
            self.assertEqual((name, dev), ("cpu", torch.device("cpu")))
            self.assertTrue(note.startswith("TPU 패키지(torch_xla)가 없어 CPU로 학습했습니다."), note)
            self.assertIn("python -m learnbox.setup_accel", note)
            self.assertEqual(tinylm._pick_device("auto")[2], note)  # auto도 '지정 안 함'
            self.assertEqual(tinylm._pick_device("cpu"), ("cpu", torch.device("cpu"), None))  # 요청이 우선

    def test_train_on_missing_tpu_falls_back_to_cpu(self):
        tmp = Path(tempfile.mkdtemp(prefix="lb-tputiny-pick-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        tr, ev = write_corpus(tmp / "data", n_train=12)
        with mock.patch.object(accel, "_probe_tpu", return_value=(None, "not_installed")):
            info = tinylm.train(tr, ev, tmp / "m", ARCH, steps=3, device="tpu", **KW)
        self.assertEqual(info["device"], "cpu")
        self.assertEqual(info["dtype"], "float32")
        self.assertEqual(len([n for n in info["notes"] if n.startswith("TPU 패키지(torch_xla)가 없어")]), 1,
                         info["notes"])
        saved = json.loads((tmp / "m" / "info.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["device"], saved["dtype"]), ("cpu", "float32"))


@unittest.skipUnless(HAS_TORCH, "torch 필요")
class RetieTest(unittest.TestCase):
    def _count(self, m):
        return sum(p.numel() for p in m.parameters())

    def test_retie_after_broken_tie(self):
        import torch

        m = tinylm.build_model(ARCH, vocab=50)
        tied = self._count(m)
        self.assertIs(m.head.weight, m.tok.weight)
        self.assertFalse(tinylm._retie_head(m))  # 이미 묶여 있으면 아무것도 안 함
        # 장치 변환으로 공유가 풀린 상태 흉내: 값은 같고 객체만 다른 매개변수
        m.head.weight = torch.nn.Parameter(m.tok.weight.detach().clone())
        self.assertIsNot(m.head.weight, m.tok.weight)
        self.assertEqual(self._count(m), tied + 50 * ARCH["d_model"])
        self.assertTrue(tinylm._retie_head(m))
        self.assertIs(m.head.weight, m.tok.weight)
        self.assertEqual(self._count(m), tied)
        self.assertFalse(tinylm._retie_head(m))
        x = torch.randint(0, 50, (2, 8))
        self.assertEqual(tuple(m(x).shape), (2, 8, 50))

    def test_retie_after_non_dense_device_move(self):
        """meta 장치는 XLA처럼 '얕은 복사 호환'이 아니라서 .to()가 매개변수를 새로 만든다 (TPU와 같은 원리)."""
        m = tinylm.build_model(ARCH, vocab=50)
        tied = self._count(m)
        m.to("meta")
        if m.head.weight is m.tok.weight:
            self.skipTest("이 torch에서는 meta 이동이 가중치 공유를 풀지 않는다")
        self.assertGreater(self._count(m), tied)
        self.assertTrue(tinylm._retie_head(m))
        self.assertIs(m.head.weight, m.tok.weight)
        self.assertEqual(self._count(m), tied)

    def test_retie_ignores_models_without_head(self):
        from torch import nn

        self.assertFalse(tinylm._retie_head(nn.Linear(2, 2)))


@unittest.skipUnless(HAS_TORCH, "torch 필요")
class FakeTpuTest(_EnvMixin, unittest.TestCase):
    """accel.pick이 'tpu'를 돌려주게 해서 tinylm의 TPU 경로를 실제 torch_xla 없이 확인한다.

    장치 객체는 CPU라서 accel의 다른 함수는 CPU처럼 동작한다. 정밀도는 TPU처럼 bf16 자동 혼합 정밀도
    (CPU autocast)로, step_end/to_cpu/optimizer_step은 호출을 기록하며 실제 함수를 부른다.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="lb-tputiny-fake-"))
        cls.train_path, cls.eval_path = write_corpus(cls.tmp / "data")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        import torch

        self.clean_env()
        self.calls = []
        real_step_end, real_to_cpu, real_opt_step = accel.step_end, accel.to_cpu, accel.optimizer_step

        def step_end(dev):
            self.calls.append("step_end")
            return real_step_end(dev)

        def to_cpu(obj, dev=None):
            self.calls.append(("to_cpu", dev))
            return real_to_cpu(obj, dev)

        def optimizer_step(opt, scaler, dev=None):
            self.calls.append("optimizer_step")
            return real_opt_step(opt, scaler, dev)

        self.cpu = torch.device("cpu")
        self.patchers = []
        for name, new in (("pick", lambda device=None, **k: ("tpu", self.cpu, None)),
                          ("tiny_amp_dtype", lambda dev: torch.bfloat16),
                          ("step_end", step_end), ("to_cpu", to_cpu), ("optimizer_step", optimizer_step)):
            p = mock.patch.object(accel, name, new)
            p.start()
            self.addCleanup(p.stop)
            self.patchers.append(p)

    def stop_fake(self):
        for p in self.patchers:
            p.stop()
        self.patchers = []

    def _train(self, name, steps, **kw):
        msgs = []
        info = tinylm.train(self.train_path, self.eval_path, self.tmp / name, ARCH, steps=steps,
                            progress=lambda p, m: msgs.append((p, m)), **{**KW, **kw})
        return info, msgs

    def test_step_end_once_per_step_and_tpu_messages(self):
        import torch

        steps = 6
        info, msgs = self._train("plain", steps)
        # 단계마다 optimizer_step 1번 + step_end 1번, 그리고 CPU로 옮기기 전에 step_end 1번
        self.assertEqual(self.calls.count("optimizer_step"), steps)
        self.assertEqual(self.calls.count("step_end"), steps + 1)
        self.assertEqual([c for c in self.calls if c != "step_end"], ["optimizer_step"] * steps)
        for i in range(steps):  # optimizer_step 바로 뒤에 step_end
            j = [k for k, c in enumerate(self.calls) if c == "optimizer_step"][i]
            self.assertEqual(self.calls[j + 1], "step_end")
        self.assertEqual(info["device"], "tpu")
        self.assertEqual(info["dtype"], "bfloat16")
        self.assertEqual(info["steps"], steps)
        texts = [m for _, m in msgs]
        self.assertIn(f"학습 전 평가 중 · {COMPILE_MSG}", texts)
        self.assertIn(f"소형 AI 모델 학습 시작 · TPU · {COMPILE_MSG}", texts)
        step_msgs = [m for m in texts if re.match(r"소형 AI 모델 학습 \d+/", m)]
        self.assertEqual(len(step_msgs), steps)
        self.assertTrue(all(re.match(rf"소형 AI 모델 학습 \d+/{steps} · TPU · loss \d", m) for m in step_msgs),
                        step_msgs)
        ps = [p for p, _ in msgs]
        self.assertEqual(ps, sorted(ps))  # 진행률은 줄지 않는다
        self.assertEqual(info["notes"], [])
        self.assertTrue(math.isfinite(info["eval_loss_after"]))
        # 결과 모델은 CPU fp32, 가중치 공유 유지
        sd = model_state(self.tmp / "plain")
        self.assertTrue(all(v.device.type == "cpu" and v.dtype == torch.float32 for v in sd.values()))
        self.assertTrue(torch.equal(sd["head.weight"], sd["tok.weight"]))

    def test_saves_go_through_to_cpu_and_resume_on_cpu(self):
        import torch

        root = self.tmp / "save-ck"
        stop = {"n": 0}

        def stop_requested():
            stop["n"] += 1
            return stop["n"] >= 4  # 4단계에서 저장하고 끝내기

        info, _ = self._train("saved", 10, checkpoint_dir=str(root), checkpoint_every_steps=2,
                              checkpoint_every_sec=0, stop_requested=stop_requested)
        self.assertTrue(info["stopped_early"])
        self.assertEqual(info["steps"], 4)
        # 저장 2번(2단계, 4단계): 저장마다 모델·옵티마이저 상태를 TPU 장치 기준으로 to_cpu
        to_cpu = [c for c in self.calls if isinstance(c, tuple)]
        self.assertEqual(to_cpu, [("to_cpu", self.cpu)] * 4)
        # 저장 직전에도 step_end: 단계 4번 + 저장 2번 + CPU로 옮기기 전 1번
        self.assertEqual(self.calls.count("step_end"), 4 + 2 + 1)
        for k, c in enumerate(self.calls):
            if isinstance(c, tuple) and (k == 0 or not isinstance(self.calls[k - 1], tuple)):
                self.assertEqual(self.calls[k - 1], "step_end")  # 저장의 첫 to_cpu 바로 앞은 step_end
        ck = tinylm._read_tiny_checkpoint(root)
        self.assertEqual(set(ck["state"]), STATE_KEYS)  # 상태 키는 그대로
        self.assertEqual(ck["meta"]["device"], "tpu")
        self.assertEqual(ck["meta"]["step"], 4)
        self.assertIsNone(ck["state"]["cuda_rng"])
        self.assertIsNone(ck["state"]["scaler"])  # bf16: 스케일러 없음
        self.assertTrue(all(v.device.type == "cpu" for v in ck["state"]["model"].values()))
        # TPU 체크포인트를 실제 CPU에서 이어서 학습 (가짜 장치 해제)
        self.stop_fake()
        info2 =tinylm.train(None, None, self.tmp / "saved-cpu", None, resume_checkpoint=str(root), threads=1,
                             device="cpu")
        self.assertEqual(info2["steps"], 10)
        self.assertEqual(info2["resumed_from_step"], 4)
        self.assertEqual(info2["device"], "cpu")
        self.assertEqual(info2["dtype"], "float32")
        self.assertTrue(math.isfinite(info2["eval_loss_after"]))
        self.assertIsInstance(torch.load(self.tmp / "saved-cpu" / "model.pt", weights_only=True), dict)

    def test_resume_on_tpu_shows_compile_message(self):
        root = self.tmp / "resume-ck"
        self._train("first", 6, checkpoint_dir=str(root), checkpoint_every_steps=3, checkpoint_every_sec=0,
                    stop_requested=lambda: True)
        msgs = []
        info = tinylm.train(None, None, self.tmp / "resumed", None, resume_checkpoint=str(root), threads=1,
                            progress=lambda p, m: msgs.append((p, m)))
        self.assertEqual(info["device"], "tpu")
        self.assertEqual(info["steps"], 6)
        self.assertIn(f"체크포인트에서 이어서 학습합니다 (1/6단계부터) · {COMPILE_MSG}", [m for _, m in msgs])
        self.assertFalse(any(m.startswith("학습 전 평가 중") for _, m in msgs))

    def test_broken_tie_is_retied_on_both_moves(self):
        import torch

        orig = tinylm.build_model
        moves = []

        def fake_build(arch, vocab=None):
            m = orig(arch, vocab)
            real_to = m.to

            def to(*a, **k):  # torch_xla처럼 장치 변환 때마다 가중치 공유가 풀린다
                r = real_to(*a, **k)
                # 풀린 사본을 0으로 채워, 다시 묶지 않으면 결과에서 바로 드러나게 한다
                m.head.weight = torch.nn.Parameter(torch.zeros_like(m.head.weight))
                moves.append(a)
                return r

            m.to = to
            return m

        root = self.tmp / "tie-ck"
        with mock.patch.object(tinylm, "build_model", side_effect=fake_build):
            info, _ = self._train("tie", 4, checkpoint_dir=str(root), checkpoint_every_steps=4,
                                  checkpoint_every_sec=0)
        self.assertEqual(len(moves), 2)  # 장치로 1번, CPU로 1번
        ref = orig(ARCH, vocab=info["vocab_size"])
        self.assertEqual(info["params"], sum(p.numel() for p in ref.parameters()))
        sd = model_state(self.tmp / "tie")
        self.assertTrue(torch.equal(sd["head.weight"], sd["tok.weight"]))
        ck = tinylm._read_tiny_checkpoint(root)
        self.assertEqual(len(ck["state"]["opt"]["state"]), len(list(ref.parameters())))
        self.assertTrue(torch.equal(ck["state"]["model"]["head.weight"], ck["state"]["model"]["tok.weight"]))

    def test_recompile_warning_only_with_xla_debug(self):
        counts = iter([5, 8])
        os.environ["LEARNBOX_XLA_DEBUG"] = "1"
        with mock.patch.object(accel, "compile_count", side_effect=lambda: next(counts)) as cc:
            info, _ = self._train("dbg", 5)
        self.assertEqual(cc.call_count, 2)
        warn = [n for n in info["notes"] if "다시 컴파일" in n]
        self.assertEqual(len(warn), 1, info["notes"])
        self.assertIn("3번", warn[0])
        os.environ.pop("LEARNBOX_XLA_DEBUG")
        with mock.patch.object(accel, "compile_count", return_value=5) as cc:
            info, _ = self._train("nodbg", 5)
        self.assertEqual(cc.call_count, 0)
        self.assertEqual(info["notes"], [])


if __name__ == "__main__":
    unittest.main()
