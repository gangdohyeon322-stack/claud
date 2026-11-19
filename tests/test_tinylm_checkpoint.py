"""소형 AI(tinylm) 학습 중간 저장 · 일시정지 · 끝내기 · 재개 테스트 (docs/SPEC-pause-resume.md §1, §7).

작은 구조(d32 L2 h4 T16, CPU, seed 0)로 빠르게 돈다. vocab_size는 LIMITS 최솟값(100)을 쓰고,
실제 어휘는 말뭉치의 글자 수만큼(100 미만)이다.

실행: python -m unittest tests.test_tinylm_checkpoint -v
"""
import importlib.util
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from learnbox import checkpoint, tinylm

HAS_TORCH = importlib.util.find_spec("torch") is not None

SENTS = [
    "딥 러닝은 여러 층의 인공 신경망을 사용하는 기계 학습 방법이다.",
    "김치찌개는 한국의 대표적인 음식이며 돼지고기와 두부를 넣어 끓인다.",
    "서울은 대한민국의 수도이며 인구가 가장 많은 도시이다.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
]
ARCH = {"d_model": 32, "n_layer": 2, "n_head": 4, "seq_len": 16, "vocab_size": 100}
STEPS = 30
KW = dict(batch_size=8, lr=3e-3, threads=1, seed=0, onnx_int8=False, device="cpu")
META_KEYS = {"kind", "step", "steps", "arch", "batch_size", "lr", "seed", "train_path", "eval_path", "parent",
             "parent_path", "parent_total", "parent_train_seq", "eval_before", "elapsed", "train_seconds", "resumes",
             "notes", "onnx_int8", "max_seconds"}
STATE_KEYS = {"model", "opt", "scaler", "gen", "torch_rng", "cuda_rng", "recent", "B", "T"}


class Crash(Exception):
    """학습 중 강제 중단(저장장치 분리·전원 끊김 흉내)."""


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


def crash_at(step: int, start: int = 0):
    """pause_point: start 다음 단계부터 세어 step단계가 끝난 직후 Crash."""
    n = [start]

    def pp(save):
        n[0] += 1
        if n[0] == step:
            raise Crash()
    return pp


def state_dict(model_dir):
    import torch

    return torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=True)["state_dict"]


def max_diff(a, b) -> float:
    assert set(a) == set(b)
    return max(float((a[k].float() - b[k].float()).abs().max()) for k in a)


@unittest.skipUnless(HAS_TORCH, "torch 필요")
class TinyCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="lb-tinyck-"))
        cls.train_path, cls.eval_path = write_corpus(cls.tmp / "corpus")
        cls.arch = tinylm.resolve_arch("custom", ARCH)
        # 끊김 없는 기준 실행
        cls.base = tinylm.train(cls.train_path, cls.eval_path, cls.tmp / "base", cls.arch, steps=STEPS, **KW)
        cls.base_sd = state_dict(cls.tmp / "base")
        # 12단계마다 저장, 17단계에서 끊김 -> 12단계 체크포인트 (다른 테스트가 복사해서 쓴다)
        cls.ck12 = cls.tmp / "ck12"
        cls.ck12_events = []
        try:
            tinylm.train(cls.train_path, cls.eval_path, cls.tmp / "crashed", cls.arch, steps=STEPS, **KW,
                         checkpoint_dir=str(cls.ck12), checkpoint_every_steps=12, checkpoint_every_sec=0,
                         pause_point=crash_at(17), on_checkpoint=cls.ck12_events.append)
        except Crash:
            pass
        else:
            raise AssertionError("17단계에서 끊겨야 합니다")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def copy_ck12(self, name: str) -> Path:
        dst = self.tmp / name
        shutil.copytree(self.ck12, dst)
        return dst

    # ------------------------------------------------------------------------------------------------------------
    def test_crash_left_checkpoint(self):
        self.assertEqual([e["step"] for e in self.ck12_events], [12])
        ev = self.ck12_events[0]
        self.assertEqual(set(ev), {"path", "step", "saved_at", "bytes", "steps"})
        self.assertEqual(ev["steps"], STEPS)
        self.assertEqual(ev["path"], str(self.ck12))
        self.assertGreater(ev["bytes"], 0)
        s = checkpoint.summary(self.ck12)
        self.assertEqual((s["kind"], s["step"], s["steps"]), ("tiny", 12, STEPS))
        self.assertFalse((self.tmp / "crashed").exists())  # 끊긴 실행은 모델 폴더를 만들지 않았다
        self.assertNotIn("resumed_from_step", self.base)   # 기본 실행 info는 그대로
        self.assertNotIn("resumes", self.base)

    def test_resume_equals_uninterrupted(self):
        import torch

        root = self.copy_ck12("ck-eq")
        msgs = []
        info = tinylm.train(None, None, self.tmp / "resumed", None, resume_checkpoint=str(root),
                            checkpoint_dir=str(root), threads=1, device="cpu",
                            progress=lambda p, m: msgs.append((p, m)))
        self.assertLessEqual(max_diff(self.base_sd, state_dict(self.tmp / "resumed")), 1e-5)
        self.assertEqual(info["resumed_from_step"], 12)
        self.assertEqual(info["resumes"], 1)
        for k in ("steps", "total_steps", "batch_size", "lr", "train_seq_len", "vocab_size", "arch", "eval_loss_before",
                  "train_chars", "eval_source", "parent", "parent_path", "stopped_early"):
            self.assertEqual(info[k], self.base[k], k)
        self.assertAlmostEqual(info["train_loss"], self.base["train_loss"], places=4)
        self.assertAlmostEqual(info["eval_loss_after"], self.base["eval_loss_after"], places=4)
        self.assertEqual(info["notes"], [])  # 기본값만 넘겼으므로 '무시한 요청 값' 안내도 없다
        # 학습 전 평가·어휘 만들기는 다시 하지 않는다, 진행률은 줄지 않는다
        joined = " ".join(m for _, m in msgs)
        self.assertNotIn("학습 전 평가", joined)
        self.assertNotIn("어휘 만드는 중", joined)
        self.assertIn("12/30단계부터", joined)
        ps = [p for p, _ in msgs]
        self.assertEqual(ps, sorted(ps))
        self.assertEqual(json.loads((self.tmp / "resumed" / "vocab.json").read_text(encoding="utf-8")),
                         json.loads((self.tmp / "base" / "vocab.json").read_text(encoding="utf-8")))
        # 함수는 체크포인트를 지우지 않는다 (작업 계층의 몫)
        self.assertIsNotNone(checkpoint.load(root))
        # 연달아 두 번 끊겨도 같다: 12 -> (24 저장, 26 끊김) -> 24 -> 30
        root2 = self.copy_ck12("ck-chain")
        events = []
        with self.assertRaises(Crash):
            tinylm.train(None, None, self.tmp / "chain-1", None, resume_checkpoint=str(root2), checkpoint_dir=str(root2),
                         checkpoint_every_steps=12, checkpoint_every_sec=0, threads=1, device="cpu",
                         pause_point=crash_at(26, start=12), on_checkpoint=events.append)
        self.assertEqual([e["step"] for e in events], [24])
        meta, d = checkpoint.load(root2)
        self.assertEqual((meta["step"], meta["resumes"]), (24, 1))
        st = torch.load(d / "state.pt", map_location="cpu", weights_only=True)
        self.assertEqual(set(st), STATE_KEYS)
        info2 = tinylm.train(None, None, self.tmp / "chain-2", None, resume_checkpoint=str(root2), threads=1,
                             device="cpu")
        self.assertLessEqual(max_diff(self.base_sd, state_dict(self.tmp / "chain-2")), 1e-5)
        self.assertEqual((info2["resumed_from_step"], info2["resumes"]), (24, 2))
        self.assertEqual((info2["steps"], info2["total_steps"]), (STEPS, STEPS))
        self.assertGreaterEqual(info2["seconds"], round(meta["elapsed"], 1) - 0.05)  # 누적 시간

    def test_pause_save_and_finish(self):
        root = self.tmp / "ck-pause"
        events, got = [], []
        calls = [0]

        def pp(save):
            calls[0] += 1
            if calls[0] == 5:  # 5단계에서 일시정지: 저장 후 잠시 대기
                got.append(save())
                got.append(save())  # 같은 단계에서 다시 부르면 다시 쓰지 않고 같은 요약
                time.sleep(1.2)

        t = time.time()
        info = tinylm.train(self.train_path, self.eval_path, self.tmp / "paused", self.arch, steps=20, **KW,
                            checkpoint_dir=str(root), checkpoint_every_steps=0, checkpoint_every_sec=0,
                            pause_point=pp, stop_requested=lambda: calls[0] >= 7, on_checkpoint=events.append)
        wall = time.time() - t
        self.assertEqual(got[0]["step"], 5)
        self.assertEqual(got[0]["steps"], 20)
        self.assertIs(got[0], got[1])
        self.assertEqual([e["step"] for e in events], [5, 7])  # 일시정지 저장 + 끝내기 전 저장
        self.assertTrue(info["stopped_early"])
        self.assertEqual((info["steps"], info["total_steps"]), (7, 7))
        self.assertIn("사용자가 7/20단계에서 학습을 끝냈습니다.", info["notes"])
        for name in ("model.pt", "vocab.json", "model.onnx", "info.json"):
            self.assertTrue((self.tmp / "paused" / name).exists(), name)
        self.assertEqual(checkpoint.summary(root)["step"], 7)  # 정상 종료여도 함수는 지우지 않는다
        # 일시정지 시간(1.2초)은 학습 시간·경과 시간에서 빠진다
        self.assertLessEqual(info["seconds"], wall - 1.1)
        self.assertLessEqual(info["train_seconds"], wall - 1.1)

        # checkpoint_dir가 없으면 save()는 None, 끝내기는 그대로 동작
        got.clear()
        calls[0] = 0
        info = tinylm.train(self.train_path, self.eval_path, self.tmp / "paused-nock", self.arch, steps=20, **KW,
                            pause_point=pp, stop_requested=lambda: calls[0] >= 6)
        self.assertEqual(got, [None, None])
        self.assertTrue(info["stopped_early"])
        self.assertEqual(info["steps"], 6)
        self.assertIn("사용자가 6/20단계에서 학습을 끝냈습니다.", info["notes"])

    def test_step_and_time_based_saving(self):
        import torch

        root = self.tmp / "ck-steps"
        events = []
        tinylm.train(self.train_path, self.eval_path, self.tmp / "by-steps", self.arch, steps=10, **KW,
                     checkpoint_dir=str(root), checkpoint_every_steps=4, checkpoint_every_sec=0,
                     on_checkpoint=events.append)
        self.assertEqual([e["step"] for e in events], [4, 8])
        self.assertTrue(all(e["steps"] == 10 for e in events))
        self.assertEqual(sorted(p.name for p in root.iterdir()), ["current"])  # prev/.tmp 없음
        self.assertEqual(sorted(p.name for p in (root / "current").iterdir()),
                         ["checkpoint.json", "state.pt", "vocab.json"])
        meta, d = checkpoint.load(root)
        self.assertTrue(META_KEYS <= set(meta), META_KEYS - set(meta))
        self.assertEqual((meta["kind"], meta["step"], meta["steps"], meta["resumes"]), ("tiny", 8, 10, 0))
        self.assertEqual(meta["arch"], self.arch)
        self.assertEqual(meta["train_path"], str(Path(self.train_path).resolve()))
        self.assertEqual(meta["eval_path"], str(Path(self.eval_path).resolve()))
        self.assertEqual((meta["batch_size"], meta["seed"], meta["onnx_int8"]), (8, 0, False))
        self.assertIsNone(meta["parent"])
        self.assertAlmostEqual(meta["eval_before"], self.base["eval_loss_before"], places=3)
        st = torch.load(d / "state.pt", map_location="cpu", weights_only=True)
        self.assertEqual(set(st), STATE_KEYS)
        self.assertEqual((st["B"], st["T"]), (8, 16))
        self.assertIsNone(st["scaler"])
        self.assertIsNone(st["cuda_rng"])
        self.assertEqual(len(st["recent"]), 8)

        # 시간 기준: 아주 짧은 간격이면 매 단계 저장
        events = []
        tinylm.train(self.train_path, self.eval_path, self.tmp / "by-time", self.arch, steps=5, **KW,
                     checkpoint_dir=str(self.tmp / "ck-time"), checkpoint_every_steps=0, checkpoint_every_sec=1e-9,
                     on_checkpoint=events.append)
        self.assertEqual([e["step"] for e in events], [1, 2, 3, 4, 5])
        # 둘 다 0이면 예약 저장 없음 (일시정지/끝내기 때만)
        events = []
        tinylm.train(self.train_path, self.eval_path, self.tmp / "no-sched", self.arch, steps=5, **KW,
                     checkpoint_dir=str(self.tmp / "ck-none"), checkpoint_every_steps=0, checkpoint_every_sec=0,
                     on_checkpoint=events.append)
        self.assertEqual(events, [])
        self.assertIsNone(checkpoint.load(self.tmp / "ck-none"))

    def test_corrupt_current_falls_back_to_prev(self):
        root = self.tmp / "ck-corrupt"
        stash = self.tmp / "ck-corrupt-stash"

        def keep_first(summary):
            if summary["step"] == 4:
                shutil.copytree(root / "current", stash)

        tinylm.train(self.train_path, self.eval_path, self.tmp / "corrupt-src", self.arch, steps=10, **KW,
                     checkpoint_dir=str(root), checkpoint_every_steps=4, checkpoint_every_sec=0,
                     on_checkpoint=keep_first)
        self.assertEqual(checkpoint.load(root)[0]["step"], 8)
        shutil.copytree(stash, root / "prev")  # 바꿔 끼우던 중 끊긴 상황: current(8) + prev(4)
        sp = root / "current" / "state.pt"
        sp.write_bytes(sp.read_bytes()[: sp.stat().st_size // 2])  # current 상태 파일이 반쯤 쓰임
        info = tinylm.train(None, None, self.tmp / "from-prev", None, resume_checkpoint=str(root), threads=1,
                            device="cpu")
        self.assertEqual(info["resumed_from_step"], 4)
        self.assertEqual(info["steps"], 10)
        (root / "current" / "state.pt").unlink()  # 파일 일부가 없어도 prev
        info = tinylm.train(None, None, self.tmp / "from-prev2", None, resume_checkpoint=str(root), threads=1,
                            device="cpu")
        self.assertEqual(info["resumed_from_step"], 4)
        shutil.rmtree(root / "prev")
        with self.assertRaises(ValueError) as cm:  # current 손상 + prev 없음
            tinylm.train(None, None, self.tmp / "nope", None, resume_checkpoint=str(root))
        self.assertIn("손상", str(cm.exception))
        shutil.rmtree(root)
        with self.assertRaises(ValueError) as cm:  # 체크포인트가 아예 없음
            tinylm.train(None, None, self.tmp / "nope", None, resume_checkpoint=str(root))
        self.assertIn("체크포인트를 찾을 수 없습니다", str(cm.exception))
        self.assertFalse((self.tmp / "nope").exists())

    def test_missing_or_changed_corpus(self):
        tr, ev = write_corpus(self.tmp / "corpus-gone")
        root = self.tmp / "ck-gone"
        with self.assertRaises(Crash):
            tinylm.train(tr, ev, self.tmp / "gone", self.arch, steps=10, **KW, checkpoint_dir=str(root),
                         checkpoint_every_steps=5, checkpoint_every_sec=0, pause_point=crash_at(6))
        backup = Path(tr).read_bytes()
        Path(tr).unlink()
        with self.assertRaises(ValueError) as cm:
            tinylm.train(None, None, self.tmp / "gone-r", None, resume_checkpoint=str(root))
        self.assertEqual(str(cm.exception),
                         f"체크포인트에 기록된 학습 자료가 없습니다: {Path(tr).resolve()}. 이어서 할 수 없습니다.")
        Path(tr).write_bytes(backup)
        Path(ev).unlink()
        with self.assertRaises(ValueError) as cm:
            tinylm.train(None, None, self.tmp / "gone-r", None, resume_checkpoint=str(root))
        self.assertIn("체크포인트에 기록된 학습 자료가 없습니다", str(cm.exception))
        self.assertIn("eval.jsonl", str(cm.exception))
        self.assertFalse((self.tmp / "gone-r").exists())
        # 자료가 바뀌었으면 이어서 하되 안내를 남긴다
        write_corpus(self.tmp / "corpus-gone", n_train=45)
        info = tinylm.train(None, None, self.tmp / "gone-r", None, resume_checkpoint=str(root), threads=1,
                            device="cpu")
        self.assertEqual((info["resumed_from_step"], info["steps"]), (5, 10))
        self.assertTrue(any("학습 자료가 바뀌어" in n for n in info["notes"]), info["notes"])

    def test_checkpoint_settings_win_over_arguments(self):
        root = self.copy_ck12("ck-args")
        other = {**ARCH, "d_model": 64}
        tr2, ev2 = write_corpus(self.tmp / "corpus-other", n_train=10)
        info = tinylm.train(tr2, ev2, self.tmp / "args", other, steps=99, batch_size=4, lr=1e-3,
                            resume_checkpoint=str(root), threads=1, device="cpu")
        self.assertEqual(info["arch"], self.arch)
        self.assertEqual((info["steps"], info["batch_size"], info["lr"]), (STEPS, 8, 3e-3))
        self.assertEqual(info["train_chars"], self.base["train_chars"])  # 체크포인트에 기록된 자료를 썼다
        self.assertLessEqual(max_diff(self.base_sd, state_dict(self.tmp / "args")), 1e-5)
        note = [n for n in info["notes"] if "무시한 요청 값" in n]
        self.assertEqual(len(note), 1, info["notes"])
        for part in ("구조 d_model 64", "steps 99", "batch_size 4", "lr 0.001", "학습 자료"):
            self.assertIn(part, note[0])
        # 체크포인트와 같은 값을 넘기면 안내 없음
        root2 = self.copy_ck12("ck-args-same")
        info = tinylm.train(self.train_path, self.eval_path, self.tmp / "args-same", dict(self.arch), steps=STEPS,
                            resume_checkpoint=str(root2), **KW)
        self.assertEqual(info["notes"], [])

    def test_save_failure_keeps_training(self):
        """저장장치가 가득 차거나 빠져 체크포인트를 못 써도 학습은 계속한다 (안내 한 줄, on_checkpoint 없음)."""
        events, got, calls = [], [], [0]

        def pp(save):
            calls[0] += 1
            if calls[0] == 3:
                got.append(save())

        with mock.patch.object(tinylm, "_write_tiny_checkpoint", side_effect=OSError(28, "No space left on device")):
            info = tinylm.train(self.train_path, self.eval_path, self.tmp / "save-fail", self.arch, steps=6, **KW,
                                checkpoint_dir=str(self.tmp / "ck-fail"), checkpoint_every_steps=2,
                                checkpoint_every_sec=0, pause_point=pp, on_checkpoint=events.append)
        self.assertEqual(calls[0], 6)  # 매 단계 한 번
        self.assertEqual((info["steps"], info["stopped_early"]), (6, False))
        self.assertEqual(events, [])
        self.assertEqual(got, [None])
        self.assertEqual(len([n for n in info["notes"] if "중간 저장에 실패" in n]), 1, info["notes"])

    def test_wrong_kind_rejected(self):
        root = self.tmp / "ck-lora"
        tmp = checkpoint.begin(root)
        (tmp / "trainer.pt").write_bytes(b"x")
        checkpoint.commit(root, tmp, {"kind": "lora", "step": 3, "steps": 10})
        with self.assertRaises(ValueError) as cm:
            tinylm.train(None, None, self.tmp / "lora-x", None, resume_checkpoint=str(root))
        self.assertIn("소형 AI 모델의 체크포인트가 아닙니다", str(cm.exception))

    def test_out_of_memory_halving_keeps_batch_in_checkpoint(self):
        import torch

        orig = tinylm.build_model

        def fake_build(arch, vocab=None):
            m = orig(arch, vocab)
            real_forward = m.forward

            def fwd(idx):
                if m.training and idx.shape[0] > 2:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
                return real_forward(idx)

            m.forward = fwd
            return m

        with mock.patch.object(tinylm, "build_model", side_effect=fake_build):
            ref = tinylm.train(self.train_path, self.eval_path, self.tmp / "oom-ref", self.arch, steps=8, **KW)
            root = self.tmp / "ck-oom"
            with self.assertRaises(Crash):
                tinylm.train(self.train_path, self.eval_path, self.tmp / "oom-x", self.arch, steps=8, **KW,
                             checkpoint_dir=str(root), checkpoint_every_steps=3, checkpoint_every_sec=0,
                             pause_point=crash_at(5))
        self.assertEqual(ref["batch_size"], 2)
        meta, d = checkpoint.load(root)
        self.assertEqual(meta["step"], 3)
        self.assertEqual(torch.load(d / "state.pt", map_location="cpu", weights_only=True)["B"], 2)
        self.assertTrue(any("메모리가 부족" in n for n in meta["notes"]))
        # 메모리 부족이 없는 환경에서 재개해도 줄인 배치(2)를 그대로 쓴다 -> 결과 같음
        info = tinylm.train(None, None, self.tmp / "oom-r", None, resume_checkpoint=str(root), threads=1,
                            device="cpu")
        self.assertEqual(info["batch_size"], 2)
        self.assertEqual(info["steps"], 8)
        self.assertTrue(any("메모리가 부족" in n for n in info["notes"]))
        self.assertLessEqual(max_diff(state_dict(self.tmp / "oom-ref"), state_dict(self.tmp / "oom-r")), 1e-5)


if __name__ == "__main__":
    unittest.main()
