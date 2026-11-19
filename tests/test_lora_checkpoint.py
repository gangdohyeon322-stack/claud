"""train_pc.train_lora 중간 저장(체크포인트)·일시정지·재개 테스트 (docs/SPEC-pause-resume.md §1, §7).

작은 공개 테스트 모델 hf-internal-testing/tiny-random-gpt2로 CPU에서 몇 단계만 학습한다.
모델이 캐시에 없고 내려받기도 안 되면 건너뛴다(오프라인 환경).
"""
import json
import math
import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from learnbox import checkpoint, train_pc

TINY = "hf-internal-testing/tiny-random-gpt2"
STEPS = 8

TEXTS = [
    "딥 러닝은 여러 층의 인공 신경망을 사용해 데이터에서 특징을 스스로 학습하는 기계 학습 방법이다.",
    "LearnBox는 허용된 시간대에만 웹 문서를 모으고, 사용자가 가져온 파일을 검색하고 학습한다.",
    "Transformers use self-attention to model relationships between tokens in a sequence.",
    "서울은 대한민국의 수도이며 한강이 도시를 가로질러 흐른다. 인구는 약 천만 명이다.",
    "LoRA freezes the base model and trains small low-rank matrices added to linear layers.",
    "광합성은 식물이 빛 에너지를 이용해 이산화탄소와 물로 포도당과 산소를 만드는 과정이다.",
]
HANGUL = re.compile(r"[가-힣]")
META_KEYS = {"kind", "step", "steps", "base_model", "lr", "seq_len", "batch_size", "grad_accum", "lora_r",
             "lora_alpha", "qa_path", "train_path", "eval_path", "seed", "dtype", "max_train_tokens",
             "gradient_checkpointing", "parent", "parent_path", "parent_total", "eval_before", "elapsed",
             "resumes", "notes"}


class Crash(Exception):
    """학습 도중 끊김(저장장치 분리 등)을 흉내 내는 예외."""


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def crash_at(n: int, save_first: bool):
    """pause_point: n번째 호출(이번 실행의 n번째 단계)에서 (save_first면 저장한 뒤) 예외로 끊는다."""
    calls = {"n": 0}

    def pp(save):
        calls["n"] += 1
        if calls["n"] == n:
            if save_first:
                save()
            raise Crash()
    return pp


def adapter_weights(d: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(Path(d) / "adapter_model.safetensors"))


class LoraCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import peft  # noqa: F401
            import safetensors  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"학습 라이브러리 없음: {e}") from e
        try:
            train_pc._load_tokenizer(TINY)
        except Exception as e:  # 캐시에 없고 내려받기도 실패
            raise unittest.SkipTest(f"테스트 모델을 불러올 수 없음: {e}") from e
        cls.tmp = Path(tempfile.mkdtemp(prefix="lb-lora-ckpt-"))
        cls.train = cls.tmp / "train.jsonl"
        cls.eval = cls.tmp / "eval.jsonl"
        cls.qa = cls.tmp / "qa.jsonl"
        _write_jsonl(cls.train, [{"text": t * 3, "title": f"t{i}", "url": f"u{i}"} for i, t in enumerate(TEXTS * 3)])
        _write_jsonl(cls.eval, [{"text": TEXTS[0] * 2, "title": "e", "url": "e"}])
        _write_jsonl(cls.qa, [
            {"question": "대한민국의 수도는?", "answer": "서울입니다.", "source_url": "u3"},
            {"question": "LoRA는 무엇을 학습하나요?", "answer": "작은 저랭크 행렬만 학습합니다.", "source_url": "u4"},
            {"question": "광합성의 산물은?", "answer": "포도당과 산소입니다.", "source_url": "u5"},
        ])
        cls.kw = dict(max_steps=STEPS, lr=1e-3, seq_len=32, batch_size=2, grad_accum=2, lora_r=4, lora_alpha=8,
                      qa_path=str(cls.qa), threads=1, seed=3)
        # 기준: 끊김 없이 8단계 (새 인자 없이 = 지금까지와 같은 동작)
        cls.full_out = cls.tmp / "full"
        cls.full = train_pc.train_lora(str(cls.train), str(cls.eval), cls.full_out, TINY, **cls.kw)
        # 3단계에서 저장한 뒤 끊김
        cls.ck3 = cls.tmp / "ck3"
        cls.crash_out = cls.tmp / "crash3"
        try:
            train_pc.train_lora(str(cls.train), str(cls.eval), cls.crash_out, TINY, checkpoint_dir=str(cls.ck3),
                                checkpoint_every_sec=0, pause_point=crash_at(3, save_first=True), **cls.kw)
        except Crash:
            pass
        else:
            raise AssertionError("pause_point의 예외가 전파되지 않았습니다")

    @classmethod
    def tearDownClass(cls):
        train_pc.unload()
        if getattr(cls, "tmp", None):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    # --- 도우미 ---
    def copy_ck3(self, name: str) -> Path:
        dst = self.tmp / name
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(self.ck3, dst)
        return dst

    def resume(self, root: Path, out_name: str, **kw) -> dict:
        args = dict(threads=1, checkpoint_dir=str(root), checkpoint_every_sec=0)
        args.update(kw)
        return train_pc.train_lora(None, None, self.tmp / out_name, "", resume_checkpoint=str(root), **args)

    def assert_same_as_full(self, out: Path, info: dict) -> None:
        a, b = adapter_weights(self.full_out), adapter_weights(out)
        self.assertEqual(set(a), set(b))
        diff = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
        self.assertLessEqual(diff, 1e-5, f"LoRA 가중치 최대 차이 {diff}")
        self.assertEqual(info["steps"], STEPS)
        self.assertEqual(info["total_steps"], STEPS)
        self.assertEqual(len(info["loss_history"]), len(self.full["loss_history"]))
        for x, y in zip(info["loss_history"], self.full["loss_history"], strict=True):
            self.assertAlmostEqual(x, y, places=3)
        self.assertEqual(info["loss_history"][-3:], self.full["loss_history"][-3:])
        self.assertEqual(info["eval_loss_before"], self.full["eval_loss_before"])
        self.assertAlmostEqual(info["eval_loss_after"], self.full["eval_loss_after"], places=3)
        self.assertEqual(info["qa_samples_used"], self.full["qa_samples_used"])
        self.assertFalse(info["stopped_early"])

    def assert_korean_error(self, cm, *parts: str) -> None:
        msg = str(cm.exception)
        self.assertRegex(msg, HANGUL)
        for p in parts:
            self.assertIn(p, msg)

    # --- 체크포인트 내용 ---
    def test_checkpoint_contents(self):
        import torch

        got = checkpoint.load(self.ck3)
        self.assertIsNotNone(got)
        meta, d = got
        self.assertEqual(d.name, "current")
        self.assertEqual(META_KEYS - set(meta), set())
        self.assertEqual((meta["kind"], meta["step"], meta["steps"]), ("lora", 3, STEPS))
        self.assertEqual(meta["base_model"], TINY)
        self.assertEqual((meta["lora_r"], meta["lora_alpha"], meta["seed"], meta["seq_len"]), (4, 8, 3, 32))
        self.assertEqual(meta["train_path"], str(self.train.resolve()))
        self.assertEqual(meta["eval_path"], str(self.eval.resolve()))
        self.assertEqual(meta["qa_path"], str(self.qa.resolve()))
        self.assertEqual(meta["dtype"], "float32")
        self.assertEqual(meta["resumes"], 0)
        self.assertAlmostEqual(meta["eval_before"], self.full["eval_loss_before"], places=4)
        self.assertGreater(meta["elapsed"], 0)
        self.assertTrue((d / "adapter" / "adapter_config.json").is_file())
        self.assertTrue((d / "adapter" / "adapter_model.safetensors").is_file())
        keys = list(adapter_weights(d / "adapter"))
        self.assertTrue(keys and all("lora_" in k for k in keys), keys[:5])  # 임베딩 저장 안 함
        tr = torch.load(d / "trainer.pt", map_location="cpu", weights_only=True)
        for k in ("opt", "py_rng", "torch_rng", "cuda_rng", "history", "qa_used"):
            self.assertIn(k, tr)
        self.assertIsNone(tr["cuda_rng"])
        self.assertEqual(len(tr["history"]), 3)
        self.assertEqual([round(v, 4) for v in tr["history"]], self.full["loss_history"][:3])
        s = checkpoint.summary(self.ck3)
        self.assertEqual((s["kind"], s["step"], s["steps"]), ("lora", 3, STEPS))
        self.assertGreater(s["bytes"], 0)
        self.assertFalse((self.crash_out / "info.json").exists())  # 끊긴 실행은 결과 모델을 남기지 않음

    # --- 끊김 없이 8단계 == 3단계 저장 → 재개 → 8단계 ---
    def test_resume_matches_uninterrupted(self):
        root = self.copy_ck3("ck-resume")
        meta, _ = checkpoint.load(root)
        events = []
        # 호출자가 넘긴 다른 설정·resume_from은 무시하고 체크포인트 값으로 이어서 한다
        info = self.resume(root, "resumed", max_steps=100, lr=0.5, seq_len=64, resume_from=str(self.full_out),
                           progress=lambda p, m: events.append((p, m)))
        self.assert_same_as_full(self.tmp / "resumed", info)
        self.assertEqual(info["resumed_from_step"], 3)
        self.assertEqual(info["resumes"], 1)
        self.assertEqual(info["base_model"], TINY)
        self.assertGreaterEqual(info["seconds"], round(meta["elapsed"], 1))
        notes = "\n".join(info["notes"])
        self.assertIn("넘겨받은 값 무시", notes)
        self.assertIn("max_steps 100→8", notes)
        self.assertIn("resume_from", notes)
        self.assertIn("3/8단계 체크포인트에서 이어서", notes)
        self.assertIsNone(info["parent"])
        msgs = [m for _, m in events]
        self.assertFalse(any("학습 전 평가" in m for m in msgs))  # 학습 전 평가는 저장된 값 사용
        self.assertFalse(any("LoRA 학습 3/8" in m for m in msgs))
        self.assertTrue(any("LoRA 학습 4/8" in m for m in msgs))
        ps = [p for p, _ in events]
        self.assertEqual(ps, sorted(ps))
        # 학습 함수는 정상 종료해도 체크포인트를 지우지 않는다 (지우는 것은 작업 계층)
        self.assertIsNotNone(checkpoint.load(root))

    # --- 단계 기준 저장 + on_checkpoint + 두 번 끊겼다 이어서 하기 ---
    def test_periodic_saves_and_chained_resumes(self):
        root = self.tmp / "ck-chain"
        seen = []
        with self.assertRaises(Crash):  # 5단계에서 저장 없이 끊김 → 마지막 저장은 4단계
            train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "chain-1", TINY, checkpoint_dir=str(root),
                                checkpoint_every_steps=2, checkpoint_every_sec=0, on_checkpoint=seen.append,
                                pause_point=crash_at(5, save_first=False), **self.kw)
        self.assertEqual([s["step"] for s in seen], [2, 4])
        for s in seen:
            self.assertEqual(set(s), {"path", "step", "saved_at", "bytes", "steps"})
            self.assertEqual(s["steps"], STEPS)
            self.assertEqual(s["path"], str(root))
            self.assertGreater(s["bytes"], 0)
        self.assertEqual(checkpoint.summary(root)["step"], 4)
        seen.clear()
        with self.assertRaises(Crash):  # 재개 1: 5,6(저장),7단계에서 끊김
            self.resume(root, "chain-2", checkpoint_every_steps=2, on_checkpoint=seen.append,
                        pause_point=crash_at(3, save_first=False))
        self.assertEqual([s["step"] for s in seen], [6])
        meta, _ = checkpoint.load(root)
        self.assertEqual((meta["step"], meta["resumes"]), (6, 1))
        info = self.resume(root, "chain-3")  # 재개 2: 끝까지
        self.assertEqual(info["resumed_from_step"], 6)
        self.assertEqual(info["resumes"], 2)
        self.assert_same_as_full(self.tmp / "chain-3", info)

    # --- 지금까지 저장하고 끝내기 ---
    def test_stop_requested_finishes_early(self):
        root = self.tmp / "ck-stop"
        seen, calls = [], {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] >= 3

        out = self.tmp / "stopped"
        info = train_pc.train_lora(str(self.train), str(self.eval), out, TINY, checkpoint_dir=str(root),
                                   checkpoint_every_sec=0, stop_requested=stop, on_checkpoint=seen.append, **self.kw)
        self.assertTrue(info["stopped_early"])
        self.assertEqual((info["steps"], info["total_steps"]), (3, 3))
        self.assertEqual(len(info["loss_history"]), 3)
        self.assertIn("사용자가 3/8단계에서 학습을 끝냈습니다.", info["notes"])
        self.assertTrue(math.isfinite(info["eval_loss_after"]))
        self.assertTrue((out / "adapter_model.safetensors").is_file())
        saved = json.loads((out / "info.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["stopped_early"])
        # 끝내기 때도 체크포인트를 남긴다 (3단계) — 3단계까지의 가중치는 끊김 실험의 3단계 저장본과 같다
        self.assertEqual([s["step"] for s in seen], [3])
        _, d = checkpoint.load(root)
        a, b = adapter_weights(out), adapter_weights(d / "adapter")
        self.assertLessEqual(max(float((a[k] - b[k]).abs().max()) for k in a), 1e-6)

    def test_stop_and_pause_without_checkpoint_dir(self):
        got = []

        def pp(save):
            got.append(save())

        info = train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "nodir", TINY,
                                   pause_point=pp, stop_requested=lambda: True, **self.kw)
        self.assertEqual(got, [None])  # checkpoint_dir가 없으면 save()는 None
        self.assertTrue(info["stopped_early"])
        self.assertEqual(info["steps"], 1)
        self.assertNotIn("resumes", info)
        self.assertNotIn("resumed_from_step", info)

    # --- pause_point의 save(): 같은 단계는 한 번만 쓰기, 일시정지 시간은 학습 시간·ETA에서 제외 ---
    def test_pause_point_save_and_paused_time(self):
        root = self.tmp / "ck-pause"
        seen, results, events = [], [], []
        pause_sec = 2.0

        def pp(save):
            if len(results) == 0 and len(events) and any("LoRA 학습 2/" in m for _, m in events):
                results.append(save())
                results.append(save())  # 이미 저장했으면 다시 쓰지 않고 같은 요약
                time.sleep(pause_sec)

        t = time.time()
        info = train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "paused", TINY,
                                   checkpoint_dir=str(root), checkpoint_every_sec=0, pause_point=pp,
                                   on_checkpoint=seen.append, progress=lambda p, m: events.append((p, m)),
                                   **self.kw)
        wall = time.time() - t
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["step"], 2)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], seen[0])
        self.assertLessEqual(info["seconds"], wall - pause_sec + 0.5)
        meta, _ = checkpoint.load(root)
        self.assertEqual(meta["step"], 2)
        self.assertLess(meta["elapsed"], wall - pause_sec + 0.5)
        # 일시정지 바로 다음 단계의 남은 시간에 멈춘 시간이 섞이지 않는다
        m3 = next(m for _, m in events if "LoRA 학습 3/" in m)
        eta = re.search(r"남은 시간 약 (\d+)초", m3)
        self.assertIsNotNone(eta, m3)
        self.assertLess(int(eta.group(1)), pause_sec)

    def test_time_based_saves(self):
        root = self.tmp / "ck-time"
        seen = []
        kw = dict(self.kw, max_steps=3)
        info = train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "timed", TINY, checkpoint_dir=str(root),
                                   checkpoint_every_sec=1e-6, on_checkpoint=seen.append, **kw)
        self.assertEqual([s["step"] for s in seen], [1, 2, 3])
        self.assertEqual(info["steps"], 3)
        self.assertEqual(checkpoint.summary(root)["step"], 3)
        self.assertFalse((root / ".tmp").exists())
        self.assertFalse((root / "prev").exists())

    # --- 깨진/맞지 않는 체크포인트 ---
    def test_missing_or_corrupt_checkpoint(self):
        empty = self.tmp / "ck-empty"
        empty.mkdir(exist_ok=True)
        for root in (empty, self.tmp / "ck-does-not-exist"):
            with self.assertRaises(ValueError) as cm:
                self.resume(root, "never")
            self.assert_korean_error(cm, "체크포인트", str(root))

        bad = self.copy_ck3("ck-bad-trainer")
        (bad / "current" / "trainer.pt").write_bytes(b"not a torch file")
        with self.assertRaises(ValueError) as cm:
            self.resume(bad, "never")
        self.assert_korean_error(cm, "손상")

        no_adapter = self.copy_ck3("ck-no-adapter")
        (no_adapter / "current" / "adapter" / "adapter_model.safetensors").unlink()
        with self.assertRaises(ValueError) as cm:
            self.resume(no_adapter, "never")
        self.assert_korean_error(cm, "손상")
        self.assertFalse((self.tmp / "never").exists())

    def test_wrong_kind(self):
        root = self.copy_ck3("ck-kind")
        mp = root / "current" / "checkpoint.json"
        meta = json.loads(mp.read_text(encoding="utf-8"))
        meta["kind"] = "tiny"
        mp.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            self.resume(root, "never")
        self.assert_korean_error(cm, "LoRA")

    def test_missing_train_file(self):
        root = self.copy_ck3("ck-gone")
        gone = str(self.tmp / "gone" / "train.jsonl")
        mp = root / "current" / "checkpoint.json"
        meta = json.loads(mp.read_text(encoding="utf-8"))
        meta["train_path"] = gone
        mp.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            self.resume(root, "never")
        self.assertEqual(str(cm.exception), f"체크포인트에 기록된 학습 자료가 없습니다: {gone}. 이어서 할 수 없습니다.")

    def test_resume_uses_stored_settings(self):
        root = self.copy_ck3("ck-gc")
        mp = root / "current" / "checkpoint.json"
        meta = json.loads(mp.read_text(encoding="utf-8"))
        meta["gradient_checkpointing"] = True  # 처음 실행이 메모리 부족으로 체크포인팅을 켰던 경우
        mp.write_text(json.dumps(meta), encoding="utf-8")
        info = self.resume(root, "gc", gradient_checkpointing=False, dtype="bfloat16", seed=99)
        self.assertTrue(info["gradient_checkpointing"])
        self.assertEqual(info["dtype"], "float32")
        notes = "\n".join(info["notes"])
        for part in ("gradient_checkpointing False→True", "dtype bfloat16→float32", "seed 99→3"):
            self.assertIn(part, notes)
        self.assert_same_as_full(self.tmp / "gc", info)  # 활성값 체크포인팅은 결과를 바꾸지 않는다

    def test_corrupt_current_falls_back_to_prev(self):
        root = self.copy_ck3("ck-prev")
        shutil.copytree(root / "current", root / "prev")
        (root / "current" / "trainer.pt").write_bytes(b"\x00" * 64)  # current는 깨짐, prev는 온전
        info = self.resume(root, "from-prev")
        self.assertEqual(info["resumed_from_step"], 3)
        self.assert_same_as_full(self.tmp / "from-prev", info)


if __name__ == "__main__":
    unittest.main()
