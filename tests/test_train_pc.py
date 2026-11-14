"""train_pc (LoRA) 테스트. 무거운 부분은 작은 공개 테스트 모델 hf-internal-testing/tiny-random-gpt2로 확인한다.

모델이 캐시에 없으면 한 번 내려받기를 시도하고, 실패하면 건너뛴다(오프라인 환경).
"""
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from learnbox import train_pc

TINY = "hf-internal-testing/tiny-random-gpt2"

TEXTS = [
    "딥 러닝은 여러 층의 인공 신경망을 사용해 데이터에서 특징을 스스로 학습하는 기계 학습 방법이다.",
    "LearnBox는 허용된 시간대에만 웹 문서를 모으고, 사용자가 가져온 파일을 검색하고 학습한다.",
    "Transformers use self-attention to model relationships between tokens in a sequence.",
    "서울은 대한민국의 수도이며 한강이 도시를 가로질러 흐른다. 인구는 약 천만 명이다.",
    "LoRA freezes the base model and trains small low-rank matrices added to linear layers.",
    "광합성은 식물이 빛 에너지를 이용해 이산화탄소와 물로 포도당과 산소를 만드는 과정이다.",
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class EstimateTest(unittest.TestCase):
    def test_known_model(self):
        e = train_pc.estimate_lora("Qwen/Qwen2.5-0.5B-Instruct", 8, 512, 2)
        self.assertEqual(e["base_params_b"], 0.5)
        self.assertTrue(e["known"])
        self.assertEqual(e["size"], "0.5B")
        # Qwen2.5-0.5B all-linear r=8 ≈ 4.4M
        self.assertTrue(3_000_000 < e["trainable_params_est"] < 6_000_000, e["trainable_params_est"])
        self.assertGreater(e["ram_gb_est"], 0.5 * 4)  # fp32 가중치보다 큼
        self.assertGreater(e["vram_gb_est"], 0.5 * 2)
        self.assertGreater(e["ram_gb_est"], e["vram_gb_est"])
        self.assertTrue(5 < e["adapter_mb_est"] < 40, e["adapter_mb_est"])
        for k in ("base_params_b", "trainable_params_est", "ram_gb_est", "vram_gb_est", "adapter_mb_est"):
            self.assertIn(k, e)

    def test_scaling(self):
        small = train_pc.estimate_lora("Qwen/Qwen2.5-0.5B-Instruct", 8, 512, 2)
        big = train_pc.estimate_lora("Qwen/Qwen2.5-7B-Instruct", 8, 512, 2)
        self.assertGreater(big["ram_gb_est"], small["ram_gb_est"] + 20)
        r16 = train_pc.estimate_lora("Qwen/Qwen2.5-0.5B-Instruct", 16, 512, 2)
        self.assertEqual(r16["trainable_params_est"], 2 * small["trainable_params_est"])
        longer = train_pc.estimate_lora("Qwen/Qwen2.5-0.5B-Instruct", 8, 1024, 4)
        self.assertGreater(longer["ram_gb_est"], small["ram_gb_est"])

    def test_unknown_model(self):
        e = train_pc.estimate_lora("someone/mystery-model", 8, 256, 1)
        self.assertIsNone(e["base_params_b"])
        self.assertIsNone(e["trainable_params_est"])
        self.assertIsInstance(e["ram_gb_est"], float)
        self.assertIsInstance(e["vram_gb_est"], float)
        self.assertIsInstance(e["adapter_mb_est"], float)
        self.assertTrue(e["note"])

    def test_size_from_name(self):
        self.assertEqual(train_pc.estimate_lora("org/Foo-7B-chat", 8, 512, 2)["base_params_b"], 7.0)
        self.assertEqual(train_pc.params_from_name("HuggingFaceTB/SmolLM2-135M-Instruct"), 0.135)
        self.assertEqual(train_pc.params_from_name("meta-llama/Llama-3.1-8B"), 8.0)
        self.assertEqual(train_pc.params_from_name("Qwen/Qwen2.5-1.5B"), 1.5)
        self.assertIsNone(train_pc.params_from_name("gpt2"))

    def test_profile_fit(self):
        e = train_pc.estimate_lora("Qwen/Qwen2.5-7B-Instruct", 8, 512, 2,
                                   profile={"ram_gb": 8.0, "gpu": {"type": None, "vram_gb": None}})
        self.assertFalse(e["fits"]["cpu"])
        self.assertIsNone(e["recommended_device"])
        e = train_pc.estimate_lora("Qwen/Qwen2.5-0.5B-Instruct", 8, 512, 2,
                                   profile={"ram_gb": 64.0, "gpu": {"type": "cuda", "vram_gb": 24.0}})
        self.assertTrue(e["fits"]["gpu"])
        self.assertEqual(e["recommended_device"], "cuda")

    def test_base_models_list(self):
        ids = [m["id"] for m in train_pc.BASE_MODELS]
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", ids)
        for m in train_pc.BASE_MODELS:
            self.assertEqual(set(m), {"id", "size", "params_b", "min_ram_gb", "min_vram_gb", "license"})
        self.assertIs(train_pc.train_pc, train_pc.train_lora)


class CitationTest(unittest.TestCase):
    CTX = [{"title": "가", "url": "u1", "content": "x", "doc_id": 1},
           {"title": "나", "url": "u2", "content": "y"},
           {"title": "다", "url": "u3", "content": "z"}]

    def test_markers(self):
        c = train_pc.extract_citations("서울은 수도입니다 [2]. 또한 [1, 3] 참고. [9]는 없음 [2]", self.CTX)
        self.assertEqual([r["index"] for r in c], [2, 1, 3])
        self.assertEqual(c[0], {"index": 2, "title": "나", "url": "u2"})
        self.assertEqual(c[1]["doc_id"], 1)

    def test_fallback_all(self):
        c = train_pc.extract_citations("근거 표시 없음", self.CTX)
        self.assertEqual([r["index"] for r in c], [1, 2, 3])

    def test_no_contexts_skips_model(self):
        out = train_pc.answer("질문", [], "does/not-exist")
        self.assertEqual(out["citations"], [])
        self.assertTrue(out["answer"])


class TinyModelTest(unittest.TestCase):
    """실제 학습/이어 학습/답변 (tiny-random-gpt2, CPU, 몇 단계)."""

    @classmethod
    def setUpClass(cls):
        try:
            import peft  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"학습 라이브러리 없음: {e}")
        try:
            cls.tok = train_pc._load_tokenizer(TINY)
        except Exception as e:  # 캐시에 없고 내려받기도 실패
            raise unittest.SkipTest(f"테스트 모델을 불러올 수 없음: {e}")
        cls.tmp = Path(tempfile.mkdtemp(prefix="lb-lora-"))
        cls.train = cls.tmp / "train.jsonl"
        cls.eval = cls.tmp / "eval.jsonl"
        cls.qa = cls.tmp / "qa.jsonl"
        _write_jsonl(cls.train, [{"text": t * 3, "title": f"t{i}", "url": f"u{i}"} for i, t in enumerate(TEXTS * 3)])
        _write_jsonl(cls.eval, [{"text": TEXTS[0] * 2, "title": "e", "url": "e"}])
        _write_jsonl(cls.qa, [
            {"question": "대한민국의 수도는?", "answer": "서울입니다.", "source_url": "u3"},
            {"question": "LoRA는 무엇을 학습하나요?", "answer": "작은 저랭크 행렬만 학습합니다.", "source_url": "u4"},
            {"question": "광합성의 산물은?", "answer": "포도당과 산소입니다.", "source_url": "u5"},
            {"question": "", "answer": "빈 질문은 건너뜀", "source_url": ""},
        ])
        cls.events = []
        cls.out1 = cls.tmp / "lora-1"
        cls.info1 = train_pc.train_lora(
            str(cls.train), str(cls.eval), cls.out1, TINY, max_steps=3, lr=1e-3, seq_len=64, batch_size=2,
            grad_accum=2, lora_r=4, lora_alpha=8, qa_path=str(cls.qa), threads=2,
            progress=lambda p, m: cls.events.append((p, m)), use_qa=True)

    @classmethod
    def tearDownClass(cls):
        train_pc.unload()
        if getattr(cls, "tmp", None):
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_train_info(self):
        info = self.info1
        self.assertEqual(info["kind"], "lora")
        self.assertEqual(info["base_model"], TINY)
        self.assertEqual(info["steps"], 3)
        self.assertEqual(info["total_steps"], 3)
        self.assertIsNone(info["parent"])
        self.assertEqual((info["lora_r"], info["lora_alpha"]), (4, 8))
        self.assertEqual(info["qa_examples"], 3)
        self.assertGreater(info["train_blocks"], 0)
        self.assertEqual(info["seq_len"], 64)
        self.assertEqual(info["threads"], 2)
        self.assertGreater(info["trainable_params"], 0)
        self.assertGreater(info["total_params"], info["trainable_params"])
        for k in ("train_loss", "eval_loss_before", "eval_loss_after"):
            self.assertTrue(math.isfinite(info[k]), k)
        self.assertEqual(info["ignored_params"], ["use_qa"])
        for k in ("device", "seconds", "files"):
            self.assertIn(k, info)

    def test_files_saved(self):
        files = self.info1["files"]
        self.assertGreater(files.get("adapter_model.safetensors", 0), 0)
        self.assertIn("adapter_config.json", files)
        for name, size in files.items():
            self.assertEqual((self.out1 / name).stat().st_size, size)
        saved = json.loads((self.out1 / "info.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["total_steps"], 3)
        self.assertEqual(train_pc.adapter_base_model(self.out1), TINY)

    def test_progress(self):
        ps = [p for p, _ in self.events]
        self.assertTrue(ps)
        self.assertEqual(ps, sorted(ps))
        self.assertAlmostEqual(ps[-1], 1.0)
        self.assertTrue(any("LoRA 학습 3/3" in m for _, m in self.events))

    def test_resume_increments_total_steps(self):
        out2 = self.tmp / "lora-2"
        info = train_pc.train_lora(str(self.train), str(self.eval), out2, TINY, max_steps=2, seq_len=64,
                                   batch_size=1, grad_accum=1, lora_r=16, resume_from=str(self.out1),
                                   threads=1, gradient_checkpointing=True)
        self.assertEqual(info["steps"], 2)
        self.assertEqual(info["total_steps"], 5)
        self.assertEqual(info["parent"], "lora-1")
        self.assertEqual(info["lora_r"], 4)  # 기존 어댑터 설정 유지
        self.assertEqual(info["qa_examples"], 0)
        self.assertTrue(info["gradient_checkpointing"])
        self.assertTrue((out2 / "adapter_model.safetensors").is_file())
        # base_model 생략 시 어댑터에서 읽음
        out3 = self.tmp / "lora-3"
        info3 = train_pc.train_lora(str(self.train), str(self.eval), out3, "", max_steps=1, seq_len=32,
                                    batch_size=1, grad_accum=1, resume_from=str(out2), threads=1)
        self.assertEqual(info3["total_steps"], 6)
        self.assertEqual(info3["base_model"], TINY)

    def test_resume_base_mismatch(self):
        with self.assertRaises(ValueError):
            train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "bad", "Qwen/Qwen2.5-0.5B-Instruct",
                                max_steps=1, resume_from=str(self.out1))

    def test_cancel(self):
        class Stop(Exception):
            pass

        calls = {"n": 0}

        def check():
            calls["n"] += 1
            if calls["n"] > 3:
                raise Stop()

        out = self.tmp / "lora-cancel"
        with self.assertRaises(Stop):
            train_pc.train_lora(str(self.train), str(self.eval), out, TINY, max_steps=50, seq_len=32,
                                batch_size=1, grad_accum=1, lora_r=2, threads=1, check_cancel=check)
        self.assertFalse((out / "info.json").exists())

    def test_qa_only_training(self):
        empty = self.tmp / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        info = train_pc.train_lora(str(empty), str(empty), self.tmp / "lora-qa", TINY, max_steps=1, seq_len=64,
                                   batch_size=2, grad_accum=1, lora_r=2, qa_path=str(self.qa), threads=1)
        self.assertEqual(info["train_blocks"], 0)
        self.assertEqual(info["qa_examples"], 3)
        with self.assertRaises(ValueError):
            train_pc.train_lora(str(empty), str(empty), self.tmp / "lora-none", TINY, max_steps=1, threads=1)

    def test_qa_format_fallback(self):
        self.assertFalse(getattr(self.tok, "chat_template", None))
        ids, labels = train_pc.format_qa(self.tok, "수도는?", "서울")
        self.assertEqual(len(ids), len(labels))
        text = self.tok.decode(ids)
        self.assertIn("질문: 수도는?", text)
        self.assertIn("답변:", text)
        n_prompt = sum(1 for v in labels if v == -100)
        self.assertGreater(n_prompt, 0)
        self.assertEqual(labels[n_prompt:], ids[n_prompt:])
        self.assertEqual(ids[-1], self.tok.eos_token_id)

    def test_answer(self):
        ctx = [{"title": "서울", "url": "https://example.org/seoul", "content": TEXTS[3], "doc_id": 4},
               {"title": "광합성", "url": "https://example.org/ps", "content": TEXTS[5] * 30}]
        out = train_pc.answer("대한민국의 수도는?", ctx, TINY, adapter_dir=str(self.out1), max_new_tokens=8)
        self.assertEqual(set(out), {"answer", "citations"})
        self.assertIsInstance(out["answer"], str)
        self.assertIsInstance(out["citations"], list)
        self.assertTrue(1 <= len(out["citations"]) <= 2)
        for c in out["citations"]:
            self.assertEqual({"index", "title", "url"} - set(c), set())
        # 어댑터만 주면 기반 모델을 어댑터에서 읽고, 캐시된 모델을 재사용
        out2 = train_pc.answer("수도?", ctx, None, model_dir=str(self.out1), max_new_tokens=4)
        self.assertIsInstance(out2["answer"], str)
        self.assertEqual(len(train_pc._CACHE), 1)
        # 긴 자료도 모델 최대 길이(512)에 맞게 잘린다
        long_ctx = [{"title": f"t{i}", "url": f"u{i}", "content": TEXTS[i % 6] * 40} for i in range(6)]
        out3 = train_pc.answer("질문", long_ctx, TINY, max_new_tokens=300)
        self.assertIsInstance(out3["answer"], str)


if __name__ == "__main__":
    unittest.main()
