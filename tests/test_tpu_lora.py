"""train_pc.train_lora의 장치 계층(accel) 연동 테스트 (docs/PRD-tpu-training.md 6.3~6.8).

TPU·GPU가 없는 환경에서 돈다:
- 정밀도 결정(_train_precision)·메모리 계획(_plan)·적합성(_fit)은 accel 함수를 흉내 내어 단위 검사한다.
- 실제 학습은 작은 공개 테스트 모델 hf-internal-testing/tiny-random-gpt2로 CPU에서 몇 단계만 한다
  (모델이 캐시에 없고 내려받기도 안 되면 건너뛴다).
- TPU 경로(고정 모양 패딩, 단계마다 step_end, CPU 사본 저장, 장치 라벨, 컴파일 안내, 재컴파일 경고)는
  accel.pick이 ("tpu", CPU torch.device)를 돌려주게 해서 CPU에서 끝까지 실행해 본다(sync만 실제로 일어나지 않음).
- fp16 스케일러 경로(n_ok==0 보호, nan 연속 멈춤, 스케일러 상태 저장·복원)는 CUDA가 없으므로 켜진 CPU용
  GradScaler(torch.amp.GradScaler("cpu"))로 대신 확인한다. 실제 CUDA fp16 autocast는 여기서 검사하지 못한다.
"""
import contextlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learnbox import accel, checkpoint, train_pc  # noqa: E402

TINY = "hf-internal-testing/tiny-random-gpt2"
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_XLA = importlib.util.find_spec("torch_xla") is not None
SEQ = 32

TEXTS = [
    "딥 러닝은 여러 층의 인공 신경망을 사용해 데이터에서 특징을 스스로 학습하는 기계 학습 방법이다.",
    "LearnBox는 허용된 시간대에만 웹 문서를 모으고, 사용자가 가져온 파일을 검색하고 학습한다.",
    "Transformers use self-attention to model relationships between tokens in a sequence.",
    "서울은 대한민국의 수도이며 한강이 도시를 가로질러 흐른다. 인구는 약 천만 명이다.",
    "LoRA freezes the base model and trains small low-rank matrices added to linear layers.",
    "광합성은 식물이 빛 에너지를 이용해 이산화탄소와 물로 포도당과 산소를 만드는 과정이다.",
]
EST = train_pc.estimate_lora("Qwen/Qwen2.5-0.5B-Instruct", 8, 512, 2)

_ENV_PATCH = None


def setUpModule():
    # 이 테스트는 환경변수 기본값(지정 안 함)을 전제로 한다.
    global _ENV_PATCH
    env = {k: v for k, v in os.environ.items() if k not in ("LEARNBOX_AMP", "LEARNBOX_DEVICE", "LEARNBOX_XLA_DEBUG")}
    _ENV_PATCH = mock.patch.dict(os.environ, env, clear=True)
    _ENV_PATCH.start()


def tearDownModule():
    if _ENV_PATCH is not None:
        _ENV_PATCH.stop()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _adapter(d: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(Path(d) / "adapter_model.safetensors"))


def _need_tiny_model():
    """학습 라이브러리와 테스트 모델이 없으면 SkipTest."""
    try:
        import peft  # noqa: F401
        import safetensors  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise unittest.SkipTest(f"학습 라이브러리 없음: {e}") from e
    try:
        return train_pc._load_tokenizer(TINY)
    except Exception as e:  # 캐시에 없고 내려받기도 실패
        raise unittest.SkipTest(f"테스트 모델을 불러올 수 없음: {e}") from e


class _Data:
    """학습·평가·Q&A 파일 (클래스마다 임시 폴더)."""

    @classmethod
    def make_data(cls, prefix: str):
        cls.tmp = Path(tempfile.mkdtemp(prefix=prefix))
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

    @classmethod
    def drop_data(cls):
        train_pc.unload()
        if getattr(cls, "tmp", None):
            shutil.rmtree(cls.tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 서명·import
# ---------------------------------------------------------------------------
class SignatureTest(unittest.TestCase):
    def test_amp_is_real_keyword(self):
        p = inspect.signature(train_pc.train_lora).parameters
        self.assertIn("amp", p)  # **ignored로 들어가면 안 된다
        self.assertIs(p["amp"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(p["amp"].default)
        self.assertIn("device", p)

    def test_import_does_not_load_torch(self):
        code = "import sys; import learnbox.train_pc; print('torch' in sys.modules, 'torch_xla' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(out.stdout.strip(), "False False", out.stderr)


# ---------------------------------------------------------------------------
# 메모리 적합성 (_fit / estimate_lora): 기존 profile 결과는 그대로, TPU·Intel GPU 추가
# ---------------------------------------------------------------------------
def _old_fit(est: dict, profile: dict) -> dict:
    """변경 전 train_pc._fit 그대로 (회귀 비교용)."""
    ram = profile.get("ram_gb")
    gpu = profile.get("gpu") or {}
    vram = gpu.get("vram_gb")
    fits_cpu = None if not ram else ram >= est["ram_gb_est"]
    fits_cpu_bf16 = None if not ram else ram >= est["ram_gb_est_bf16"]
    fits_gpu = None if not (gpu.get("type") == "cuda" and vram) else vram >= est["vram_gb_est"]
    if gpu.get("type") == "mps" and ram:
        fits_gpu = ram * 0.7 >= est["ram_gb_est"]
    if fits_gpu:
        rec, device = "GPU로 학습할 수 있습니다.", gpu.get("type")
    elif fits_cpu:
        rec, device = "CPU로 학습할 수 있지만 느립니다. 단계 수를 줄이거나 작은 모델을 쓰세요.", "cpu"
    elif fits_cpu_bf16:
        rec, device = "메모리가 빠듯해 CPU에서 bf16으로 학습합니다(자동). 매우 느릴 수 있습니다.", "cpu"
    elif ram is None:
        rec, device = "메모리 정보를 알 수 없습니다.", None
    else:
        rec, device = ("이 PC에는 너무 큰 설정입니다. 더 작은 모델(0.5B)을 고르거나 seq_len·batch를 줄이세요.",
                       None)
    return {"fits": {"cpu": fits_cpu, "cpu_bf16": fits_cpu_bf16, "gpu": fits_gpu},
            "recommended_device": device, "recommendation": rec}


class FitTest(unittest.TestCase):
    MODELS = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "mystery/x")
    NO_TPU = {"available": {"auto": True, "cuda": False, "tpu": False, "xpu": False, "mps": False, "cpu": True},
              "tpu": {"installed": False, "likely": False, "hbm_gb": None, "ok": False}}
    TPU = {"available": {"auto": True, "cuda": False, "tpu": True, "xpu": False, "mps": False, "cpu": True},
           "tpu": {"installed": True, "likely": True, "hbm_gb": 16.0, "ok": False}}

    def test_existing_profiles_unchanged(self):
        profiles = [
            {"ram_gb": 8.0, "gpu": {"type": None, "vram_gb": None}},
            {"ram_gb": 64.0, "gpu": {"type": "cuda", "vram_gb": 24.0}},
            {"ram_gb": 16.0, "gpu": {"type": "cuda", "vram_gb": 4.0}},
            {"ram_gb": 32.0, "gpu": {"type": "mps", "vram_gb": None}},
            {"ram_gb": None, "gpu": {}},
            {"ram_gb": 3.0},
        ]
        for prof in profiles:
            # devices.pc_profile()에 TPU 정보(accel.info())가 붙어도 TPU가 없으면 결과가 같아야 한다
            for p in (prof, {**prof, "accel": self.NO_TPU}):
                for m in self.MODELS:
                    with self.subTest(profile=p, model=m):
                        est = train_pc.estimate_lora(m, 8, 512, 2)
                        got = train_pc.estimate_lora(m, 8, 512, 2, profile=p)
                        self.assertEqual({k: got[k] for k in ("fits", "recommended_device", "recommendation")},
                                         _old_fit(est, p))
                        self.assertEqual(set(got) - set(est), {"fits", "recommended_device", "recommendation"})
        # cuda GPU가 있으면 TPU가 보여도 예전 판단 그대로 (auto 순서: cuda → tpu)
        p = {"ram_gb": 64.0, "gpu": {"type": "cuda", "vram_gb": 4.0}, "accel": self.TPU}
        est = train_pc.estimate_lora(self.MODELS[2], 8, 512, 2)
        self.assertEqual(train_pc._fit(est, p), _old_fit(est, p))
        # profile도 device도 없으면 fits를 붙이지 않는다 (예전 그대로)
        self.assertNotIn("fits", train_pc.estimate_lora(self.MODELS[0]))

    def test_tpu_device_uses_hint_memory(self):
        with mock.patch.object(accel, "tpu_hint", return_value={"likely": True, "hbm_gb": 16.0}):
            e = train_pc.estimate_lora("Qwen/Qwen2.5-0.5B-Instruct", 8, 512, 2, device="tpu")
            self.assertIs(e["fits"]["tpu"], True)
            self.assertEqual(e["recommended_device"], "tpu")
            self.assertEqual(e["tpu_hbm_gb"], 16.0)
            big = train_pc.estimate_lora("Qwen/Qwen2.5-7B-Instruct", 8, 2048, 8, device="tpu")
            self.assertIs(big["fits"]["tpu"], False)
            self.assertIsNone(big["recommended_device"])
            self.assertIn("줄이세요", big["recommendation"])

    def test_tpu_unknown_memory_means_measure(self):
        with mock.patch.object(accel, "tpu_hint", return_value={"likely": False, "hbm_gb": None}):
            e = train_pc.estimate_lora("Qwen/Qwen2.5-1.5B-Instruct", 8, 512, 2, device="tpu",
                                       profile={"ram_gb": 32.0, "gpu": {"type": None}})
        self.assertIsNone(e["fits"]["tpu"])
        self.assertEqual(e["recommended_device"], "tpu")
        self.assertIn("측정 필요", e["recommendation"])

    def test_tpu_from_profile_without_device(self):
        p = {"ram_gb": 47.0, "gpu": {"type": None, "vram_gb": None}, "accel": self.TPU}
        e = train_pc.estimate_lora("Qwen/Qwen2.5-3B-Instruct", 8, 512, 2, profile=p)
        self.assertIs(e["fits"]["tpu"], True)
        self.assertEqual(e["recommended_device"], "tpu")
        # TPU에 안 들어가면 (device를 직접 고르지 않았으므로) 예전처럼 CPU 판단으로
        e = train_pc.estimate_lora("Qwen/Qwen2.5-7B-Instruct", 8, 4096, 8, profile=p)
        self.assertIs(e["fits"]["tpu"], False)
        self.assertNotEqual(e["recommended_device"], "tpu")

    def test_xpu_profile(self):
        p = {"ram_gb": 16.0, "gpu": {"type": "xpu", "vram_gb": 16.0}}
        e = train_pc.estimate_lora("Qwen/Qwen2.5-0.5B-Instruct", 8, 512, 2, profile=p)
        self.assertIs(e["fits"]["gpu"], True)
        self.assertEqual(e["recommended_device"], "xpu")


# ---------------------------------------------------------------------------
# 메모리 계획·정밀도 결정
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH, "torch 필요")
class PlanTest(unittest.TestCase):
    def test_tpu_plan_bf16_and_checkpointing_by_hbm(self):
        with mock.patch.object(accel, "mem_total_gb", return_value=16.0) as mem:
            dt, ck, notes = train_pc._plan("tpu", "auto", {**EST, "vram_gb_est": 20.0}, True, tdev="xla-dev")
            self.assertEqual((dt, ck), ("bfloat16", True))
            self.assertTrue(any("체크포인팅" in n for n in notes))
            mem.assert_called_with("xla-dev")  # torch 장치로 메모리를 읽는다
            self.assertEqual(train_pc._plan("tpu", "auto", {**EST, "vram_gb_est": 4.0}, True), ("bfloat16", False, []))
            # 직접 준 dtype이 우선 (예전과 같음)
            self.assertEqual(train_pc._plan("tpu", "float32", {**EST, "vram_gb_est": 4.0}, True)[0], "float32")
        with mock.patch.object(accel, "mem_total_gb", return_value=None):
            self.assertEqual(train_pc._plan("tpu", "auto", {**EST, "vram_gb_est": 99.0}, True), ("bfloat16", False, []))

    def test_xpu_plan(self):
        with mock.patch.object(accel, "native_bf16", return_value=True), \
                mock.patch.object(accel, "mem_total_gb", return_value=8.0):
            self.assertEqual(train_pc._plan("xpu", "auto", {**EST, "vram_gb_est": 2.0}, True), ("bfloat16", False, []))
            self.assertTrue(train_pc._plan("xpu", "auto", {**EST, "vram_gb_est": 7.5}, True)[1])
        with mock.patch.object(accel, "native_bf16", return_value=False), \
                mock.patch.object(accel, "mem_total_gb", return_value=None):
            self.assertEqual(train_pc._plan("xpu", "auto", EST, True)[0], "float32")
            self.assertEqual(train_pc._plan("xpu", "auto", EST, False)[0], "float16")


@unittest.skipUnless(HAS_TORCH, "torch 필요")
class PrecisionTest(unittest.TestCase):
    """_train_precision: _plan(예전 규칙) → accel.lora_precision 연결."""

    def plan(self, dtype="bfloat16"):
        # T4 + 최신 torch: 예전 _plan은 에뮬레이션 bf16을 고른다. 부를 때마다 새 notes 목록을 돌려준다.
        return mock.patch.object(train_pc, "_plan", side_effect=lambda *a, **k: (dtype, False, []))

    def test_cuda_without_native_bf16_defaults_to_fp16(self):
        with self.plan() as plan, mock.patch.object(accel, "native_bf16", return_value=False):
            dt, amp, ck, notes = train_pc._train_precision("cuda", "auto", EST, "auto")
            self.assertEqual((dt, amp, ck), ("float16", True, False))
            self.assertTrue(any("LEARNBOX_AMP=off" in n for n in notes))
            self.assertEqual(plan.call_args.args[1], "auto")
            # off = 예전 방식(_plan 결과 그대로), fp32 = float32, fp16 = fp16 혼합 정밀도
            self.assertEqual(train_pc._train_precision("cuda", "auto", EST, "off"), ("bfloat16", False, False, []))
            self.assertEqual(train_pc._train_precision("cuda", "auto", EST, "fp32")[:2], ("float32", False))
            self.assertEqual(train_pc._train_precision("cuda", "auto", EST, "fp16")[:2], ("float16", True))
            # 직접 준 dtype이 우선, float16이면 혼합 정밀도
            self.assertEqual(train_pc._train_precision("cuda", "float16", EST, "auto")[:2], ("float16", True))
            self.assertEqual(train_pc._train_precision("cuda", "fp32", EST, "fp16")[:2], ("float32", False))
        with self.plan(), mock.patch.object(accel, "native_bf16", return_value=True):
            self.assertEqual(train_pc._train_precision("cuda", "auto", EST, "auto"), ("bfloat16", False, False, []))

    def test_resume_same_device_keeps_stored_dtype(self):
        with self.plan("float16") as plan, mock.patch.object(accel, "native_bf16", return_value=True):
            got = train_pc._train_precision("cuda", "float16", EST, "auto", ck_meta={"device": "cuda", "dtype": "float16"})
            self.assertEqual(got, ("float16", True, False, []))
            self.assertEqual(plan.call_args.args[1], "float16")
            # 장치 기록이 없는 예전 체크포인트도 같은 장치로 본다
            self.assertEqual(train_pc._train_precision("cuda", "float16", EST, "auto", ck_meta={})[:2], ("float16", True))

    def test_resume_fp16_with_amp_off_replans(self):
        with self.plan("bfloat16") as plan, mock.patch.object(accel, "native_bf16", return_value=False):
            dt, amp, _, notes = train_pc._train_precision("cuda", "float16", EST, "off", ck_meta={"device": "cuda"})
            self.assertEqual((dt, amp), ("bfloat16", False))
            self.assertEqual(plan.call_args.args[1], "auto")
            self.assertTrue(any("정밀도 설정(off)" in n and "float16→bfloat16" in n for n in notes), notes)
            dt, amp, _, _ = train_pc._train_precision("cuda", "float16", EST, "fp32", ck_meta={"device": "cuda"})
            self.assertEqual((dt, amp), ("float32", False))

    def test_resume_on_other_device_replans(self):
        with self.plan("bfloat16") as plan, mock.patch.object(accel, "native_bf16", return_value=False):
            dt, amp, _, notes = train_pc._train_precision("cuda", "bfloat16", EST, "auto", ck_meta={"device": "tpu"})
            self.assertEqual((dt, amp), ("float16", True))  # TPU bf16 → T4: fp16 혼합 정밀도
            self.assertEqual(plan.call_args.args[1], "auto")
            self.assertTrue(any("장치가 바뀌어" in n and "bfloat16→float16" in n for n in notes), notes)
            dt, amp, _, notes = train_pc._train_precision("tpu", "float16", EST, "auto", ck_meta={"device": "cuda"})
            self.assertEqual((dt, amp), ("bfloat16", False))  # T4 fp16 → TPU: bf16
            self.assertTrue(any("float16→bfloat16" in n for n in notes), notes)

    def test_tpu_policy(self):
        with mock.patch.object(accel, "mem_total_gb", return_value=None):
            self.assertEqual(train_pc._train_precision("tpu", "auto", EST, "auto"), ("bfloat16", False, False, []))
            dt, amp, _, notes = train_pc._train_precision("tpu", "auto", EST, "fp16")
            self.assertEqual((dt, amp), ("bfloat16", False))
            self.assertTrue(notes)
            self.assertEqual(train_pc._train_precision("tpu", "float16", EST, "auto")[:2], ("bfloat16", False))
            self.assertEqual(train_pc._train_precision("tpu", "auto", EST, "fp32")[:2], ("float32", False))

    def test_cpu_unchanged_by_modes(self):
        with mock.patch.object(train_pc, "_mem_available_gb", return_value=64.0):
            for mode in ("auto", "fp16", "off", "fp32"):
                self.assertEqual(train_pc._train_precision("cpu", "auto", EST, mode), ("float32", False, False, []))
                self.assertEqual(train_pc._train_precision("mps", "auto", EST, mode), ("float32", False, False, []))
            # CPU → CPU 재개: 새 안내 문구 없음
            self.assertEqual(train_pc._train_precision("cpu", "float32", EST, "auto", ck_meta={"device": "cpu"}),
                             ("float32", False, False, []))
            # GPU fp16 → CPU: 저장된 fp16을 쓰지 않고 다시 고른다
            dt, amp, _, notes = train_pc._train_precision("cpu", "float16", EST, "auto", ck_meta={"device": "cuda:0"})
            self.assertEqual((dt, amp), ("float32", False))
            self.assertTrue(any("float16→float32" in n for n in notes), notes)
        self.assertFalse(train_pc._ckpt_moved({"device": "cuda:1"}, "cuda"))
        self.assertTrue(train_pc._ckpt_moved({"device": "xla:0"}, "cpu"))
        self.assertFalse(train_pc._ckpt_moved({"device": "xla:0"}, "tpu"))
        self.assertFalse(train_pc._ckpt_moved(None, "cpu"))

    def test_cpu_mps_resume_fp16_ignores_amp_mode(self):
        # cpu/mps에서 float16은 fp16 혼합 정밀도가 아니다: off/fp32로 재개해도 저장된 dtype 그대로, 새 안내 없음(예전과 같음)
        with mock.patch.object(train_pc, "_mem_available_gb", return_value=64.0):
            for dev, mode in (("cpu", "off"), ("cpu", "fp32"), ("mps", "off"), ("mps", "fp32")):
                got = train_pc._train_precision(dev, "float16", EST, mode, ck_meta={"device": dev})
                self.assertEqual(got, ("float16", False, False, []), (dev, mode))
                self.assertEqual(got, train_pc._train_precision(dev, "float16", EST, mode))  # 새로 시작할 때와 같다


# ---------------------------------------------------------------------------
# 고정 모양 패딩 (_collate pad_to)
# ---------------------------------------------------------------------------
@unittest.skipUnless(HAS_TORCH, "torch 필요")
class CollateTest(unittest.TestCase):
    def examples(self):
        import torch

        a = torch.arange(10, 15)                       # 블록 (labels=None → ids)
        b = torch.arange(20, 23)
        lab = torch.tensor([-100, -100, 22])           # Q&A: 질문 부분 -100
        return [(a, None), (b, lab)]

    def test_shapes_and_mask(self):
        import torch

        ex = self.examples()
        x0, a0, y0 = train_pc._collate(torch, ex, 7, "cpu")
        self.assertEqual(tuple(x0.shape), (2, 5))
        x, att, y = train_pc._collate(torch, ex, 7, torch.device("cpu"), pad_to=SEQ)
        self.assertEqual(tuple(x.shape), (2, SEQ))
        self.assertTrue(torch.equal(x[:, :5], x0) and torch.equal(att[:, :5], a0) and torch.equal(y[:, :5], y0))
        self.assertTrue(bool((x[:, 5:] == 7).all()))
        self.assertTrue(bool((att[:, 5:] == 0).all()))
        self.assertTrue(bool((y[:, 5:] == -100).all()))
        self.assertEqual(y[1, :3].tolist(), [-100, -100, 22])
        self.assertEqual(y[0, :5].tolist(), list(range(10, 15)))
        # pad_to가 더 짧으면 자르지 않는다
        self.assertEqual(tuple(train_pc._collate(torch, ex, 7, "cpu", pad_to=3)[0].shape), (2, 5))

    def test_padding_does_not_change_loss(self):
        _need_tiny_model()
        import torch
        from transformers import AutoModelForCausalLM

        torch.manual_seed(0)
        model = train_pc._from_pretrained(AutoModelForCausalLM, TINY).eval()
        g = torch.Generator().manual_seed(1)
        V = model.config.vocab_size
        blk = (torch.randint(0, V, (20,), generator=g), None)
        qa_ids = torch.randint(0, V, (13,), generator=g)
        qa = (qa_ids, torch.cat([torch.full((6,), -100), qa_ids[6:]]))
        with torch.no_grad():
            for batch in ([blk], [qa], [blk, qa]):
                x, att, y = train_pc._collate(torch, batch, 0, "cpu")
                xp, attp, yp = train_pc._collate(torch, batch, 0, "cpu", pad_to=SEQ)
                l0 = model(input_ids=x, attention_mask=att, labels=y).loss
                l1 = model(input_ids=xp, attention_mask=attp, labels=yp).loss
                self.assertAlmostEqual(float(l0), float(l1), places=5)


# ---------------------------------------------------------------------------
# 어댑터 저장 (TPU: CPU 사본 state_dict)
# ---------------------------------------------------------------------------
class SaveAdapterTest(unittest.TestCase):
    def test_tpu_save_matches_plain_save(self):
        _need_tiny_model()
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        torch.manual_seed(0)
        base = train_pc._from_pretrained(AutoModelForCausalLM, TINY)
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules="all-linear", task_type="CAUSAL_LM"))
        with torch.no_grad():  # B가 0이 아니게 (저장 값 비교가 의미 있도록)
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.add_(0.01)
        tmp = Path(tempfile.mkdtemp(prefix="lb-tpu-save-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        train_pc._save_adapter(model, tmp / "cpu", "cpu")
        train_pc._save_adapter(model, tmp / "tpu", "tpu")
        a, b = _adapter(tmp / "cpu"), _adapter(tmp / "tpu")
        self.assertTrue(a and set(a) == set(b))
        self.assertTrue(all("lora_" in k for k in b))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)
        self.assertTrue((tmp / "tpu" / "adapter_config.json").is_file())


@contextlib.contextmanager
def _xla_like_move():
    """XLA 장치로 옮길 때처럼 Module._apply가 set_data 대신 모듈마다 새 Parameter를 만들게 한다(CPU에서 흉내)."""
    import torch

    old = torch.__future__.get_overwrite_module_params_on_conversion()
    torch.__future__.set_overwrite_module_params_on_conversion(True)
    try:
        yield
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(old)


class RetieTest(_Data, unittest.TestCase):
    """TPU로 옮기면 풀리는 묶인 임베딩(tie_word_embeddings)을 다시 묶는다. 다른 장치는 model.to만(예전 그대로)."""

    @classmethod
    def setUpClass(cls):
        _need_tiny_model()
        cls.make_data("lb-tpu-lora-tie-")

    @classmethod
    def tearDownClass(cls):
        cls.drop_data()

    @staticmethod
    def count(model) -> int:
        return sum(p.numel() for p in model.parameters())

    def test_move_reties_only_on_tpu(self):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        cpu = torch.device("cpu")
        base = train_pc._from_pretrained(AutoModelForCausalLM, TINY)
        self.assertTrue(train_pc._embeddings_tied(base))
        n0 = self.count(base)
        with _xla_like_move():
            plain = train_pc._from_pretrained(AutoModelForCausalLM, TINY)
            train_pc._to_train_device(plain, "cpu", cpu)  # 흉내가 실제로 묶음을 푸는지(= TPU에서 일어나는 일)
            self.assertFalse(train_pc._embeddings_tied(plain))
            self.assertGreater(self.count(plain), n0)
            train_pc._to_train_device(base, "tpu", cpu)
            self.assertTrue(train_pc._embeddings_tied(base))
            self.assertEqual(self.count(base), n0)
            model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules="all-linear",
                                                    task_type="CAUSAL_LM"))
            train_pc._to_train_device(model, "tpu", cpu)  # PEFT를 씌운 뒤 두 번째 이동
            self.assertTrue(train_pc._embeddings_tied(model))
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.assertEqual(self.count(model), n0 + trainable)

    def test_fake_tpu_training_keeps_one_embedding(self):
        import torch
        from transformers import AutoModelForCausalLM

        n0 = self.count(train_pc._from_pretrained(AutoModelForCausalLM, TINY))
        kw = dict(max_steps=1, lr=1e-3, seq_len=SEQ, batch_size=2, grad_accum=1, lora_r=4, lora_alpha=8, threads=1,
                  seed=6)
        with _xla_like_move(), \
                mock.patch.object(accel, "pick", return_value=("tpu", torch.device("cpu"), None)), \
                mock.patch.object(accel, "mem_total_gb", return_value=16.0):
            info = train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "tpu", TINY, **kw)
        self.assertEqual(info["device"], "tpu")
        self.assertEqual(info["total_params"], n0 + info["trainable_params"])


class TorchTpuDropoutTest(_Data, unittest.TestCase):
    """TorchTPU 알려진 문제(#6966): 체크포인팅 + dropout이면 기울기가 틀리므로 dropout을 끈다. 다른 경우는 그대로."""

    @classmethod
    def setUpClass(cls):
        _need_tiny_model()
        cls.make_data("lb-tpu-lora-drop-")

    @classmethod
    def tearDownClass(cls):
        cls.drop_data()

    def test_disable_dropout_helper(self):
        import torch

        m = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(2, 2), torch.nn.Dropout(0.0))
        m[1].attention_dropout = 0.2
        self.assertEqual(train_pc._disable_dropout(m), 2)
        self.assertEqual(m[0].p, 0.0)
        self.assertEqual(m[1].attention_dropout, 0.0)
        self.assertEqual(train_pc._disable_dropout(m), 0)

    def run_fake_tpu(self, backend, ckpt):
        import torch

        kw = dict(max_steps=1, lr=1e-3, seq_len=SEQ, batch_size=2, grad_accum=1, lora_r=4, lora_alpha=8, threads=1,
                  seed=6, gradient_checkpointing=ckpt)
        seen = {}
        real = train_pc._disable_dropout

        def spy(model):
            n = real(model)
            seen["n"] = n
            seen["ps"] = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
            return n

        with mock.patch.object(accel, "pick", return_value=("tpu", torch.device("cpu"), None)), \
                mock.patch.object(accel, "tpu_backend", return_value=backend), \
                mock.patch.object(accel, "step_end"), \
                mock.patch.object(accel, "mem_total_gb", return_value=16.0), \
                mock.patch.object(train_pc, "_disable_dropout", side_effect=spy):
            info = train_pc.train_lora(str(self.train), str(self.eval), self.tmp / f"{backend}-{ckpt}", TINY, **kw)
        return info, seen

    def test_torch_tpu_with_checkpointing_disables_dropout(self):
        info, seen = self.run_fake_tpu("torch_tpu", True)
        self.assertGreater(seen["n"], 0)
        self.assertTrue(all(p == 0.0 for p in seen["ps"]))
        self.assertTrue(any("dropout을 끄고" in n for n in info["notes"]))

    def test_other_cases_keep_dropout(self):
        for backend, ckpt in (("torch_tpu", False), ("torch_xla", True)):
            info, seen = self.run_fake_tpu(backend, ckpt)
            self.assertNotIn("n", seen, (backend, ckpt))
            self.assertFalse(any("dropout을 끄고" in n for n in info["notes"]))


# ---------------------------------------------------------------------------
# 실제 CPU 학습: 정밀도 모드·장치 지정은 CPU 결과를 바꾸지 않는다, 체크포인트 키
# ---------------------------------------------------------------------------
class CpuTrainTest(_Data, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _need_tiny_model()
        cls.make_data("lb-tpu-lora-cpu-")
        cls.kw = dict(max_steps=4, lr=1e-3, seq_len=SEQ, batch_size=2, grad_accum=2, lora_r=4, lora_alpha=8,
                      qa_path=str(cls.qa), threads=1, seed=5)
        cls.ck = cls.tmp / "ck"
        cls.base_out = cls.tmp / "base"
        cls.base = train_pc.train_lora(str(cls.train), str(cls.eval), cls.base_out, TINY, checkpoint_dir=str(cls.ck),
                                       checkpoint_every_steps=2, checkpoint_every_sec=0, **cls.kw)

    @classmethod
    def tearDownClass(cls):
        cls.drop_data()

    def assert_same(self, out: Path, info: dict) -> None:
        import torch

        for k in ("loss_history", "eval_loss_before", "eval_loss_after", "dtype", "device", "train_loss"):
            self.assertEqual(info[k], self.base[k], k)
        a, b = _adapter(self.base_out), _adapter(out)
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)  # CPU는 비트 단위로 같아야 한다

    def test_base_info_and_checkpoint_keys(self):
        import torch

        self.assertEqual((self.base["device"], self.base["dtype"]), ("cpu", "float32"))
        self.assertIs(self.base["amp"], False)
        self.assertEqual(self.base["notes"], [])  # CPU에서는 새 안내 문구 없음
        self.assertNotIn("ignored_params", self.base)
        meta, d = checkpoint.load(self.ck)
        self.assertEqual((meta["step"], meta["device"], meta["amp"], meta["dtype"]), (4, "cpu", False, "float32"))
        tr = torch.load(d / "trainer.pt", map_location="cpu", weights_only=True)
        self.assertIn("scaler", tr)
        self.assertIsNone(tr["scaler"])  # 혼합 정밀도가 아니면 None
        self.assertIsNone(tr["cuda_rng"])
        saved = json.loads((self.base_out / "info.json").read_text(encoding="utf-8"))
        self.assertIs(saved["amp"], False)

    def test_amp_modes_and_device_env_do_not_change_cpu_results(self):
        runs = [("amp-fp16", {"amp": "fp16"}, {}),
                ("env-fp16", {}, {"LEARNBOX_AMP": "fp16"}),
                ("amp-off", {"amp": "off"}, {"LEARNBOX_DEVICE": "cpu"}),
                ("amp-fp32", {"amp": "fp32", "device": "cpu"}, {"LEARNBOX_AMP": "1"})]
        for name, extra, env in runs:
            with self.subTest(name), mock.patch.dict(os.environ, env):
                out = self.tmp / name
                info = train_pc.train_lora(str(self.train), str(self.eval), out, TINY, **extra, **self.kw)
                self.assertNotIn("ignored_params", info)
                self.assertEqual(info["notes"], [])
                self.assertIs(info["amp"], False)
                self.assert_same(out, info)

    def test_cpu_resume_matches_and_adds_no_device_notes(self):
        # 2단계 체크포인트를 새로 만든다 (stop_requested → 저장 후 끝내기)
        root2 = self.tmp / "ck2"
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] >= 2

        train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "stop2", TINY, checkpoint_dir=str(root2),
                            checkpoint_every_sec=0, stop_requested=stop, **self.kw)
        self.assertEqual(checkpoint.summary(root2)["step"], 2)
        out = self.tmp / "resumed"
        same = {k: None for k in ("max_steps", "lr", "seq_len", "batch_size", "grad_accum", "lora_r", "lora_alpha")}
        info = train_pc.train_lora(None, None, out, "", resume_checkpoint=str(root2), threads=1, seed=5, **same)
        # CPU → CPU 재개: 장치·정밀도 안내 없이 예전과 같은 문구만
        self.assertEqual(info["notes"], ["2/4단계 체크포인트에서 이어서 학습했습니다."])
        a, b = _adapter(self.base_out), _adapter(out)
        self.assertLessEqual(max(float((a[k] - b[k]).abs().max()) for k in a), 1e-5)
        self.assertEqual(info["loss_history"][-2:], self.base["loss_history"][-2:])
        self.assertEqual(info["eval_loss_before"], self.base["eval_loss_before"])

    @unittest.skipIf(HAS_XLA, "torch_xla가 설치된 환경")
    def test_unavailable_tpu_falls_back_to_cpu_with_note(self):
        out = self.tmp / "want-tpu"
        info = train_pc.train_lora(str(self.train), str(self.eval), out, TINY, device="tpu", **self.kw)
        self.assertEqual(info["device"], "cpu")
        self.assertEqual(len(info["notes"]), 1)
        self.assertIn("torch_xla", info["notes"][0])
        self.assertEqual(info["loss_history"], self.base["loss_history"])  # 학습 결과는 같다


# ---------------------------------------------------------------------------
# TPU 고정 모양 패딩을 CPU에서: 평가 손실 가중치는 실제 토큰 수 기준
# ---------------------------------------------------------------------------
class PaddingPathTest(_Data, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = _need_tiny_model()
        cls.make_data("lb-tpu-lora-pad-")
        # 평가 자료: 마지막 블록이 seq_len보다 짧은(8 이상) 길이를 고른다 → TPU에서는 그 블록이 패딩된다
        cls.n_eval_tokens = None
        for j in range(1, 40):
            text = " ".join(TEXTS[i % len(TEXTS)] for i in range(j))
            n = len(tok(text, add_special_tokens=False)["input_ids"]) + 1  # + eos
            if n < 8 * SEQ and n % SEQ >= 8:
                _write_jsonl(cls.eval, [{"text": text}])
                cls.n_eval_tokens = n
                break
        if cls.n_eval_tokens is None:
            raise unittest.SkipTest("짧은 마지막 블록이 있는 평가 자료를 만들지 못함")
        cls.kw = dict(max_steps=2, lr=1e-3, seq_len=SEQ, batch_size=2, grad_accum=2, lora_r=4, lora_alpha=8,
                      qa_path=str(cls.qa), threads=1, seed=2)

    @classmethod
    def tearDownClass(cls):
        cls.drop_data()

    def test_padded_eval_equals_unpadded(self):
        plain = train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "plain", TINY, **self.kw)
        shapes = []
        real = train_pc._collate

        def spy(torch, examples, pad_id, device, pad_to=None):
            out = real(torch, examples, pad_id, device, pad_to)
            shapes.append((tuple(out[0].shape), max(int(e[0].shape[0]) for e in examples)))
            return out

        with mock.patch.object(accel, "fixed_shapes", return_value=True), \
                mock.patch.object(train_pc, "_collate", side_effect=spy):
            padded = train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "padded", TINY, **self.kw)
        self.assertTrue(shapes)
        self.assertTrue(all(s[1] == SEQ for s, _ in shapes), shapes)  # 학습·평가 모두 [B, seq_len]
        self.assertTrue(any(n < SEQ for _, n in shapes))  # 실제로 짧은 예시(꼬리 블록·Q&A)를 패딩했다
        self.assertEqual(padded["eval_blocks"], plain["eval_blocks"])
        self.assertEqual(plain["eval_blocks"], self.n_eval_tokens // SEQ + 1)
        # info는 소수 4자리로 반올림되어 있으므로 반올림 경계 차이(1e-4)까지 허용
        self.assertAlmostEqual(padded["eval_loss_before"], plain["eval_loss_before"], delta=2e-4)


# ---------------------------------------------------------------------------
# TPU 경로 전체를 CPU에서 (accel.pick → ("tpu", cpu 장치))
# ---------------------------------------------------------------------------
class FakeTpuTest(_Data, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _need_tiny_model()
        import torch

        cls.make_data("lb-tpu-lora-fake-")
        cls.kw = dict(max_steps=4, lr=1e-3, seq_len=SEQ, batch_size=2, grad_accum=2, lora_r=4, lora_alpha=8,
                      qa_path=str(cls.qa), threads=1, seed=4)
        cls.events, cls.shapes = [], []
        real = train_pc._collate

        def spy(torch_, examples, pad_id, device, pad_to=None):
            out = real(torch_, examples, pad_id, device, pad_to)
            cls.shapes.append(tuple(out[0].shape))
            return out

        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] >= 3

        cls.ck = cls.tmp / "ck"
        cls.out = cls.tmp / "tpu"
        counts = iter([5, 7])
        with mock.patch.object(accel, "pick", return_value=("tpu", torch.device("cpu"), None)), \
                mock.patch.object(accel, "compile_count", side_effect=lambda: next(counts, 7)), \
                mock.patch.object(accel, "step_end", wraps=accel.step_end) as step_end, \
                mock.patch.object(accel, "mem_total_gb", return_value=16.0), \
                mock.patch.object(train_pc, "_collate", side_effect=spy), \
                mock.patch.dict(os.environ, {"LEARNBOX_XLA_DEBUG": "1"}):
            cls.info = train_pc.train_lora(str(cls.train), str(cls.eval), cls.out, TINY, checkpoint_dir=str(cls.ck),
                                           checkpoint_every_steps=2, checkpoint_every_sec=0, stop_requested=stop,
                                           progress=lambda p, m: cls.events.append((p, m)), **cls.kw)
            cls.step_end_calls = step_end.call_count

    @classmethod
    def tearDownClass(cls):
        cls.drop_data()

    def test_info_and_progress(self):
        info = self.info
        self.assertEqual((info["device"], info["dtype"], info["amp"]), ("tpu", "bfloat16", False))
        self.assertEqual(info["steps"], 3)
        self.assertTrue(info["stopped_early"])
        msgs = [m for _, m in self.events]
        self.assertTrue(any(m.startswith("LoRA 학습 1/4 · TPU · loss") for m in msgs), msgs)
        self.assertTrue(any(m == f"학습 전 평가 중 · {train_pc.TPU_COMPILE_MSG}" for m in msgs), msgs)
        ps = [p for p, _ in self.events]
        self.assertEqual(ps, sorted(ps))
        self.assertAlmostEqual(ps[-1], 1.0)
        self.assertTrue(any("TPU 재컴파일이 3단계 이후 2번" in n for n in info["notes"]), info["notes"])

    def test_fixed_shapes_and_sync(self):
        self.assertTrue(self.shapes)
        self.assertTrue(all(s[1] == SEQ for s in self.shapes), set(self.shapes))
        # 단계마다 1번 + 저장(2단계, 3단계 끝내기) 때마다 + 최종 저장 1번
        self.assertGreaterEqual(self.step_end_calls, 3 + 2 + 1)

    def test_checkpoint_is_cpu_and_resumes_on_cpu(self):
        import torch

        meta, d = checkpoint.load(self.ck)
        self.assertEqual((meta["step"], meta["device"], meta["dtype"], meta["amp"]), (3, "tpu", "bfloat16", False))
        self.assertEqual(meta["n_eval"], 16)
        tr = torch.load(d / "trainer.pt", map_location="cpu", weights_only=True)
        self.assertIsNone(tr["scaler"])
        self.assertIsNone(tr["cuda_rng"])
        for st in tr["opt"]["state"].values():
            for v in st.values():
                if torch.is_tensor(v):
                    self.assertEqual(v.device.type, "cpu")
        ck_w, out_w = _adapter(d / "adapter"), _adapter(self.out)
        self.assertTrue(ck_w and all("lora_" in k for k in ck_w))
        self.assertEqual(set(ck_w), set(out_w))
        for k in ck_w:  # 3단계 끝내기 저장본 == 최종 어댑터
            self.assertTrue(torch.equal(ck_w[k], out_w[k]), k)
        # TPU에서 저장 → CPU에서 재개: 다른 장치 안내 + 정밀도 다시 고르기
        info = train_pc.train_lora(None, None, self.tmp / "on-cpu", "", resume_checkpoint=str(self.ck), threads=1)
        self.assertEqual((info["device"], info["dtype"], info["steps"]), ("cpu", "float32", 4))
        notes = "\n".join(info["notes"])
        self.assertIn("저장했던 장치(tpu)와 다른 장치(cpu)에서 이어서 학습합니다.", notes)
        self.assertIn("bfloat16→float32", notes)
        self.assertIn("3/4단계 체크포인트에서 이어서", notes)


# ---------------------------------------------------------------------------
# fp16 혼합 정밀도 경로 (켜진 CPU GradScaler로 대신 확인)
# ---------------------------------------------------------------------------
class ScalerPathTest(_Data, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _need_tiny_model()
        import torch

        try:
            torch.amp.GradScaler("cpu", init_scale=8.0)
        except Exception as e:  # torch < 2.3
            raise unittest.SkipTest(f"CPU GradScaler 없음: {e}") from e
        cls.make_data("lb-tpu-lora-amp-")
        cls.kw = dict(lr=1e-3, seq_len=16, batch_size=1, grad_accum=2, lora_r=2, lora_alpha=4, threads=1, seed=1)

    @classmethod
    def tearDownClass(cls):
        cls.drop_data()

    def amp_on(self, init_scale: float, made: list):
        """use_amp=True로 만들고, 켜진 CPU 스케일러를 쓴다 (CUDA fp16 autocast는 CPU에서 흉내 내지 않음)."""
        import torch

        def make(dev, enabled):
            s = torch.amp.GradScaler("cpu", init_scale=init_scale, enabled=bool(enabled))
            made.append(s)
            return s

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(train_pc, "_train_precision",
                                              side_effect=lambda *a, **k: ("float32", True, False, [])))
        stack.enter_context(mock.patch.object(accel, "make_scaler", side_effect=make))
        stack.enter_context(mock.patch.object(accel, "autocast", side_effect=lambda d, dt: contextlib.nullcontext()))
        return stack

    @staticmethod
    def nonfinite_losses():
        """손실(0차원 텐서)만 nan/inf로 보이게 한다."""
        import torch

        real = torch.isfinite

        def fake(x):
            r = real(x)
            return torch.zeros_like(r) if r.dim() == 0 else r
        return mock.patch.object(torch, "isfinite", side_effect=fake)

    def test_scaler_state_saved_and_restored(self):
        import torch

        root = self.tmp / "ck-scale"
        made: list = []
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return calls["n"] >= 2

        with self.amp_on(1024.0, made):
            info = train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "amp1", TINY, max_steps=4,
                                       checkpoint_dir=str(root), checkpoint_every_sec=0, stop_requested=stop, **self.kw)
        self.assertIs(info["amp"], True)
        meta, d = checkpoint.load(root)
        self.assertEqual((meta["step"], meta["amp"]), (2, True))
        tr = torch.load(d / "trainer.pt", map_location="cpu", weights_only=True)
        self.assertEqual(tr["scaler"]["scale"], 1024.0)
        tr["scaler"]["scale"] = 128.0  # 줄어든 스케일에서 이어서 하는지 보려고 바꿔 둔다
        torch.save(tr, d / "trainer.pt")
        made.clear()
        with self.amp_on(65536.0, made):
            info = train_pc.train_lora(None, None, self.tmp / "amp2", "", resume_checkpoint=str(root), threads=1)
        self.assertEqual(info["steps"], 4)
        self.assertEqual(made[-1].get_scale(), 128.0)

    def test_all_nonfinite_steps_are_skipped_then_stop(self):
        root = self.tmp / "ck-nan"
        events = []
        with self.amp_on(1024.0, []), self.nonfinite_losses():
            with self.assertRaises(ValueError) as cm:  # 예전에는 GradScaler AssertionError로 죽었다
                train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "nan", TINY, max_steps=30,
                                    checkpoint_dir=str(root), checkpoint_every_steps=100, checkpoint_every_sec=0,
                                    progress=lambda p, m: events.append(m), **self.kw)
        msg = str(cm.exception)
        self.assertIn(f"{train_pc.NAN_STREAK_LIMIT}단계 연속 nan/inf", msg)
        self.assertIn("LEARNBOX_AMP=off", msg)
        self.assertTrue(any(m.startswith(f"LoRA 학습 {train_pc.NAN_STREAK_LIMIT}/30") for m in events))
        self.assertFalse(any(m.startswith(f"LoRA 학습 {train_pc.NAN_STREAK_LIMIT + 1}/30") for m in events))
        meta, _ = checkpoint.load(root)  # 멈추기 전에 저장 → 이어서 할 수 있다
        self.assertEqual((meta["step"], meta["amp"]), (train_pc.NAN_STREAK_LIMIT, True))
        self.assertFalse((self.tmp / "nan" / "info.json").exists())

    def test_without_amp_nonfinite_keeps_old_behaviour(self):
        with self.nonfinite_losses():  # 혼합 정밀도가 꺼져 있으면(CPU) 예전처럼 끝까지 (멈추지 않음)
            info = train_pc.train_lora(str(self.train), str(self.eval), self.tmp / "nan-off", TINY, max_steps=3,
                                       **self.kw)
        self.assertEqual(info["steps"], 3)
        self.assertEqual(info["loss_history"], [])
        self.assertIs(info["amp"], False)


if __name__ == "__main__":
    unittest.main()
