"""LoRA 모델: 공개 소형 언어모델(Qwen2.5 0.5B~7B 등)에 학습 파일로 LoRA 추가 학습 + 검색 근거 답변(RAG).

- 학습: 말뭉치를 seq_len 토큰 블록으로 묶어 이어 학습(continued pretraining)하고, Q&A 파일이 있으면
  대화 형식 예시를 섞는다. 기존 어댑터 폴더(resume_from)에서 이어 학습할 수 있다(total_steps 누적).
- PC 성능 차이 고려: CUDA > Apple MPS > CPU 자동 선택, 메모리가 모자라면 bf16 / 활성값 체크포인팅을
  자동으로 켜고, CPU 스레드 수를 지정할 수 있다. estimate_lora()로 필요한 메모리를 미리 계산한다.
- 기반 모델은 먼저 로컬 캐시에서 찾고(인터넷 없이 동작), 없을 때만 내려받는다.
- 무거운 라이브러리(torch, transformers, peft)는 함수 안에서만 불러온다.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import threading
import time
from array import array
from datetime import datetime, timezone
from pathlib import Path

GB = 1024 ** 3
MB = 1024 ** 2

BASE_MODELS = [  # UI에 표시. 사용자는 다른 Hugging Face 모델 id를 직접 입력할 수도 있다.
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "size": "0.5B", "params_b": 0.5, "min_ram_gb": 4, "min_vram_gb": 2,
     "license": "Apache-2.0"},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "size": "1.5B", "params_b": 1.5, "min_ram_gb": 8, "min_vram_gb": 4,
     "license": "Apache-2.0"},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "size": "3B", "params_b": 3.0, "min_ram_gb": 14, "min_vram_gb": 8,
     "license": "Qwen Research (비상업)"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "size": "7B", "params_b": 7.0, "min_ram_gb": 30, "min_vram_gb": 16,
     "license": "Apache-2.0"},
]

# 알려진 모델 구조 (LoRA 파라미터 수·활성값 메모리 추정용).
# d=hidden, L=layers, heads/kv_heads/head_dim, i=intermediate, V=vocab, gated=SwiGLU MLP 여부
_ARCH = {
    "qwen/qwen2.5-0.5b": {"d": 896, "L": 24, "heads": 14, "kv_heads": 2, "head_dim": 64, "i": 4864,
                          "V": 151936, "gated": True},
    "qwen/qwen2.5-1.5b": {"d": 1536, "L": 28, "heads": 12, "kv_heads": 2, "head_dim": 128, "i": 8960,
                          "V": 151936, "gated": True},
    "qwen/qwen2.5-3b": {"d": 2048, "L": 36, "heads": 16, "kv_heads": 2, "head_dim": 128, "i": 11008,
                        "V": 151936, "gated": True},
    "qwen/qwen2.5-7b": {"d": 3584, "L": 28, "heads": 28, "kv_heads": 4, "head_dim": 128, "i": 18944,
                        "V": 152064, "gated": True},
}
_GATED_TYPES = {"llama", "qwen2", "qwen3", "qwen2_moe", "mistral", "mixtral", "gemma", "gemma2", "gemma3",
                "gemma3_text", "phi3", "olmo", "olmo2", "exaone", "granite", "cohere", "internlm2", "smollm3"}

SYSTEM_PROMPT = "당신은 사용자가 학습시킨 자료를 근거로 한국어로 정확하게 답하는 도우미입니다."
ANSWER_INSTRUCTION = (
    "아래 자료만 근거로 질문에 한국어로 답하세요. 근거가 된 자료 번호를 [1]처럼 표시하고, "
    "자료에 없으면 모른다고 답하세요."
)
NO_CONTEXT_ANSWER = "관련 자료를 찾지 못했습니다. 먼저 자료를 수집하거나 파일을 가져오세요."

_CACHE: dict = {}          # 추론용 모델 캐시: 한 번에 하나만 메모리에 올린다.
_LOCK = threading.RLock()  # 추론 모델 로드/생성 직렬화


def _noop_progress(p: float, m: str) -> None:
    return None


def _noop() -> None:
    return None


# ---------------------------------------------------------------------------
# 모델 크기·메모리 추정
# ---------------------------------------------------------------------------
def _known(base_model: str) -> dict | None:
    key = (base_model or "").strip().lower()
    for m in BASE_MODELS:
        if m["id"].lower() == key:
            return m
    return None


def _arch_key(base_model: str) -> str:
    key = (base_model or "").strip().lower()
    for suffix in ("-instruct", "-chat", "-it"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def params_from_name(model_id: str) -> float | None:
    """모델 id 안의 크기 표기("0.5B", "7b", "135M", "8x7B")에서 파라미터 수(십억 단위)를 읽는다."""
    name = (model_id or "").lower()
    m = re.search(r"(?<![\w.])(\d+)x(\d+(?:\.\d+)?)b(?![a-z])", name)
    if m:
        return round(float(m.group(1)) * float(m.group(2)), 3)
    m = re.search(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([bm])(?![a-z])", name)
    if not m:
        return None
    val = float(m.group(1))
    if val <= 0:
        return None
    return round(val if m.group(2) == "b" else val / 1000.0, 4)


def _hub_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return Path(home) / "hub"


def _local_config(model_id: str) -> dict | None:
    """로컬 폴더 또는 Hugging Face 캐시에 있는 config.json (네트워크 없이, transformers 없이)."""
    if not model_id:
        return None
    cands: list[Path] = []
    try:
        p = Path(model_id).expanduser()
        if p.is_dir():
            cands.append(p / "config.json")
        elif "/" in model_id and not p.is_absolute():
            snap = _hub_cache_dir() / ("models--" + model_id.replace("/", "--")) / "snapshots"
            if snap.is_dir():
                cands.extend(sorted(snap.glob("*/config.json"), key=lambda f: f.stat().st_mtime, reverse=True))
    except OSError:
        return None
    for c in cands:
        try:
            data = json.loads(c.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            continue
    return None


def _arch_from_config(cfg: dict) -> dict | None:
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    d = tc.get("hidden_size") or tc.get("n_embd") or tc.get("d_model")
    L = tc.get("num_hidden_layers") or tc.get("n_layer") or tc.get("num_layers")
    if not isinstance(d, int) or not isinstance(L, int) or d <= 0 or L <= 0:
        return None
    heads = tc.get("num_attention_heads") or tc.get("n_head") or 1
    kv_heads = tc.get("num_key_value_heads") or heads
    head_dim = tc.get("head_dim") or max(1, d // max(1, heads))
    i = tc.get("intermediate_size") or tc.get("n_inner") or 4 * d
    V = tc.get("vocab_size") or 50000
    mtype = str(tc.get("model_type") or cfg.get("model_type") or "").lower()
    act = str(tc.get("hidden_act") or "").lower()
    gated = mtype in _GATED_TYPES or act in ("silu", "swiglu")
    arch = {"d": d, "L": L, "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "i": i, "V": V,
            "gated": gated}
    tied = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", True))
    q = heads * head_dim
    kv = kv_heads * head_dim
    per_layer = d * q + 2 * d * kv + q * d + (3 if gated else 2) * d * i
    total = V * d * (1 if tied is not False else 2) + L * per_layer
    if tc.get("n_positions"):
        total += int(tc["n_positions"]) * d
    arch["params_b"] = round(total / 1e9, 4)
    return arch


def _arch_heuristic(params_b: float) -> dict:
    """크기만 알 때의 대략적 구조 (GPT 계열 평균)."""
    d = max(256, int(round(1250 * math.sqrt(params_b) / 128.0)) * 128)
    V = 50000
    L = max(4, int(round(max(params_b * 1e9 - V * d, 12 * d * d) / (12 * d * d))))
    heads = max(1, d // 64)
    return {"d": d, "L": L, "heads": heads, "kv_heads": heads, "head_dim": 64, "i": 4 * d, "V": V,
            "gated": False, "params_b": params_b}


def _lora_params(arch: dict, r: int) -> int:
    """target_modules="all-linear"일 때 학습되는 LoRA 파라미터 수: 각 Linear(in,out)마다 r*(in+out)."""
    d, i = arch["d"], arch["i"]
    q = arch["heads"] * arch["head_dim"]
    kv = arch["kv_heads"] * arch["head_dim"]
    attn = (d + q) + 2 * (d + kv) + (q + d)
    mlp = (3 if arch["gated"] else 2) * (d + i)
    return int(r * arch["L"] * (attn + mlp))


def _resolve_size(base_model: str) -> tuple[float | None, dict | None, str | None]:
    """(params_b, arch, 출처 "list"|"config"|"name"|None)"""
    known = _known(base_model)
    cfg = None
    arch = _ARCH.get(_arch_key(base_model))
    if arch is None:
        cfg = _local_config(base_model)
        arch = _arch_from_config(cfg) if cfg else None
    if known:
        return float(known["params_b"]), arch or _arch_heuristic(float(known["params_b"])), "list"
    if arch and arch.get("params_b"):
        return float(arch["params_b"]), arch, "config"
    pb = params_from_name(base_model)
    if pb:
        return pb, arch or _arch_heuristic(pb), "name"
    return None, None, None


def estimate_lora(base_model: str, lora_r: int = 8, seq_len: int = 512, batch_size: int = 2,
                  profile: dict | None = None) -> dict:
    """LoRA 학습에 필요한 메모리·어댑터 크기 추정.

    RAM(CPU 학습, fp32) ≈ 기반 모델 params×4B + LoRA 가중치/기울기/AdamW 상태 + 활성값 + 로짓 + 1GB.
    VRAM(GPU, bf16) ≈ params×2B + LoRA/옵티마이저 + 활성값 + 로짓 + CUDA 컨텍스트.
    profile(devices.pc_profile 결과)을 주면 이 PC에서 가능한지(fits)와 권장 문구를 덧붙인다.
    """
    base_model = str(base_model or "").strip()
    r = max(1, int(lora_r or 8))
    seq = max(16, int(seq_len or 512))
    bs = max(1, int(batch_size or 1))
    params_b, arch, source = _resolve_size(base_model)
    tokens = bs * seq
    if arch is None:  # 크기를 모름: 가중치를 제외한 최소 추정치 (일반적인 1B급 구조 가정)
        a = _arch_heuristic(1.0)
        trainable = None
        lora_state = 0.0
        weights = 0.0
    else:
        a = arch
        trainable = _lora_params(arch, r)
        lora_state = trainable * 16.0  # fp32 가중치 + 기울기 + AdamW m, v
        weights = params_b * 1e9 if params_b else 0.0
    act_units = tokens * a["d"] * a["L"]
    logits = tokens * a["V"]
    ram = weights * 4 + lora_state + act_units * 68 + logits * 4 * 3 + 1.0 * GB
    ram_bf16 = weights * 2 + lora_state + act_units * 34 + logits * (2 + 4 + 4) + 1.0 * GB
    vram = weights * 2 + lora_state + act_units * 34 + logits * (2 + 4 + 4) + 0.6 * GB
    known = _known(base_model)
    out = {
        "base_model": base_model,
        "known": known is not None,
        "size": known["size"] if known else (f"{params_b:g}B" if params_b else None),
        "size_source": source,
        "base_params_b": params_b,
        "trainable_params_est": trainable,
        "ram_gb_est": round(ram / GB, 2),
        "ram_gb_est_bf16": round(ram_bf16 / GB, 2),
        "vram_gb_est": round(vram / GB, 2),
        "adapter_mb_est": round((trainable or 0) * 4 / MB, 2),
        "lora_r": r, "seq_len": seq, "batch_size": bs,
        "min_ram_gb": known["min_ram_gb"] if known else None,
        "min_vram_gb": known["min_vram_gb"] if known else None,
        "license": known["license"] if known else None,
        "note": None if params_b else (
            "모델 크기를 알 수 없어 기반 모델 가중치를 뺀 최소 추정치입니다. "
            "모델 이름에 크기(예: 1.5B)가 있거나 한 번 내려받으면 더 정확해집니다."),
    }
    if profile:
        out.update(_fit(out, profile))
    return out


def _fit(est: dict, profile: dict) -> dict:
    ram = profile.get("ram_gb")
    gpu = profile.get("gpu") or {}
    vram = gpu.get("vram_gb")
    fits_cpu = None if not ram else ram >= est["ram_gb_est"]
    fits_cpu_bf16 = None if not ram else ram >= est["ram_gb_est_bf16"]
    fits_gpu = None if not (gpu.get("type") == "cuda" and vram) else vram >= est["vram_gb_est"]
    if gpu.get("type") == "mps" and ram:
        fits_gpu = ram * 0.7 >= est["ram_gb_est"]  # Apple 통합 메모리
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


# ---------------------------------------------------------------------------
# 장치·메모리 계획
# ---------------------------------------------------------------------------
def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        pass
    try:
        from .devices import pc_profile

        ram = pc_profile().get("ram_gb")
        return float(ram) * 0.8 if ram else None
    except Exception:  # devices 모듈이 없거나 실패해도 학습은 계속
        return None


def _vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / GB
    except Exception:
        pass
    return None


_DTYPE_NAMES = {"float32": "float32", "fp32": "float32", "bfloat16": "bfloat16", "bf16": "bfloat16",
                "float16": "float16", "fp16": "float16"}


def _plan(device: str, dtype: str | None, est: dict, training: bool) -> tuple[str, bool, list[str]]:
    """(dtype 이름, 활성값 체크포인팅 여부, 안내 문구들)"""
    import torch

    notes: list[str] = []
    want = _DTYPE_NAMES.get(str(dtype or "auto").lower())
    if want is None:
        if device == "cuda":
            want = "bfloat16" if torch.cuda.is_bf16_supported() else ("float32" if training else "float16")
        elif device == "mps":
            want = "float32" if training else "float16"
        else:
            want = "float32"
            avail = _mem_available_gb()
            need = est["ram_gb_est"] if training else (est["base_params_b"] or 0) * 4 + 1.0
            need_bf16 = est["ram_gb_est_bf16"] if training else (est["base_params_b"] or 0) * 2 + 1.0
            if avail and need > avail * 0.9 and need_bf16 <= avail * 0.9:
                want = "bfloat16"
                notes.append(f"메모리 여유({avail:.1f}GB)가 부족해 bf16으로 불러옵니다.")
    checkpointing = False
    if training:
        if device == "cuda":
            vram = _vram_gb()
            checkpointing = bool(vram and est["vram_gb_est"] > vram * 0.85)
        elif device == "cpu":
            avail = _mem_available_gb()
            need = est["ram_gb_est_bf16"] if want == "bfloat16" else est["ram_gb_est"]
            checkpointing = bool(avail and need > avail * 0.75)
        if checkpointing:
            notes.append("메모리를 아끼려고 활성값 체크포인팅을 켭니다(조금 느려짐).")
    return want, checkpointing, notes


def _set_threads(threads: int | None) -> int:
    import torch

    try:
        n = int(threads or 0)
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        n = max(1, (os.cpu_count() or 2) - 1)
    torch.set_num_threads(n)
    return n


# ---------------------------------------------------------------------------
# 불러오기
# ---------------------------------------------------------------------------
def _from_pretrained(loader, name: str, on_download=None, **kw):
    """로컬 캐시 먼저(오프라인 동작), 없으면 내려받기. 실패하면 한국어 ValueError."""
    try:
        return loader.from_pretrained(name, local_files_only=True, **kw)
    except MemoryError:
        raise
    except Exception as e:  # 캐시에 없음 → 내려받기 시도
        local_err = e
    if Path(name).expanduser().exists():
        raise ValueError(f"모델 폴더를 불러오지 못했습니다({name}): {local_err}") from local_err
    if on_download:
        on_download()
    try:
        return loader.from_pretrained(name, **kw)
    except MemoryError:
        raise
    except Exception as e:
        raise ValueError(f"기반 모델을 불러오지 못했습니다({name}): {e} — 인터넷 연결과 모델 이름을 확인하세요.") from e


def _load_tokenizer(name: str, fallback: str | None = None, on_download=None):
    from transformers import AutoTokenizer

    try:
        tok = _from_pretrained(AutoTokenizer, name, on_download=on_download)
    except ValueError:
        if not fallback or fallback == name:
            raise
        tok = _from_pretrained(AutoTokenizer, fallback, on_download=on_download)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
    return tok


def _pad_id(tok) -> int:
    for v in (tok.pad_token_id, tok.eos_token_id, tok.unk_token_id):
        if v is not None:
            return int(v)
    return 0


def _max_positions(config) -> int | None:
    tc = getattr(config, "text_config", None) or config
    for attr in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
        v = getattr(tc, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return None


def _has_tokenizer(path: Path) -> bool:
    return any((path / f).exists() for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model"))


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def read_info(model_dir: str | Path) -> dict | None:
    """모델 폴더의 info.json (없으면 None)."""
    return _read_json(Path(model_dir) / "info.json")


def adapter_base_model(adapter_dir: str | Path) -> str | None:
    """어댑터가 만들어진 기반 모델 id (info.json → adapter_config.json 순)."""
    d = Path(adapter_dir)
    info = read_info(d) or {}
    if info.get("base_model"):
        return str(info["base_model"])
    acfg = _read_json(d / "adapter_config.json") or {}
    return acfg.get("base_model_name_or_path") or None


def _same_model(a: str, b: str) -> bool:
    a, b = str(a).strip().rstrip("/\\").lower(), str(b).strip().rstrip("/\\").lower()
    return a == b or Path(a).name == Path(b).name


def _enable_checkpointing(model) -> None:
    kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    fn = getattr(model, "activation_checkpointing_enable", None)
    try:
        if fn is not None:
            try:
                fn(**kwargs)
            except TypeError:
                fn()
        else:
            model.gradient_checkpointing_enable(**kwargs)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    except (ValueError, NotImplementedError, AttributeError):
        pass  # 체크포인팅을 지원하지 않는 모델: 그냥 진행


# ---------------------------------------------------------------------------
# 학습 데이터
# ---------------------------------------------------------------------------
def _tokenize_texts(tok, texts: list[str], max_tokens: int, check_cancel, report=None) -> tuple[array, int]:
    """텍스트들을 eos로 이어 붙인 토큰 id 배열(int32). max_tokens에 닿으면 멈춘다."""
    ids = array("i")
    eos = tok.eos_token_id
    used = 0
    step = 64
    for s in range(0, len(texts), step):
        check_cancel()
        batch = [t for t in texts[s:s + step] if t and t.strip()]
        if batch:
            enc = tok(batch, add_special_tokens=False, verbose=False)["input_ids"]
            for seq in enc:
                ids.extend(seq)
                if eos is not None:
                    ids.append(int(eos))
                used += 1
                if len(ids) >= max_tokens:
                    return ids, used
        if report:
            report(min(1.0, (s + step) / max(1, len(texts))))
    return ids, used


def _make_blocks(n_tokens: int, seq_len: int, min_len: int = 8) -> list[tuple[int, int]]:
    """seq_len 단위로 자른 (start, end) 구간. 마지막 짧은 조각은 min_len 이상일 때만."""
    blocks = [(s, s + seq_len) for s in range(0, n_tokens - seq_len + 1, seq_len)]
    tail = len(blocks) * seq_len
    if n_tokens - tail >= min_len:
        blocks.append((tail, n_tokens))
    return blocks


def load_qa(qa_path: str | Path | None, limit: int = 20000) -> list[dict]:
    """Q&A JSONL ({"question","answer","source_url"}) 읽기. 파일이 없으면 빈 목록."""
    if not qa_path:
        return []
    p = Path(qa_path)
    if not p.is_file():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            q, a = str(r.get("question") or "").strip(), str(r.get("answer") or "").strip()
            if q and a:
                rows.append({"question": q, "answer": a, "source_url": r.get("source_url") or ""})
            if len(rows) >= limit:
                break
    return rows


def format_qa(tok, question: str, answer: str) -> tuple[list[int], list[int]]:
    """Q&A 한 쌍 → (input_ids, labels). 질문(프롬프트) 부분은 labels=-100.

    채팅 템플릿이 있으면 tokenizer.apply_chat_template, 없으면 "질문: ...\\n답변: ..." 형식.
    """
    eos = tok.eos_token_id
    if getattr(tok, "chat_template", None):
        try:
            user = [{"role": "user", "content": question}]
            prompt = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
            full = tok.apply_chat_template(user + [{"role": "assistant", "content": answer}], tokenize=False)
            if isinstance(prompt, str) and isinstance(full, str):
                if full.startswith(prompt):
                    p_ids = tok(prompt, add_special_tokens=False, verbose=False)["input_ids"]
                    a_ids = tok(full[len(prompt):], add_special_tokens=False, verbose=False)["input_ids"]
                    return list(p_ids) + list(a_ids), [-100] * len(p_ids) + list(a_ids)
                ids = list(tok(full, add_special_tokens=False, verbose=False)["input_ids"])
                return ids, list(ids)
        except Exception:  # 템플릿이 깨져 있으면 기본 형식으로
            pass
    p_ids = list(tok(f"질문: {question}\n답변: ", add_special_tokens=False, verbose=False)["input_ids"])
    a_ids = list(tok(answer, add_special_tokens=False, verbose=False)["input_ids"])
    if eos is not None:
        a_ids.append(int(eos))
    return p_ids + a_ids, [-100] * len(p_ids) + a_ids


def _collate(torch, examples: list[tuple], pad_id: int, device: str):
    """examples: [(ids LongTensor, labels LongTensor|None)] → input_ids, attention_mask, labels (오른쪽 패딩)."""
    L = max(int(e[0].shape[0]) for e in examples)
    B = len(examples)
    x = torch.full((B, L), pad_id, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    y = torch.full((B, L), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(examples):
        n = int(ids.shape[0])
        x[i, :n] = ids
        att[i, :n] = 1
        y[i, :n] = ids if lab is None else lab
    return x.to(device), att.to(device), y.to(device)


def _fmt_eta(sec: float) -> str:
    if sec < 60:
        return f"{max(0, int(sec))}초"
    if sec < 7200:
        return f"{sec / 60:.0f}분"
    return f"{sec / 3600:.1f}시간"


# ---------------------------------------------------------------------------
# 학습 중간 저장(체크포인트) — 폴더 구조·원자적 교체는 learnbox/checkpoint.py
# state_dir/adapter/ (PEFT save_pretrained) + state_dir/trainer.pt + checkpoint.json(meta)
# ---------------------------------------------------------------------------
CKPT_KIND = "lora"
_CKPT_ADAPTER = "adapter"
_CKPT_TRAINER = "trainer.pt"

# (train_lora 인자 이름, meta 키) — 재개할 때 체크포인트 값이 우선한다 (다르면 notes에 한 줄)
_CKPT_OVERRIDES = (
    ("base_model", "base_model"), ("max_steps", "steps"), ("lr", "lr"), ("seq_len", "seq_len"),
    ("batch_size", "batch_size"), ("grad_accum", "grad_accum"), ("lora_r", "lora_r"), ("lora_alpha", "lora_alpha"),
    ("qa_path", "qa_path"), ("train_path", "train_path"), ("eval_path", "eval_path"), ("seed", "seed"),
    ("dtype", "dtype"), ("max_train_tokens", "max_train_tokens"), ("gradient_checkpointing", "gradient_checkpointing"),
)


def _abs_file(path) -> str | None:
    """있는 파일이면 절대 경로 문자열, 아니면 None (체크포인트 meta 기록용)."""
    if not path:
        return None
    p = Path(path).expanduser()
    return str(p.resolve()) if p.is_file() else None


def _ckpt_same(arg: str, given, stored) -> bool:
    if arg in ("qa_path", "train_path", "eval_path"):
        if not stored:
            return False
        try:
            return Path(given).expanduser().resolve() == Path(stored).expanduser().resolve()
        except (OSError, RuntimeError, TypeError):
            return str(given) == str(stored)
    if arg == "base_model":
        return bool(stored) and _same_model(str(given), str(stored))
    if arg == "dtype":
        return _DTYPE_NAMES.get(str(given).lower()) == stored
    if arg == "gradient_checkpointing":
        return bool(given) == bool(stored)
    try:
        return math.isclose(float(given), float(stored), rel_tol=1e-9, abs_tol=0.0)
    except (TypeError, ValueError):
        return str(given) == str(stored)


def _ckpt_ignored_note(given: dict, meta: dict) -> str | None:
    """호출자가 넘긴 값 중 체크포인트 값과 다른 것들을 한 줄로 (넘기지 않은 값·auto는 비교하지 않음)."""
    diffs = []
    for arg, key in _CKPT_OVERRIDES:
        g = given.get(arg)
        if g is None or g == "" or (arg == "dtype" and str(g).lower() == "auto"):
            continue
        s = meta.get(key)
        if not _ckpt_same(arg, g, s):
            diffs.append(f"{arg} {g}→{s}")
    if not diffs:
        return None
    return "체크포인트에 기록된 설정으로 이어서 학습합니다(넘겨받은 값 무시: " + ", ".join(diffs) + ")."


def _read_trainer_state(torch, state_dir: Path) -> dict:
    """체크포인트 폴더의 어댑터 파일 확인 + trainer.pt 읽기 (CPU로). 문제가 있으면 예외."""
    adir = state_dir / _CKPT_ADAPTER
    if not (adir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"{adir / 'adapter_config.json'} 없음")
    if not any((adir / f).is_file() for f in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"{adir / 'adapter_model.safetensors'} 없음")
    tr = torch.load(state_dir / _CKPT_TRAINER, map_location="cpu", weights_only=True)
    if not isinstance(tr, dict) or not isinstance(tr.get("opt"), dict) or tr.get("py_rng") is None:
        raise ValueError(f"{_CKPT_TRAINER} 내용이 올바르지 않음")
    return tr


def _open_lora_checkpoint(root) -> tuple[dict, Path, dict]:
    """resume_checkpoint 열기 → (meta, state_dir, trainer 상태). current가 깨졌으면 prev를 쓴다. 실패는 한국어 ValueError."""
    import torch

    from . import checkpoint as ckpt

    root = Path(root).expanduser()
    got = ckpt.load(root)
    if not got:
        raise ValueError(f"이어서 할 체크포인트를 찾을 수 없거나 손상되었습니다: {root}")
    meta, d = got
    if meta.get("kind") != CKPT_KIND:
        raise ValueError(f"LoRA 학습 체크포인트가 아닙니다(종류: {meta.get('kind') or '알 수 없음'}). "
                         f"이어서 할 수 없습니다: {root}")
    cands = [(meta, d)]
    prev = root / "prev"
    if d.name == "current" and prev.is_dir():
        pm = ckpt._read_meta(prev)
        if pm is not None and pm.get("kind") == CKPT_KIND:
            cands.append((pm, prev))
    err: Exception | None = None
    for m, sd in cands:
        try:
            return m, sd, _read_trainer_state(torch, sd)
        except Exception as e:  # 파일 일부가 없거나 깨짐 → 다음 후보(prev)
            err = e
    raise ValueError(f"체크포인트 파일이 손상되어 이어서 할 수 없습니다: {root} ({err})") from err


def _set_py_rng(rnd: random.Random, state) -> None:
    """torch.save/load를 거친 random.getstate() 값을 되돌린다 (리스트로 바뀌었어도 튜플로)."""
    version, internal, gauss = state
    rnd.setstate((int(version), tuple(int(v) for v in internal), gauss))


def _restore_torch_rng(torch, tr: dict, dev: str) -> None:
    rng = tr.get("torch_rng")
    if rng is not None:
        torch.set_rng_state(rng.to("cpu", dtype=torch.uint8) if hasattr(rng, "to") else rng)
    cuda_rng = tr.get("cuda_rng")
    if cuda_rng and dev == "cuda" and torch.cuda.is_available():
        try:
            states = [s.to("cpu", dtype=torch.uint8) for s in cuda_rng]
            if len(states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(states)
            else:  # GPU 개수가 바뀜: 첫 GPU만 맞춘다
                torch.cuda.set_rng_state(states[0])
        except (RuntimeError, IndexError):
            pass


def _optimizer_to_params(torch, opt) -> None:
    """옵티마이저 상태 텐서를 각 파라미터의 장치로 (GPU에서 저장 → CPU에서 재개 등). step은 그대로 둔다."""
    for p, st in opt.state.items():
        for k, v in list(st.items()):
            if k != "step" and torch.is_tensor(v) and v.device != p.device:
                st[k] = v.to(p.device)


# ---------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------
def train_lora(train_path, eval_path, out_dir, base_model, max_steps=200, lr=2e-4, seq_len=512, batch_size=2,
               grad_accum=4, lora_r=8, lora_alpha=16, qa_path: str | None = None, resume_from: str | None = None,
               threads: int | None = None, progress=_noop_progress, check_cancel=_noop, *,
               device: str | None = None, dtype: str | None = "auto", gradient_checkpointing: bool | None = None,
               max_train_tokens: int | None = None, seed: int = 0,
               checkpoint_dir: str | None = None, checkpoint_every_steps: int = 0,
               checkpoint_every_sec: float = 600.0, resume_checkpoint: str | None = None,
               pause_point=None, stop_requested=None, on_checkpoint=None, **ignored) -> dict:
    """공개 소형 언어모델에 LoRA 추가 학습. out_dir에 어댑터 + 토크나이저 + info.json 저장.

    resume_from: 기존 어댑터 폴더에서 새 학습을 시작 (기반 모델이 같아야 함, r/alpha는 기존 어댑터 값 유지).
    추가 키워드: device("cuda"/"mps"/"cpu"), dtype("auto"/"float32"/"bfloat16"/"float16"),
    gradient_checkpointing(None=메모리 보고 자동), max_train_tokens(토큰화 상한), seed.
    알 수 없는 키워드(use_qa 등)는 무시하고 info["ignored_params"]에 기록한다.

    중간 저장·일시정지(docs/SPEC-pause-resume.md §1):
      checkpoint_dir: 체크포인트 root (None이면 저장 안 함). checkpoint_every_steps / checkpoint_every_sec: 저장 간격.
      resume_checkpoint: 이전 실행이 남긴 체크포인트 root → 그 단계 다음부터 이어서 (기록된 설정이 우선,
        resume_from보다 우선). pause_point(save): 매 단계 옵티마이저 스텝 뒤 호출(일시정지 동안 블록).
      stop_requested(): True면 지금까지 저장하고 끝내기. on_checkpoint(summary): 체크포인트를 쓸 때마다.
      학습 함수는 체크포인트를 지우지 않는다(작업 계층의 몫).
    """
    import torch

    from . import checkpoint as ckpt
    from .dataset import load_texts

    # --- 체크포인트에서 이어서 하기: 기록된 설정이 우선 ---
    ck_meta: dict | None = None
    ck_dir: Path | None = None
    ck_tr: dict | None = None
    pre_notes: list[str] = []
    if resume_checkpoint:
        ck_meta, ck_dir, ck_tr = _open_lora_checkpoint(resume_checkpoint)
        given = {"base_model": base_model, "max_steps": max_steps, "lr": lr, "seq_len": seq_len,
                 "batch_size": batch_size, "grad_accum": grad_accum, "lora_r": lora_r, "lora_alpha": lora_alpha,
                 "qa_path": qa_path, "train_path": train_path, "eval_path": eval_path, "seed": seed, "dtype": dtype,
                 "max_train_tokens": max_train_tokens, "gradient_checkpointing": gradient_checkpointing}
        m = ck_meta
        try:
            base_model = str(m.get("base_model") or "")
            max_steps, lr, seq_len = int(m["steps"]), float(m["lr"]), int(m["seq_len"])
            batch_size, grad_accum = int(m["batch_size"]), int(m["grad_accum"])
            lora_r, lora_alpha, seed = int(m["lora_r"]), int(m["lora_alpha"]), int(m.get("seed") or 0)
            dtype = m.get("dtype") or "auto"
            max_train_tokens = int(m["max_train_tokens"]) if m.get("max_train_tokens") else None
            gradient_checkpointing = bool(m.get("gradient_checkpointing"))
            start_step = int(m["step"])
            if not 0 <= start_step <= max_steps:
                raise ValueError(f"step {start_step}")
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"체크포인트 정보가 올바르지 않습니다({e}). 이어서 할 수 없습니다: "
                             f"{resume_checkpoint}") from e
        train_path, eval_path, qa_path = m.get("train_path"), m.get("eval_path"), m.get("qa_path")
        for p in (train_path, eval_path):
            if p and not Path(p).is_file():
                raise ValueError(f"체크포인트에 기록된 학습 자료가 없습니다: {p}. 이어서 할 수 없습니다.")
        note = _ckpt_ignored_note(given, m)
        if note:
            pre_notes.append(note)
        if resume_from and not (m.get("parent_path") and _ckpt_same("train_path", resume_from, m["parent_path"])):
            pre_notes.append("체크포인트에서 이어서 하므로 이어 학습할 어댑터(resume_from) 지정은 무시했습니다.")
        resume_from = None

    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError("학습 단계 수(max_steps)는 1 이상이어야 합니다.")
    lr, seq_len, batch_size = float(lr), max(16, int(seq_len)), max(1, int(batch_size))
    grad_accum, lora_r, lora_alpha = max(1, int(grad_accum)), max(1, int(lora_r)), int(lora_alpha or 2 * int(lora_r))
    base_model = str(base_model or "").strip()

    # --- 이어 학습 대상 확인 ---
    rdir = None
    parent_info: dict = {}
    if resume_from:
        rdir = Path(resume_from)
        if not (rdir / "adapter_config.json").is_file():
            raise ValueError(f"이어 학습할 LoRA 어댑터를 찾을 수 없습니다: {rdir}")
        parent_info = read_info(rdir) or {}
        parent_base = adapter_base_model(rdir)
        if not base_model:
            base_model = parent_base or ""
        elif parent_base and not _same_model(parent_base, base_model):
            raise ValueError(f"이어 학습하려는 LoRA 어댑터의 기반 모델({parent_base})이 "
                             f"선택한 기반 모델({base_model})과 다릅니다.")
    if not base_model:
        raise ValueError("기반 모델을 지정하세요.")
    if ck_meta is not None:
        parent_name, parent_path = ck_meta.get("parent"), ck_meta.get("parent_path")
        parent_total = int(ck_meta.get("parent_total") or 0)
    else:
        parent_name, parent_path = (rdir.name, str(rdir)) if rdir else (None, None)
        parent_total = int(parent_info.get("total_steps") or parent_info.get("steps") or 0) if rdir else 0

    unload()  # 답변용으로 올려 둔 모델을 내려 메모리 확보 (한 번에 하나)
    t0 = time.time()
    paused = 0.0  # 일시정지(pause_point 안)에서 보낸 시간: 학습 시간·남은 시간 계산에서 뺀다
    elapsed_before = float(ck_meta.get("elapsed") or 0.0) if ck_meta else 0.0
    resumes = int(ck_meta.get("resumes") or 0) + 1 if ck_meta else 0
    n_threads = _set_threads(threads)
    torch.manual_seed(seed)
    rnd = random.Random(seed)
    dev = device or _device()
    est = estimate_lora(base_model, lora_r, seq_len, batch_size)
    dtype_name, auto_ckpt, notes = _plan(dev, dtype, est, training=True)
    use_ckpt = auto_ckpt if gradient_checkpointing is None else bool(gradient_checkpointing)
    torch_dtype = getattr(torch, dtype_name)
    if ck_meta is not None:  # 처음 실행의 안내 문구를 이어받는다 (중복 제거)
        notes = list(dict.fromkeys([str(n) for n in (ck_meta.get("notes") or [])] + notes + pre_notes))
        if ck_meta.get("device") and ck_meta["device"] != dev:
            notes.append(f"저장했던 장치({ck_meta['device']})와 다른 장치({dev})에서 이어서 학습합니다.")

    # --- 모델 불러오기 ---
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    if ck_meta is not None:
        progress(0.0, f"체크포인트({ck_meta['step']}/{max_steps}단계)에서 이어서 학습 준비 중: "
                      f"{base_model} ({dev}, {dtype_name})")
    else:
        progress(0.0, f"기반 모델 불러오는 중: {base_model} ({dev}, {dtype_name})")

    def on_download():
        progress(0.01, f"기반 모델 내려받는 중(처음 한 번만): {base_model}")

    tok_dir = rdir if rdir else (Path(parent_path) if ck_meta is not None and parent_path else None)
    tok_src = str(tok_dir) if tok_dir and tok_dir.is_dir() and _has_tokenizer(tok_dir) else base_model
    tok = _load_tokenizer(tok_src, fallback=base_model, on_download=on_download)
    pad_id = _pad_id(tok)
    model = _from_pretrained(AutoModelForCausalLM, base_model, on_download=on_download, dtype=torch_dtype)
    max_pos = _max_positions(model.config)
    if max_pos and seq_len > max_pos:
        notes.append(f"seq_len을 모델 최대 길이 {max_pos}로 줄였습니다.")
        seq_len = max_pos
    try:
        model.config.use_cache = False
    except AttributeError:
        pass
    model.to(dev)
    if use_ckpt:
        _enable_checkpointing(model)
    check_cancel()
    if ck_dir is not None:
        try:
            model = PeftModel.from_pretrained(model, str(ck_dir / _CKPT_ADAPTER), is_trainable=True)
        except Exception as e:
            raise ValueError(f"체크포인트의 LoRA 어댑터를 불러오지 못했습니다: {e}. 이어서 할 수 없습니다.") from e
    elif rdir:
        model = PeftModel.from_pretrained(model, str(rdir), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    model.to(dev)
    pcfg = model.peft_config.get("default") if hasattr(model, "peft_config") else None
    if pcfg is not None:
        lora_r, lora_alpha = int(pcfg.r), int(pcfg.lora_alpha)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # --- 데이터 (재개 때도 같은 seed로 다시 토큰화 → 같은 블록·평가 구간) ---
    progress(0.04, "학습 자료를 토큰으로 바꾸는 중")
    train_texts = load_texts(train_path) if train_path and Path(train_path).is_file() else []
    eval_texts = load_texts(eval_path) if eval_path and Path(eval_path).is_file() else []
    need = max_steps * grad_accum * batch_size * seq_len
    cap = int(max_train_tokens) if max_train_tokens else max(need * 4, 2_000_000)
    order = list(train_texts)
    rnd.shuffle(order)  # 상한에 걸려도 매번 다른 부분이 학습되도록
    train_ids, texts_used = _tokenize_texts(tok, order, cap, check_cancel,
                                            lambda f: progress(0.04 + 0.05 * f, "학습 자료를 토큰으로 바꾸는 중"))
    n_eval = 16 if dev == "cuda" else 8
    if ck_meta is not None and ck_meta.get("n_eval"):
        n_eval = int(ck_meta["n_eval"])  # 장치가 바뀌어도 학습 전·후 평가 구간을 같게
    eval_ids, _ = _tokenize_texts(tok, eval_texts, n_eval * seq_len, check_cancel)
    train_t = torch.frombuffer(train_ids, dtype=torch.int32) if len(train_ids) else torch.zeros(0, dtype=torch.int32)
    eval_t = torch.frombuffer(eval_ids, dtype=torch.int32) if len(eval_ids) else torch.zeros(0, dtype=torch.int32)
    train_blocks = _make_blocks(len(train_ids), seq_len)
    eval_spans = [(eval_t, s, e) for s, e in _make_blocks(len(eval_ids), seq_len)][:n_eval]
    if not eval_spans:  # 평가 자료가 없으면 학습 자료 끝부분으로
        eval_spans = [(train_t, s, e) for s, e in train_blocks[-n_eval:]]

    if ck_meta is not None and qa_path and not Path(qa_path).is_file() and ck_meta.get("qa_examples"):
        notes.append(f"체크포인트에 기록된 Q&A 파일이 없어 Q&A 예시 없이 이어서 학습합니다: {qa_path}")
    qa_rows = load_qa(qa_path)
    qa_examples = []
    for r in qa_rows:
        ids, labels = format_qa(tok, r["question"], r["answer"])
        ids, labels = ids[:seq_len], labels[:seq_len]
        if sum(1 for v in labels[1:] if v != -100) >= 1:
            qa_examples.append((torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)))
    if not train_blocks and not qa_examples:
        raise ValueError("학습할 자료가 없습니다. 먼저 자료를 수집하거나 파일을 가져오세요.")
    if train_blocks and qa_examples:
        p_qa = min(0.5, max(0.25, len(qa_examples) / (len(qa_examples) + len(train_blocks))))
    else:
        p_qa = 1.0 if qa_examples else 0.0
    if ck_meta is not None:
        changed = [k for k, v in (("train_tokens", len(train_ids)), ("qa_examples", len(qa_examples)))
                   if ck_meta.get(k) is not None and int(ck_meta[k]) != v]
        if changed:
            notes.append("학습 자료가 체크포인트를 저장할 때와 달라 학습 순서가 처음과 다를 수 있습니다.")

    def block_example(t, s, e):
        return t[s:e].long(), None

    @torch.no_grad()
    def evaluate():
        if not eval_spans:
            return None
        model.eval()
        tot, cnt = 0.0, 0
        for t, s, e in eval_spans:
            check_cancel()
            x, att, y = _collate(torch, [block_example(t, s, e)], pad_id, dev)
            loss = model(input_ids=x, attention_mask=att, labels=y).loss
            if torch.isfinite(loss):
                n = int(x.shape[1]) - 1
                tot += float(loss.item()) * n
                cnt += n
        model.train()
        return tot / cnt if cnt else None

    if ck_meta is not None:
        eval_before = ck_meta.get("eval_before")  # 학습 전 평가는 처음 실행 때 값
        eval_before = float(eval_before) if isinstance(eval_before, (int, float)) else None
    else:
        progress(0.1, "학습 전 평가 중")
        eval_before = evaluate()

    # --- 학습 루프 ---
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_names = [n for n, _ in named]
    params = [p for _, p in named]
    if ck_tr is not None and ck_tr.get("param_names"):
        saved_names = [str(n) for n in ck_tr["param_names"]]
        lookup = dict(named)
        if sorted(saved_names) != sorted(param_names):
            raise ValueError("체크포인트의 LoRA 구조가 지금 모델과 맞지 않습니다. 이어서 할 수 없습니다.")
        param_names, params = saved_names, [lookup[n] for n in saved_names]  # 저장할 때와 같은 순서
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warmup = max(1, min(20, max_steps // 10))
    history: list[float] = []
    qa_used = 0
    start = 1
    if ck_tr is not None:
        try:
            opt.load_state_dict(ck_tr["opt"])
        except (ValueError, KeyError, RuntimeError) as e:
            raise ValueError(f"체크포인트의 옵티마이저 상태를 불러오지 못했습니다: {e}. 이어서 할 수 없습니다.") from e
        _optimizer_to_params(torch, opt)
        history = [float(v) for v in (ck_tr.get("history") or [])]
        qa_used = int(ck_tr.get("qa_used") or 0)
        start = int(ck_meta["step"]) + 1
        _set_py_rng(rnd, ck_tr["py_rng"])   # 토큰화(셔플) 뒤에 덮어써야 배치 순서가 이어진다
        _restore_torch_rng(torch, ck_tr, dev)
        notes.append(f"{start - 1}/{max_steps}단계 체크포인트에서 이어서 학습했습니다.")
        ck_tr = None  # 메모리 정리 (옵티마이저 상태 사본)
    model.train()

    ck_root = Path(checkpoint_dir).expanduser() if checkpoint_dir else None
    every_steps = max(0, int(checkpoint_every_steps or 0))
    every_sec = max(0.0, float(checkpoint_every_sec or 0.0))
    rec = {"train_path": _abs_file(train_path), "eval_path": _abs_file(eval_path),
           "qa_path": str(Path(qa_path).expanduser().resolve()) if qa_path else None}
    last_saved_step: int | None = None
    last_summary: dict | None = None
    last_save_t = time.monotonic()
    ckpt_errors, ckpt_last_err = 0, ""
    step = start - 1

    def save_checkpoint() -> dict | None:
        """지금 상태로 체크포인트 저장 → 요약. 이번 단계에 이미 저장했으면 그 요약. 저장 안 함/실패면 None."""
        nonlocal last_saved_step, last_summary, last_save_t, ckpt_errors, ckpt_last_err
        if ck_root is None:
            return None
        if last_saved_step == step:
            return last_summary
        try:
            tmp = ckpt.begin(ck_root)
            try:  # 임베딩은 학습하지 않으므로 저장 안 함
                model.save_pretrained(str(tmp / _CKPT_ADAPTER), save_embedding_layers=False)
            except TypeError:
                model.save_pretrained(str(tmp / _CKPT_ADAPTER))
            torch.save({
                "opt": opt.state_dict(), "py_rng": rnd.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if dev == "cuda" else None,
                "history": list(history), "qa_used": int(qa_used), "param_names": list(param_names),
            }, tmp / _CKPT_TRAINER)
            meta = {
                "kind": CKPT_KIND, "step": int(step), "steps": max_steps, "base_model": base_model, "lr": lr,
                "seq_len": seq_len, "batch_size": batch_size, "grad_accum": grad_accum,
                "lora_r": lora_r, "lora_alpha": lora_alpha, "qa_path": rec["qa_path"],
                "train_path": rec["train_path"], "eval_path": rec["eval_path"], "seed": int(seed),
                "dtype": dtype_name, "max_train_tokens": int(max_train_tokens) if max_train_tokens else None,
                "gradient_checkpointing": bool(use_ckpt), "parent": parent_name, "parent_path": parent_path,
                "parent_total": parent_total, "eval_before": eval_before,
                "elapsed": round(elapsed_before + time.time() - t0 - paused, 3), "resumes": resumes,
                "notes": list(notes),
                # 덧붙인 정보: 장치, 평가 구간 수, 자료 크기(재개 때 바뀌었는지 확인용)
                "device": dev, "n_eval": n_eval, "train_tokens": len(train_ids), "qa_examples": len(qa_examples),
            }
            summary = dict(ckpt.commit(ck_root, tmp, meta))
        except Exception as e:  # 저장장치 문제 등: 학습은 계속하고 다음 기회에 다시 저장
            ckpt_errors += 1
            ckpt_last_err = f"{type(e).__name__}: {e}"
            return None
        summary["steps"] = max_steps
        last_saved_step, last_summary, last_save_t = step, summary, time.monotonic()
        if on_checkpoint is not None:
            on_checkpoint(summary)
        return summary

    t_loop = time.time()  # paused는 루프 안(pause_point)에서만 늘어난다
    stopped_at: int | None = None
    for step in range(start, max_steps + 1):
        check_cancel()
        if step <= warmup:
            cur_lr = lr * step / warmup
        else:
            prog = (step - warmup) / max(1, max_steps - warmup)
            cur_lr = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        acc, n_ok = 0.0, 0
        for _ in range(grad_accum):
            if qa_examples and rnd.random() < p_qa:
                batch = [rnd.choice(qa_examples) for _ in range(batch_size)]
                qa_used += len(batch)
            else:
                batch = [block_example(train_t, *rnd.choice(train_blocks)) for _ in range(batch_size)]
            x, att, y = _collate(torch, batch, pad_id, dev)
            loss = model(input_ids=x, attention_mask=att, labels=y).loss
            if not torch.isfinite(loss):
                continue
            (loss / grad_accum).backward()
            acc += float(loss.item())
            n_ok += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if n_ok:
            history.append(acc / n_ok)
        el = max(0.0, time.time() - t_loop - paused)  # 일시정지 시간 제외
        eta = el / max(1, step - start + 1) * (max_steps - step)
        last = f"{history[-1]:.3f}" if history else "-"
        progress(0.12 + 0.83 * step / max_steps,
                 f"LoRA 학습 {step}/{max_steps} · loss {last} · 남은 시간 약 {_fmt_eta(eta)}")
        # (a) 정해진 간격이면 중간 저장
        if ck_root is not None and ((every_steps and step % every_steps == 0)
                                    or (every_sec and time.monotonic() - last_save_t >= every_sec)):
            save_checkpoint()
        # (b) 일시정지 지점 (요청이 있으면 저장 후 여기서 블록, 취소면 예외가 그대로 올라감)
        if pause_point is not None:
            tp = time.time()
            try:
                pause_point(save_checkpoint)
            finally:
                paused += max(0.0, time.time() - tp)
        # (c) 지금까지 저장하고 끝내기
        if stop_requested is not None and step < max_steps and stop_requested():
            save_checkpoint()
            stopped_at = step
            notes.append(f"사용자가 {step}/{max_steps}단계에서 학습을 끝냈습니다.")
            break
    if ckpt_errors:
        notes.append(f"중간 저장에 {ckpt_errors}번 실패했습니다(마지막 오류: {ckpt_last_err}).")
    done = stopped_at if stopped_at is not None else max_steps

    progress(0.95, "학습 후 평가 중")
    eval_after = evaluate()

    # --- 저장 ---
    progress(0.98, "어댑터 저장 중")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:  # 임베딩은 학습하지 않으므로 저장 안 함 (auto는 Hub에 접속해 확인하므로 오프라인에서 느려짐)
        model.save_pretrained(str(out), save_embedding_layers=False)
    except TypeError:
        model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    files = {f.relative_to(out).as_posix(): f.stat().st_size for f in sorted(out.rglob("*"))
             if f.is_file() and f.name != "info.json"}
    tail = history[-10:]
    stride = max(1, len(history) // 100)

    def rnd4(v):
        return None if v is None or not math.isfinite(v) else round(float(v), 4)

    info = {
        "kind": "lora", "base_model": base_model, "device": dev, "dtype": dtype_name,
        "steps": done, "total_steps": parent_total + done,
        "parent": parent_name, "parent_path": parent_path,
        "lora_r": lora_r, "lora_alpha": lora_alpha,
        "trainable_params": int(trainable), "total_params": int(total),
        "train_loss": rnd4(sum(tail) / len(tail)) if tail else None,
        "eval_loss_before": rnd4(eval_before), "eval_loss_after": rnd4(eval_after),
        "perplexity_after": rnd4(math.exp(min(eval_after, 30.0))) if eval_after is not None else None,
        "seconds": round(elapsed_before + time.time() - t0 - paused, 1), "threads": n_threads,
        "train_blocks": len(train_blocks), "train_tokens": len(train_ids), "train_texts": len(train_texts),
        "train_texts_used": texts_used, "eval_blocks": len(eval_spans),
        "qa_examples": len(qa_examples), "qa_samples_used": qa_used,
        "seq_len": seq_len, "batch_size": batch_size, "grad_accum": grad_accum, "lr": lr,
        "gradient_checkpointing": use_ckpt, "stopped_early": stopped_at is not None, "notes": notes,
        "loss_history": [round(v, 4) for v in history[::stride]],
        "adapter_bytes": sum(v for k, v in files.items() if k.startswith("adapter_model")),
        "files": files, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if ck_meta is not None:
        info["resumed_from_step"] = int(ck_meta["step"])
        info["resumes"] = resumes
    if ignored:
        info["ignored_params"] = sorted(ignored)
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    del model, opt, params, named
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    progress(1.0, f"LoRA 학습 완료 · eval loss {info['eval_loss_before']} → {info['eval_loss_after']}")
    return info


train_pc = train_lora  # 이전 이름 호환


# ---------------------------------------------------------------------------
# 답변 (RAG)
# ---------------------------------------------------------------------------
def unload() -> None:
    """캐시된 추론 모델을 내려 메모리를 비운다."""
    with _LOCK:
        had = bool(_CACHE)
        _CACHE.clear()
    if had:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _load_for_inference(base_model: str, adapter_dir: str | None):
    import torch
    from transformers import AutoModelForCausalLM

    adir = Path(adapter_dir).resolve() if adapter_dir else None
    mtime = 0.0
    if adir is not None:
        if not (adir / "adapter_config.json").is_file():
            raise ValueError(f"LoRA 어댑터를 찾을 수 없습니다: {adapter_dir}")
        mtime = max((f.stat().st_mtime for f in adir.glob("adapter_model*")), default=0.0)
    key = (base_model, str(adir) if adir else None, mtime)
    if key in _CACHE:
        return _CACHE[key]
    unload()  # 한 번에 하나만 메모리에 올린다
    dev = _device()
    dtype_name, _, _ = _plan(dev, "auto", estimate_lora(base_model), training=False)
    tok_src = str(adir) if adir and _has_tokenizer(adir) else base_model
    tok = _load_tokenizer(tok_src, fallback=base_model)
    model = _from_pretrained(AutoModelForCausalLM, base_model, dtype=getattr(torch, dtype_name))
    if adir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adir))
    model.to(dev).eval()
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    inner.generation_config = _greedy_config(inner, tok, 256)  # 탐욕적 디코딩을 기본값으로
    with _LOCK:
        _CACHE.clear()
        _CACHE[key] = (tok, model, dev)
    return _CACHE[key]


def build_prompt(tok, question: str, contexts: list[dict], max_context_chars: int = 1200) -> str:
    ctx = "\n\n".join(
        f"[{i}] {(c.get('title') or '').strip()}\n{(c.get('content') or '').strip()[:max_context_chars]}"
        for i, c in enumerate(contexts, 1))
    instruction = f"{ANSWER_INSTRUCTION}\n\n자료:\n{ctx}\n\n질문: {question}"
    if getattr(tok, "chat_template", None):
        try:
            text = tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": instruction}],
                tokenize=False, add_generation_prompt=True)
            if isinstance(text, str):
                return text
        except Exception:
            pass
    return instruction + "\n답변: "


def _prompt_ids(tok, question: str, contexts: list[dict], budget: int, max_context_chars: int) -> list[int]:
    templated = bool(getattr(tok, "chat_template", None))
    per = max(100, int(max_context_chars))
    while True:
        text = build_prompt(tok, question, contexts, per)
        ids = list(tok(text, add_special_tokens=not templated, verbose=False)["input_ids"])
        if len(ids) <= budget or per <= 100:
            break
        per = max(100, per // 2)
    if len(ids) > budget:
        ids = ids[-budget:]  # 질문이 있는 끝부분은 남긴다
    return ids


_CITE_RE = re.compile(r"[\[［]\s*(\d{1,3}(?:\s*[,，、]\s*\d{1,3})*)\s*[\]］]")


def extract_citations(text: str, contexts: list[dict]) -> list[dict]:
    """답변 속 [n] / [1, 2] 표시를 자료 목록과 연결. 없으면 모든 자료를 근거로 돌려준다."""
    nums: list[int] = []
    for m in _CITE_RE.finditer(text or ""):
        for n in re.findall(r"\d+", m.group(1)):
            k = int(n)
            if 1 <= k <= len(contexts) and k not in nums:
                nums.append(k)
    if not nums:
        nums = list(range(1, len(contexts) + 1))
    out = []
    for k in nums:
        c = contexts[k - 1]
        row = {"index": k, "title": c.get("title") or "", "url": c.get("url") or ""}
        for extra in ("doc_id", "chunk_id"):
            if c.get(extra) is not None:
                row[extra] = c[extra]
        out.append(row)
    return out


def _greedy_config(model, tok, max_new: int):
    """모델 기본 생성 설정을 복사해 탐욕적 디코딩으로 (샘플링 값은 지워 경고 방지)."""
    import copy

    from transformers import GenerationConfig

    base = getattr(model, "generation_config", None)
    gcfg = copy.deepcopy(base) if base is not None else GenerationConfig()
    eos = gcfg.eos_token_id
    eos_ids = [e for e in (eos if isinstance(eos, list) else [eos]) if e is not None]
    if tok.eos_token_id is not None and tok.eos_token_id not in eos_ids:
        eos_ids.append(tok.eos_token_id)
    gcfg.do_sample = False
    gcfg.temperature = None
    gcfg.top_p = None
    gcfg.top_k = None
    gcfg.max_new_tokens = max_new
    gcfg.max_length = None
    gcfg.repetition_penalty = 1.1
    gcfg.pad_token_id = _pad_id(tok)
    gcfg.eos_token_id = eos_ids or None
    return gcfg


def _clean_answer(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("답변:"):
        text = text[3:].strip()
    for stop in ("\n질문:", "\n자료:", "<|im_end|>", "<|endoftext|>"):
        i = text.find(stop)
        if i > 0:
            text = text[:i].strip()
    return text


def answer(question: str, contexts: list[dict], base_model: str | None = None, adapter_dir: str | None = None,
           max_new_tokens: int = 256, *, model_dir: str | None = None, max_context_chars: int = 1200) -> dict:
    """검색된 자료(contexts: title, content, url)를 근거로 답변.

    Returns {"answer": str, "citations": [{"index","title","url"(,"doc_id","chunk_id")}]}.
    model_dir는 adapter_dir의 이전 이름(호환용). base_model이 비어 있으면 어댑터에서 읽는다.
    """
    import torch

    adapter_dir = adapter_dir or model_dir
    contexts = [c for c in (contexts or []) if isinstance(c, dict)]
    question = str(question or "").strip()
    if not contexts:
        return {"answer": NO_CONTEXT_ANSWER, "citations": []}
    if not base_model and adapter_dir:
        base_model = adapter_base_model(adapter_dir)
    if not base_model:
        raise ValueError("기반 모델을 지정하세요.")
    with _LOCK:
        tok, model, dev = _load_for_inference(base_model, adapter_dir)
        max_pos = _max_positions(model.config) or 4096
        max_new = max(1, min(int(max_new_tokens or 256), max(16, max_pos // 2)))
        budget = max(32, min(max_pos - max_new, 3000 if dev == "cpu" else 8000))
        ids = _prompt_ids(tok, question, contexts, budget, max_context_chars)
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        att = torch.ones_like(x)
        with torch.no_grad():
            out = model.generate(input_ids=x, attention_mask=att, max_new_tokens=max_new)
        text = tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)
    text = _clean_answer(text)
    return {"answer": text, "citations": extract_citations(text, contexts)}
