"""소형 AI 모델: 학습 파일(수집·가져온 자료)만으로 처음부터 학습하는 글자 단위 GPT.

- 크기는 프리셋(nano/mini/small/base/large) 또는 직접 입력한 숫자로 정한다 (resolve_arch, estimate).
- PC에서 PyTorch로 학습하고(train), ONNX(fp32)와 int8 ONNX로 내보내 휴대폰/PC 브라우저(온보드)에서도 돌린다.
- 이전 학습 결과에 이어서 학습할 수 있다(resume_from). 예전 train_mobile.py가 만든 모델 폴더와 호환된다.

수백만~수천만 파라미터 수준이라 문장 흉내는 내지만 사실 답변 능력은 약하다. 사실 확인은 검색을 사용한다.
torch / onnx / onnxruntime 은 함수 안에서만 import 한다 (없어도 이 모듈은 로드되고 estimate 등은 동작).
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import threading
import time
import warnings
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------------------------------------------
# 크기 프리셋과 제한
# ---------------------------------------------------------------------------------------------------------------
ARCH_KEYS = ("d_model", "n_layer", "n_head", "seq_len", "vocab_size")

PRESETS: dict[str, dict] = {
    "nano": {"d_model": 128, "n_layer": 2, "n_head": 4, "seq_len": 128, "vocab_size": 2000,
             "label": "나노 (저사양 휴대폰)", "target": "phone-low",
             "desc": "가장 작은 모델입니다. 저사양 휴대폰에서도 빠르게 실행되고 PC CPU로 몇 분 만에 학습됩니다."},
    "mini": {"d_model": 256, "n_layer": 4, "n_head": 4, "seq_len": 192, "vocab_size": 3000,
             "label": "미니 (고사양 휴대폰)", "target": "phone-high",
             "desc": "고사양 휴대폰에서 실행할 수 있는 크기입니다. 나노보다 문장이 자연스럽습니다."},
    "small": {"d_model": 384, "n_layer": 6, "n_head": 6, "seq_len": 256, "vocab_size": 4000,
              "label": "스몰 (PC CPU)", "target": "pc-cpu",
              "desc": "GPU 없는 PC에서 학습하기 알맞은 크기입니다. 휴대폰에서는 느릴 수 있습니다."},
    "base": {"d_model": 512, "n_layer": 8, "n_head": 8, "seq_len": 384, "vocab_size": 6000,
             "label": "베이스 (PC GPU)", "target": "pc-gpu",
             "desc": "GPU(VRAM 6GB 이상)로 학습하는 것을 권장합니다. 더 긴 문맥(384자)을 다룹니다."},
    "large": {"d_model": 768, "n_layer": 12, "n_head": 12, "seq_len": 512, "vocab_size": 8000,
              "label": "라지 (고성능 GPU)", "target": "pc-gpu-high",
              "desc": "고성능 GPU(VRAM 16GB 이상) 전용입니다. 학습 자료가 수백만 글자 이상일 때 효과가 있습니다."},
}
LIMITS = {"d_model": (32, 2048), "n_layer": (1, 48), "n_head": (1, 32), "seq_len": (16, 2048),
          "vocab_size": (100, 20000)}

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD, UNK, BOS, EOS = 0, 1, 2, 3
MODEL_FILES = ("model.onnx", "model.int8.onnx", "model.pt", "vocab.json", "info.json")

_MB = 1024 * 1024

# 기기 등급별 기준 (estimate의 fits 판단). 휴대폰은 브라우저(onnxruntime-web)에서 int8로 실행하는 기준,
# PC는 학습(가중치+기울기+AdamW 상태+활성값) 기준. 생성 연산량은 KV 캐시 없이 문맥 전체를 다시 계산하는 ONNX 기준.
FIT_RULES = {
    "phone-low": {"int8_mb": 8, "infer_ram_mb": 200, "gen_gflops": 0.3},
    "phone-high": {"int8_mb": 40, "infer_ram_mb": 800, "gen_gflops": 2.0},
    "pc-cpu": {"train_ram_mb": 6000, "train_gflops": 1000},
    "pc-gpu": {"train_ram_mb": 5500, "train_gflops": 20000},
    "pc-gpu-high": {"train_ram_mb": 13000, "train_gflops": 60000},
}


def _to_int(key: str, value) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} 값은 정수여야 합니다 (입력: {value!r}).")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        try:
            return int(s)
        except ValueError:
            try:
                f = float(s)
            except ValueError:
                f = None
            if f is not None and f.is_integer():
                return int(f)
    raise ValueError(f"{key} 값은 정수여야 합니다 (입력: {value!r}).")


def _check_arch(arch: dict) -> None:
    for k, (lo, hi) in LIMITS.items():
        v = arch[k]
        if not lo <= v <= hi:
            raise ValueError(f"{k} 값은 {lo}~{hi} 범위여야 합니다 (입력: {v}).")
    if arch["d_model"] % arch["n_head"] != 0:
        raise ValueError(f"d_model은 n_head로 나누어떨어져야 합니다 "
                         f"(d_model={arch['d_model']}, n_head={arch['n_head']}).")


def _preset_name(arch: dict) -> str:
    for name, p in PRESETS.items():
        if all(p[k] == arch[k] for k in ARCH_KEYS):
            return name
    return "custom"


def resolve_arch(preset: str | None = None, overrides: dict | None = None, recommended: str = "mini") -> dict:
    """프리셋 이름 + 직접 입력값 -> 검증된 구조 {"preset", d_model, n_layer, n_head, seq_len, vocab_size}.

    preset이 "auto"/None이면 recommended, "custom"이면 mini에서 시작해 overrides(하나 이상 필요)를 적용한다.
    overrides의 0/None/빈 값은 프리셋 값을 그대로 쓴다. 잘못된 값은 한국어 ValueError.
    """
    name = str(preset).strip().lower() if preset not in (None, "") else "auto"
    if name in ("auto", "none", "recommended"):
        name = recommended if recommended in PRESETS else "mini"
    custom = name == "custom"
    if not custom and name not in PRESETS:
        raise ValueError(f"알 수 없는 모델 크기입니다: {preset} (nano, mini, small, base, large, custom 중에서 고르세요).")
    base = PRESETS["mini" if custom else name]
    arch = {k: base[k] for k in ARCH_KEYS}
    given = False
    for k in ARCH_KEYS:
        v = (overrides or {}).get(k)
        if v is None or v == "" or v == 0 or v == "0":
            continue
        arch[k] = _to_int(k, v)
        given = True
    if custom and not given:
        raise ValueError("직접 설정을 고르면 d_model, n_layer, n_head, seq_len, vocab_size 중 하나 이상을 입력해야 합니다.")
    _check_arch(arch)
    return {"preset": _preset_name(arch), **arch}


def _clean_arch(arch: dict) -> dict:
    """외부에서 받은 arch dict(문자열 숫자 포함 가능)를 검증해 정리."""
    if not isinstance(arch, dict):
        raise ValueError("모델 구조(arch)가 올바르지 않습니다.")
    out = {}
    for k in ARCH_KEYS:
        if arch.get(k) in (None, "", 0, "0"):
            raise ValueError(f"모델 구조에 {k} 값이 없습니다.")
        out[k] = _to_int(k, arch[k])
    _check_arch(out)
    preset = arch.get("preset")
    if not preset or preset not in PRESETS or any(PRESETS[preset][k] != out[k] for k in ARCH_KEYS):
        preset = _preset_name(out)
    return {"preset": preset, **out}


def count_params(d_model: int, n_layer: int, seq_len: int, vocab_size: int) -> int:
    """아래 TinyGPT 구조의 정확한 파라미터 수 (출력층은 토큰 임베딩과 가중치 공유)."""
    d = d_model
    per_block = 12 * d * d + 13 * d  # ln1 2d, qkv 3d²+3d, proj d²+d, ln2 2d, mlp 4d²+4d, 4d²+d
    return vocab_size * d + seq_len * d + n_layer * per_block + 2 * d


def _fmt_count(n: int) -> str:
    if n >= 100_000_000:
        return f"{n / 100_000_000:.1f}억"
    if n >= 10_000:
        return f"{n / 10_000:,.0f}만"
    return f"{n:,}"


def _fmt_mb(mb: float) -> str:
    return f"{mb / 1024:.1f}GB" if mb >= 1024 else f"{mb:.1f}MB"


def _estimate_raw(V: int, d: int, L: int, h: int, T: int, B: int) -> dict:
    params = count_params(d, L, T, V)
    matrices = V * d + T * d + L * 12 * d * d  # int8로 바뀌는 행렬(임베딩 Gather + MatMul 가중치)
    vectors = params - matrices
    fp32_bytes = params * 4 + 32 * 1024  # 출력층은 Transpose 노드로 공유 (사본 없음) + 그래프
    # int8: 행렬은 1바이트, 출력층 행렬곱용 int8 사본(V·d), 벡터(LayerNorm/bias)는 fp32, 양자화 scale 등 그래프
    int8_bytes = matrices + V * d + vectors * 4 + 40 * 1024 + L * 5 * 1024
    # 학습: 가중치 4 + 기울기 4 + AdamW 8 바이트, 활성값(층마다 약 17·B·T·d, 출력층 3·B·T·V; 메모리 효율 어텐션 기준)
    act_train = 4 * (L * 17 * B * T * d + 3 * B * T * V + 2 * B * T * d)
    train_bytes = params * 16 + act_train
    # 추론(브라우저/PC): 가중치 2벌(파일 + 런타임) + 문맥 하나의 활성값 + 런타임 여유
    act_infer = 4 * (6 * T * d + h * T * T + T * V)
    infer_bytes = fp32_bytes * 2 + act_infer + 24 * _MB
    nonemb = L * 12 * d * d
    gen_flops = 2 * T * nonemb + L * 4 * T * T * d + 2 * T * V * d   # 토큰 하나 생성(문맥 T 전체 재계산)
    train_flops = 6 * B * T * (nonemb + V * d) + L * 12 * B * T * T * d  # 학습 한 단계(순전파+역전파)
    return {"params": params, "fp32_bytes": fp32_bytes, "int8_bytes": int8_bytes, "train_bytes": train_bytes,
            "infer_bytes": infer_bytes, "gen_gflops": gen_flops / 1e9, "train_gflops": train_flops / 1e9}


def estimate(arch: dict, batch_size: int = 32) -> dict:
    """모델 크기/메모리 예상치와 기기별 적합 여부."""
    a = _clean_arch(arch)
    B = max(1, _to_int("batch_size", batch_size or 32))
    r = _estimate_raw(a["vocab_size"], a["d_model"], a["n_layer"], a["n_head"], a["seq_len"], B)
    fp32_mb, int8_mb = r["fp32_bytes"] / _MB, r["int8_bytes"] / _MB
    train_mb, infer_mb = r["train_bytes"] / _MB, r["infer_bytes"] / _MB
    fits = {}
    for cls, rule in FIT_RULES.items():
        if cls.startswith("phone"):
            fits[cls] = (int8_mb <= rule["int8_mb"] and infer_mb <= rule["infer_ram_mb"]
                         and r["gen_gflops"] <= rule["gen_gflops"])
        else:
            fits[cls] = train_mb <= rule["train_ram_mb"] and r["train_gflops"] <= rule["train_gflops"]
    label = (f"파라미터 약 {_fmt_count(r['params'])}개 · 파일 {_fmt_mb(fp32_mb)}(fp32) / {_fmt_mb(int8_mb)}(int8)"
             f" · 학습 메모리 약 {_fmt_mb(train_mb)}")
    return {
        "params": r["params"],
        "fp32_mb": round(fp32_mb, 2),
        "int8_mb": round(int8_mb, 2),
        "train_ram_mb": round(train_mb, 1),
        "infer_ram_mb": round(infer_mb, 1),
        "fits": fits,
        "label": label,
        # 참고용 추가 정보
        "batch_size": B,
        "head_dim": a["d_model"] // a["n_head"],
        "train_gflops_per_step": round(r["train_gflops"], 2),
        "gen_gflops_per_token": round(r["gen_gflops"], 3),
    }


# ---------------------------------------------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------------------------------------------
_CLS = None


def _model_cls():
    """TinyGPT 클래스 (torch를 처음 쓸 때 만든다). state_dict 키는 예전 train_mobile.py와 같다."""
    global _CLS
    if _CLS is not None:
        return _CLS
    import torch
    import torch.nn.functional as F
    from torch import nn

    has_sdpa = hasattr(F, "scaled_dot_product_attention")

    class Block(nn.Module):
        def __init__(self, d, h, ctx):
            super().__init__()
            self.h = h
            self.hd = d // h
            self.sqrt_hd = math.sqrt(d // h)
            self.ln1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

        def _heads(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
            q = q.view(B, T, self.h, self.hd).transpose(1, 2)
            k = k.view(B, T, self.h, self.hd).transpose(1, 2)
            v = v.view(B, T, self.h, self.hd).transpose(1, 2)
            return q, k, v

        def _out(self, x, y):
            B, T, C = x.shape
            x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
            return x + self.mlp(self.ln2(x))

        def forward(self, x, mask=None):
            q, k, v = self._heads(x)
            if mask is None:  # 학습/평가: 메모리 효율 어텐션
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:  # ONNX 내보내기: 단순 연산만 사용 (onnxruntime-web 호환)
                att = (q @ k.transpose(-2, -1)) / self.sqrt_hd
                att = att.masked_fill(~mask, float("-inf")).softmax(-1)
                y = att @ v
            return self._out(x, y)

        def forward_cached(self, x, past):
            q, k, v = self._heads(x)
            if past is not None:
                k = torch.cat([past[0], k], dim=2)
                v = torch.cat([past[1], v], dim=2)
            T, S = q.shape[2], k.shape[2]
            att = (q @ k.transpose(-2, -1)) / self.sqrt_hd
            m = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
            att = att.masked_fill(~m, float("-inf")).softmax(-1)
            return self._out(x, att @ v), (k, v)

    class TinyGPT(nn.Module):
        def __init__(self, vocab, d_model, n_layer, n_head, seq_len):
            super().__init__()
            self.seq_len = seq_len
            self.vocab_size = vocab
            self.fast = False  # True면 scaled_dot_product_attention 사용 (학습용)
            self.tok = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.blocks = nn.ModuleList(Block(d_model, n_head, seq_len) for _ in range(n_layer))
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.head.weight = self.tok.weight  # 가중치 공유로 크기 절약
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)),
                                 persistent=False)
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(m.weight, std=0.02)
                    if getattr(m, "bias", None) is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, idx):
            T = idx.shape[1]
            pos = torch.arange(T, device=idx.device)
            x = self.tok(idx) + self.pos(pos)
            mask = None if (self.fast and has_sdpa) else self.mask[:T, :T]
            for b in self.blocks:
                x = b(x, mask)
            return self.head(self.ln(x))

        def forward_cached(self, idx, past=None, offset=0):
            """KV 캐시를 쓰는 생성용 순전파. idx: [1, t] 새 토큰, offset: 첫 새 토큰의 위치."""
            t = idx.shape[1]
            pos = torch.arange(offset, offset + t, device=idx.device)
            x = self.tok(idx) + self.pos(pos)
            new = []
            for i, b in enumerate(self.blocks):
                x, kv = b.forward_cached(x, past[i] if past else None)
                new.append(kv)
            return self.head(self.ln(x)), new

    _CLS = TinyGPT
    return TinyGPT


def build_model(arch: dict, vocab: int | None = None):
    """arch(d_model,n_layer,n_head,seq_len[,vocab_size]) -> 새 TinyGPT."""
    return _model_cls()(int(vocab or arch["vocab_size"]), int(arch["d_model"]), int(arch["n_layer"]),
                        int(arch["n_head"]), int(arch["seq_len"]))


def build_vocab(texts, size: int) -> list[str]:
    counts: Counter = Counter()
    for t in texts:
        counts.update(t)
    for s in SPECIALS:
        counts.pop(s, None)
    chars = [c for c, _ in counts.most_common(max(0, size - len(SPECIALS)))]
    return SPECIALS + chars


def _encode_corpus(texts, stoi: dict):
    """글들을 [BOS, ..., EOS] 로 이어 붙인 int32 텐서. (unk 개수도 돌려준다)"""
    import torch
    from array import array

    buf = array("i")
    unk = 0
    for t in texts:
        buf.append(BOS)
        ids = [stoi.get(c, UNK) for c in t]
        unk += ids.count(UNK)
        buf.extend(ids)
        buf.append(EOS)
    if not buf:
        return torch.zeros(0, dtype=torch.int32), 0
    return torch.frombuffer(buf, dtype=torch.int32).clone(), unk


def _pick_device(device: str | None):
    import torch

    want = (device or "auto").lower()
    note = None
    if want in ("auto", "none", ""):
        if torch.cuda.is_available():
            return torch.device("cuda"), None
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps"), None
        return torch.device("cpu"), None
    if want.startswith("cuda") and not torch.cuda.is_available():
        note = "CUDA GPU를 찾지 못해 CPU로 학습했습니다."
        return torch.device("cpu"), note
    if want == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            return torch.device("cpu"), "Apple GPU(MPS)를 찾지 못해 CPU로 학습했습니다."
    try:
        return torch.device(want), None
    except Exception:
        return torch.device("cpu"), f"알 수 없는 장치 '{device}' 대신 CPU로 학습했습니다."


def _default_threads() -> int:
    try:
        n = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        n = os.cpu_count() or 1
    return max(1, n - 1)


def _num(x, nd: int = 4):
    """JSON에 넣을 수 있는 숫자 (NaN/무한대는 None)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def model_arch(model_dir: str | Path) -> dict:
    """모델 폴더의 구조 {"preset", d_model, n_layer, n_head, seq_len, vocab_size}.
    info.json의 arch(요청한 구조)를 우선 사용하고, 없으면(예전 폴더) model.pt의 config로 만든다."""
    d = Path(model_dir)
    info = _read_json(d / "info.json")
    arch = info.get("arch")
    if isinstance(arch, dict) and all(k in arch for k in ARCH_KEYS):
        return {"preset": arch.get("preset") or _preset_name(arch), **{k: int(arch[k]) for k in ARCH_KEYS}}
    import torch

    cfg = torch.load(d / "model.pt", map_location="cpu", weights_only=True)["config"]
    out = {k: int(cfg[k]) for k in ARCH_KEYS}
    out["vocab_size"] = max(LIMITS["vocab_size"][0], out["vocab_size"])
    return {"preset": _preset_name(out), **out}


def same_arch(a: dict, b: dict, check_vocab: bool = True) -> bool:
    keys = ARCH_KEYS if check_vocab else ARCH_KEYS[:4]
    try:
        return all(int(a[k]) == int(b[k]) for k in keys)
    except (KeyError, TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------------------------------------------
def train(train_path, eval_path, out_dir, arch: dict | None, steps: int = 1500, batch_size: int = 32,
          lr: float = 3e-3, resume_from: str | None = None, device: str | None = None, threads: int | None = None,
          progress=lambda p, m: None, check_cancel=lambda: None, onnx_int8: bool = True, seed: int = 0,
          max_seconds: float | None = None) -> dict:
    """학습 파일(train/eval JSONL {"text",...})로 소형 AI 모델을 학습하고 out_dir에 저장한다.

    resume_from: 이어 학습할 기존 모델 폴더(구조가 같아야 함; arch=None이면 그 구조를 그대로 사용).
    max_seconds: 지정하면 그 시간이 지나면 학습을 멈추고(지금까지의 결과로) 저장한다.
    """
    import torch
    import torch.nn.functional as F

    from .dataset import load_texts

    t_start = time.time()
    steps = max(1, int(steps))
    lr = float(lr)
    out = Path(out_dir)
    progress(0.0, "학습 자료 읽는 중")
    train_texts = [t for t in load_texts(train_path) if t]
    eval_texts = [t for t in load_texts(eval_path) if t] if eval_path and Path(eval_path).exists() else []
    if not train_texts:
        raise ValueError("학습할 자료가 없습니다. 먼저 자료를 수집하거나 파일을 가져오세요.")

    # 이어 학습: 부모 모델과 어휘를 불러오고 구조를 확인한다.
    parent = None
    parent_state = None
    parent_total = 0
    if resume_from:
        pdir = Path(resume_from)
        if not (pdir / "model.pt").exists() or not (pdir / "vocab.json").exists():
            raise ValueError(f"이어 학습할 모델을 찾을 수 없습니다: {pdir}")
        ckpt = torch.load(pdir / "model.pt", map_location="cpu", weights_only=True)
        pcfg = ckpt["config"]
        itos = json.loads((pdir / "vocab.json").read_text(encoding="utf-8"))["itos"]
        pinfo = _read_json(pdir / "info.json")
        parch_req = pinfo.get("arch") if isinstance(pinfo.get("arch"), dict) else None
        parent_arch = {k: int(pcfg[k]) for k in ARCH_KEYS[:4]}
        parent_arch["vocab_size"] = int(parch_req["vocab_size"]) if parch_req and parch_req.get("vocab_size") \
            else max(LIMITS["vocab_size"][0], len(itos))
        if arch is None or (isinstance(arch, dict) and str(arch.get("preset", "")).lower() in ("auto", "inherit")
                            and not any(arch.get(k) for k in ARCH_KEYS)):
            arch = {"preset": _preset_name(parent_arch), **parent_arch}
        else:
            arch = _clean_arch(arch)
            diffs = [f"{k} {parent_arch[k]}→{arch[k]}" for k in ARCH_KEYS[:4] if parent_arch[k] != arch[k]]
            if parch_req and int(parch_req.get("vocab_size", arch["vocab_size"])) != arch["vocab_size"]:
                diffs.append(f"vocab_size {parch_req['vocab_size']}→{arch['vocab_size']}")
            if diffs:
                raise ValueError("이어 학습할 모델과 구조가 다릅니다 (" + ", ".join(diffs) +
                                 "). 같은 크기로 학습하거나 새 모델로 학습하세요.")
        if int(pcfg["vocab_size"]) != len(itos):
            raise ValueError("이어 학습할 모델의 어휘 파일(vocab.json)이 모델과 맞지 않습니다.")
        parent_state = ckpt["state_dict"]
        parent_total = int(pinfo.get("total_steps") or pinfo.get("steps") or 0)
        parent = pdir.resolve().name
        parent_train_seq = int(pcfg.get("train_seq_len") or pcfg["seq_len"])
    else:
        if arch is None:
            raise ValueError("모델 구조(arch)를 지정하세요.")
        arch = _clean_arch(arch)
        progress(0.01, "어휘 만드는 중")
        itos = build_vocab(train_texts, arch["vocab_size"])
        parent_train_seq = 0
    stoi = {c: i for i, c in enumerate(itos)}
    V = len(itos)
    seq_len = arch["seq_len"]

    train_ids, unk_train = _encode_corpus(train_texts, stoi)
    eval_ids, _ = _encode_corpus(eval_texts, stoi)
    n_tok = int(train_ids.numel())
    if n_tok < 34:
        raise ValueError("학습 자료가 너무 적습니다 (최소 30글자 이상 필요).")
    T = min(seq_len, n_tok - 2)  # 자료가 문맥 길이보다 짧으면 짧은 문맥으로 학습
    eval_source = "eval"
    if eval_ids.numel() < max(34, min(T, 64) + 2):
        eval_ids = train_ids[-min(n_tok, max(T * 4, 256)):]
        eval_source = "train_tail"
    T_eval = min(T, int(eval_ids.numel()) - 2)

    notes = []
    windows = n_tok - T
    batch_req = max(1, int(batch_size))
    B = max(1, min(batch_req, windows, max(1, (n_tok - 1) // T)))
    if B < batch_req:
        notes.append(f"자료가 적어 배치 크기를 {batch_req}에서 {B}(으)로 줄였습니다.")
    if T < seq_len:
        notes.append(f"자료가 문맥 길이({seq_len})보다 짧아 {T}글자 문맥으로 학습했습니다.")

    # 장치와 스레드
    dev, dev_note = _pick_device(device)
    if dev_note:
        notes.append(dev_note)
    n_threads = max(1, int(threads)) if threads else _default_threads()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    try:
        torch.manual_seed(seed)
        model = build_model(arch, vocab=V)
        if parent_state is not None:
            model.load_state_dict(parent_state)
        model.to(dev)
        model.fast = True
        n_params = sum(p.numel() for p in model.parameters())

        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=lr, betas=(0.9, 0.95))
        use_amp = dev.type == "cuda"
        amp_dtype = torch.float32
        if use_amp:
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scale_on = use_amp and amp_dtype == torch.float16
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=scale_on)
        except (AttributeError, TypeError):  # torch < 2.3
            scaler = torch.cuda.amp.GradScaler(enabled=scale_on)

        gen = torch.Generator().manual_seed(seed + 1)

        def get_batch(ids, n, length, g):
            ix = torch.randint(0, int(ids.numel()) - length, (n,), generator=g)  # 창 [i, i+length]
            chunk = ids[ix.unsqueeze(1) + torch.arange(length + 1)].long()
            return chunk[:, :-1].to(dev), chunk[:, 1:].to(dev)

        def loss_of(x, y):
            amp = (torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp
                   else contextlib.nullcontext())
            with amp:
                logits = model(x)
            return F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1))

        eval_windows = int(eval_ids.numel()) - T_eval
        eval_bs = max(1, min(16, eval_windows))

        @torch.no_grad()
        def evaluate(n_batches=8):
            model.eval()
            g = torch.Generator().manual_seed(1234)
            losses = []
            for _ in range(n_batches):
                check_cancel()
                x, y = get_batch(eval_ids, eval_bs, T_eval, g)
                losses.append(loss_of(x, y).item())
            model.train()
            return sum(losses) / len(losses)

        progress(0.02, "학습 전 평가 중")
        eval_before = evaluate()
        model.train()
        recent: list[float] = []
        t0 = time.time()
        done = 0
        stopped_early = False
        warmup = min(100, max(1, steps // 10))
        min_lr = lr * 0.1
        report_every = max(1, steps // 200)
        step = 0
        while step < steps:
            step += 1
            check_cancel()
            if step <= warmup:
                cur_lr = lr * step / warmup
            else:
                frac = (step - warmup) / max(1, steps - warmup)
                cur_lr = min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * frac))
            for grp in opt.param_groups:
                grp["lr"] = cur_lr
            try:
                x, y = get_batch(train_ids, B, T, gen)
                loss = loss_of(x, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                lv = loss.item()
            except RuntimeError as e:  # GPU 메모리 부족 -> 배치를 절반으로 줄여 같은 단계를 다시
                if "out of memory" not in str(e).lower() or B <= 1:
                    raise
                x = y = loss = None
                opt.zero_grad(set_to_none=True)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                B = max(1, B // 2)
                notes.append(f"메모리가 부족해 배치 크기를 {B}(으)로 줄였습니다.")
                step -= 1
                continue
            if math.isfinite(lv):
                recent.append(lv)
                if len(recent) > 50:
                    recent.pop(0)
            done = step
            if step % report_every == 0 or step == steps:
                el = max(1e-6, time.time() - t0)
                avg = sum(recent) / len(recent) if recent else float("nan")
                progress(0.03 + 0.85 * step / steps,
                         f"소형 AI 모델 학습 {step}/{steps} · loss {avg:.3f} · {step * B * T / el:,.0f} 토큰/초")
            if max_seconds and time.time() - t_start > max_seconds and step < steps:
                stopped_early = True
                notes.append(f"시간 제한({max_seconds:.0f}초)으로 {step}/{steps}단계에서 멈췄습니다.")
                break
        train_seconds = time.time() - t0
        progress(0.88, "학습 후 평가 중")
        eval_after = evaluate()
        train_loss = sum(recent) / len(recent) if recent else float("nan")

        # 저장 (CPU, fp32)
        model.to("cpu")
        model.eval()
        model.fast = False
        out.mkdir(parents=True, exist_ok=True)
        config = {"vocab_size": V, "d_model": arch["d_model"], "n_layer": arch["n_layer"],
                  "n_head": arch["n_head"], "seq_len": seq_len, "train_seq_len": max(T, parent_train_seq)}
        tmp = out / "model.pt.tmp"
        torch.save({"config": config, "state_dict": model.state_dict()}, tmp)
        os.replace(tmp, out / "model.pt")
        (out / "vocab.json").write_text(json.dumps({"itos": itos}, ensure_ascii=False), encoding="utf-8")

        check_cancel()
        progress(0.9, "ONNX로 내보내는 중")
        onnx_res = export_onnx_files(model, out, seq_len, V, int8=onnx_int8, threads=n_threads,
                                     progress=lambda p, m: progress(0.9 + 0.07 * p, m))

        progress(0.97, "샘플 문장 생성 중")
        prompt_src = (eval_texts or train_texts)[0]
        prompt = prompt_src[:20]
        try:
            g = torch.Generator().manual_seed(seed + 7)
            sample = prompt + _generate_with(model, itos, prompt, 60, 0.8, 20, g, ctx=max(T, parent_train_seq))
        except Exception as e:  # 샘플은 부가 정보
            sample = f"(샘플 생성 실패: {type(e).__name__})"
    finally:
        torch.set_num_threads(old_threads)

    files = {}
    for name in ("model.onnx", "model.int8.onnx", "model.pt", "vocab.json"):
        p = out / name
        if p.exists():
            files[name] = p.stat().st_size
    eval_after_r = _num(eval_after)
    info = {
        "kind": "tiny",
        "name": out.resolve().name,
        "preset": arch["preset"],
        "arch": dict(arch),
        "params": int(n_params),
        "steps": int(done),
        "total_steps": int(parent_total + done),
        "parent": parent,
        "parent_path": str(Path(resume_from).resolve()) if resume_from else None,
        "train_loss": _num(train_loss),
        "eval_loss_before": _num(eval_before),
        "eval_loss_after": eval_after_r,
        "perplexity_after": _num(math.exp(min(eval_after, 50.0)), 3) if math.isfinite(eval_after) else None,
        "seconds": round(time.time() - t_start, 1),
        "train_seconds": round(train_seconds, 1),
        "device": dev.type,
        "threads": n_threads,
        "batch_size": B,
        "lr": lr,
        **onnx_res,
        "files": files,
        "vocab_size": V,
        "seq_len": seq_len,
        "train_seq_len": max(T, parent_train_seq),
        "train_chars": n_tok,
        "eval_chars": int(eval_ids.numel()),
        "eval_source": eval_source,
        "unk_rate": round(unk_train / max(1, n_tok), 5),
        "sample": sample,
        "sample_prompt": prompt,
        "stopped_early": stopped_early,
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    progress(1.0, f"완료 · 평가 loss {info['eval_loss_before']} → {eval_after_r}")
    return info


# ---------------------------------------------------------------------------------------------------------------
# ONNX 내보내기 / int8 변환 / 검증
# ---------------------------------------------------------------------------------------------------------------
class _DropQuantHint(logging.Filter):
    def filter(self, record):  # quantize_dynamic의 "pre-processing" 권고 경고는 숨긴다
        return "pre-processing before quantization" not in record.getMessage()


def _torch_export(model, path: Path, seq_len: int) -> tuple[bool, str | None]:
    import torch

    dummy = torch.zeros((1, min(8, seq_len)), dtype=torch.long)
    kwargs = dict(input_names=["input_ids"], output_names=["logits"], opset_version=17,
                  dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}})
    errors = []
    for extra in ({"dynamo": False}, {}):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    torch.onnx.export(model, (dummy,), str(path), **kwargs, **extra)
            return True, None
        except TypeError as e:  # 이전 torch에는 dynamo 인자가 없음
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    return False, " | ".join(errors)[:500]


def _dedupe_tied_head(path: Path) -> bool:
    """출력층이 토큰 임베딩의 전치 사본으로 따로 저장된 경우 Transpose 노드로 바꿔 파일 크기를 줄인다."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    m = onnx.load(str(path))
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    if "tok.weight" not in inits:
        return False
    tok_t = numpy_helper.to_array(inits["tok.weight"]).T
    changed = False
    for name, init in list(inits.items()):
        if name == "tok.weight" or tuple(init.dims) != tok_t.shape:
            continue
        users = [n for n in g.node if name in n.input]
        if not users or any(n.op_type != "MatMul" for n in users):
            continue
        if not np.array_equal(numpy_helper.to_array(init), tok_t):
            continue
        g.initializer.remove(init)
        g.node.insert(0, helper.make_node("Transpose", ["tok.weight"], [name], perm=[1, 0],
                                          name=f"tied_head_transpose_{name}"))
        changed = True
    if changed:
        onnx.checker.check_model(m)
        onnx.save(m, str(path))
    return changed


def _ort_session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    if threads:
        so.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def verify_onnx(path, model, seq_len: int, vocab: int, threads: int | None = None,
                lengths: tuple[int, ...] | None = None) -> tuple[bool, float | None, str | None]:
    """onnxruntime 결과와 PyTorch 결과 비교 (seq 1, 7, seq_len). -> (일치 여부, 최대 오차, 오류)"""
    import numpy as np
    import torch

    try:
        sess = _ort_session(Path(path), threads)
    except Exception as e:
        return False, None, f"onnxruntime 로드 실패: {type(e).__name__}: {e}"[:500]
    g = torch.Generator().manual_seed(42)
    lens = lengths or tuple(sorted({1, min(7, seq_len), seq_len}))
    worst = 0.0
    fast = getattr(model, "fast", False)
    model.fast = False
    try:
        for n in lens:
            x = torch.randint(0, vocab, (1, n), generator=g)
            with torch.no_grad():
                ref = model(x).numpy()
            got = sess.run(["logits"], {"input_ids": x.numpy().astype(np.int64)})[0]
            if got.shape != ref.shape:
                return False, None, f"ONNX 출력 모양이 다릅니다: {got.shape} != {ref.shape}"
            worst = max(worst, float(np.abs(got - ref).max()))
    except Exception as e:
        return False, None, f"onnxruntime 실행 실패: {type(e).__name__}: {e}"[:500]
    finally:
        model.fast = fast
    if not worst < 1e-3:
        return False, worst, f"ONNX 결과가 PyTorch와 다릅니다 (최대 오차 {worst:.2e})."
    return True, worst, None


def check_int8(path, vocab: int, seq_len: int, threads: int | None = None) -> tuple[bool, str | None]:
    """int8 ONNX가 열리고, 올바른 모양의 유한한 logits를 내는지 확인."""
    import numpy as np

    try:
        sess = _ort_session(Path(path), threads)
        for n in sorted({1, min(7, seq_len)}):
            x = (np.arange(n, dtype=np.int64) * 7 + 4) % vocab
            out = sess.run(["logits"], {"input_ids": x.reshape(1, n)})[0]
            if out.shape != (1, n, vocab) or not np.isfinite(out).all():
                return False, f"int8 ONNX 출력이 올바르지 않습니다 (모양 {out.shape})."
        return True, None
    except Exception as e:
        return False, f"int8 ONNX 확인 실패: {type(e).__name__}: {e}"[:500]


def export_onnx_files(model, out_dir, seq_len: int, vocab: int, int8: bool = True, threads: int | None = None,
                      progress=lambda p, m: None) -> dict:
    """model.onnx(fp32)와 model.int8.onnx를 만들고 검증한다. -> info 일부

    fp32: 출력층(토큰 임베딩과 가중치 공유)의 전치 사본을 Transpose 노드로 바꿔 파일을 줄인 뒤 PyTorch와 비교 검증.
    int8: 원본 내보내기 파일을 onnxruntime.quantization.quantize_dynamic(QInt8)으로 변환 (출력층도 int8 행렬곱).
    """
    import importlib.util
    import shutil

    out = Path(out_dir)
    path, path8 = out / "model.onnx", out / "model.int8.onnx"
    res = {"onnx": False, "onnx_int8": False, "onnx_verified": False, "onnx_error": None,
           "onnx_max_diff": None, "onnx_int8_error": None, "onnx_note": None}
    for p in (path, path8):
        if p.exists():
            p.unlink()
    if importlib.util.find_spec("onnx") is None:
        res["onnx_error"] = "onnx 패키지가 없어 ONNX로 내보내지 못했습니다: pip install onnx"
        return res
    raw = out / "model.raw.onnx.tmp"
    tmp = out / "model.onnx.tmp"
    try:
        ok, err = _torch_export(model, raw, seq_len)
        if not ok:
            res["onnx_error"] = err
            return res
        has_ort = importlib.util.find_spec("onnxruntime") is not None
        shutil.copyfile(raw, tmp)
        try:
            _dedupe_tied_head(tmp)
        except Exception:
            shutil.copyfile(raw, tmp)
        if has_ort:
            progress(0.3, "ONNX 결과 검증 중")
            v_ok, diff, v_err = verify_onnx(tmp, model, seq_len, vocab, threads)
            if not v_ok:  # 중복 제거 전 파일로 다시 확인
                shutil.copyfile(raw, tmp)
                v_ok, diff, v_err = verify_onnx(tmp, model, seq_len, vocab, threads)
            res["onnx_verified"], res["onnx_max_diff"], res["onnx_error"] = v_ok, diff, v_err
        else:
            res["onnx_note"] = "onnxruntime이 없어 ONNX 검증과 int8 변환을 건너뛰었습니다: pip install onnxruntime"
        os.replace(tmp, path)
        res["onnx"] = True

        if int8 and has_ort:
            progress(0.6, "int8 모델로 변환 중")
            flt = _DropQuantHint()
            root = logging.getLogger()
            root.addFilter(flt)
            try:
                from onnxruntime.quantization import QuantType, quantize_dynamic

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    quantize_dynamic(str(raw), str(path8), weight_type=QuantType.QInt8)
                ok8, err8 = check_int8(path8, vocab, seq_len, threads)
                if not ok8 and path8.exists():
                    path8.unlink()
                res["onnx_int8"], res["onnx_int8_error"] = ok8, err8
            except Exception as e:
                if path8.exists():
                    path8.unlink()
                res["onnx_int8_error"] = f"int8 변환 실패: {type(e).__name__}: {e}"[:500]
            finally:
                root.removeFilter(flt)
    finally:
        for p in (tmp, raw):
            if p.exists():
                p.unlink()
    progress(1.0, "ONNX 완료")
    return res


# ---------------------------------------------------------------------------------------------------------------
# 불러오기 / 생성
# ---------------------------------------------------------------------------------------------------------------
_LOAD_LOCK = threading.Lock()
_LOAD_CACHE: "OrderedDict[tuple[str, int], tuple]" = OrderedDict()
_LOAD_MAX = 3


def load(model_dir) -> tuple:
    """모델 폴더 -> (model, itos). (경로, model.pt 수정 시각) 기준으로 캐시한다."""
    import torch

    d = Path(model_dir).resolve()
    pt = d / "model.pt"
    if not pt.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {pt}")
    key = (str(d), pt.stat().st_mtime_ns)
    with _LOAD_LOCK:
        hit = _LOAD_CACHE.get(key)
        if hit is not None:
            _LOAD_CACHE.move_to_end(key)
            return hit
    ckpt = torch.load(pt, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = _model_cls()(int(cfg["vocab_size"]), int(cfg["d_model"]), int(cfg["n_layer"]), int(cfg["n_head"]),
                         int(cfg["seq_len"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.ctx_len = int(cfg.get("train_seq_len") or cfg["seq_len"])
    itos = json.loads((d / "vocab.json").read_text(encoding="utf-8"))["itos"]
    if len(itos) != int(cfg["vocab_size"]):
        raise ValueError("어휘 파일(vocab.json)이 모델과 맞지 않습니다.")
    val = (model, itos)
    with _LOAD_LOCK:
        for k in [k for k in _LOAD_CACHE if k[0] == str(d)]:  # 같은 폴더의 옛 버전 제거
            del _LOAD_CACHE[k]
        _LOAD_CACHE[key] = val
        while len(_LOAD_CACHE) > _LOAD_MAX:
            _LOAD_CACHE.popitem(last=False)
    return val


def _generate_with(model, itos: list[str], prompt: str, max_new: int, temperature: float, top_k: int,
                   generator=None, ctx: int | None = None) -> str:
    import torch

    stoi = {c: i for i, c in enumerate(itos)}
    V = len(itos)
    ctx = max(2, min(int(ctx or model.seq_len), model.seq_len))
    ids = [BOS] + [stoi.get(c, UNK) for c in prompt]
    window = ids[-ctx:]
    out: list[str] = []
    banned = torch.tensor([i for i in (PAD, UNK, BOS) if i < V], dtype=torch.long)
    keep = max(1, ctx * 3 // 4)
    with torch.no_grad():
        logits, past = model.forward_cached(torch.tensor([window]), None, 0)
        pos = len(window)
        for _ in range(max(0, int(max_new))):
            lg = logits[0, -1].float().clone()
            if banned.numel() and V > len(banned) + 1:
                lg[banned] = float("-inf")
            if temperature is None or temperature <= 0:
                nxt = int(torch.argmax(lg))
            else:
                lg = lg / max(float(temperature), 1e-4)
                k = max(1, min(int(top_k or V), V))
                v, i = torch.topk(lg, k)
                nxt = int(i[torch.multinomial(v.softmax(-1), 1, generator=generator)])
            if nxt == EOS:
                break
            window.append(nxt)
            out.append(itos[nxt] if nxt >= len(SPECIALS) else "")
            if pos >= ctx:  # 문맥이 가득 차면 최근 3/4만 남기고 다시 계산
                window = window[-keep:]
                logits, past = model.forward_cached(torch.tensor([window]), None, 0)
                pos = len(window)
            else:
                logits, past = model.forward_cached(torch.tensor([[nxt]]), past, pos)
                pos += 1
    return "".join(out)


def generate(model_dir, prompt: str, max_new: int = 120, temperature: float = 0.8, top_k: int = 20,
             seed: int | None = None) -> str:
    """PC(PyTorch)에서 이어 쓰기. 프롬프트 뒤에 생성된 글자만 돌려준다."""
    import torch

    model, itos = load(model_dir)
    g = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    return _generate_with(model, itos, prompt or "", max_new, temperature, top_k, g,
                          ctx=getattr(model, "ctx_len", model.seq_len))
