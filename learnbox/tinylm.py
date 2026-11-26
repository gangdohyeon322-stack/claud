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

from . import accel

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


_TPU_COMPILE_MSG = "TPU 컴파일 중(처음 한 번, 몇 분 걸릴 수 있음)"


def _pick_device(device: str | None):
    """학습 장치 -> (정식 이름, torch.device, 안내 문구 또는 None).

    device가 None/'auto'면 LEARNBOX_DEVICE 환경변수를 보고, 그것도 없으면 자동(cuda → tpu → xpu → mps → cpu).
    고르기는 accel.pick이 한다(cuda/mps/cpu 결과와 안내 문구는 예전과 같다). 직접 준 값은 그대로 넘겨
    '알 수 없는 장치' 안내에 사용자가 쓴 값이 그대로 보이게 한다.
    """
    req = device if accel.normalize(device) != "auto" else accel.requested_device(None)
    if accel.normalize(req).startswith(("tpu:", "xla:")):
        # 번호 붙은 TPU 이름('xla:0', 'tpu:0')도 TPU를 확인하게 한다 (accel.pick은 'xla:0'을 확인 없이 받아들여
        # torch_xla가 없으면 model.to에서 죽는다). 쓸 수 없으면 CPU와 'TPU 패키지 없음' 안내.
        req = "tpu"
    return accel.pick(req)


def _tpu_oom_error(e: BaseException) -> RuntimeError:
    """TPU 메모리 부족 오류 (TPU에서는 배치를 줄여 다시 하지 않는다: 실패한 단계의 연산이 남아 있어 복구가 안전하지 않다)."""
    return RuntimeError("TPU 메모리가 부족해 학습을 멈췄습니다. 배치 크기·문맥 길이·모델 크기를 줄이세요. "
                        f"({type(e).__name__}: {str(e)[:200]})")


def _retie_head(model) -> bool:
    """출력층과 임베딩의 가중치 공유(head.weight = tok.weight)가 풀렸으면 다시 묶는다. 묶었으면 True.

    torch_xla(TPU) 장치로 옮기거나 거기서 CPU로 되돌리면 매개변수가 새로 만들어져 공유가 풀린다.
    cuda/mps/cpu에서는 공유가 유지되므로 아무것도 하지 않는다.
    """
    head, tok = getattr(model, "head", None), getattr(model, "tok", None)
    if head is None or tok is None or head.weight is tok.weight:
        return False
    head.weight = tok.weight
    return True


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
# 학습 중간 저장(체크포인트) 도우미 — 파일 구조와 원자적 교체는 learnbox.checkpoint가 맡는다.
#   state_dir/state.pt   {"model","opt","scaler","gen","torch_rng","cuda_rng","recent","B","T"}
#   state_dir/vocab.json {"itos": [...]}
# ---------------------------------------------------------------------------------------------------------------
_CKPT_STATE_KEYS = ("model", "opt", "scaler", "gen", "torch_rng", "cuda_rng", "recent", "B", "T")


def _write_tiny_checkpoint(root, state: dict, itos: list[str], meta: dict) -> dict:
    """state.pt와 vocab.json을 임시 폴더에 쓰고 확정한다. -> checkpoint.commit 반환값"""
    import shutil

    import torch

    from . import checkpoint as ckpt

    tmp = ckpt.begin(root)
    try:
        torch.save(state, tmp / "state.pt")
        (tmp / "vocab.json").write_text(json.dumps({"itos": itos}, ensure_ascii=False), encoding="utf-8")
        return ckpt.commit(root, tmp, meta)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def _read_tiny_checkpoint(root) -> dict:
    """가장 최근의 온전한 소형 AI 체크포인트 -> {"meta", "dir", "state", "itos"}.

    current의 상태 파일이 깨졌으면(checkpoint.json은 멀쩡해도) prev를 대신 읽는다. 없으면 한국어 ValueError."""
    import torch

    from . import checkpoint as ckpt

    got = ckpt.load(root)
    if got is None:
        raise ValueError(f"이어서 할 체크포인트를 찾을 수 없습니다: {root}")
    cands = [got]
    prev = Path(root) / "prev"
    if got[1].name == "current" and prev.is_dir():
        pmeta = ckpt._read_meta(prev)
        if pmeta is not None:
            cands.append((pmeta, prev))
    errors = []
    for meta, d in cands:
        if meta.get("kind") != "tiny":
            raise ValueError(f"소형 AI 모델의 체크포인트가 아닙니다 (종류: {meta.get('kind')}). 이어서 할 수 없습니다.")
        try:
            state = torch.load(d / "state.pt", map_location="cpu", weights_only=True)
            itos = json.loads((d / "vocab.json").read_text(encoding="utf-8"))["itos"]
            missing = [k for k in _CKPT_STATE_KEYS if k not in state]
            if missing:
                raise KeyError(", ".join(missing))
            if int(state["model"]["tok.weight"].shape[0]) != len(itos):
                raise ValueError("어휘 파일이 모델과 맞지 않습니다")
            _clean_arch(meta.get("arch"))
            int(meta["step"]), int(meta["steps"]), str(meta["train_path"])
        except Exception as e:
            errors.append(f"{d.name}: {type(e).__name__}: {e}")
            continue
        return {"meta": meta, "dir": d, "state": state, "itos": itos}
    raise ValueError("체크포인트 파일이 손상되어 이어서 할 수 없습니다 (" + "; ".join(errors)[:400] + ").")


def _same_path(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except (TypeError, OSError):
        return False


_TRAIN_DEFAULTS = {"steps": 1500, "batch_size": 32, "lr": 3e-3, "seed": 0, "onnx_int8": True, "max_seconds": None}


def _ignored_requests(meta: dict, arch, steps, batch_size, lr, seed, onnx_int8, max_seconds, train_path, eval_path,
                      resume_from) -> list[str]:
    """체크포인트에서 이어서 할 때 무시되는 호출자 값(체크포인트와 다른 것)을 사람이 읽을 문장 조각으로.
    train()의 기본값 그대로인 인자는 '지정하지 않음'으로 보고 비교하지 않는다."""
    d = _TRAIN_DEFAULTS
    steps = None if steps == d["steps"] else steps
    batch_size = None if batch_size == d["batch_size"] else batch_size
    lr = None if lr == d["lr"] else lr
    seed = None if seed == d["seed"] else seed
    out = []
    if arch is not None and not (isinstance(arch, dict) and str(arch.get("preset", "")).lower() in ("auto", "inherit")
                                 and not any(arch.get(k) for k in ARCH_KEYS)):
        try:
            want = _clean_arch(arch)
        except ValueError:
            want = None
        if want is None or not same_arch(want, meta["arch"]):
            got = meta["arch"]
            out.append("구조 " + (", ".join(f"{k} {want[k]}" for k in ARCH_KEYS if want[k] != int(got[k]))
                                if want else "(올바르지 않음)"))
    for name, given, stored in (("steps", steps, meta.get("steps")), ("batch_size", batch_size, meta.get("batch_size")),
                                ("seed", seed, meta.get("seed"))):
        try:
            if given is not None and stored is not None and int(given) != int(stored):
                out.append(f"{name} {given}")
        except (TypeError, ValueError):
            out.append(f"{name} {given}")
    try:
        if lr is not None and meta.get("lr") is not None and not math.isclose(float(lr), float(meta["lr"]),
                                                                              rel_tol=1e-9):
            out.append(f"lr {lr}")
    except (TypeError, ValueError):
        out.append(f"lr {lr}")
    if onnx_int8 != d["onnx_int8"] and "onnx_int8" in meta and bool(onnx_int8) != bool(meta["onnx_int8"]):
        out.append(f"onnx_int8 {bool(onnx_int8)}")
    if max_seconds and "max_seconds" in meta and max_seconds != meta["max_seconds"]:
        out.append(f"max_seconds {max_seconds}")
    if train_path and not _same_path(train_path, meta.get("train_path")):
        out.append(f"학습 자료 {train_path}")
    if eval_path and meta.get("eval_path") and not _same_path(eval_path, meta["eval_path"]):
        out.append(f"평가 자료 {eval_path}")
    if resume_from and not (meta.get("parent_path") and _same_path(resume_from, meta["parent_path"])):
        out.append(f"이어 학습할 모델 {Path(str(resume_from)).name}")
    return out


def _fmt_eta(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 60:
        return "1분 미만"
    h, m = divmod(sec // 60, 60)
    if h:
        return f"{h}시간 {m}분" if m else f"{h}시간"
    return f"{m}분"


# ---------------------------------------------------------------------------------------------------------------
# 학습
# ---------------------------------------------------------------------------------------------------------------
def train(train_path, eval_path, out_dir, arch: dict | None, steps: int = 1500, batch_size: int = 32,
          lr: float = 3e-3, resume_from: str | None = None, device: str | None = None, threads: int | None = None,
          progress=lambda p, m: None, check_cancel=lambda: None, onnx_int8: bool = True, seed: int = 0,
          max_seconds: float | None = None, checkpoint_dir: str | None = None, checkpoint_every_steps: int = 0,
          checkpoint_every_sec: float = 600.0, resume_checkpoint: str | None = None, pause_point=None,
          stop_requested=None, on_checkpoint=None) -> dict:
    """학습 파일(train/eval JSONL {"text",...})로 소형 AI 모델을 학습하고 out_dir에 저장한다.

    resume_from: 이어 학습할 기존 모델 폴더(구조가 같아야 함; arch=None이면 그 구조를 그대로 사용).
    max_seconds: 지정하면 그 시간이 지나면 학습을 멈추고(지금까지의 결과로) 저장한다.
    device: 학습 장치 auto|cuda|tpu|xpu|mps|cpu (None/'auto'면 LEARNBOX_DEVICE → 자동). 쓸 수 없으면 CPU로 학습하고
    notes에 이유를 남긴다. 장치별 차이(정밀도·TPU 실행 신호·CPU 저장)는 learnbox.accel이 맡는다.

    중간 저장(체크포인트, docs/SPEC-pause-resume.md §1):
    checkpoint_dir: 체크포인트 root (None이면 저장하지 않음). checkpoint_every_steps / checkpoint_every_sec:
    N단계 / N초(일시정지 시간 제외)마다 저장, 0이면 그 기준은 끔.
    resume_checkpoint: 이전 실행이 남긴 체크포인트 root. 주면 그 단계 다음부터 이어서 하며, 체크포인트에 기록된
    값(구조·steps·batch_size·lr·seed·자료 경로·부모 모델 등)이 호출자의 같은 인자보다 우선한다(다르면 notes에 기록).
    train_path/eval_path/arch는 None이어도 된다.
    pause_point(save): 매 단계 옵티마이저 스텝(과 예약된 저장) 직후 1회 호출. save()는 지금 상태를 저장하고
    요약을 돌려준다(이번 단계에 이미 저장했으면 그 요약, checkpoint_dir가 None이면 None). 안에서 보낸 시간은
    학습 시간·남은 시간·토큰/초에서 뺀다. 발생한 예외(Cancelled 등)는 그대로 전파한다.
    stop_requested(): True면 저장한 뒤 남은 단계를 건너뛰고 지금까지의 결과로 평가·내보내기를 마친다.
    on_checkpoint(summary): 체크포인트를 쓸 때마다 checkpoint.commit 반환값 + "steps"로 호출.
    """
    import torch
    import torch.nn.functional as F

    from .dataset import load_texts

    t_start = time.time()
    out = Path(out_dir)
    rs = None  # 체크포인트에서 이어서 할 때의 저장 내용
    notes: list[str] = []
    if resume_checkpoint:
        progress(0.0, "체크포인트 읽는 중")
        rs = _read_tiny_checkpoint(resume_checkpoint)
        rmeta = rs["meta"]
        ignored = _ignored_requests(rmeta, arch, steps, batch_size, lr, seed, onnx_int8, max_seconds, train_path,
                                    eval_path, resume_from)
        arch = _clean_arch(rmeta["arch"])
        steps, batch_size, lr = int(rmeta["steps"]), int(rmeta.get("batch_size") or batch_size), rmeta["lr"]
        seed = int(rmeta.get("seed") or 0)
        onnx_int8 = bool(rmeta.get("onnx_int8", onnx_int8))
        max_seconds = rmeta.get("max_seconds", max_seconds)
        train_path, eval_path = rmeta["train_path"], rmeta.get("eval_path")
        for p in (train_path, eval_path):
            if p and not Path(p).is_file():
                raise ValueError(f"체크포인트에 기록된 학습 자료가 없습니다: {p}. 이어서 할 수 없습니다.")
        notes = [str(n) for n in (rmeta.get("notes") or [])]
        if ignored:
            notes.append("체크포인트에 기록된 설정으로 이어서 학습했습니다 (무시한 요청 값: " + ", ".join(ignored) + ").")
    steps = max(1, int(steps))
    lr = float(lr)
    progress(0.0, "학습 자료 읽는 중")
    train_texts = [t for t in load_texts(train_path) if t]
    eval_used = bool(eval_path and Path(eval_path).exists())
    eval_texts = [t for t in load_texts(eval_path) if t] if eval_used else []
    if not train_texts:
        raise ValueError("학습할 자료가 없습니다. 먼저 자료를 수집하거나 파일을 가져오세요.")
    # 체크포인트에 남길 자료 경로 (재개할 때 같은 자료를 다시 읽는다)
    train_abs = str(Path(train_path).resolve())
    eval_abs = str(Path(eval_path).resolve()) if eval_used else None

    # 이어 학습: 부모 모델과 어휘를 불러오고 구조를 확인한다.
    parent = None
    parent_state = None
    parent_total = 0
    parent_path = str(Path(resume_from).resolve()) if resume_from else None
    if rs is not None:  # 체크포인트에서 이어서: 부모 정보와 어휘는 체크포인트의 것을 쓴다 (어휘를 다시 만들지 않음)
        itos = list(rs["itos"])
        parent = rmeta.get("parent")
        parent_path = rmeta.get("parent_path")
        parent_total = int(rmeta.get("parent_total") or 0)
        parent_train_seq = int(rmeta.get("parent_train_seq") or 0)
    elif resume_from:
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
    if rs is None:
        T = min(seq_len, n_tok - 2)  # 자료가 문맥 길이보다 짧으면 짧은 문맥으로 학습
    else:  # 저장할 때의 실제 문맥 길이 (자료가 그 사이 줄었으면 맞춰 줄인다)
        T = max(1, min(int(rs["state"]["T"]), seq_len, n_tok - 2))
    eval_source = "eval"
    if eval_ids.numel() < max(34, min(T, 64) + 2):
        eval_ids = train_ids[-min(n_tok, max(T * 4, 256)):]
        eval_source = "train_tail"
    T_eval = min(T, int(eval_ids.numel()) - 2)

    windows = n_tok - T
    batch_req = max(1, int(batch_size))
    if rs is None:
        B = max(1, min(batch_req, windows, max(1, (n_tok - 1) // T)))
        if B < batch_req:
            notes.append(f"자료가 적어 배치 크기를 {batch_req}에서 {B}(으)로 줄였습니다.")
        if T < seq_len:
            notes.append(f"자료가 문맥 길이({seq_len})보다 짧아 {T}글자 문맥으로 학습했습니다.")
    else:  # 저장할 때의 실제 배치 (메모리 부족으로 줄인 것 포함)
        B = max(1, min(int(rs["state"]["B"]), windows, max(1, (n_tok - 1) // T)))
        if rmeta.get("train_chars") is not None and int(rmeta["train_chars"]) != n_tok:
            notes.append("체크포인트를 저장한 뒤 학습 자료가 바뀌어, 끊기기 전과 똑같이 이어지지는 않습니다.")

    # 장치와 스레드 (장치: 요청 → LEARNBOX_DEVICE → 자동. 쓸 수 없으면 CPU와 안내)
    dev_name, dev, dev_note = _pick_device(device)
    if dev_note and dev_note not in notes:
        notes.append(dev_note)
    on_tpu = dev_name == "tpu"
    # 진행 문구의 장치 표시는 tpu/xpu일 때만 (cuda/mps/cpu 문구는 예전 그대로)
    dev_label = f" · {accel.LABELS[dev_name]}" if dev_name in ("tpu", "xpu") else ""
    xla_debug = on_tpu and os.environ.get("LEARNBOX_XLA_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
    n_threads = max(1, int(threads)) if threads else _default_threads()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(n_threads)
    # 시간: 이전 실행(체크포인트)에서 쓴 시간 + 이번 실행에서 일시정지(pause_point 안) 시간을 뺀 시간
    prev_elapsed = float(rmeta.get("elapsed") or 0.0) if rs is not None else 0.0
    prev_train = float(rmeta.get("train_seconds") or 0.0) if rs is not None else 0.0
    resumes = int(rmeta.get("resumes") or 0) + 1 if rs is not None else 0
    paused = 0.0
    try:
        torch.manual_seed(seed)
        model = build_model(arch, vocab=V)
        if rs is not None:
            model.load_state_dict(rs["state"]["model"])
        elif parent_state is not None:
            model.load_state_dict(parent_state)
        model.to(dev)
        _retie_head(model)  # TPU로 옮기면 가중치 공유가 풀린다 (다른 장치에서는 아무것도 안 함)
        model.fast = True
        n_params = sum(p.numel() for p in model.parameters())

        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=lr, betas=(0.9, 0.95))
        if rs is not None:  # 장치가 바뀌었어도(GPU 저장 -> CPU 재개 등) 상태 텐서를 매개변수 장치로 옮긴다
            opt.load_state_dict(rs["state"]["opt"])
            for st in opt.state.values():
                for k, v in st.items():
                    if k != "step" and torch.is_tensor(v) and v.device != dev:
                        st[k] = v.to(dev)
        # 자동 혼합 정밀도: cuda는 예전 그대로(bf16 가능하면 bf16, 아니면 fp16+스케일러), tpu bf16(스케일러 없음),
        # xpu bf16/fp16, mps/cpu는 쓰지 않음(None).
        amp_dtype = accel.tiny_amp_dtype(dev)
        scaler = accel.make_scaler(dev, amp_dtype == torch.float16)
        if rs is not None and rs["state"].get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(rs["state"]["scaler"])

        gen = torch.Generator().manual_seed(seed + 1)
        if rs is not None:
            gen.set_state(rs["state"]["gen"])

        def get_batch(ids, n, length, g):
            ix = torch.randint(0, int(ids.numel()) - length, (n,), generator=g)  # 창 [i, i+length]
            chunk = ids[ix.unsqueeze(1) + torch.arange(length + 1)].long()
            return chunk[:, :-1].to(dev), chunk[:, 1:].to(dev)

        def loss_of(x, y):
            with accel.autocast(dev, amp_dtype):  # 혼합 정밀도를 안 쓰면 nullcontext (예전 cpu/mps와 같음)
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

        if rs is None:
            # TPU: 학습 전 평가가 첫 컴파일 시점이다
            progress(0.02, "학습 전 평가 중" + (f" · {_TPU_COMPILE_MSG}" if on_tpu else ""))
            eval_before = evaluate()
            if on_tpu:  # 학습 단계 그래프도 처음 한 번 컴파일한다
                progress(0.03, f"소형 AI 모델 학습 시작{dev_label} · {_TPU_COMPILE_MSG}")
            recent: list[float] = []
            step = 0
        else:  # 학습 전 평가는 다시 하지 않는다 (처음 실행의 값)
            eb = rmeta.get("eval_before")
            eval_before = float(eb) if eb is not None else float("nan")
            recent = [float(v) for v in rs["state"]["recent"]]
            step = min(int(rmeta["step"]), steps)
            torch.set_rng_state(rs["state"]["torch_rng"])
            cuda_rng = rs["state"].get("cuda_rng")
            if cuda_rng and dev.type == "cuda":
                with contextlib.suppress(Exception):
                    for i in range(min(len(cuda_rng), torch.cuda.device_count())):
                        torch.cuda.set_rng_state(cuda_rng[i], i)
            # TPU: 재개하면 첫 단계가 첫 컴파일 시점이다
            progress(0.03 + 0.85 * step / steps, f"체크포인트에서 이어서 학습합니다 ({step}/{steps}단계부터)"
                     + (f" · {_TPU_COMPILE_MSG}" if on_tpu else ""))
        model.train()
        start_step = step
        t0 = time.time()
        done = step
        stopped_early = False
        warmup = min(100, max(1, steps // 10))
        min_lr = lr * 0.1
        report_every = max(1, steps // 200)
        cc_base = None  # LEARNBOX_XLA_DEBUG=1일 때 3단계 뒤의 XLA 컴파일 횟수
        # TPU: 단계 손실(장치 텐서)을 모아 두었다가 진행 표시 간격·저장·끝에서만 읽는다(PRD 6.2). 단계마다
        # .item()을 부르면 매 단계 장치를 기다려야 해 다음 단계 준비와 겹쳐 돌지 못한다. 다른 장치는 늘 비어 있다.
        pending: list = []

        def keep_loss(v: float) -> None:
            if math.isfinite(v):
                recent.append(v)
                if len(recent) > 50:
                    recent.pop(0)

        def flush_losses() -> None:
            """모아 둔 TPU 손실을 순서대로 읽어 recent에 넣는다. 하나씩 읽는다(이미 실행된 값을 옮기기만 하고,
            torch.stack처럼 길이마다 새 그래프를 컴파일하지 않는다). 지연 실행 오류는 여기서 드러난다."""
            if not pending:
                return
            vals = pending[:]
            pending.clear()
            try:
                for t in vals:
                    keep_loss(t.item())
            except RuntimeError as e:
                if accel.is_oom(e):
                    raise _tpu_oom_error(e) from e
                raise

        ck_steps = max(0, int(checkpoint_every_steps or 0))
        ck_sec = max(0.0, float(checkpoint_every_sec or 0.0))
        pause_at = None  # pause_point 안에 들어간 시각 (그 안에서 저장하면 이 시각까지의 시간으로 기록)
        last_ck = {"step": None, "summary": None, "active": 0.0, "failed": False}

        def clock() -> tuple[float, float]:
            """(이번 실행의 경과 시간, 이번 실행의 학습 시간) — 일시정지 시간 제외"""
            now = pause_at if pause_at is not None else time.time()
            return now - t_start - paused, now - t0 - paused

        def save():
            """지금 상태를 체크포인트로 저장하고 요약을 돌려준다 (이번 단계에 이미 저장했으면 그 요약)."""
            if not checkpoint_dir:
                return None
            if last_ck["step"] == step and last_ck["summary"] is not None:
                return last_ck["summary"]
            accel.step_end(dev)  # TPU: 모아 둔 연산을 실행한 뒤 저장 (다른 장치는 아무것도 안 함)
            flush_losses()  # TPU: recent를 저장하기 전에 남은 손실을 읽는다
            el, tr = clock()
            # TPU 텐서는 CPU 사본으로 저장한다 (TPU가 없는 곳에서도 읽히게). 다른 장치는 예전처럼 그대로.
            # TPU는 keep_vars로 매개변수 객체를 그대로 넘겨, 묶인 가중치(tok/head)가 같은 CPU 사본 하나로 저장되게 한다
            # (state_dict는 키마다 detach()로 새 객체를 만들어 to_cpu가 같은 텐서로 알아보지 못한다).
            state = {"model": accel.to_cpu(model.state_dict(keep_vars=on_tpu), dev),
                     "opt": accel.to_cpu(opt.state_dict(), dev),
                     "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                     "gen": gen.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": accel.rng_state(dev),
                     "recent": list(recent), "B": int(B), "T": int(T)}
            meta = {"kind": "tiny", "step": int(step), "steps": int(steps), "arch": dict(arch),
                    "batch_size": int(batch_req), "lr": lr, "seed": int(seed),
                    "train_path": train_abs, "eval_path": eval_abs,
                    "parent": parent, "parent_path": parent_path, "parent_total": int(parent_total),
                    "parent_train_seq": int(parent_train_seq),
                    "eval_before": float(eval_before) if math.isfinite(eval_before) else None,
                    "elapsed": round(prev_elapsed + el, 3), "train_seconds": round(prev_train + tr, 3),
                    "resumes": int(resumes), "notes": list(notes), "onnx_int8": bool(onnx_int8),
                    "max_seconds": max_seconds, "train_chars": n_tok, "device": dev_name}
            try:
                res = _write_tiny_checkpoint(checkpoint_dir, state, itos, meta)
            except (OSError, RuntimeError) as e:  # 저장장치 가득 참/빠짐 등: 학습은 계속하고 다음 기회에 다시 시도
                last_ck["active"] = clock()[1]
                if not last_ck["failed"]:
                    last_ck["failed"] = True
                    notes.append(f"중간 저장에 실패했습니다 ({step}단계): {type(e).__name__}: {str(e)[:200]}")
                progress(0.03 + 0.85 * step / steps, f"중간 저장 실패 ({step}/{steps}단계) · 학습은 계속합니다")
                return None
            summary = {**res, "steps": int(steps)}
            last_ck.update(step=step, summary=summary, active=clock()[1])
            if on_checkpoint is not None:
                on_checkpoint(summary)
            return summary

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
                accel.optimizer_step(opt, scaler, dev)
                scaler.update()
                accel.step_end(dev)  # TPU: 이 단계의 연산을 실행 (실행·컴파일 오류도 여기서 잡힌다)
                if on_tpu:  # 손실은 진행 표시 간격에서만 읽는다 (flush_losses)
                    pending.append(loss.detach())
                else:
                    lv = loss.item()
            except RuntimeError as e:  # 장치 메모리 부족 -> 배치를 절반으로 줄여 같은 단계를 다시
                if on_tpu and accel.is_oom(e):
                    # TPU: 실패한 큰 배치의 갱신이 이미 기록돼(AdamW step도 늘어남) 다시 시도하면 그 연산 위에
                    # 더 큰 그래프를 다시 컴파일하거나 갱신이 두 번 들어간다. 오류로 끝낸다(체크포인트로 재개 가능).
                    raise _tpu_oom_error(e) from e
                if not accel.is_oom(e) or B <= 1:
                    raise
                x = y = loss = None
                opt.zero_grad(set_to_none=True)
                accel.empty_cache(dev)
                B = max(1, B // 2)
                notes.append(f"메모리가 부족해 배치 크기를 {B}(으)로 줄였습니다.")
                step -= 1
                continue
            if not on_tpu:
                keep_loss(lv)
            done = step
            if xla_debug and step - start_step == 3:
                cc_base = accel.compile_count()
            if step % report_every == 0 or step == steps:
                flush_losses()  # TPU: 여기서만 손실을 읽는다 (다른 장치는 아무것도 안 함)
                el = max(1e-6, time.time() - t0 - paused)
                avg = sum(recent) / len(recent) if recent else float("nan")
                ran = max(1, step - start_step)
                msg = (f"소형 AI 모델 학습 {step}/{steps}{dev_label} · loss {avg:.3f} · "
                       f"{(step - start_step) * B * T / el:,.0f} 토큰/초")
                if step < steps:
                    msg += f" · 남은 시간 약 {_fmt_eta(el / ran * (steps - step))}"
                progress(0.03 + 0.85 * step / steps, msg)
            # (a) 예약된 중간 저장 (단계 / 시간 기준, 시간은 일시정지 제외)
            if checkpoint_dir and ((ck_steps and step % ck_steps == 0)
                                   or (ck_sec and clock()[1] - last_ck["active"] >= ck_sec)):
                save()
            # (b) 일시정지 지점: 안에서 블록되는 시간은 학습 시간에서 뺀다
            if pause_point is not None:
                pause_at = time.time()
                try:
                    pause_point(save)
                finally:
                    waited = time.time() - pause_at
                    paused += waited
                    pause_at = None
                if waited >= 1.0:
                    progress(0.03 + 0.85 * step / steps,
                             f"소형 AI 모델 학습 {step}/{steps}{dev_label} · 다시 시작했습니다")
            # (c) 지금까지 저장하고 끝내기
            if stop_requested is not None and step < steps and stop_requested():
                save()
                stopped_early = True
                notes.append(f"사용자가 {step}/{steps}단계에서 학습을 끝냈습니다.")
                break
            if max_seconds and prev_elapsed + (time.time() - t_start - paused) > max_seconds and step < steps:
                stopped_early = True
                notes.append(f"시간 제한({max_seconds:.0f}초)으로 {step}/{steps}단계에서 멈췄습니다.")
                break
        flush_losses()  # TPU: 끝내기·시간 제한으로 멈췄을 때 남은 손실 (다른 장치는 아무것도 안 함)
        if xla_debug and cc_base is not None:  # LEARNBOX_XLA_DEBUG=1 (TPU만): 3단계 뒤에도 다시 컴파일했는지
            cc_end = accel.compile_count()
            if cc_end is not None and cc_end > cc_base:
                notes.append(f"TPU에서 3단계 이후 다시 컴파일한 횟수: {cc_end - cc_base}번 "
                             "(입력 모양이 바뀌면 느려집니다).")
        train_seconds = prev_train + (time.time() - t0 - paused)
        progress(0.88, "학습 후 평가 중")
        eval_after = evaluate()
        train_loss = sum(recent) / len(recent) if recent else float("nan")

        # 저장 (CPU, fp32)
        accel.step_end(dev)  # TPU: 남은 연산을 실행한 뒤 CPU로 옮긴다 (다른 장치는 아무것도 안 함)
        model.to("cpu")
        _retie_head(model)  # TPU에서 CPU로 옮기면 가중치 공유가 다시 풀린다
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
        "parent_path": parent_path,
        "train_loss": _num(train_loss),
        "eval_loss_before": _num(eval_before),
        "eval_loss_after": eval_after_r,
        "perplexity_after": _num(math.exp(min(eval_after, 50.0)), 3) if math.isfinite(eval_after) else None,
        "seconds": round(prev_elapsed + time.time() - t_start - paused, 1),
        "train_seconds": round(train_seconds, 1),
        "device": dev_name,
        "dtype": str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "float32",
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
    if rs is not None:  # 체크포인트에서 이어서 한 경우만
        info["resumed_from_step"] = int(start_step)
        info["resumes"] = int(resumes)
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
