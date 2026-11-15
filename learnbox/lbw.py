"""LearnBox 가중치 파일(.lbw): PC(PyTorch) <-> 브라우저 온디바이스 학습 공용 형식. (SPEC §2)

바이너리, little-endian:  magic "LBWEIGHT"(8) | version u32 | hdr_len u32 | JSON 헤더 | float32 텐서들(행우선).
numpy만 필요하다. torch는 torch를 쓰는 함수 안에서만 import 한다.
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

from . import tinylm

MAGIC = b"LBWEIGHT"
VERSION = 1
SPECIALS = tinylm.SPECIALS
# 특수 토큰 4개 + 글자 몇 개. tinylm.LIMITS의 vocab_size 하한(100)은 "요청한 최대 어휘"라 실제 어휘와 다르다.
MIN_VOCAB = 5
_HEAD = struct.Struct("<8sII")


def tensor_spec(arch: dict) -> list[tuple[str, tuple[int, ...]]]:
    """arch -> [(이름, shape)] SPEC 순서. head.weight는 tok.weight와 공유라 없다."""
    d, L, T, V = int(arch["d_model"]), int(arch["n_layer"]), int(arch["seq_len"]), int(arch["vocab_size"])
    out = [("tok.weight", (V, d)), ("pos.weight", (T, d))]
    for i in range(L):
        p = f"blocks.{i}."
        out += [(p + "ln1.weight", (d,)), (p + "ln1.bias", (d,)),
                (p + "qkv.weight", (3 * d, d)), (p + "qkv.bias", (3 * d,)),
                (p + "proj.weight", (d, d)), (p + "proj.bias", (d,)),
                (p + "ln2.weight", (d,)), (p + "ln2.bias", (d,)),
                (p + "mlp.0.weight", (4 * d, d)), (p + "mlp.0.bias", (4 * d,)),
                (p + "mlp.2.weight", (d, 4 * d)), (p + "mlp.2.bias", (d,))]
    out += [("ln.weight", (d,)), ("ln.bias", (d,))]
    return out


def _arch_of(arch: dict) -> dict:
    return {k: int(arch[k]) for k in tinylm.ARCH_KEYS}


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):        # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.ascontiguousarray(x, dtype="<f4")


def build_lbw(state: dict, arch: dict, itos: list[str], meta: dict | None = None) -> bytes:
    arch = _arch_of(arch)
    header = dict(meta or {})
    header.update({"format": "lbw", "arch": arch, "itos": list(itos), "tensors": []})
    parts = []
    for name, shape in tensor_spec(arch):
        if name not in state:
            raise ValueError(f"가중치에 {name} 텐서가 없습니다.")
        a = _np(state[name])
        if tuple(a.shape) != tuple(shape):
            raise ValueError(f"{name}의 모양이 맞지 않습니다: {tuple(a.shape)} (필요: {tuple(shape)})")
        header["tensors"].append({"name": name, "shape": list(shape)})
        parts.append(a.tobytes())
    hdr = json.dumps(header, ensure_ascii=False).encode("utf-8")
    return _HEAD.pack(MAGIC, VERSION, len(hdr)) + hdr + b"".join(parts)


def write_lbw(path, state: dict, arch: dict, itos: list[str], meta: dict | None = None) -> None:
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(build_lbw(state, arch, itos, meta))
    tmp.replace(p)


def read_lbw(data: bytes) -> tuple[dict, dict]:
    """bytes -> (header, {이름: float32 ndarray}). 형식이 깨졌으면 한국어 ValueError."""
    if len(data) < _HEAD.size:
        raise ValueError("가중치 파일이 너무 짧습니다.")
    magic, version, hl = _HEAD.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("LearnBox 가중치 파일(.lbw)이 아닙니다 (magic 불일치).")
    if version != VERSION:
        raise ValueError(f"지원하지 않는 .lbw 버전입니다: {version}")
    if hl <= 0 or _HEAD.size + hl > len(data):
        raise ValueError("가중치 파일의 헤더 길이가 올바르지 않습니다.")
    try:
        header = json.loads(data[_HEAD.size:_HEAD.size + hl].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("가중치 파일의 헤더를 읽을 수 없습니다.")
    if not isinstance(header, dict) or not isinstance(header.get("tensors"), list):
        raise ValueError("가중치 파일의 헤더에 tensors 목록이 없습니다.")
    off = _HEAD.size + hl
    tensors: dict = {}
    for t in header["tensors"]:
        try:
            name, shape = str(t["name"]), tuple(int(x) for x in t["shape"])
            if any(x < 0 for x in shape):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise ValueError("가중치 파일의 텐서 설명이 올바르지 않습니다.")
        if name in tensors:
            raise ValueError(f"텐서 {name}이(가) 중복되었습니다.")
        n = math.prod(shape)
        if off + 4 * n > len(data):
            raise ValueError(f"가중치 파일이 {name} 텐서 중간에서 끝났습니다.")
        tensors[name] = np.frombuffer(data, dtype="<f4", count=n, offset=off).reshape(shape)
        off += 4 * n
    if off != len(data):
        raise ValueError(f"가중치 파일 끝에 남는 데이터가 {len(data) - off}바이트 있습니다.")
    return header, tensors


def validate_lbw(header: dict, tensors: dict, max_params: int) -> list[str]:
    """한국어 오류 목록 (비면 통과)."""
    errs: list[str] = []
    if header.get("format") not in (None, "lbw"):
        errs.append(f"알 수 없는 형식입니다: {header.get('format')}")
    arch = header.get("arch")
    if not isinstance(arch, dict):
        return errs + ["헤더에 모델 구조(arch)가 없습니다."]
    clean = {}
    for k, (lo, hi) in tinylm.LIMITS.items():
        if k == "vocab_size":
            lo = MIN_VOCAB      # 실제 어휘는 자료의 글자 수라 구조 하한(100)보다 작을 수 있다
        v = arch.get(k)
        if isinstance(v, bool) or not isinstance(v, int):
            errs.append(f"모델 구조의 {k} 값이 정수가 아닙니다.")
        elif not lo <= v <= hi:
            errs.append(f"{k} 값은 {lo}~{hi} 범위여야 합니다 (입력: {v}).")
        else:
            clean[k] = v
    if errs:
        return errs
    if clean["d_model"] % clean["n_head"]:
        return [f"d_model은 n_head로 나누어떨어져야 합니다 (d_model={clean['d_model']}, n_head={clean['n_head']})."]
    params = tinylm.count_params(clean["d_model"], clean["n_layer"], clean["seq_len"], clean["vocab_size"])
    if params > max_params:
        return [f"파라미터가 너무 많습니다: {params:,}개 (최대 {max_params:,}개)."]

    itos = header.get("itos")
    if not isinstance(itos, list) or not all(isinstance(s, str) for s in itos):
        errs.append("헤더의 어휘(itos)가 올바르지 않습니다.")
    else:
        if len(itos) != clean["vocab_size"]:
            errs.append(f"어휘 수({len(itos)})가 vocab_size({clean['vocab_size']})와 다릅니다.")
        if list(itos[:4]) != SPECIALS:
            errs.append("어휘의 앞 4개는 <pad>, <unk>, <bos>, <eos> 여야 합니다.")
        if len(set(itos)) != len(itos):
            errs.append("어휘에 중복된 글자가 있습니다.")

    spec = tensor_spec(clean)
    names = {n for n, _ in spec}
    for n in tensors:
        if n not in names:
            errs.append(f"알 수 없는 텐서가 있습니다: {n}")
    for n, shape in spec:
        a = tensors.get(n)
        if a is None:
            errs.append(f"텐서가 없습니다: {n}")
            continue
        if tuple(a.shape) != shape:
            errs.append(f"{n}의 모양이 맞지 않습니다: {tuple(a.shape)} (필요: {shape})")
        elif not np.isfinite(a).all():
            errs.append(f"{n}에 NaN 또는 Inf 값이 있습니다.")
    return errs


def state_from_torch_dir(model_dir) -> tuple[dict, dict, list[str]]:
    """model.pt + vocab.json -> (arch, {이름: ndarray}, itos)."""
    import torch

    d = Path(model_dir)
    ckpt = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    arch = _arch_of(cfg)
    sd = ckpt["state_dict"]
    state = {}
    for name, shape in tensor_spec(arch):
        if name not in sd:
            raise ValueError(f"model.pt에 {name} 텐서가 없습니다.")
        state[name] = sd[name].detach().cpu().numpy().astype("<f4")
    itos = json.loads((d / "vocab.json").read_text(encoding="utf-8"))["itos"]
    return arch, state, itos


def lbw_from_model_dir(model_dir, meta: dict | None = None) -> bytes:
    arch, state, itos = state_from_torch_dir(model_dir)
    meta = dict(meta or {})
    d = Path(model_dir)
    try:
        cfg = json.loads((d / "info.json").read_text(encoding="utf-8"))
    except Exception:
        cfg = {}
    meta.setdefault("train_seq_len", cfg.get("train_seq_len") or arch["seq_len"])
    meta.setdefault("total_steps", cfg.get("total_steps"))
    meta.setdefault("parent", cfg.get("name") or d.name)
    return build_lbw(state, arch, itos, meta)


def torch_state_from_lbw(tensors: dict) -> dict:
    """{이름: ndarray} -> {이름: torch.Tensor} (복사본). head는 tok.weight와 공유하므로 만들지 않는다."""
    import torch

    return {n: torch.from_numpy(np.array(a, dtype=np.float32)) for n, a in tensors.items()}


def load_torch_model(header: dict, tensors: dict):
    """검증된 lbw -> eval 모드 TinyGPT. load_state_dict 대신 텐서를 직접 복사한다(헤드는 tok와 공유)."""
    import torch

    model = tinylm.build_model(header["arch"])
    ts = torch_state_from_lbw(tensors)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, shape in tensor_spec(header["arch"]):
            params[n].copy_(ts[n])
    model.eval()
    model.fast = False
    return model
