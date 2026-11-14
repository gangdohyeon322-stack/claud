"""휴대폰(저사양)용 초소형 언어 모델: 글자 단위 GPT를 처음부터 학습하고 ONNX로 내보낸다.

CPU만으로 학습 가능한 크기이며, 휴대폰 브라우저에서 onnxruntime-web으로 실행된다.
수백만 파라미터 수준이라 문장 흉내는 내지만 사실 답변 능력은 없다. 사실 확인은 검색을 사용한다.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path

PRESETS = {
    "tiny": {"d_model": 128, "n_layer": 2, "n_head": 4, "seq_len": 128},
    "small": {"d_model": 256, "n_layer": 4, "n_head": 4, "seq_len": 128},
}
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def build_vocab(texts: list[str], size: int) -> list[str]:
    counts = Counter()
    for t in texts:
        counts.update(t)
    chars = [c for c, _ in counts.most_common(max(0, size - len(SPECIALS)))]
    return SPECIALS + chars


def _model_cls():
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, d, h, ctx):
            super().__init__()
            self.h = h
            self.ln1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.ln2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.register_buffer("mask", torch.tril(torch.ones(ctx, ctx, dtype=torch.bool)), persistent=False)

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
            q = q.view(B, T, self.h, C // self.h).transpose(1, 2)
            k = k.view(B, T, self.h, C // self.h).transpose(1, 2)
            v = v.view(B, T, self.h, C // self.h).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.h)
            att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(-1)
            y = (att @ v).transpose(1, 2).reshape(B, T, C)
            x = x + self.proj(y)
            return x + self.mlp(self.ln2(x))

    class TinyGPT(nn.Module):
        def __init__(self, vocab, d_model, n_layer, n_head, seq_len):
            super().__init__()
            self.seq_len = seq_len
            self.tok = nn.Embedding(vocab, d_model)
            self.pos = nn.Embedding(seq_len, d_model)
            self.blocks = nn.ModuleList(Block(d_model, n_head, seq_len) for _ in range(n_layer))
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab, bias=False)
            self.head.weight = self.tok.weight  # 가중치 공유로 크기 절약
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(m.weight, std=0.02)
                    if getattr(m, "bias", None) is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, idx):
            T = idx.shape[1]
            pos = torch.arange(T, device=idx.device)
            x = self.tok(idx) + self.pos(pos)
            for b in self.blocks:
                x = b(x)
            return self.head(self.ln(x))

    return TinyGPT


def train_mobile(train_path: str, eval_path: str, out_dir: str | Path, preset: str = "small", steps: int = 2000,
                 batch_size: int = 32, lr: float = 3e-3, vocab_size: int = 3000,
                 progress=lambda p, m: None, check_cancel=lambda: None) -> dict:
    import torch
    import torch.nn.functional as F

    from .dataset import load_texts

    torch.manual_seed(0)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    arch = PRESETS[preset]
    seq_len = arch["seq_len"]
    train_texts, eval_texts = load_texts(train_path), load_texts(eval_path)
    itos = build_vocab(train_texts, vocab_size)
    stoi = {c: i for i, c in enumerate(itos)}

    def encode(texts):
        ids = []
        for t in texts:
            ids.append(2)
            ids.extend(stoi.get(c, 1) for c in t)
            ids.append(3)
        return torch.tensor(ids, dtype=torch.long)

    train_ids, eval_ids = encode(train_texts), encode(eval_texts)
    if len(train_ids) <= seq_len + 1:
        raise ValueError("학습 자료가 너무 적습니다.")
    if len(eval_ids) <= seq_len + 1:
        eval_ids = train_ids[-(seq_len * 4):]

    def batch(ids, n):
        ix = torch.randint(0, len(ids) - seq_len - 1, (n,))
        x = torch.stack([ids[i:i + seq_len] for i in ix])
        y = torch.stack([ids[i + 1:i + seq_len + 1] for i in ix])
        return x, y

    model = _model_cls()(len(itos), **arch)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    @torch.no_grad()
    def evaluate(n=8):
        model.eval()
        g = torch.Generator().manual_seed(1)
        losses = []
        for _ in range(n):
            ix = torch.randint(0, len(eval_ids) - seq_len - 1, (16,), generator=g)
            x = torch.stack([eval_ids[i:i + seq_len] for i in ix])
            y = torch.stack([eval_ids[i + 1:i + seq_len + 1] for i in ix])
            losses.append(F.cross_entropy(model(x).view(-1, len(itos)), y.view(-1)).item())
        model.train()
        return sum(losses) / len(losses)

    eval_before = evaluate()
    t0 = time.time()
    loss_val = float("nan")
    for step in range(1, steps + 1):
        check_cancel()
        cur_lr = lr * min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        x, y = batch(train_ids, batch_size)
        loss = F.cross_entropy(model(x).view(-1, len(itos)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_val = loss.item()
        if step % 10 == 0 or step == steps:
            progress(step / steps, f"모바일 모델 학습 {step}/{steps} · loss {loss_val:.3f}")
    eval_after = evaluate()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    config = {"vocab_size": len(itos), **arch}
    torch.save({"config": config, "state_dict": model.state_dict()}, out / "model.pt")
    (out / "vocab.json").write_text(json.dumps({"itos": itos}, ensure_ascii=False), encoding="utf-8")
    onnx_ok, onnx_err = export_onnx(model, out / "model.onnx", seq_len)
    info = {
        "kind": "mobile", "preset": preset, "params": n_params, "steps": steps, "train_loss": round(loss_val, 4),
        "eval_loss_before": round(eval_before, 4), "eval_loss_after": round(eval_after, 4),
        "seconds": round(time.time() - t0, 1), "onnx": onnx_ok, "onnx_error": onnx_err, **config,
        "train_chars": int(len(train_ids)),
    }
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def export_onnx(model, path: Path, seq_len: int) -> tuple[bool, str | None]:
    import torch

    dummy = torch.zeros((1, 8), dtype=torch.long)
    kwargs = dict(input_names=["input_ids"], output_names=["logits"], opset_version=17,
                  dynamic_axes={"input_ids": {1: "seq"}, "logits": {1: "seq"}})
    errors = []
    for extra in ({"dynamo": False}, {}):
        try:
            torch.onnx.export(model, (dummy,), str(path), **kwargs, **extra)
            return True, None
        except TypeError as e:  # 이전 torch에는 dynamo 인자가 없음
            errors.append(str(e))
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
    return False, " | ".join(errors)[:500]


def load(model_dir: str | Path):
    import torch

    d = Path(model_dir)
    ckpt = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model = _model_cls()(**_args(ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    itos = json.loads((d / "vocab.json").read_text(encoding="utf-8"))["itos"]
    return model, itos


def _args(cfg: dict) -> dict:
    return {"vocab": cfg["vocab_size"], "d_model": cfg["d_model"], "n_layer": cfg["n_layer"],
            "n_head": cfg["n_head"], "seq_len": cfg["seq_len"]}


def generate(model_dir: str | Path, prompt: str, max_new: int = 120, temperature: float = 0.8, top_k: int = 20) -> str:
    import torch

    model, itos = load(model_dir)
    stoi = {c: i for i, c in enumerate(itos)}
    ids = [2] + [stoi.get(c, 1) for c in prompt]
    out = []
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-model.seq_len:]])
            logits = model(x)[0, -1] / max(temperature, 1e-4)
            v, i = torch.topk(logits, min(top_k, logits.numel()))
            nxt = int(i[torch.multinomial(v.softmax(-1), 1)])
            if nxt == 3:
                break
            ids.append(nxt)
            out.append(itos[nxt] if nxt >= len(SPECIALS) else "")
    return "".join(out)
