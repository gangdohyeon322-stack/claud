"""PC용 학습: 공개 소형 모델에 수집 자료로 LoRA 파인튜닝. GPU(CUDA/Apple MPS)가 있으면 자동 사용."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

_CACHE: dict = {}


def _device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_pc(train_path: str, eval_path: str, out_dir: str | Path, base_model: str, max_steps: int = 200,
             lr: float = 2e-4, seq_len: int = 512, batch_size: int = 2, grad_accum: int = 4, lora_r: int = 8,
             progress=lambda p, m: None, check_cancel=lambda: None) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .dataset import load_texts

    torch.manual_seed(0)
    device = _device()
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    progress(0.0, f"기반 모델 불러오는 중: {base_model} ({device})")
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    def blocks(texts):
        eos = tok.eos_token_id
        ids = []
        for t in texts:
            ids.extend(tok(t, add_special_tokens=False)["input_ids"] + ([eos] if eos is not None else []))
        return [ids[i:i + seq_len] for i in range(0, max(1, len(ids) - seq_len + 1), seq_len)] or [ids]

    train_blocks = blocks(load_texts(train_path))
    eval_blocks = blocks(load_texts(eval_path))[:16]

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for b in eval_blocks:
            x = torch.tensor([b], device=device)
            losses.append(model(input_ids=x, labels=x).loss.item())
        model.train()
        return sum(losses) / max(1, len(losses))

    eval_before = evaluate()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    rnd = random.Random(0)
    t0, loss_val = time.time(), float("nan")
    for step in range(1, max_steps + 1):
        check_cancel()
        cur_lr = lr * min(1.0, step / 10) * 0.5 * (1 + math.cos(math.pi * step / max_steps))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        acc = 0.0
        for _ in range(grad_accum):
            batch = [rnd.choice(train_blocks) for _ in range(batch_size)]
            L = min(len(b) for b in batch)
            x = torch.tensor([b[:L] for b in batch], device=device)
            loss = model(input_ids=x, labels=x).loss / grad_accum
            loss.backward()
            acc += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_val = acc
        el = time.time() - t0
        eta = el / step * (max_steps - step)
        progress(step / max_steps, f"PC 모델 학습 {step}/{max_steps} · loss {loss_val:.3f} · 남은 시간 약 {eta/60:.0f}분")
    eval_after = evaluate()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    info = {
        "kind": "pc", "base_model": base_model, "device": device, "steps": max_steps, "lora_r": lora_r,
        "trainable_params": trainable, "total_params": total, "train_loss": round(loss_val, 4),
        "eval_loss_before": round(eval_before, 4), "eval_loss_after": round(eval_after, 4),
        "seconds": round(time.time() - t0, 1), "train_blocks": len(train_blocks), "seq_len": seq_len,
    }
    (out / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def _load(model_dir: str | None, base_model: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    key = (model_dir, base_model)
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()  # 메모리 절약: 한 번에 하나만 올림
    device = _device()
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tok = AutoTokenizer.from_pretrained(model_dir or base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    if model_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_dir)
    model.to(device).eval()
    _CACHE[key] = (tok, model, device)
    return _CACHE[key]


def answer(question: str, contexts: list[dict], base_model: str, model_dir: str | None = None,
           max_new_tokens: int = 256) -> str:
    """검색된 자료를 근거로 답변 (RAG)."""
    import torch

    tok, model, device = _load(model_dir, base_model)
    ctx = "\n\n".join(f"[{i + 1}] {c['title']}\n{c['content'][:1200]}" for i, c in enumerate(contexts))
    instruction = (
        "아래 자료만 근거로 질문에 한국어로 답하세요. 근거가 된 자료 번호를 [1]처럼 표시하고, "
        "자료에 없으면 모른다고 답하세요.\n\n" f"자료:\n{ctx}\n\n질문: {question}"
    )
    if getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template([{"role": "user", "content": instruction}], tokenize=False,
                                         add_generation_prompt=True)
    else:
        prompt = instruction + "\n답변: "
    enc = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
