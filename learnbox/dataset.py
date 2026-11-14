"""수집 자료 -> 학습용 말뭉치(JSONL)."""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path


def build_corpus(db, out_dir: Path, holdout: float = 0.05, seed: int = 0) -> dict:
    rows = [r for r in db.iter_chunks() if len(r["content"]) >= 50]
    if not rows:
        raise ValueError("학습할 자료가 없습니다. 먼저 수집하세요.")
    rnd = random.Random(seed)
    rnd.shuffle(rows)
    n_eval = max(1, int(len(rows) * holdout)) if len(rows) > 20 else 1
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = out_dir / f"corpus-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train.jsonl", rows[n_eval:] or rows), ("eval.jsonl", rows[:n_eval])):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"text": r["content"], "title": r["title"], "url": r["url"]}, ensure_ascii=False) + "\n")
    return {"dir": str(out), "train": len(rows) - n_eval, "eval": n_eval, "chars": sum(len(r["content"]) for r in rows)}


def load_texts(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f if line.strip()]
