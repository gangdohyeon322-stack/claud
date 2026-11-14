"""엔진 조립: 설정, DB, 수집기, 작업 관리자, 학습/내보내기 핸들러."""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

from .collector import collect_links, collect_search
from .config import Config
from .dataset import build_corpus
from .db import DB
from .export import export
from .fetcher import Fetcher
from .jobs import AutoCollector, JobManager
from .timegate import describe, is_allowed, next_allowed


def _coerce(defaults: dict, given: dict) -> dict:
    out = dict(defaults)
    for k, v in (given or {}).items():
        if k in defaults and v not in (None, ""):
            out[k] = type(defaults[k])(v) if not isinstance(defaults[k], str) else str(v)
    return out


class App:
    def __init__(self, data_dir: str | Path):
        self.cfg = Config(data_dir)
        self.db = DB(self.cfg.db_path)
        self.fetcher = Fetcher(self.cfg)
        self.jobs = JobManager(self.db, {
            "collect_links": self._collect_links,
            "collect_search": self._collect_search,
            "train_pc": self._train_pc,
            "train_mobile": self._train_mobile,
            "export": self._export,
        })
        self.auto = AutoCollector(self.cfg, self.db, self.jobs)

    def start_background(self) -> None:
        self.jobs.start()
        self.auto.start()

    # ---------- 상태 ----------
    def gate(self) -> dict:
        w, tz = self.cfg["allowed_windows"], self.cfg["timezone"]
        allowed = is_allowed(w, tz)
        nxt = None if allowed else next_allowed(w, tz)
        return {"allowed": allowed, "windows": describe(w), "timezone": tz,
                "next_allowed": nxt.isoformat(timespec="minutes") if nxt else None}

    @staticmethod
    def capabilities() -> dict:
        caps = {"torch": importlib.util.find_spec("torch") is not None,
                "transformers": importlib.util.find_spec("transformers") is not None,
                "peft": importlib.util.find_spec("peft") is not None, "device": "cpu"}
        if caps["torch"]:
            import torch

            if torch.cuda.is_available():
                caps["device"] = f"cuda ({torch.cuda.get_device_name(0)})"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                caps["device"] = "mps (Apple)"
        return caps

    def status(self) -> dict:
        return {"counts": self.db.counts(), "gate": self.gate(), "auto_collect": self.cfg["auto_collect"],
                "search_provider": self.cfg["search"]["provider"], "capabilities": self.capabilities(),
                "jobs": self.db.list_jobs(8)}

    # ---------- 작업 핸들러 ----------
    def _collect_links(self, ctx, p):
        return collect_links(ctx, self.cfg, self.db, self.fetcher, list(p.get("urls", [])),
                             depth=min(int(p.get("depth", 0)), 2), same_domain=bool(p.get("same_domain", True)))

    def _collect_search(self, ctx, p):
        return collect_search(ctx, self.cfg, self.db, self.fetcher, str(p["query"]), p.get("limit"))

    def _new_model_dir(self, kind: str) -> tuple[str, Path]:
        name = f"{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return name, self.cfg.models_dir / name

    def _train_pc(self, ctx, p):
        from .train_pc import train_pc

        opts = _coerce(self.cfg["pc_training"], p)
        ctx.progress(0.0, "말뭉치 만드는 중")
        corpus = build_corpus(self.db, self.cfg.datasets_dir)
        name, out = self._new_model_dir("pc")
        info = train_pc(f"{corpus['dir']}/train.jsonl", f"{corpus['dir']}/eval.jsonl", out,
                        progress=ctx.progress, check_cancel=ctx.check_cancel, **opts)
        info["corpus"] = corpus
        info["model_id"] = self.db.add_model("pc", name, str(out), info)
        return info

    def _train_mobile(self, ctx, p):
        from .train_mobile import train_mobile

        opts = _coerce(self.cfg["mobile_training"], p)
        ctx.progress(0.0, "말뭉치 만드는 중")
        corpus = build_corpus(self.db, self.cfg.datasets_dir)
        name, out = self._new_model_dir("mobile")
        info = train_mobile(f"{corpus['dir']}/train.jsonl", f"{corpus['dir']}/eval.jsonl", out,
                            progress=ctx.progress, check_cancel=ctx.check_cancel, **opts)
        info["corpus"] = corpus
        info["model_id"] = self.db.add_model("mobile", name, str(out), info)
        return info

    def _export(self, ctx, p):
        return export(self.cfg, self.db, str(p["target"]), list(p.get("include", [])), str(p.get("actor", "admin")),
                      progress=ctx.progress, check_cancel=ctx.check_cancel)

    # ---------- 질문 ----------
    def ask(self, question: str, model_id: int | None = None, k: int = 4) -> dict:
        results = self.db.search(question, k)
        out = {"question": question, "sources": [{"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
                                                 for r in results], "answer": None, "model": None}
        if not model_id:
            return out
        m = self.db.get_model(int(model_id))
        if not m:
            raise ValueError("모델을 찾을 수 없습니다.")
        out["model"] = m["name"]
        if m["kind"] == "pc":
            from .train_pc import answer

            if not results:
                out["answer"] = "관련 자료를 찾지 못했습니다. 먼저 자료를 수집하세요."
            else:
                out["answer"] = answer(question, results, m["info"]["base_model"], m["path"])
        else:
            from .train_mobile import generate

            out["answer"] = generate(m["path"], question)
            out["note"] = "초소형 모바일 모델의 생성 결과입니다. 사실 확인은 출처를 보세요."
        return out
