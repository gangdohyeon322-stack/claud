"""엔진 조립: 설정, DB, 수집기, 작업 관리자, 학습/내보내기 핸들러, 질문 답변(검색/소형 AI/LoRA/온라인)."""
from __future__ import annotations

import importlib
import importlib.util
import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path

from .autolearn import AutoLearner, record_collect
from .collector import collect_links, collect_search
from .config import Config
from .dataset import build_corpus
from .db import DB
from .export import export
from .fetcher import Fetcher
from .jobs import AutoCollector, JobManager
from .timegate import describe, is_allowed, next_allowed

ENGINES = ("search", "tiny", "lora", "online")
ARCH_KEYS = ("d_model", "n_layer", "n_head", "seq_len", "vocab_size")
TINY_KINDS = ("tiny", "mobile")   # 'mobile'은 예전 이름 (DB가 시작할 때 tiny로 바꾼다)
LORA_KINDS = ("lora", "pc")
NO_MODEL_MSG = {"tiny": "학습한 소형 AI 모델이 없습니다. [학습] 탭에서 소형 AI 모델을 학습하세요.",
                "lora": "학습한 LoRA 모델이 없습니다. [학습] 탭에서 LoRA 모델을 학습하세요."}


def _mod(name: str):
    """learnbox 하위 모듈을 필요할 때 불러온다 (torch 등 무거운 의존성은 이렇게 늦게 불러온다)."""
    return importlib.import_module(f"{__package__}.{name}")


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _int(value, default, name: str = "", lo: int | None = None, hi: int | None = None) -> int:
    v = default if value in (None, "") else value
    try:
        v = int(float(v))
    except (TypeError, ValueError):
        raise ValueError(f"'{name or '값'}'이(가) 올바른 숫자가 아닙니다: {v!r}") from None
    if lo is not None and v < lo:
        raise ValueError(f"'{name or '값'}'은(는) {lo} 이상이어야 합니다 (입력: {v}).")
    if hi is not None and v > hi:
        raise ValueError(f"'{name or '값'}'은(는) {hi} 이하여야 합니다 (입력: {v}).")
    return v


def _float(value, default, name: str = "", lo: float | None = None) -> float:
    v = default if value in (None, "") else value
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"'{name or '값'}'이(가) 올바른 숫자가 아닙니다: {v!r}") from None
    if lo is not None and v <= lo:
        raise ValueError(f"'{name or '값'}'은(는) {lo}보다 커야 합니다 (입력: {v}).")
    return v


def _bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on", "y", "예")
    return bool(v)


def _sub(progress, lo: float, hi: float):
    def f(p, m=""):
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = 0.0
        progress(lo + (hi - lo) * max(0.0, min(1.0, p)), m)
    return f


class App:
    def __init__(self, data_dir: str | Path):
        self.cfg = Config(data_dir)
        self.db = DB(self.cfg.db_path)
        self.fetcher = Fetcher(self.cfg)
        self.jobs = JobManager(self.db, {
            "collect_links": self._collect_links,
            "collect_search": self._collect_search,
            "ingest_path": self._ingest_path,
            "index_embeddings": self._index_embeddings,
            "build_qa": self._build_qa,
            "train_tiny": self._train_tiny,
            "train_lora": self._train_lora,
            "autolearn_train": self._autolearn_train,
            "finalize_learning": self._finalize_learning,
            "export": self._export,
        })
        self.auto = AutoCollector(self.cfg, self.db, self.jobs)   # 설정 auto_collect (단순 예약 수집)
        self.autolearn = AutoLearner(self)                        # 자율 학습 (수집 + 학습 + 학습 종료)
        self._profile_warm = threading.Event()

    def start_background(self) -> None:
        self.jobs.start()
        if self.auto.ident is None:
            self.auto.start()
        self.autolearn.start()
        threading.Thread(target=self._warm_profile, name="profile-warm", daemon=True).start()

    def _warm_profile(self) -> None:
        """PC 성능 확인은 처음에 torch를 불러와 몇 초 걸리므로, 서버가 켜질 때 미리 한 번 해 둔다."""
        try:
            _mod("devices").pc_profile()
        except Exception:
            traceback.print_exc()
        finally:
            self._profile_warm.set()

    # ---------- 상태 ----------
    def gate(self) -> dict:
        w, tz = self.cfg["allowed_windows"], self.cfg["timezone"]
        allowed = is_allowed(w, tz)
        nxt = None if allowed else next_allowed(w, tz)
        return {"allowed": allowed, "windows": describe(w), "timezone": tz,
                "next_allowed": nxt.isoformat(timespec="minutes") if nxt else None}

    def capabilities(self) -> dict:
        """설치된 패키지와 학습 장치. PC 성능 확인이 끝나기 전에는 패키지 설치 여부만 빠르게 알려 준다."""
        caps = {"torch": _has("torch"), "transformers": _has("transformers"), "peft": _has("peft"),
                "onnxruntime": _has("onnxruntime"), "anthropic": _has("anthropic"), "pypdf": _has("pypdf"),
                "device": "cpu", "tier": None}
        if self._profile_warm.is_set():
            try:
                prof = _mod("devices").pc_profile()
                gpu = prof.get("gpu") or {}
                if gpu.get("type") == "cuda":
                    caps["device"] = f"cuda ({gpu.get('name') or 'GPU'})"
                elif gpu.get("type") == "mps":
                    caps["device"] = "mps (Apple)"
                caps["tier"] = prof.get("tier")
            except Exception:
                pass
        return caps

    def autolearn_compact(self) -> dict:
        try:
            st = self.autolearn.status()
        except Exception:
            return {"active": False, "phase": "idle", "session_id": None}
        s = st.get("session")
        return {"active": st["active"], "phase": st["phase"], "session_id": s["id"] if s else None,
                "session_status": s["status"] if s else None, "next_collect_at": st.get("next_collect_at"),
                "next_train_check_at": st.get("next_train_check_at"),
                "new_chars_since_train": st.get("new_chars_since_train", 0)}

    def status(self) -> dict:
        return {"counts": self.db.counts(), "gate": self.gate(), "auto_collect": self.cfg["auto_collect"],
                "search_provider": self.cfg["search"]["provider"], "capabilities": self.capabilities(),
                "jobs": self.db.list_jobs(8), "online": _mod("online").status(self.cfg),
                "autolearn": self.autolearn_compact(), "unread": self.db.unread_notifications()}

    def device_info(self) -> dict:
        """PC 성능, 추천, 소형 AI 크기 목록(크기 예상 포함), LoRA 기반 모델 목록, 입력 범위."""
        devices, tinylm, train_pc = _mod("devices"), _mod("tinylm"), _mod("train_pc")
        pc = devices.pc_profile()
        presets = {}
        for name, pr in tinylm.PRESETS.items():
            presets[name] = {**pr, "estimate": tinylm.estimate(tinylm.resolve_arch(name), 32)}
        return {"pc": pc, "recommend": devices.recommend(pc), "presets": presets,
                "base_models": train_pc.BASE_MODELS, "limits": {k: list(v) for k, v in tinylm.LIMITS.items()}}

    def estimate(self, body: dict) -> dict:
        """모델 크기/메모리 예상. body: {"kind":"tiny", preset, d_model…, batch_size} 또는 {"kind":"lora", ...}"""
        kind = str(body.get("kind") or "tiny")
        devices, tinylm = _mod("devices"), _mod("tinylm")
        if kind == "lora":
            lt = self.cfg["lora_training"]
            base = str(body.get("base_model") or lt.get("base_model") or "").strip()
            if not base:
                raise ValueError("기반 모델을 지정하세요.")
            return _mod("train_pc").estimate_lora(
                base, _int(body.get("lora_r"), lt.get("lora_r", 8), "lora_r", 1, 512),
                _int(body.get("seq_len"), lt.get("seq_len", 512), "seq_len", 16, 32768),
                _int(body.get("batch_size"), lt.get("batch_size", 2), "batch_size", 1, 512), profile=devices.pc_profile())
        if kind != "tiny":
            raise ValueError("kind는 tiny 또는 lora여야 합니다.")
        tt = self.cfg["tiny_training"]
        preset = str(body.get("preset") or "auto").strip().lower()
        arch = tinylm.resolve_arch(preset, {k: body.get(k) for k in ARCH_KEYS},
                                   recommended=devices.recommend()["tiny_preset"] if preset == "auto" else "mini")
        batch = _int(body.get("batch_size"), tt.get("batch_size", 32), "batch_size", 1, 4096)
        return {"arch": arch, "estimate": tinylm.estimate(arch, batch)}

    # ---------- 수집 작업 ----------
    def _record(self, p: dict, stats: dict) -> None:
        """자율 학습 세션이 낸 수집 작업이면 세션 통계에 반영한다."""
        sid = p.get("session_id")
        if sid in (None, ""):
            return
        try:
            record_collect(self, int(sid), stats)
        except Exception:
            traceback.print_exc()

    def _collect_links(self, ctx, p):
        auto = p.get("session_id") not in (None, "")
        stats = collect_links(ctx, self.cfg, self.db, self.fetcher, list(p.get("urls", [])),
                              depth=min(int(p.get("depth", 0)), 2), same_domain=bool(p.get("same_domain", True)),
                              source="auto" if auto else "link", max_pages=p.get("max_pages"))
        self._record(p, stats)
        return stats

    def _collect_search(self, ctx, p):
        auto = p.get("session_id") not in (None, "")
        stats = collect_search(ctx, self.cfg, self.db, self.fetcher, str(p["query"]), p.get("limit"),
                               source="auto" if auto else "search")
        self._record(p, stats)
        return stats

    # ---------- 가벼운 작업 ----------
    def _ingest_path(self, ctx, p):
        return _mod("ingest").ingest_path(self.cfg, self.db, str(p["path"]), actor=str(p.get("actor") or "admin"),
                                          progress=ctx.progress, check_cancel=ctx.check_cancel)

    def _index_embeddings(self, ctx, p):
        return _mod("search").index_embeddings(self.cfg, self.db, progress=ctx.progress, check_cancel=ctx.check_cancel)

    def _build_qa(self, ctx, p):
        return _mod("online").build_qa_dataset(
            self.cfg, self.db, self.cfg.datasets_dir / "qa.jsonl",
            limit_chunks=_int(p.get("limit_chunks"), 50, "limit_chunks", 1, 5000),
            per_chunk=_int(p.get("per_chunk"), 2, "per_chunk", 1, 10),
            progress=ctx.progress, check_cancel=ctx.check_cancel)

    # ---------- 학습 작업 ----------
    def _new_model_dir(self, kind: str) -> tuple[str, Path]:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for i in range(1, 1000):
            name = f"{kind}-{stamp}" + (f"-{i}" if i > 1 else "")
            path = self.cfg.models_dir / name
            try:
                path.mkdir()
            except FileExistsError:
                continue
            return name, path
        raise RuntimeError("모델 폴더 이름을 만들 수 없습니다.")

    def _resume_model(self, model_id, kinds: tuple[str, ...], label: str) -> dict | None:
        if model_id in (None, "", 0, "0"):
            return None
        try:
            m = self.db.get_model(int(model_id))
        except (TypeError, ValueError):
            m = None
        if not m or m["kind"] not in kinds:
            raise ValueError(f"이어 학습할 {label}을(를) 찾을 수 없습니다 (번호 {model_id}).")
        if not Path(m["path"]).is_dir():
            raise ValueError(f"이어 학습할 {label}의 폴더가 없습니다: {m['path']}")
        return m

    def _threads(self, p: dict) -> int | None:
        t = _int(p.get("threads"), (self.cfg["autolearn"] or {}).get("threads") or 0, "threads", 0, 1024)
        return t or None

    def plan_tiny(self, p: dict) -> dict:
        """소형 AI 모델 학습 요청을 검사하고 구조/이어 학습 대상/단계를 정한다 (잘못되면 한국어 ValueError)."""
        tt, tinylm = self.cfg["tiny_training"], _mod("tinylm")
        resume = self._resume_model(p.get("resume_from_model_id"), TINY_KINDS, "소형 AI 모델")
        explicit = str(p.get("preset") or "").strip().lower()
        preset = explicit or str(tt.get("preset") or "auto").strip().lower()
        preset = {"tiny": "nano"}.get(preset, preset)    # 예전 train_mobile의 크기 이름
        src = p if explicit else {**{k: tt.get(k) for k in ARCH_KEYS}, **{k: p[k] for k in ARCH_KEYS if k in p}}
        overrides = {k: src.get(k) for k in ARCH_KEYS if src.get(k) not in (None, "", 0, "0")}
        if resume and preset in ("auto", "") and not overrides:
            arch = None                                   # 이어 학습: 부모 모델의 구조를 그대로 쓴다
        else:
            rec = _mod("devices").recommend()["tiny_preset"] if preset in ("auto", "") else "mini"
            arch = tinylm.resolve_arch(preset, overrides or None, recommended=rec)
        return {
            "arch": arch, "resume": resume,
            "steps": _int(p.get("steps"), tt.get("steps", 1500), "steps", 1, 10_000_000),
            "batch_size": _int(p.get("batch_size"), tt.get("batch_size", 32), "batch_size", 1, 4096),
            "lr": _float(p.get("lr"), tt.get("lr", 3e-3), "lr", 0.0),
            "threads": self._threads(p),
        }

    def _train_tiny(self, ctx, p):
        plan = self.plan_tiny(p)
        ctx.progress(0.0, "말뭉치 만드는 중")
        corpus = build_corpus(self.db, self.cfg.datasets_dir)
        name, out = self._new_model_dir("tiny")
        try:
            info = _mod("tinylm").train(
                f"{corpus['dir']}/train.jsonl", f"{corpus['dir']}/eval.jsonl", out, plan["arch"],
                steps=plan["steps"], batch_size=plan["batch_size"], lr=plan["lr"],
                resume_from=plan["resume"]["path"] if plan["resume"] else None, threads=plan["threads"],
                progress=_sub(ctx.progress, 0.03, 1.0), check_cancel=ctx.check_cancel)
        except BaseException:
            shutil.rmtree(out, ignore_errors=True)
            raise
        info = dict(info)
        info["corpus"] = corpus
        info["model_id"] = self.db.add_model("tiny", name, str(out), info)
        self.db.notify("success", "소형 AI 모델 학습을 마쳤습니다",
                       f"{name} · {info.get('steps')}단계 · 평가 loss {_fmt(info.get('eval_loss_before'))} → "
                       f"{_fmt(info.get('eval_loss_after'))}", {"kind": "train_tiny", "model_id": info["model_id"]})
        return info

    def plan_lora(self, p: dict) -> dict:
        lt = self.cfg["lora_training"]
        resume = self._resume_model(p.get("resume_from_model_id"), LORA_KINDS, "LoRA 모델")
        base = str(p.get("base_model") or "").strip()
        if not base and not resume:
            base = str(lt.get("base_model") or "").strip()   # 이어 학습이면 비워 두면 어댑터의 기반 모델을 쓴다
        if not base and not resume:
            raise ValueError("기반 모델을 지정하세요.")
        use_qa = _bool(p["use_qa"]) if "use_qa" in p else _bool(lt.get("use_qa", True))
        qa = self.cfg.datasets_dir / "qa.jsonl"
        return {
            "resume": resume, "base_model": base,
            "max_steps": _int(p.get("max_steps"), lt.get("max_steps", 200), "max_steps", 1, 10_000_000),
            "lr": _float(p.get("lr"), lt.get("lr", 2e-4), "lr", 0.0),
            "seq_len": _int(p.get("seq_len"), lt.get("seq_len", 512), "seq_len", 16, 32768),
            "batch_size": _int(p.get("batch_size"), lt.get("batch_size", 2), "batch_size", 1, 512),
            "grad_accum": _int(p.get("grad_accum"), lt.get("grad_accum", 4), "grad_accum", 1, 1024),
            "lora_r": _int(p.get("lora_r"), lt.get("lora_r", 8), "lora_r", 1, 512),
            "lora_alpha": _int(p.get("lora_alpha"), lt.get("lora_alpha", 16), "lora_alpha", 1, 4096),
            "qa_path": str(qa) if use_qa and qa.is_file() else None, "threads": self._threads(p),
        }

    def _train_lora(self, ctx, p):
        plan = self.plan_lora(p)
        ctx.progress(0.0, "말뭉치 만드는 중")
        corpus = build_corpus(self.db, self.cfg.datasets_dir)
        name, out = self._new_model_dir("lora")
        try:
            info = _mod("train_pc").train_lora(
                f"{corpus['dir']}/train.jsonl", f"{corpus['dir']}/eval.jsonl", out, plan["base_model"],
                max_steps=plan["max_steps"], lr=plan["lr"], seq_len=plan["seq_len"], batch_size=plan["batch_size"],
                grad_accum=plan["grad_accum"], lora_r=plan["lora_r"], lora_alpha=plan["lora_alpha"],
                qa_path=plan["qa_path"], resume_from=plan["resume"]["path"] if plan["resume"] else None,
                threads=plan["threads"], progress=_sub(ctx.progress, 0.03, 1.0), check_cancel=ctx.check_cancel)
        except BaseException:
            shutil.rmtree(out, ignore_errors=True)
            raise
        info = dict(info)
        info["corpus"] = corpus
        info["model_id"] = self.db.add_model("lora", name, str(out), info)
        self.db.notify("success", "LoRA 모델 학습을 마쳤습니다",
                       f"{name} · {info.get('steps')}단계 · 평가 loss {_fmt(info.get('eval_loss_before'))} → "
                       f"{_fmt(info.get('eval_loss_after'))}", {"kind": "train_lora", "model_id": info["model_id"]})
        return info

    def _autolearn_train(self, ctx, p):
        return _mod("autolearn").run_autolearn_train(self, int(p["session_id"]), progress=ctx.progress,
                                                     check_cancel=ctx.check_cancel)

    def _finalize_learning(self, ctx, p):
        res = _mod("autolearn").run_finalize(self, int(p["session_id"]), progress=ctx.progress,
                                             check_cancel=ctx.check_cancel)
        # 작업 기록(DB)에는 파일 전체 목록 대신 화면용 목록(문서별 마크다운은 한 줄)만 남긴다
        slim = dict(res)
        slim["files"] = _mod("results").display_files(res["files"])
        return slim

    def _export(self, ctx, p):
        return export(self.cfg, self.db, str(p["target"]), list(p.get("include", [])), str(p.get("actor", "admin")),
                      progress=ctx.progress, check_cancel=ctx.check_cancel)

    # ---------- 질문 ----------
    def _pick_model(self, model_id, kinds: tuple[str, ...], label: str) -> dict:
        if model_id in (None, "", 0, "0"):
            for m in self.db.list_models():     # 최신순
                if m["kind"] in kinds and Path(m["path"]).is_dir():
                    return m
            raise ValueError(NO_MODEL_MSG[label])
        try:
            m = self.db.get_model(int(model_id))
        except (TypeError, ValueError):
            m = None
        if not m or m["kind"] not in kinds:
            raise ValueError("모델을 찾을 수 없습니다." if not m else
                             f"이 모델은 {'소형 AI' if label == 'tiny' else 'LoRA'} 모델이 아닙니다.")
        if not Path(m["path"]).is_dir():
            raise ValueError(f"모델 폴더가 없습니다: {m['name']}")
        return m

    def _source_rows(self, q: str, contexts: list[dict]) -> list[dict]:
        """답변 근거 조각 -> 검색 결과와 같은 모양의 행 (snippet에 강조 표시)."""
        search = _mod("search")
        pq = search.parse_query(q)
        needles = list(dict.fromkeys(pq["terms"] + pq["phrases"] + search.relaxed_terms(pq)))
        rows = []
        for c in contexts:
            content = c.get("content") or ""
            try:
                snippet = search.make_snippet(content, needles)
            except Exception:
                snippet = content[:200]
            rows.append({"doc_id": c.get("doc_id"), "chunk_id": c.get("chunk_id"), "url": c.get("url"),
                         "title": c.get("title") or "", "source": c.get("source"), "snippet": snippet,
                         "content": content[:1500]})
        return rows

    def ask(self, question: str, engine: str = "search", model_id=None, sources: list[str] | None = None,
            k: int | None = None) -> dict:
        """학습한 자료를 근거로 답한다. engine: search(검색만) | tiny(소형 AI) | lora | online(온라인 모델).

        반환 {"engine","answer","citations","sources","extract","model","note"}. 근거 번호 [n]은 sources의 n번째.
        """
        q = (question or "").strip()
        if not q:
            raise ValueError("질문을 입력하세요.")
        engine = str(engine or "search").strip().lower()
        if engine == "onboard":
            raise ValueError("온보드 모델은 이 기기의 브라우저에서 실행됩니다. 웹 화면에서 '온보드'를 고르세요.")
        if engine not in ENGINES:
            raise ValueError(f"알 수 없는 답변 방식입니다: {engine} (search, tiny, lora, online 중 하나)")
        search = _mod("search")
        src = sources or None
        out = {"engine": engine, "answer": None, "citations": [], "sources": [], "extract": [], "model": None,
               "note": None}

        if engine == "search":
            res = search.search(self.cfg, self.db, q, limit=5, sources=src)
            out["sources"], out["extract"] = res["results"], res.get("extract") or []
            out["answer"], out["citations"] = self._extract_answer(out["extract"], out["sources"])
            out["note"] = (None if out["answer"] else "학습한 자료에서 관련 내용을 찾지 못했습니다.")
            return out

        k = _int(k, self.cfg["online"].get("max_context_chunks", 8) if engine == "online" else 4, "k", 1, 30)
        contexts = search.retrieve(self.cfg, self.db, q, k=k, sources=src)
        out["sources"] = self._source_rows(q, contexts)
        out["extract"] = search.extractive_answer(q, contexts, 3)

        if engine == "online":
            online = _mod("online")
            out["model"] = online.status(self.cfg)["model"]
            try:
                r = online.ask(self.cfg, q, contexts)
            except online.OnlineError as e:
                if e.kind == "refusal":   # 거절은 오류가 아니라 '답이 없음'으로 보여 준다
                    out["note"] = str(e)
                    return out
                raise
            out.update(answer=r["answer"], citations=r["citations"], model=r.get("model") or out["model"])
            out["note"] = "검색된 자료 일부가 Anthropic API로 전송되었습니다."
            return out

        if engine == "lora":
            m = self._pick_model(model_id, LORA_KINDS, "lora")
            out["model"] = m["name"]
            r = _mod("train_pc").answer(q, contexts, (m.get("info") or {}).get("base_model") or None, m["path"])
            out.update(answer=r["answer"], citations=r["citations"])
            out["note"] = "LoRA 모델이 검색된 자료를 근거로 만든 답변입니다. 사실 확인은 출처를 보세요."
            return out

        # tiny: 글자 단위 소형 모델은 질문에 답하지 못하므로, 가장 알맞은 문장에 이어 쓴다 (온보드 모델과 같은 방식)
        m = self._pick_model(model_id, TINY_KINDS, "tiny")
        out["model"] = m["name"]
        best = out["extract"][0] if out["extract"] else None
        prompt = best["sentence"] if best else q
        text = _mod("tinylm").generate(m["path"], prompt, max_new=120)
        if best:
            ans, cites = self._extract_answer([best], out["sources"])
            out["answer"] = f"{ans}\n{text}".strip() if ans else f"{prompt}{text}"
            out["citations"] = cites
            out["note"] = ("소형 AI 모델이 가장 알맞은 문장에 이어 쓴 글입니다. 첫 문장은 자료의 일부이고, 뒤는 모델이 "
                           "지어낸 글이므로 사실 확인은 출처를 보세요.")
        else:
            out["answer"] = f"{prompt}{text}"
            out["note"] = "관련 자료를 찾지 못해 질문에 이어 쓴 글입니다. 사실이 아닐 수 있습니다."
        return out

    @staticmethod
    def _extract_answer(extract: list[dict], sources: list[dict]) -> tuple[str | None, list[dict]]:
        """핵심 문장들을 이어 붙이고 각 문장 뒤에 근거 번호 [n]을 붙인다."""
        by_doc = {}
        for i, s in enumerate(sources, 1):
            by_doc.setdefault(s.get("doc_id"), i)
        parts, cites, seen = [], [], set()
        for e in extract:
            idx = by_doc.get(e.get("doc_id"))
            parts.append(e["sentence"] + (f" [{idx}]" if idx else ""))
            if idx and idx not in seen:
                seen.add(idx)
                s = sources[idx - 1]
                cites.append({"index": idx, "title": s.get("title") or "", "url": s.get("url"),
                              "doc_id": s.get("doc_id"), "chunk_id": s.get("chunk_id")})
        return ("\n".join(parts) if parts else None), cites


def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "-"
