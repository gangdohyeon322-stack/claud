"""엔진 조립: 설정, DB, 수집기, 작업 관리자, 학습/내보내기 핸들러, 질문 답변(검색/소형 AI/LoRA/온라인)."""
from __future__ import annotations

import importlib
import math
import importlib.util
import re
import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from . import accel, checkpoint
from .autolearn import AutoLearner, record_collect
from .collector import collect_links, collect_search
from .config import Config, amp_value, device_value
from .dataset import build_corpus
from .db import DB
from .export import export
from .fetcher import Fetcher
from .jobs import PAUSABLE_KINDS, AutoCollector, Cancelled, JobManager, canonical_kind
from .timegate import describe, is_allowed, next_allowed

ENGINES = ("search", "tiny", "lora", "online")
CKPT_KIND = {"train_tiny": "tiny", "train_lora": "lora"}     # 작업 종류 -> 체크포인트 meta["kind"]
LIVE_STATUSES = ("queued", "running", "paused")
ARCH_KEYS = ("d_model", "n_layer", "n_head", "seq_len", "vocab_size")
TINY_KINDS = ("tiny", "mobile")   # 'mobile'은 예전 이름 (DB가 시작할 때 tiny로 바꾼다)
LORA_KINDS = ("lora", "pc")
NO_MODEL_MSG = {"tiny": "학습한 소형 AI 모델이 없습니다. [학습] 탭에서 소형 AI 모델을 학습하세요.",
                "lora": "학습한 LoRA 모델이 없습니다. [학습] 탭에서 LoRA 모델을 학습하세요."}


class LbwInvalid(ValueError):
    """업로드한 .lbw가 검증을 통과하지 못함. errors: 한국어 사유 목록."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("가중치 파일이 올바르지 않습니다: " + "; ".join(self.errors[:3])
                         + (f" 외 {len(self.errors) - 3}건" if len(self.errors) > 3 else ""))


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


def _nonneg(value, name: str) -> float:
    """0 이상의 숫자 (빈 값은 0)."""
    if value in (None, ""):
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}'이(가) 올바른 숫자가 아닙니다: {value!r}") from None
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"'{name}'은(는) 0 이상이어야 합니다 (입력: {value}).")
    return v


def _set_final(ctx, result: dict) -> None:
    """작업이 정상 반환하지 못했을 때 남길 result (JobContext.final_result)."""
    try:
        ctx.final_result = result
    except AttributeError:
        pass


def _same_path(a, b) -> bool:
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except (OSError, ValueError):
        return str(a) == str(b)


_MODEL_DIR_RE = re.compile(r"^(tiny|lora)-\d{8}-\d{6}(-\d+)?$")    # _new_model_dir가 만드는 이름
_RESUMABLE_HINT = re.compile(r"\s*·\s*이어서 재개할 수 있습니다 \([^)]*\)\s*$")
_RESUMED_BY = re.compile(r"\s*·?\s*작업 #\d+\(으\)로 이어서 재개함\s*$")
STATUS_KO = {"queued": "대기", "running": "진행 중", "paused": "일시정지", "done": "완료", "failed": "실패",
             "cancelled": "취소", "blocked": "시간 밖 중지", "interrupted": "중단"}


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
            "build_chat": self._build_chat,
            "train_tiny": self._train_tiny,
            "train_lora": self._train_lora,
            "autolearn_train": self._autolearn_train,
            "finalize_learning": self._finalize_learning,
            "export": self._export,
        })
        self.jobs.checkpoint_of = self.job_checkpoint             # 재시작 때 '이어서 재개' 안내용
        self.auto = AutoCollector(self.cfg, self.db, self.jobs)   # 설정 auto_collect (단순 예약 수집)
        self.autolearn = AutoLearner(self)                        # 자율 학습 (수집 + 학습 + 학습 종료)
        self._profile_warm = threading.Event()
        self._restart_lock = threading.Lock()

    def start_background(self) -> None:
        self.remove_orphan_model_dirs()     # 작업 스레드를 켜기 전에 (지금 만드는 중인 폴더와 헷갈리지 않게)
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
                devices = _mod("devices")
                prof = devices.pc_profile()
                gpu = prof.get("gpu") or {}
                if gpu.get("type") == "cuda":
                    amd = " · AMD ROCm" if gpu.get("vendor") == "amd" or gpu.get("rocm") else ""
                    caps["device"] = f"cuda ({gpu.get('name') or 'GPU'}{amd})"
                elif gpu.get("type") == "xpu":
                    caps["device"] = f"xpu ({gpu.get('name') or 'Intel GPU'})"
                elif gpu.get("type") == "mps":
                    caps["device"] = "mps (Apple)"
                else:
                    ok, tpu = devices.tpu_usable(prof)
                    if ok:       # GPU가 없고 TPU 런타임 + torch_xla (TPU는 학습을 시작할 때 연다)
                        caps["device"] = f"tpu ({tpu.get('accelerator_type') or 'TPU'})"
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
                "base_models": train_pc.BASE_MODELS, "limits": {k: list(v) for k, v in tinylm.LIMITS.items()},
                "accel": self._accel_devices(pc), "defaults": self.training_defaults()}

    def training_defaults(self) -> dict:
        """학습 폼의 '자동'이 실제로 쓸 값: 환경변수(LEARNBOX_DEVICE, LEARNBOX_AMP) → 설정 training_device,
        lora_training.amp → auto. 학습 요청에서 device/amp를 빼면 서버가 이 순서로 정한다(잘못된 값은 auto)."""
        out = {}
        for key, raw, check, resolve in (
                ("training_device", self.cfg.get("training_device"), device_value, accel.requested_device),
                ("amp", (self.cfg["lora_training"] or {}).get("amp"), amp_value, accel.requested_amp)):
            try:
                cfg_value = check(raw)
            except ValueError:
                cfg_value = "auto"
            try:
                out[key] = check(resolve(None, cfg_value))
            except ValueError:
                out[key] = cfg_value
        return out

    @staticmethod
    def _accel_devices(pc: dict) -> dict:
        """학습 폼의 장치 선택용: 장치별 사용 가능 여부·이름, '자동'이 고를 장치 (TPU는 열지 않고 짐작만)."""
        devices = _mod("devices")
        acc = pc.get("accel") if isinstance(pc.get("accel"), dict) else {}
        avail = acc.get("available") if isinstance(acc.get("available"), dict) else None
        if avail is None:
            try:
                avail = accel.available(probe_tpu=False)
            except Exception:
                avail = {"auto": True, "cpu": True}
        gpu = pc.get("gpu") or {}
        tpu = acc.get("tpu") if isinstance(acc.get("tpu"), dict) else {}
        mismatch = tpu.get("mismatch") or devices.xla_mismatch(acc)
        if mismatch and avail.get("tpu") and not tpu.get("ok"):     # torch·torch_xla 버전 불일치: pick이 CPU를 고른다
            avail = {**avail, "tpu": False}
        rocm = gpu.get("rocm") or (True if acc.get("rocm") else None)
        names = {"cuda": gpu.get("name") if gpu.get("type") == "cuda" else None,
                 "xpu": acc.get("xpu_name") or (gpu.get("name") if gpu.get("type") == "xpu" else None),
                 "mps": gpu.get("name") if gpu.get("type") == "mps" else None,
                 "tpu": tpu.get("accelerator_type")}
        labels = {"auto": "자동", "cuda": "GPU (AMD ROCm)" if rocm else "GPU", "tpu": "TPU", "xpu": "Intel Arc",
                  "mps": "Mac", "cpu": "CPU"}
        out = {}
        for d in accel.DEVICES:
            out[d] = {"available": bool(avail.get(d, d in ("auto", "cpu"))), "label": labels[d], "name": names.get(d)}
        if tpu.get("likely") and not tpu.get("installed"):
            out["tpu"]["note"] = "TPU 패키지(torch_tpu)가 없습니다: python -m learnbox.setup_accel"
        elif tpu.get("likely") and mismatch and not out["tpu"]["available"]:
            out["tpu"]["note"] = mismatch
        auto = next((d for d in ("cuda", "tpu", "xpu", "mps") if out[d]["available"]), "cpu")
        cuda_bf16 = None
        if out["cuda"]["available"]:
            try:
                cuda_bf16 = bool(accel.native_bf16("cuda"))
            except Exception:
                cuda_bf16 = None
        return {"auto": auto, "devices": out, "cuda_bf16": cuda_bf16, "rocm": rocm,
                "torch_version": acc.get("torch_version") or pc.get("torch_version"),
                "torch_xla_version": acc.get("torch_xla_version"),
                "torch_tpu_version": acc.get("torch_tpu_version"),
                "tpu": {**{k: tpu.get(k) for k in ("installed", "likely", "ok", "accelerator_type", "hbm_gb", "backend", "packages")},
                        "mismatch": mismatch or None}}

    def estimate(self, body: dict) -> dict:
        """모델 크기/메모리 예상. body: {"kind":"tiny", preset, d_model…, batch_size} 또는 {"kind":"lora", ...}

        device(선택, auto|cuda|tpu|xpu|mps|cpu): 그 장치 기준 판단을 덧붙인다 (auto·없음이면 예전 결과 그대로)."""
        kind = str(body.get("kind") or "tiny")
        devices, tinylm = _mod("devices"), _mod("tinylm")
        dev = device_value(body.get("device"), "device")
        if kind == "lora":
            lt = self.cfg["lora_training"]
            base = str(body.get("base_model") or lt.get("base_model") or "").strip()
            if not base:
                raise ValueError("기반 모델을 지정하세요.")
            train_pc = _mod("train_pc")
            kw = {"profile": devices.pc_profile()}
            if dev != "auto":       # auto·없음이면 예전 호출 그대로
                kw["device"] = dev
            return train_pc.estimate_lora(
                base, _int(body.get("lora_r"), lt.get("lora_r", 8), "lora_r", 1, 512),
                _int(body.get("seq_len"), lt.get("seq_len", 512), "seq_len", 16, 32768),
                _int(body.get("batch_size"), lt.get("batch_size", 2), "batch_size", 1, 512), **kw)
        if kind != "tiny":
            raise ValueError("kind는 tiny 또는 lora여야 합니다.")
        tt = self.cfg["tiny_training"]
        preset = str(body.get("preset") or "auto").strip().lower()
        arch = tinylm.resolve_arch(preset, {k: body.get(k) for k in ARCH_KEYS},
                                   recommended=devices.recommend()["tiny_preset"] if preset == "auto" else "mini")
        batch = _int(body.get("batch_size"), tt.get("batch_size", 32), "batch_size", 1, 4096)
        out = {"arch": arch, "estimate": tinylm.estimate(arch, batch)}
        if dev == "tpu":     # TPU 메모리(HBM)를 알면 들어감/안 들어감, 모르면 None(측정 필요) (PRD 6.8)
            _, tpu = devices.tpu_usable(devices.pc_profile())
            try:
                hbm = float(tpu.get("total_gb") or tpu.get("hbm_gb") or accel.tpu_hint().get("hbm_gb") or 0) or None
            except (TypeError, ValueError):
                hbm = None
            need = out["estimate"].get("train_ram_mb")
            out["device_fit"] = {"device": "tpu", "mem_gb": hbm,
                                 "fits": None if not (hbm and need is not None) else need <= hbm * 1024 * 0.75}
        return out

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

    def _build_chat(self, ctx, p):
        """대화 학습 데이터(chat.jsonl): 챗봇 이름·말투의 정체성 대화 + (있으면) Q&A를 한 턴 대화로."""
        chat = _mod("chat")
        ctx.progress(0.1, "대화 학습 데이터 만드는 중")
        include_qa = _bool(p["include_qa"]) if "include_qa" in p else True
        qa_rows = _mod("train_pc").load_qa(self.cfg.datasets_dir / "qa.jsonl") if include_qa else []
        r = chat.build_dataset(self.cfg["chatbot"], self.cfg.datasets_dir / "chat.jsonl", qa_rows)
        ctx.progress(1.0, f"대화 {r['dialogs']}개 (정체성 {r['identity']}, Q&A {r['qa']}) · 이름 {r['name']}")
        return r

    def chat_info(self) -> dict:
        """대화 화면용: 페르소나, 대화 데이터 유무, 쓸 수 있는 LoRA 모델."""
        chat = _mod("chat")
        path = self.cfg.datasets_dir / "chat.jsonl"
        models = [{"id": m["id"], "name": m["name"], "kind": m["kind"]} for m in self.db.list_models()
                  if m["kind"] in LORA_KINDS and Path(m["path"]).is_dir()]
        c = self.cfg["chatbot"] or {}
        return {"persona": chat.persona(c), "use_context": bool(c.get("use_context", True)),
                "max_history": int(c.get("max_history") or chat.MAX_HISTORY),
                "dataset": {"exists": path.is_file(), "dialogs": len(chat.load_dialogs(path)) if path.is_file() else 0},
                "models": models}

    def chat(self, messages, engine: str = "lora", model_id=None, use_context=None) -> dict:
        """대화 모드 답변 (docs/PRD-chatbot.md). 페르소나 시스템 메시지 + 이전 대화 + (선택) 학습 자료.

        반환 {"engine","answer","citations","sources","model","persona","note"}. 근거 번호 [n]은 sources의 n번째."""
        chat = _mod("chat")
        c = self.cfg["chatbot"] or {}
        msgs = chat.clean_messages(messages, int(c.get("max_history") or chat.MAX_HISTORY))
        engine = str(engine or "lora").strip().lower()
        if engine != "lora":
            raise ValueError("대화 모드는 지금 LoRA 모델만 지원합니다 (engine: lora).")
        persona = chat.persona(c)
        use_ctx = _bool(use_context) if use_context is not None else bool(c.get("use_context", True))
        contexts = []
        if use_ctx:
            contexts = _mod("search").retrieve(self.cfg, self.db, msgs[-1]["content"], k=3)
        m = self._pick_model(model_id, LORA_KINDS, "lora")
        r = _mod("train_pc").chat(msgs, (m.get("info") or {}).get("base_model") or None, m["path"],
                                  system=chat.system_prompt(persona, with_context=bool(contexts)),
                                  contexts=contexts,
                                  max_new_tokens=_int(c.get("max_new_tokens"), 256, "max_new_tokens", 16, 2048))
        return {"engine": "lora", "answer": r["answer"], "citations": r["citations"],
                "sources": self._source_rows(msgs[-1]["content"], contexts) if contexts else [],
                "model": m["name"], "persona": persona["name"], "note": None}

    def _identity_test(self, base_model: str | None, adapter_dir: str) -> dict:
        """학습 뒤 정체성 시험: 시스템 메시지 없이(모델 기본 지시문) 이름을 말하는 비율. 실패해도 학습은 성공."""
        chat = _mod("chat")
        train_pc = _mod("train_pc")
        name = chat.persona(self.cfg["chatbot"])["name"]
        answers = []
        for q in chat.EVAL_QUESTIONS:
            r = train_pc.chat([{"role": "user", "content": q}], base_model, adapter_dir, system=None, max_new_tokens=48)
            answers.append(r["answer"])
        return {"name": name, "score": round(chat.identity_score(answers, name), 3),
                "questions": list(chat.EVAL_QUESTIONS), "answers": answers}

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

    def remove_orphan_model_dirs(self) -> list[str]:
        """이전 실행이 강제로 끝나(전원 끊김·강제 종료) 정리하지 못한 **빈** 학습 모델 폴더(tiny-*/lora-*)를 지운다.
        등록된 모델 폴더와 내용이 있는 폴더는 건드리지 않는다. 반환: 지운 폴더 이름."""
        removed: list[str] = []
        try:
            registered = {Path(m["path"]).resolve() for m in self.db.list_models() if m.get("path")}
            for d in self.cfg.models_dir.iterdir():
                if (d.is_dir() and _MODEL_DIR_RE.match(d.name) and d.resolve() not in registered
                        and not any(d.iterdir())):
                    d.rmdir()
                    removed.append(d.name)
        except OSError:
            pass
        return removed

    # ---------- 휴대폰 온디바이스 학습 (.lbw) ----------
    def ensure_lbw(self, model_row: dict) -> Path:
        """tiny 모델 폴더의 model.lbw를 돌려준다. 없으면 model.pt에서 만들어 폴더에 캐시한다."""
        d = Path(model_row["path"])
        out = d / "model.lbw"
        if out.is_file() and out.stat().st_mtime >= (d / "model.pt").stat().st_mtime:
            return out
        lbw = _mod("lbw")
        info = model_row.get("info") or {}
        meta = {"parent": model_row.get("name"), "parent_model_id": model_row.get("id"),
                "total_steps": info.get("total_steps"), "train_seq_len": info.get("train_seq_len")}
        data = lbw.lbw_from_model_dir(d, {k: v for k, v in meta.items() if v is not None})
        tmp = out.with_name("model.lbw.tmp")
        tmp.write_bytes(data)
        tmp.replace(out)
        return out

    def register_device_model(self, data: bytes, name_hint: str | None, actor: str) -> dict:
        """기기에서 학습한 .lbw를 검증해 PC 모델로 등록한다 (ONNX 재생성 포함). 실패하면 폴더를 지운다."""
        import json as _json
        import time as _time
        from datetime import timezone

        lbw, tinylm = _mod("lbw"), _mod("tinylm")
        mt = self.cfg["mobile_training"] or {}
        try:
            header, tensors = lbw.read_lbw(data)
        except ValueError as e:
            raise LbwInvalid([str(e)])
        errs = lbw.validate_lbw(header, tensors, int(mt.get("max_params", 2_000_000)))
        if errs:
            raise LbwInvalid(errs)
        import torch   # 없으면 ImportError -> 서버가 pip 안내와 함께 400

        arch = header["arch"]
        V, T = int(arch["vocab_size"]), int(arch["seq_len"])
        name, out = self._new_model_dir("tiny")
        name2 = name + "-device"
        n = 1
        while out.with_name(name2).exists():
            n += 1
            name2 = f"{name}-{n}-device"
        try:
            out2 = out.with_name(name2)
            out.rename(out2)
            out, name = out2, name2
            model = lbw.load_torch_model(header, tensors)
            train_seq = int(header.get("train_seq_len") or T)
            cfg = {"vocab_size": V, "d_model": arch["d_model"], "n_layer": arch["n_layer"],
                   "n_head": arch["n_head"], "seq_len": T, "train_seq_len": train_seq}
            torch.save({"config": cfg, "state_dict": model.state_dict()}, out / "model.pt")
            (out / "vocab.json").write_text(_json.dumps({"itos": header["itos"]}, ensure_ascii=False), encoding="utf-8")
            (out / "model.lbw").write_bytes(data)
            threads = self._threads({})
            onnx_res = tinylm.export_onnx_files(model, out, T, V, int8=True, threads=threads)
            files = {n: (out / n).stat().st_size for n in ("model.onnx", "model.int8.onnx", "model.pt", "vocab.json",
                                                            "model.lbw") if (out / n).exists()}
            params = tinylm.count_params(arch["d_model"], arch["n_layer"], T, V)
            full = {"preset": tinylm._preset_name(arch), **{k: int(arch[k]) for k in ARCH_KEYS}}
            ev_after = header.get("eval_loss_after")
            try:
                ppl = round(math.exp(min(float(ev_after), 50.0)), 3) if ev_after is not None else None
            except (TypeError, ValueError):
                ppl = None
            dev = header.get("device")
            info = {
                "kind": "tiny", "name": name, "preset": full["preset"], "arch": full, "params": int(params),
                "steps": header.get("steps"), "total_steps": header.get("total_steps", header.get("steps")),
                "parent": header.get("parent"), "parent_model_id": header.get("parent_model_id"),
                "trained_on": "device", "device": dev,
                "train_loss": header.get("train_loss"), "eval_loss_before": header.get("eval_loss_before"),
                "eval_loss_after": ev_after, "perplexity_after": ppl,
                "train_chars": header.get("train_chars"), "seconds": header.get("seconds"),
                "vocab_size": V, "seq_len": T, "train_seq_len": train_seq,
                "created_at": header.get("created_at") or datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "uploaded_by": actor, **onnx_res, "files": files,
            }
            if name_hint:
                clean = "".join(c for c in name_hint if c.isprintable())[:60].strip()
                if clean:
                    info["display_name"] = clean
            (out / "info.json").write_text(_json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            mid = self.db.add_model("tiny", name, str(out), info)
        except BaseException:
            shutil.rmtree(out, ignore_errors=True)
            raise
        info["model_id"] = mid
        self.db.notify("success", "휴대폰에서 학습한 모델을 등록했습니다",
                       f"{name} · {info.get('steps')}단계 · 평가 loss {_fmt(info.get('eval_loss_before'))} → "
                       f"{_fmt(ev_after)}", {"kind": "model_upload", "model_id": mid})
        self.db.audit(actor, "model_upload", {"model_id": mid, "name": name, "bytes": len(data),
                                              "steps": info.get("steps")})
        return {"model_id": mid, "name": name, "info": info}

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

    def _train_device(self, p: dict) -> str:
        """학습 장치: 요청 device → LEARNBOX_DEVICE → 설정 training_device → auto (PRD 6.1).
        요청 값이 잘못되면 한국어 ValueError. 환경변수·설정 값은 학습 함수(accel.pick)가 못 쓰면 CPU + 안내로 처리한다."""
        return accel.requested_device(device_value(p.get("device"), "device"), self.cfg.get("training_device"))

    def _train_amp(self, p: dict) -> str:
        """LoRA 정밀도 모드: 요청 amp → LEARNBOX_AMP → 설정 lora_training.amp → auto (PRD 6.4)."""
        return accel.requested_amp(amp_value(p.get("amp"), "amp"), (self.cfg["lora_training"] or {}).get("amp"))

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
            "threads": self._threads(p), "device": self._train_device(p),
        }

    # ---------- 학습 중간 저장 · 일시정지 · 이어서 재개 (docs/SPEC-pause-resume.md) ----------
    def checkpoint_base(self) -> Path:
        """체크포인트 폴더들의 위치: 설정 training_checkpoint.dir, 비어 있으면 데이터 폴더/checkpoints."""
        d = str((self.cfg["training_checkpoint"] or {}).get("dir") or "").strip()
        return Path(d).expanduser().resolve() if d else self.cfg.data_dir / "checkpoints"

    def checkpoint_root(self, job_id) -> Path | None:
        """작업의 체크포인트 root (…/job-<id>). 중간 저장을 껐으면(enabled False) None."""
        if not _bool((self.cfg["training_checkpoint"] or {}).get("enabled", True)):
            return None
        return self.checkpoint_base() / f"job-{int(job_id)}"

    def _checkpoint_kwargs(self, ctx, p: dict) -> tuple[Path | None, dict]:
        """학습 함수에 넘길 중간 저장·일시정지 인자 -> (체크포인트 root, kwargs).

        params.resume_checkpoint가 있으면 그 root를 그대로 계속 쓴다(새 작업 폴더로 옮기지 않음).
        params.checkpoint_minutes(명령줄 --checkpoint-minutes)는 설정의 간격보다 우선 (0이면 시간 기준 저장 끔)."""
        tc = self.cfg["training_checkpoint"] or {}
        enabled = _bool(tc.get("enabled", True))
        given = p.get("checkpoint_minutes")
        minutes = _nonneg(given if given not in (None, "") else tc.get("every_minutes", 10), "checkpoint_minutes")
        every_steps = int(_nonneg(tc.get("every_steps", 0), "every_steps"))
        resume = str(p.get("resume_checkpoint") or "").strip() or None
        job_id = getattr(ctx, "id", None)
        root = Path(resume) if resume else (self.checkpoint_root(job_id) if job_id is not None else None)
        if not enabled:      # 꺼져 있으면 정기 저장은 하지 않는다 (이어서 하는 작업은 일시정지·끝내기 때만 저장)
            minutes, every_steps = 0.0, 0
        kw = {"checkpoint_dir": str(root) if root else None, "checkpoint_every_steps": every_steps,
              "checkpoint_every_sec": minutes * 60.0, "pause_point": getattr(ctx, "pause_point", None),
              "stop_requested": getattr(ctx, "stop_requested", None),
              "on_checkpoint": getattr(ctx, "on_checkpoint", None)}
        if resume:
            kw["resume_checkpoint"] = resume
        return root, kw

    def _run_training(self, ctx, kind: str, root: Path | None, call) -> tuple[str, Path, dict]:
        """모델 폴더를 만들고 call(out)으로 학습한다. 실패하면 모델 폴더는 지우고, 체크포인트는
        취소면 지우고(사용자가 버린 것) 그 밖의 예외면 보존해 작업 result에 남긴다(이어서 재개)."""
        name, out = self._new_model_dir(kind)
        try:
            info = call(out)
        except Cancelled:
            shutil.rmtree(out, ignore_errors=True)
            self._remove_checkpoint(root)
            _set_final(ctx, {"checkpoint": None, "checkpoint_removed": True})
            raise
        except BaseException:
            shutil.rmtree(out, ignore_errors=True)
            try:
                s = checkpoint.summary(root) if root else None
            except OSError:
                s = None
            if s:
                _set_final(ctx, {"checkpoint": s, "resumable": True})
            raise
        return name, out, dict(info)

    @staticmethod
    def _remove_checkpoint(root: Path | None) -> bool:
        """체크포인트 root를 지운다. 체크포인트 폴더 모양(current/prev/.tmp만 있음)일 때만 지운다 (잘못된 경로 보호)."""
        if not root:
            return False
        r = Path(root)
        if not r.is_dir():
            return False
        if any(c.name not in ("current", "prev", ".tmp") for c in r.iterdir()):
            return False
        checkpoint.remove(r)
        return True

    def _resume_meta(self, root: str | None) -> dict:
        try:
            got = checkpoint.load(root) if root else None
        except OSError:
            got = None
        return got[0] if got else {}

    def _train_tiny(self, ctx, p):
        tinylm = _mod("tinylm")
        root, ck = self._checkpoint_kwargs(ctx, p)
        common = {"progress": _sub(ctx.progress, 0.03, 1.0), "check_cancel": ctx.check_cancel, **ck}
        if ck.get("resume_checkpoint"):      # 체크포인트에서 이어서: 기록된 자료·설정을 쓴다 (말뭉치를 새로 만들지 않음)
            ctx.progress(0.0, "체크포인트에서 이어서 학습 준비 중")
            meta = self._resume_meta(ck["resume_checkpoint"])
            corpus = {"dir": str(Path(meta["train_path"]).parent), "resumed": True} if meta.get("train_path") else None
            threads, device = self._threads(p), self._train_device(p)

            def call(out):   # 소형 AI에는 amp를 넘기지 않는다 (정밀도는 장치로 정해짐)
                return tinylm.train(None, None, out, None, threads=threads, device=device, **common)
        else:
            plan = self.plan_tiny(p)
            ctx.progress(0.0, "말뭉치 만드는 중")
            corpus = build_corpus(self.db, self.cfg.datasets_dir)

            def call(out):
                return tinylm.train(
                    f"{corpus['dir']}/train.jsonl", f"{corpus['dir']}/eval.jsonl", out, plan["arch"],
                    steps=plan["steps"], batch_size=plan["batch_size"], lr=plan["lr"],
                    resume_from=plan["resume"]["path"] if plan["resume"] else None, threads=plan["threads"],
                    device=plan["device"], **common)
        name, out, info = self._run_training(ctx, "tiny", root, call)
        info["corpus"] = corpus
        if p.get("resumed_from_job") not in (None, ""):
            info["resumed_from_job"] = p.get("resumed_from_job")
        info["model_id"] = self.db.add_model("tiny", name, str(out), info)
        self._remove_checkpoint(root)       # 완료(또는 지금까지 저장하고 끝내기): 체크포인트는 더 필요 없다
        self.db.notify("success", "소형 AI 모델 학습을 마쳤습니다",
                       f"{name} · {info.get('steps')}단계 · 평가 loss {_fmt(info.get('eval_loss_before'))} → "
                       f"{_fmt(info.get('eval_loss_after'))}" + (" · 일찍 끝냄" if info.get("stopped_early") else ""),
                       {"kind": "train_tiny", "model_id": info["model_id"]})
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
        use_chat = _bool(p["use_chat"]) if "use_chat" in p else _bool(lt.get("use_chat", True))
        qa = self.cfg.datasets_dir / "qa.jsonl"
        chat_file = self.cfg.datasets_dir / "chat.jsonl"
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
            "chat_path": str(chat_file) if use_chat and chat_file.is_file() else None,
            "device": self._train_device(p), "amp": self._train_amp(p),
        }

    def _train_lora(self, ctx, p):
        train_pc = _mod("train_pc")
        root, ck = self._checkpoint_kwargs(ctx, p)
        common = {"progress": _sub(ctx.progress, 0.03, 1.0), "check_cancel": ctx.check_cancel, **ck}
        if ck.get("resume_checkpoint"):      # 기록된 설정이 우선이므로 설정 값은 넘기지 않는다(None)
            ctx.progress(0.0, "체크포인트에서 이어서 학습 준비 중")
            meta = self._resume_meta(ck["resume_checkpoint"])
            corpus = {"dir": str(Path(meta["train_path"]).parent), "resumed": True} if meta.get("train_path") else None
            threads, device, amp = self._threads(p), self._train_device(p), self._train_amp(p)

            def call(out):
                return train_pc.train_lora(
                    None, None, out, None, max_steps=None, lr=None, seq_len=None, batch_size=None, grad_accum=None,
                    lora_r=None, lora_alpha=None, qa_path=None, resume_from=None, threads=threads, device=device,
                    amp=amp, **common)
        else:
            plan = self.plan_lora(p)
            ctx.progress(0.0, "말뭉치 만드는 중")
            corpus = build_corpus(self.db, self.cfg.datasets_dir)

            def call(out):
                return train_pc.train_lora(
                    f"{corpus['dir']}/train.jsonl", f"{corpus['dir']}/eval.jsonl", out, plan["base_model"],
                    max_steps=plan["max_steps"], lr=plan["lr"], seq_len=plan["seq_len"], batch_size=plan["batch_size"],
                    grad_accum=plan["grad_accum"], lora_r=plan["lora_r"], lora_alpha=plan["lora_alpha"],
                    qa_path=plan["qa_path"], resume_from=plan["resume"]["path"] if plan["resume"] else None,
                    threads=plan["threads"], device=plan["device"], amp=plan["amp"],
                    **({"chat_path": plan["chat_path"]} if plan["chat_path"] else {}), **common)
        name, out, info = self._run_training(ctx, "lora", root, call)
        info["corpus"] = corpus
        if info.get("chat_examples"):   # 대화 데이터로 학습했으면 정체성 시험 (PRD-chatbot 5.5)
            ctx.progress(0.99, "정체성 시험 중 (챗봇이 이름을 말하는지)")
            try:
                info["identity_test"] = self._identity_test(info.get("base_model"), str(out))
            except Exception as e:  # 시험 실패는 학습 실패가 아니다
                info["identity_test"] = {"error": f"{type(e).__name__}: {e}"[:200]}
        if p.get("resumed_from_job") not in (None, ""):
            info["resumed_from_job"] = p.get("resumed_from_job")
        info["model_id"] = self.db.add_model("lora", name, str(out), info)
        self._remove_checkpoint(root)
        self.db.notify("success", "LoRA 모델 학습을 마쳤습니다",
                       f"{name} · {info.get('steps')}단계 · 평가 loss {_fmt(info.get('eval_loss_before'))} → "
                       f"{_fmt(info.get('eval_loss_after'))}" + (" · 일찍 끝냄" if info.get("stopped_early") else ""),
                       {"kind": "train_lora", "model_id": info["model_id"]})
        return info

    # ---------- 작업 목록 표시 · 이어서 재개 · 체크포인트 삭제 ----------
    def _job_ckpt_root(self, job: dict) -> Path | None:
        """작업이 쓰는(썼던) 체크포인트 root: result.checkpoint.path → params.resume_checkpoint → 기본 위치/job-<id>."""
        res = job.get("result") if isinstance(job.get("result"), dict) else {}
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        ck = res.get("checkpoint") if isinstance(res.get("checkpoint"), dict) else {}
        root = ck.get("path") or params.get("resume_checkpoint")
        if root:
            return Path(str(root))
        return self.checkpoint_base() / f"job-{int(job['id'])}"

    def job_checkpoint(self, job: dict) -> dict | None:
        """학습 작업의 체크포인트 요약 (디스크에서 읽을 수 있는 것만). 다른 작업이 이어받았으면(resumed_by) None."""
        kind = canonical_kind(str(job.get("kind") or ""))
        if kind not in PAUSABLE_KINDS:
            return None
        res = job.get("result") if isinstance(job.get("result"), dict) else {}
        if res.get("resumed_by"):
            return None
        root = self._job_ckpt_root(job)
        try:
            s = checkpoint.summary(root) if root else None
        except OSError:
            s = None
        if s and s.get("kind") not in (None, CKPT_KIND[kind]):
            return None
        return s

    def annotate_job(self, job: dict) -> dict:
        """작업 목록용: checkpoint 요약과 can_pause/can_resume/can_finish/can_restart (+요청 상태)를 붙인다."""
        j = dict(job)
        kind = canonical_kind(str(j.get("kind") or ""))
        train = kind in PAUSABLE_KINDS
        st = j.get("status")
        jid = j.get("id")
        res = j["result"] if isinstance(j.get("result"), dict) else {}
        ck = self.job_checkpoint(j) if train and st in ("running", "paused", "interrupted", "failed") else None
        active = train and self.jobs.is_active(jid)
        pause_req = active and jid in self.jobs.pause_flags
        finish_req = active and jid in self.jobs.finish_flags
        j.update(
            checkpoint=ck, pause_requested=bool(pause_req), finish_requested=bool(finish_req),
            can_pause=bool(active and st == "running" and not pause_req and not finish_req),
            can_resume=bool(active and (st == "paused" or pause_req) and not finish_req),
            can_finish=bool(active and st in ("running", "paused") and not finish_req),
            can_restart=bool(train and st in ("interrupted", "failed") and ck and not res.get("resumed_by")),
        )
        return j

    def list_jobs(self, limit: int = 40) -> list[dict]:
        return [self.annotate_job(j) for j in self.db.list_jobs(limit)]

    def _live_root_user(self, root: Path, exclude: int | None = None) -> dict | None:
        """이 체크포인트 root를 쓰고 있는 대기/실행/일시정지 중 학습 작업."""
        with self.db.conn() as c:
            rows = [self.db._job(r) for r in c.execute(
                "SELECT * FROM jobs WHERE status IN ('queued','running','paused') AND kind IN "
                "('train_tiny','train_lora','train_mobile','train_pc')")]
        for j in rows:
            if j["id"] != exclude and _same_path(self._job_ckpt_root(j), root):
                return j
        return None

    def restart_job(self, job_id, actor: str = "admin", enqueue: bool = True, device=None, amp=None) -> int:
        """중단/실패한 학습 작업을 체크포인트에서 이어서 하는 새 작업을 만든다 -> 새 작업 번호.

        새 작업 params = 원래 params + resume_checkpoint(원래 root 그대로) + resumed_from_job.
        장치(device)는 이어받지 않는다: 기본은 지금 장치(요청 → LEARNBOX_DEVICE → 설정)에서 재개하고,
        device를 주면 그 장치로 (PRD 6.9). 정밀도(amp)는 이어받고, amp를 주면 그 값으로 바꾼다
        (예: fp16 nan으로 멈춘 작업을 amp="off"로 이어서).
        enqueue=False면 줄에 넣지 않는다 (명령줄이 현재 스레드에서 직접 실행)."""
        dev = device_value(device, "device")     # 잘못된 값이면 새 작업을 만들기 전에 ValueError
        amp_new = amp_value(amp, "amp") if amp not in (None, "") else None
        with self._restart_lock:
            try:
                job = self.db.get_job(int(job_id))
            except (TypeError, ValueError):
                job = None
            if not job:
                raise ValueError("작업이 없습니다.")
            kind = canonical_kind(job["kind"])
            if kind not in PAUSABLE_KINDS:
                raise ValueError("소형 AI·LoRA 학습 작업만 이어서 재개할 수 있습니다.")
            st = job["status"]
            if st == "paused":
                raise ValueError("일시정지된 작업입니다. '재개'를 누르세요.")
            if st in ("queued", "running"):
                raise ValueError("아직 진행 중인 작업입니다.")
            if st not in ("interrupted", "failed"):
                raise ValueError(f"중단되었거나 실패한 작업만 이어서 재개할 수 있습니다 (지금 상태: {STATUS_KO.get(st, st)}).")
            res = dict(job["result"]) if isinstance(job.get("result"), dict) else {}
            if res.get("resumed_by"):
                raise ValueError(f"이미 작업 #{res['resumed_by']}(으)로 이어서 재개했습니다. 그 작업에서 이어서 하세요.")
            root = self._job_ckpt_root(job)
            got = checkpoint.load(root) if root else None
            if not got:
                raise ValueError("이어서 할 체크포인트가 없습니다. 처음부터 다시 학습하세요.")
            if got[0].get("kind") != CKPT_KIND[kind]:
                raise ValueError("이 작업의 체크포인트가 아닙니다 (종류가 다릅니다). 이어서 할 수 없습니다.")
            other = self._live_root_user(root, exclude=job["id"])
            if other:
                raise ValueError(f"작업 #{other['id']}이(가) 이미 이 체크포인트로 학습 중입니다.")
            params = {**(job.get("params") if isinstance(job.get("params"), dict) else {}),
                      "resume_checkpoint": str(root), "resumed_from_job": job["id"]}
            params.pop("device", None)
            if dev != "auto":
                params["device"] = dev
            if amp_new is not None:
                params["amp"] = amp_new
            # 작업 목록의 요청자는 다른 작업처럼 역할만 ('admin@127.0.0.1' -> 'admin'); 감사 기록에는 그대로 남긴다
            owner = str(actor).split("@", 1)[0] or str(actor)
            new_id = self.jobs.submit(kind, params, owner) if enqueue else self.db.create_job(kind, params, owner)
            res["resumed_by"] = new_id
            # 원래 작업의 '이어서 재개할 수 있습니다' 안내를 '작업 #N(으)로 이어서 재개함'으로 바꾼다 (실패 사유는 그대로)
            base = _RESUMABLE_HINT.sub("", str(job.get("message") or ""))
            msg = (f"{base} · " if base else "") + f"작업 #{new_id}(으)로 이어서 재개함"
            self.db.update_job(job["id"], result=res, message=msg[:500])
        self.db.audit(actor, "job_restart", {"job_id": job["id"], "new_job_id": new_id, "step": got[0].get("step"),
                                             "steps": got[0].get("steps"),
                                             **({"device": dev} if dev != "auto" else {}),
                                             **({"amp": amp_new} if amp_new is not None else {})})
        return new_id

    def discard_checkpoint(self, job_id, actor: str = "admin") -> dict:
        """중단/실패한 학습 작업의 체크포인트를 지운다 (이어서 재개할 수 없게 됨)."""
        try:
            job = self.db.get_job(int(job_id))
        except (TypeError, ValueError):
            job = None
        if not job:
            raise ValueError("작업이 없습니다.")
        if canonical_kind(job["kind"]) not in PAUSABLE_KINDS:
            raise ValueError("학습 작업이 아닙니다.")
        if job["status"] in LIVE_STATUSES:
            raise ValueError("진행 중인 작업의 체크포인트는 지울 수 없습니다. 먼저 취소하거나 끝내세요.")
        res = dict(job["result"]) if isinstance(job.get("result"), dict) else {}
        if res.get("resumed_by"):
            raise ValueError(f"이 체크포인트는 작업 #{res['resumed_by']}이(가) 이어받았습니다. 그 작업에서 지우세요.")
        root = self._job_ckpt_root(job)
        other = self._live_root_user(root, exclude=job["id"]) if root else None
        if other:
            raise ValueError(f"작업 #{other['id']}이(가) 이 체크포인트로 학습 중이라 지울 수 없습니다.")
        s = checkpoint.summary(root) if root else None
        if not (root and Path(root).is_dir()):
            raise ValueError("지울 체크포인트가 없습니다.")
        if not self._remove_checkpoint(root):
            raise ValueError(f"체크포인트 폴더가 아닌 것 같아 지우지 않았습니다: {root}")
        res.update(checkpoint=None, checkpoint_discarded=True)
        res.pop("resumable", None)
        # '이어서 재개할 수 있습니다 (N/M단계)' 안내가 남아 있으면 지우고 삭제했다고 적는다 (사유는 그대로)
        base = _RESUMABLE_HINT.sub("", str(job.get("message") or ""))
        msg = (f"{base} · " if base else "") + "체크포인트 삭제함"
        self.db.update_job(job["id"], result=res, message=msg[:500])
        self.db.audit(actor, "checkpoint_discard", {"job_id": job["id"], "step": (s or {}).get("step"),
                                                    "bytes": (s or {}).get("bytes")})
        return {"ok": True, "bytes": (s or {}).get("bytes") or 0}

    def cancel_job(self, job_id: int) -> None:
        """작업 취소 (일시정지 중이어도 깨워서 취소). 시작 전에 취소한 '이어서 재개' 작업은 원래 작업이
        다시 이어서 재개할 수 있게 되돌린다 (체크포인트는 그대로)."""
        before = self.db.get_job(job_id)
        self.jobs.cancel(job_id)
        if not before or before["status"] != "queued" or canonical_kind(before["kind"]) not in PAUSABLE_KINDS:
            return
        after = self.db.get_job(job_id)
        src = (before.get("params") or {}).get("resumed_from_job") if isinstance(before.get("params"), dict) else None
        if not src or not after or after["status"] != "cancelled":
            return
        try:
            orig = self.db.get_job(int(src))
        except (TypeError, ValueError):
            orig = None
        res = orig.get("result") if orig and isinstance(orig.get("result"), dict) else None
        if res is not None and res.get("resumed_by") == job_id:
            res = dict(res)
            res.pop("resumed_by")
            self.db.update_job(orig["id"], result=res)
            orig = self.db.get_job(orig["id"])
            base = _RESUMED_BY.sub("", str(orig.get("message") or ""))
            ck = self.job_checkpoint(orig)
            if ck:
                base = (f"{base} · " if base else "") + f"이어서 재개할 수 있습니다 ({ck.get('step')}/{ck.get('steps')}단계)"
            self.db.update_job(orig["id"], message=base[:500])

    def suspend_training(self, timeout: float = 120.0, reason: str = "서버를 종료해 중단됨") -> list[int]:
        """이 프로세스에서 실행 중인 학습 작업을 저장하고 '중단됨'으로 멈춘다 (서버 종료 전). 반환: 요청한 작업 번호."""
        ids = [i for i in self.jobs.active_training() if self.jobs.interrupt(i, reason)]
        end = time.monotonic() + max(0.0, float(timeout))
        while ids and time.monotonic() < end and any(self.jobs.is_active(i) for i in ids):
            time.sleep(0.2)
        return ids

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
            k: int | None = None, history: list[dict] | None = None) -> dict:
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
                r = online.ask(self.cfg, q, contexts, history=history)  # 이전 대화(대화 모드)는 온라인 모델만 받는다
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
