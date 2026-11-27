"""백그라운드 작업 관리. 수집(collect) · 가벼운 작업(light) · 무거운 작업(heavy: 학습/내보내기)을 따로 줄 세워 동시에 돈다."""
from __future__ import annotations

import queue
import threading
import time
import traceback
from datetime import datetime

from .db import DB, now_iso
from .timegate import is_allowed

# 작업 종류 -> 줄(lane). 같은 줄의 작업은 차례로, 다른 줄의 작업은 동시에 돈다.
#  collect: 인터넷 수집 / light: 파일 가져오기·색인·Q&A 데이터 / heavy: 학습·학습 종료·내보내기
LANES = {
    "collect_links": "collect",
    "collect_search": "collect",
    "ingest_path": "light",
    "index_embeddings": "light",
    "build_qa": "light",
    "build_chat": "light",
    "train_tiny": "heavy",
    "train_lora": "heavy",
    "autolearn_train": "heavy",
    "finalize_learning": "heavy",
    "export": "heavy",
    # 이전 이름 (submit이 새 이름으로 바꿔 저장한다)
    "train_mobile": "heavy",
    "train_pc": "heavy",
}
ALIASES = {"train_mobile": "train_tiny", "train_pc": "train_lora"}
# 일시정지 · 지금까지 저장하고 끝내기 · 체크포인트에서 이어서 재개를 지원하는 작업 (docs/SPEC-pause-resume.md)
PAUSABLE_KINDS = ("train_tiny", "train_lora")
WAIT_SEC = 0.5     # 일시정지 중 재개/취소/끝내기 요청을 확인하는 간격


def canonical_kind(kind: str) -> str:
    return ALIASES.get(kind, kind)


def _local_hm(iso: str | None) -> str:
    """저장 시각(UTC ISO) -> 이 PC 시각 'HH:MM' (읽지 못하면 원문)."""
    if not iso:
        return "-"
    try:
        return datetime.fromisoformat(str(iso)).astimezone().strftime("%H:%M")
    except ValueError:
        return str(iso)


def checkpoint_steps(ck: dict | None) -> str:
    """체크포인트 요약 -> 'N/M' (단계 정보가 없으면 '')."""
    if not isinstance(ck, dict) or ck.get("step") is None:
        return ""
    return f"{ck['step']}/{ck.get('steps') if ck.get('steps') is not None else '?'}"


def default_checkpoint_of(job: dict) -> dict | None:
    """작업 행 -> 디스크에 남아 있는 체크포인트 요약 (result.checkpoint.path 또는 params.resume_checkpoint 기준)."""
    from . import checkpoint

    res = job.get("result") if isinstance(job.get("result"), dict) else {}
    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    ck = res.get("checkpoint") if isinstance(res.get("checkpoint"), dict) else {}
    root = ck.get("path") or params.get("resume_checkpoint")
    return checkpoint.summary(root) if root else None


def missing_package_message(e: ImportError) -> str:
    """선택 패키지(torch, transformers, onnxruntime, anthropic …)가 없을 때 사용자에게 보여 줄 한국어 안내."""
    name = (getattr(e, "name", None) or str(e)).split(".")[0]
    return f"필요한 패키지가 설치되어 있지 않습니다: {name} (pip install -r requirements-train.txt)"


class Cancelled(Exception):
    pass


class Interrupted(Exception):
    """저장한 뒤 멈춤 (나중에 체크포인트에서 이어서 재개). 작업 상태는 interrupted가 된다. 메시지는 사유."""


class JobContext:
    def __init__(self, manager: "JobManager", job_id: int):
        self.m, self.id = manager, job_id
        self._last = 0.0
        self.last_checkpoint: dict | None = None   # on_checkpoint로 받은 마지막 요약
        # 핸들러가 정상 반환하지 못했을 때(취소/중단/실패) 작업 result로 남길 값 (예: 보존한 체크포인트 요약)
        self.final_result: dict | None = None

    def check_cancel(self) -> None:
        if self.id in self.m.cancel_flags:
            raise Cancelled()

    def progress(self, p: float, message: str = "") -> None:
        now = time.monotonic()
        if now - self._last < 0.5 and p < 1:  # DB 쓰기 과다 방지
            return
        self._last = now
        if self.m.echo:
            print(f"  {p * 100:5.1f}% {message}", flush=True)
        self.m.db.update_job(self.id, progress=round(max(0.0, min(1.0, p)), 4), message=message[:500])

    # ----- 일시정지 · 끝내기 · 중간 저장 (학습 함수의 pause_point / stop_requested / on_checkpoint) -----
    def stop_requested(self) -> bool:
        """'지금까지 저장하고 끝내기' 요청이 있으면 True."""
        return self.id in self.m.finish_flags

    def on_checkpoint(self, summary: dict) -> None:
        """체크포인트를 쓸 때마다: 작업 result에 마지막 저장 정보를 남긴다 (완료 시 최종 result가 덮어씀)."""
        ck = dict(summary or {})
        self.last_checkpoint = ck
        self.m.db.update_job(self.id, result={"checkpoint": ck})

    def pause_point(self, save) -> None:
        """학습 함수가 매 단계 끝에 부른다. 일시정지 요청이 있으면 save()로 저장하고, 재개/취소/끝내기가 올 때까지 기다린다.

        취소면 Cancelled, 저장 후 중단(interrupt)이면 Interrupted를 올린다. 끝내기면 플래그를 둔 채 돌아온다."""
        m, jid = self.m, self.id
        if jid not in m.pause_flags and jid not in m.interrupt_flags:
            return
        if jid in m.cancel_flags:
            raise Cancelled()
        if jid in m.finish_flags and jid not in m.interrupt_flags:
            m.pause_flags.discard(jid)       # 끝내는 중이면 멈추지 않는다 (학습 함수가 stop_requested로 마무리)
            return
        summary = save() if callable(save) else None
        if jid in m.interrupt_flags:
            raise Interrupted(m.interrupt_flags.get(jid) or "중단됨")
        if summary:
            msg = f"일시정지됨 · {checkpoint_steps(summary)}단계 · {_local_hm(summary.get('saved_at'))} 저장"
        else:
            msg = "일시정지됨 · 체크포인트 없음"
        m.db.update_job(jid, status="paused", message=msg)
        if m.echo:
            print(f"  {msg}", flush=True)
        while True:
            if jid in m.cancel_flags:
                raise Cancelled()
            if jid in m.interrupt_flags:
                raise Interrupted(m.interrupt_flags.get(jid) or "중단됨")
            if jid in m.finish_flags:
                m.pause_flags.discard(jid)
                m.db.update_job(jid, status="running", message="지금까지 저장하고 끝내는 중")
                return
            if jid not in m.pause_flags:
                m.db.update_job(jid, status="running", message="다시 시작합니다")
                return
            with m._cond:
                m._cond.wait(WAIT_SEC)


class JobManager:
    def __init__(self, db: DB, handlers: dict):
        self.db = db
        self.handlers = handlers  # kind -> fn(ctx, params) -> result dict
        self.cancel_flags: set[int] = set()
        self.pause_flags: set[int] = set()       # 일시정지 요청 (다음 pause_point에서 저장하고 멈춤)
        self.finish_flags: set[int] = set()      # 지금까지 저장하고 끝내기
        self.interrupt_flags: dict[int, str] = {}   # 저장하고 멈춘 뒤 '중단됨'으로 끝내기 (명령줄 Ctrl+C, 서버 종료)
        self.active: dict[int, JobContext] = {}  # 이 프로세스에서 실행 중인 작업
        # 작업 행 -> 디스크에 남은 체크포인트 요약 (App이 체크포인트 폴더 규칙을 아는 함수로 바꿔 끼운다)
        self.checkpoint_of = default_checkpoint_of
        self.echo = False  # CLI에서 진행 상황 출력
        self.queues = {lane: queue.Queue() for lane in set(LANES.values())}
        self._threads: list[threading.Thread] = []
        self._cond = threading.Condition()

    def start(self, mark_interrupted: bool = True) -> None:
        """줄마다 작업 스레드를 하나씩 시작한다. 이전 실행에서 중단된 작업은 '중단됨'으로 표시하고,
        체크포인트가 남은 학습 작업은 '이어서 재개할 수 있다'고 알린다."""
        if self._threads:
            return
        if mark_interrupted:
            for job in self.db.mark_interrupted(self.checkpoint_of):
                ck = job.get("checkpoint")
                if canonical_kind(job["kind"]) in PAUSABLE_KINDS and ck:
                    self.db.notify("warning", "학습이 중단되었습니다",
                                   f"작업 탭에서 '이어서 재개'를 누르면 {checkpoint_steps(ck)}단계부터 계속합니다.",
                                   {"kind": "train_interrupted", "job_id": job["id"]})
        for lane, q in self.queues.items():
            t = threading.Thread(target=self._worker, args=(q,), name=f"lane-{lane}", daemon=True)
            t.start()
            self._threads.append(t)

    def submit(self, kind: str, params: dict, actor: str) -> int:
        kind = canonical_kind(kind)
        if kind not in self.handlers:
            raise ValueError(f"알 수 없는 작업: {kind}")
        job_id = self.db.create_job(kind, params, actor)
        self.queues[LANES[kind]].put(job_id)
        return job_id

    def _wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def cancel(self, job_id: int) -> None:
        self.cancel_flags.add(job_id)
        job = self.db.get_job(job_id)
        if job and job["status"] == "queued":
            self.db.update_job(job_id, status="cancelled", finished_at=now_iso())
        self._wake()   # 일시정지 중인 작업도 깨워서 취소되게

    # ----- 일시정지 · 재개 · 끝내기 -----
    def is_active(self, job_id: int) -> bool:
        return job_id in self.active

    def _pausable(self, job_id: int, what: str) -> dict:
        job = self.db.get_job(job_id)
        if not job:
            raise ValueError("작업이 없습니다.")
        if canonical_kind(job["kind"]) not in PAUSABLE_KINDS:
            raise ValueError(f"소형 AI·LoRA 학습 작업만 {what} 수 있습니다.")
        if job["status"] == "queued":
            raise ValueError(f"아직 시작하지 않은 작업은 {what} 수 없습니다. 시작한 뒤에 다시 시도하거나 취소하세요.")
        if job["status"] not in ("running", "paused"):
            raise ValueError(f"실행 중인 작업이 아닙니다 (상태: {job['status']}).")
        if job_id not in self.active:
            raise ValueError("이 작업은 다른 곳(명령줄 등)에서 실행 중이라 여기서 바꿀 수 없습니다.")
        return job

    def pause(self, job_id: int) -> None:
        """실행 중인 학습 작업 일시정지 요청: 지금 단계를 마치면 저장하고 멈춘다."""
        job = self._pausable(job_id, "일시정지할")
        if job["status"] == "paused":
            raise ValueError("이미 일시정지된 작업입니다.")
        if job_id in self.finish_flags:
            raise ValueError("지금까지 저장하고 끝내는 중이라 일시정지할 수 없습니다.")
        if job_id in self.pause_flags:
            raise ValueError("이미 일시정지를 요청했습니다. 지금 단계를 마치면 저장하고 멈춥니다.")
        self.pause_flags.add(job_id)
        self.db.update_job(job_id, message="일시정지 요청됨 · 지금 단계를 마치면 저장하고 멈춥니다")
        self._wake()

    def resume(self, job_id: int) -> None:
        """일시정지된(또는 일시정지를 요청한) 작업을 다시 진행한다."""
        job = self._pausable(job_id, "재개할")
        if job["status"] != "paused" and job_id not in self.pause_flags:
            raise ValueError("일시정지된 작업이 아닙니다.")
        self.pause_flags.discard(job_id)
        self._wake()

    def finish(self, job_id: int) -> None:
        """'지금까지 저장하고 끝내기': 남은 단계를 건너뛰고 지금까지의 결과로 모델을 마무리한다."""
        self._pausable(job_id, "끝낼")
        if job_id in self.finish_flags:
            raise ValueError("이미 끝내는 중입니다.")
        self.finish_flags.add(job_id)
        self.db.update_job(job_id, message="지금까지 저장하고 끝내는 중")
        self._wake()

    def interrupt(self, job_id: int, reason: str = "중단됨") -> bool:
        """저장하고 멈춘 뒤 '중단됨'으로 끝낸다 (나중에 이어서 재개). 이 프로세스에서 실행 중인 학습 작업만. 반환: 요청했는지."""
        job = self.db.get_job(job_id)
        if not job or canonical_kind(job["kind"]) not in PAUSABLE_KINDS or job_id not in self.active:
            return False
        self.interrupt_flags[job_id] = reason
        self._wake()
        return True

    def active_training(self) -> list[int]:
        return [i for i in list(self.active) if (j := self.db.get_job(i))
                and canonical_kind(j["kind"]) in PAUSABLE_KINDS]

    def run_inline(self, kind: str, params: dict, actor: str = "cli") -> dict:
        """CLI용: 같은 핸들러를 현재 스레드에서 실행."""
        kind = canonical_kind(kind)
        if kind not in self.handlers:
            raise ValueError(f"알 수 없는 작업: {kind}")
        job_id = self.db.create_job(kind, params, actor)
        self.echo = True
        self._run(job_id)
        return self.db.get_job(job_id)

    def run_existing(self, job_id: int) -> dict | None:
        """CLI용: 이미 만들어진(queued) 작업을 현재 스레드에서 끝까지 실행하고 진행 상황을 출력한다."""
        self.echo = True
        self._run(job_id)
        return self.db.get_job(job_id)

    def _worker(self, q: queue.Queue) -> None:
        while True:
            job_id = q.get()
            try:
                self._run(job_id)
            finally:
                q.task_done()

    def _run(self, job_id: int) -> None:
        job = self.db.get_job(job_id)
        if not job or job["status"] != "queued":
            return
        self.db.update_job(job_id, status="running", started_at=now_iso())
        ctx = JobContext(self, job_id)
        self.active[job_id] = ctx
        try:
            result = self.handlers[canonical_kind(job["kind"])](ctx, job["params"] or {})
            status = "blocked" if isinstance(result, dict) and result.get("stopped") else "done"
            msg = result.get("stopped") if status == "blocked" else "완료"
            self.db.update_job(job_id, status=status, progress=1.0, message=msg, result=result, finished_at=now_iso())
        except Cancelled:
            extra = {"result": ctx.final_result} if ctx.final_result is not None else {}
            self.db.update_job(job_id, status="cancelled", message="취소됨", finished_at=now_iso(), **extra)
        except Interrupted as e:
            ck = (ctx.final_result or {}).get("checkpoint")
            msg = (str(e) or "중단됨") + (f" · 이어서 재개할 수 있습니다 ({checkpoint_steps(ck)}단계)" if ck else "")
            extra = {"result": ctx.final_result} if ctx.final_result is not None else {}
            self.db.update_job(job_id, status="interrupted", message=msg[:500], finished_at=now_iso(), **extra)
        except Exception as e:
            # 사용자에게 보여 줄 한국어 메시지(ValueError, OnlineError)는 그대로, 그 밖의 오류는 종류를 앞에 붙인다
            plain = isinstance(e, ValueError) or type(e).__name__ == "OnlineError"
            msg = str(e) if plain and str(e) else f"{type(e).__name__}: {e}"
            if isinstance(e, ImportError):
                msg = missing_package_message(e)
            self.db.update_job(
                job_id, status="failed", message=msg[:500],
                result={**(ctx.final_result or {}), "traceback": traceback.format_exc()[-4000:]}, finished_at=now_iso(),
            )
        finally:
            self.active.pop(job_id, None)
            self.cancel_flags.discard(job_id)
            self.pause_flags.discard(job_id)
            self.finish_flags.discard(job_id)
            self.interrupt_flags.pop(job_id, None)


class AutoCollector(threading.Thread):
    """허용 시간대에, 일정 간격으로 예약 목록을 하나씩 조금씩 수집한다 (설정 auto_collect, 학습은 하지 않음).

    자율 학습 세션이 진행 중이면 같은 일을 자율 학습이 하므로 이 수집기는 쉰다."""

    def __init__(self, cfg, db: DB, jobs: JobManager):
        super().__init__(name="auto-collect", daemon=True)
        self.cfg, self.db, self.jobs = cfg, db, jobs
        self.last_run = 0.0

    def run(self) -> None:
        while True:
            time.sleep(30)
            try:
                self.tick()
            except Exception:
                traceback.print_exc()

    def tick(self) -> int | None:
        ac = self.cfg["auto_collect"]
        if not ac.get("enabled"):
            return None
        if time.time() - self.last_run < float(ac.get("interval_min", 60)) * 60:
            return None
        if self.db.active_session():
            return None
        if not is_allowed(self.cfg["allowed_windows"], self.cfg["timezone"]):
            return None
        if self.db.running_jobs(("collect_links", "collect_search")):
            return None
        item = self.db.next_watch()
        if not item:
            return None
        self.last_run = time.time()
        self.db.touch_watch(item["id"])
        if item["kind"] == "search":
            return self.jobs.submit("collect_search", {"query": item["value"]}, "auto")
        return self.jobs.submit("collect_links", {"urls": [item["value"]], "depth": item["depth"]}, "auto")
