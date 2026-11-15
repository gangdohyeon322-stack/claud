"""백그라운드 작업 관리. 수집(collect) · 가벼운 작업(light) · 무거운 작업(heavy: 학습/내보내기)을 따로 줄 세워 동시에 돈다."""
from __future__ import annotations

import queue
import threading
import time
import traceback

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


def canonical_kind(kind: str) -> str:
    return ALIASES.get(kind, kind)


def missing_package_message(e: ImportError) -> str:
    """선택 패키지(torch, transformers, onnxruntime, anthropic …)가 없을 때 사용자에게 보여 줄 한국어 안내."""
    name = (getattr(e, "name", None) or str(e)).split(".")[0]
    return f"필요한 패키지가 설치되어 있지 않습니다: {name} (pip install -r requirements-train.txt)"


class Cancelled(Exception):
    pass


class JobContext:
    def __init__(self, manager: "JobManager", job_id: int):
        self.m, self.id = manager, job_id
        self._last = 0.0

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


class JobManager:
    def __init__(self, db: DB, handlers: dict):
        self.db = db
        self.handlers = handlers  # kind -> fn(ctx, params) -> result dict
        self.cancel_flags: set[int] = set()
        self.echo = False  # CLI에서 진행 상황 출력
        self.queues = {lane: queue.Queue() for lane in set(LANES.values())}
        self._threads: list[threading.Thread] = []

    def start(self, mark_interrupted: bool = True) -> None:
        """줄마다 작업 스레드를 하나씩 시작한다. 이전 실행에서 중단된 작업은 '중단됨'으로 표시한다."""
        if self._threads:
            return
        if mark_interrupted:
            self.db.mark_interrupted()
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

    def cancel(self, job_id: int) -> None:
        self.cancel_flags.add(job_id)
        job = self.db.get_job(job_id)
        if job and job["status"] == "queued":
            self.db.update_job(job_id, status="cancelled", finished_at=now_iso())

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
        try:
            result = self.handlers[canonical_kind(job["kind"])](ctx, job["params"] or {})
            status = "blocked" if isinstance(result, dict) and result.get("stopped") else "done"
            msg = result.get("stopped") if status == "blocked" else "완료"
            self.db.update_job(job_id, status=status, progress=1.0, message=msg, result=result, finished_at=now_iso())
        except Cancelled:
            self.db.update_job(job_id, status="cancelled", message="취소됨", finished_at=now_iso())
        except Exception as e:
            # 사용자에게 보여 줄 한국어 메시지(ValueError, OnlineError)는 그대로, 그 밖의 오류는 종류를 앞에 붙인다
            plain = isinstance(e, ValueError) or type(e).__name__ == "OnlineError"
            msg = str(e) if plain and str(e) else f"{type(e).__name__}: {e}"
            if isinstance(e, ImportError):
                msg = missing_package_message(e)
            self.db.update_job(
                job_id, status="failed", message=msg[:500],
                result={"traceback": traceback.format_exc()[-4000:]}, finished_at=now_iso(),
            )
        finally:
            self.cancel_flags.discard(job_id)


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
