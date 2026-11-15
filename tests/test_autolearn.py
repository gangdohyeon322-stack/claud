"""자율 학습(autolearn) 테스트: 네트워크 없이, 무거운 학습 모듈은 가짜(stub)로 바꿔 빠르게 돈다.

실행: python -m unittest tests.test_autolearn -v
"""
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from learnbox import autolearn
from learnbox.autolearn import AutoLearner, discover_topics, record_collect, run_autolearn_train, run_finalize
from learnbox.config import Config
from learnbox.db import DB
from learnbox.export import verify

HAS_TORCH = importlib.util.find_spec("torch") is not None

KO_DOCS = [
    ("인공지능 - 위키백과", "인공지능은 사람의 학습 능력을 컴퓨터로 구현한 기술이다. 인공지능의 연구 분야는 넓다. "
     "기계학습은 인공지능을 구현하는 방법이며 신경망을 사용한다. 신경망은 여러 층으로 이루어진다. "
     "2024년 3000개의 신경망 모델이 공개되었다. 기계학습에서 데이터가 중요하다."),
    ("기계학습 입문", "기계학습은 데이터에서 규칙을 배우는 방법이다. 기계학습의 예로 분류와 회귀가 있다. "
     "신경망을 쓰는 기계학습을 딥러닝이라고 한다. 딥러닝은 인공지능 발전을 이끌었다. 딥러닝에서는 GPU가 쓰인다."),
    ("김치찌개 만들기", "김치찌개는 한국의 대표 음식이다. 김치찌개에 돼지고기와 두부를 넣는다. "
     "김치찌개를 끓일 때 김치가 잘 익어야 한다. 두부는 마지막에 넣는다."),
]


def add_docs(db, docs=KO_DOCS, source="link", prefix="https://ex.com/d"):
    for i, (title, text) in enumerate(docs):
        db.add_document(f"{prefix}{i}", title, (text + "\n") * 3, source)


def backdate_docs(db, iso="2020-01-01T00:00:00+00:00"):
    """시각이 초 단위라 빠른 테스트에서는 '같은 초'가 생긴다. 기존 문서를 과거로 돌려 기준 시각과 겹치지 않게 한다."""
    with db.conn() as c:
        c.execute("UPDATE documents SET fetched_at=?", (iso,))


class FakeJobs:
    """JobManager 대역: 제출하면 DB에 queued 작업을 만들고 기록만 한다."""

    def __init__(self, db):
        self.db = db
        self.submitted = []
        self.cancelled = []

    def submit(self, kind, params, actor):
        job_id = self.db.create_job(kind, params, actor)
        self.submitted.append({"kind": kind, "params": params, "actor": actor, "id": job_id})
        return job_id

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        self.db.update_job(job_id, status="cancelled")

    def kinds(self):
        return [s["kind"] for s in self.submitted]


class FakeApp:
    def __init__(self, data_dir):
        self.cfg = Config(data_dir)
        self.db = DB(self.cfg.db_path)
        self.jobs = FakeJobs(self.db)
        self.allowed = True

    def gate(self):
        return {"allowed": self.allowed, "windows": "매일 00:00~24:00", "timezone": self.cfg["timezone"],
                "next_allowed": None if self.allowed else
                (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat(timespec="minutes")}


# ------------------------------------------------------------------ 가짜 학습 모듈
def fake_modules(calls: dict, lora_error: Exception | None = RuntimeError("LoRA 실패 (테스트)"),
                 online_available: bool = False, recommend_error: bool = False):
    calls.setdefault("tiny", [])
    calls.setdefault("lora", [])
    calls.setdefault("embeddings", 0)
    calls.setdefault("qa", 0)

    devices = types.ModuleType("learnbox.devices")

    def recommend(profile=None):
        if recommend_error:
            raise RuntimeError("no device info")
        return {"tiny_preset": "nano", "tiny_batch_size": 8, "threads": 3, "notes": []}
    devices.recommend = recommend

    tinylm = types.ModuleType("learnbox.tinylm")
    presets = {"nano": (128, 2, 4, 128, 2000), "mini": (256, 4, 4, 192, 3000), "small": (384, 6, 6, 256, 4000)}

    def resolve_arch(preset=None, overrides=None, recommended="mini"):
        name = preset if preset not in (None, "", "auto") else recommended
        if name == "custom":
            name = "mini"
            if not overrides:
                raise ValueError("직접 설정을 고르면 값을 입력해야 합니다.")
        if name not in presets:
            raise ValueError(f"알 수 없는 모델 크기입니다: {preset}")
        arch = dict(zip(("d_model", "n_layer", "n_head", "seq_len", "vocab_size"), presets[name]))
        for k, v in (overrides or {}).items():
            if v not in (None, "", 0):
                arch[k] = int(v)
        if arch["d_model"] % arch["n_head"]:
            raise ValueError("d_model은 n_head로 나누어떨어져야 합니다")
        given = any(arch[k] != presets.get(name, (0,) * 5)[i] for i, k in
                    enumerate(("d_model", "n_layer", "n_head", "seq_len", "vocab_size")))
        return {"preset": "custom" if given else name, **arch}

    def train(train_path, eval_path, out_dir, arch, steps=1500, batch_size=32, lr=3e-3, resume_from=None,
              device=None, threads=None, progress=lambda p, m: None, check_cancel=lambda: None):
        assert Path(train_path).is_file() and Path(eval_path).is_file()
        calls["tiny"].append({"arch": dict(arch), "steps": steps, "batch_size": batch_size, "resume_from": resume_from,
                              "threads": threads, "out": str(out_dir)})
        progress(0.5, "학습 중")
        check_cancel()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.onnx").write_bytes(b"onnx" * 100)
        (out / "model.int8.onnx").write_bytes(b"int8" * 30)
        (out / "vocab.json").write_text(json.dumps({"itos": ["<pad>", "<unk>", "<bos>", "<eos>", "가"]}),
                                        encoding="utf-8")
        parent_total = 0
        if resume_from:
            parent_total = json.loads((Path(resume_from) / "info.json").read_text(encoding="utf-8"))["total_steps"]
        info = {"kind": "tiny", "preset": arch.get("preset"), "arch": dict(arch), "params": 12345, "steps": steps,
                "total_steps": parent_total + steps, "parent": Path(resume_from).name if resume_from else None,
                "train_loss": 2.0, "eval_loss_before": 5.5, "eval_loss_after": 3.25, "files": {"model.onnx": 400}}
        (out / "info.json").write_text(json.dumps(info), encoding="utf-8")
        return info
    tinylm.resolve_arch = resolve_arch
    tinylm.train = train
    tinylm.estimate = lambda arch, batch_size=32: {"params": 1, "fp32_mb": 1.0}

    train_pc = types.ModuleType("learnbox.train_pc")

    def train_lora(train_path, eval_path, out_dir, base_model, **kw):
        calls["lora"].append({"base_model": base_model, **{k: v for k, v in kw.items() if k not in
                                                           ("progress", "check_cancel")}})
        if lora_error:
            raise lora_error
        out = Path(out_dir)
        (out / "adapter_config.json").write_text("{}", encoding="utf-8")
        return {"kind": "lora", "base_model": base_model, "steps": kw.get("max_steps"), "total_steps": kw.get("max_steps"),
                "eval_loss_before": 2.5, "eval_loss_after": 2.0, "trainable_params": 10, "total_params": 100}
    train_pc.train_lora = train_lora

    search = types.ModuleType("learnbox.search")

    def index_embeddings(cfg, db, progress=lambda p, m: None, check_cancel=lambda: None, batch_size=None):
        calls["embeddings"] += 1
        return {"indexed": 7, "total": 7, "model": "fake-e5"}
    search.index_embeddings = index_embeddings

    online = types.ModuleType("learnbox.online")
    online.status = lambda cfg: {"available": online_available, "reason": None if online_available else "키 없음",
                                 "model": "claude-opus-5-5", "key_source": None}

    def build_qa_dataset(cfg, db, out_path, limit_chunks=50, per_chunk=2, progress=lambda p, m: None,
                         check_cancel=lambda: None):
        calls["qa"] += 1
        Path(out_path).write_text(json.dumps({"question": "q", "answer": "a", "source_url": "u"}) + "\n",
                                  encoding="utf-8")
        return {"path": str(out_path), "added": 1, "total": 1, "errors": 0}
    online.build_qa_dataset = build_qa_dataset

    return {"learnbox.devices": devices, "learnbox.tinylm": tinylm, "learnbox.train_pc": train_pc,
            "learnbox.search": search, "learnbox.online": online}


class Base(unittest.TestCase):
    targets = {"tiny": True, "lora": False, "embeddings": False, "qa": False}

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.app = FakeApp(self.tmp / "data")
        self.app.cfg.data["autolearn"]["targets"] = dict(self.targets)
        self.calls = {}
        self.mods = mock.patch.dict(sys.modules, fake_modules(self.calls, **self.module_opts()))
        self.mods.start()
        self.al = AutoLearner(self.app)
        self.now = datetime.now(timezone.utc)

    def module_opts(self):
        return {}

    def tearDown(self):
        self.al.stop()
        self.mods.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def session(self):
        return self.app.db.active_session()

    def stats(self, sid=None):
        sid = sid or self.session()["id"]
        return self.app.db.get_session(sid)["stats"]

    def finish_jobs(self, kinds=None):
        for j in self.app.jobs.submitted:
            if kinds is None or j["kind"] in kinds:
                job = self.app.db.get_job(j["id"])
                if job["status"] in ("queued", "running"):
                    self.app.db.update_job(j["id"], status="done")


# ------------------------------------------------------------------ 세션 시작/종료
class SessionTest(Base):
    def test_start_resolves_config_and_notifies(self):
        s = self.al.start_session("admin", {"collect_interval_min": "5", "targets": {"lora": "true"},
                                            "pages_per_cycle": 3})
        self.assertEqual(s["status"], "running")
        c = s["config"]
        self.assertEqual(c["collect_interval_min"], 5)
        self.assertIsInstance(c["collect_interval_min"], int)
        self.assertTrue(c["targets"]["lora"])
        self.assertTrue(c["targets"]["tiny"])  # 나머지 대상은 기본값 유지
        self.assertEqual(c["pages_per_cycle"], 3)
        # preset auto -> devices.recommend()["tiny_preset"] (가짜: nano), 구조가 저장된다
        self.assertEqual(c["tiny_preset_requested"], "auto")
        self.assertEqual(c["tiny_preset"], "nano")
        self.assertEqual(c["tiny_arch"]["d_model"], 128)
        self.assertEqual(c["recommended"], {"tiny_preset": "nano", "source": "devices"})
        self.assertEqual(c["tiny_batch_size"], 8)  # 추천 배치가 더 작으면 그것을 쓴다
        self.assertEqual(s["stats"]["cycles"], 0)
        self.assertEqual(s["stats"]["models"], [])
        notes = self.app.db.list_notifications()
        self.assertEqual(notes[0]["title"], "자율 학습을 시작했습니다")
        self.assertEqual(notes[0]["level"], "info")
        with self.assertRaisesRegex(ValueError, "이미 자율 학습이 진행 중입니다."):
            self.al.start_session("admin")

    def test_explicit_preset_and_invalid_values(self):
        s = self.al.start_session("admin", {"tiny_preset": "custom", "tiny_arch": {"d_model": 192, "n_head": 4}})
        self.assertEqual(s["config"]["tiny_arch"]["d_model"], 192)
        self.assertEqual(s["config"]["tiny_preset"], "custom")
        self.app.db.update_session(s["id"], status="ended")
        with self.assertRaises(ValueError):
            self.al.start_session("admin", {"tiny_preset": "custom", "tiny_arch": {"d_model": 100, "n_head": 3}})
        with self.assertRaisesRegex(ValueError, "steps_per_cycle"):
            self.al.start_session("admin", {"steps_per_cycle": "많이"})
        self.assertIsNone(self.app.db.active_session())

    def test_stop_cancels_session_collects_and_submits_finalize(self):
        with self.assertRaisesRegex(ValueError, "진행 중인 자율 학습이 없습니다"):
            self.al.stop_session("admin")
        s = self.al.start_session("admin")
        jobs = self.app.jobs
        mine = jobs.submit("collect_search", {"query": "a", "session_id": s["id"]}, "autolearn")
        other = jobs.submit("collect_links", {"urls": ["https://x"], "depth": 0}, "admin")
        queued_train = jobs.submit("autolearn_train", {"session_id": s["id"]}, "autolearn")
        r = self.al.stop_session("admin")
        self.assertEqual(r["session_id"], s["id"])
        self.assertIn(mine, jobs.cancelled)
        self.assertIn(queued_train, jobs.cancelled)
        self.assertNotIn(other, jobs.cancelled)
        fin = jobs.submitted[-1]
        self.assertEqual((fin["kind"], fin["params"], fin["id"]), ("finalize_learning", {"session_id": s["id"]},
                                                                 r["job_id"]))
        sess = self.app.db.get_session(s["id"])
        self.assertEqual(sess["status"], "finishing")
        self.assertEqual(sess["stats"]["finalize_job_id"], r["job_id"])
        self.assertEqual(self.al.status()["phase"], "finishing")
        with self.assertRaises(ValueError):
            self.al.stop_session("admin")
        with self.assertRaises(ValueError):  # 마무리 중에도 새 세션은 시작할 수 없다
            self.al.start_session("admin")
        # 마무리 작업이 사라지면(엔진 재시작) tick이 다시 제출한다
        self.app.db.update_job(r["job_id"], status="interrupted")
        out = self.al.tick()
        self.assertIn("finalize_resubmit", out["actions"])
        self.assertEqual(jobs.kinds().count("finalize_learning"), 2)
        self.assertEqual(self.al.tick()["actions"], ["finishing"])  # 이미 있으면 다시 내지 않음


class RecommendFallbackTest(Base):
    def module_opts(self):
        return {"recommend_error": True}

    def test_fallback_mini(self):
        s = self.al.start_session("admin")
        self.assertEqual(s["config"]["tiny_preset"], "mini")
        self.assertEqual(s["config"]["recommended"]["source"], "fallback")
        self.assertEqual(s["config"]["tiny_arch"]["d_model"], 256)


# ------------------------------------------------------------------ tick: 수집
class TickCollectTest(Base):
    def setUp(self):
        super().setUp()
        self.app.cfg.data["autolearn"]["targets"] = {"tiny": False, "lora": False, "embeddings": False, "qa": False}

    def test_no_session_does_nothing(self):
        self.assertEqual(self.al.tick()["actions"], [])
        self.assertEqual(self.app.jobs.submitted, [])

    def test_collect_when_window_open_with_dedupe_and_interval(self):
        db = self.app.db
        db.add_watch("search", "인공지능")
        s = self.al.start_session("admin", {"collect_interval_min": 10, "pages_per_cycle": 4})
        out = self.al.tick(self.now)
        self.assertIn("collect_search:인공지능", out["actions"])
        j = self.app.jobs.submitted[-1]
        self.assertEqual(j["kind"], "collect_search")
        self.assertEqual(j["params"], {"query": "인공지능", "limit": 4, "session_id": s["id"]})
        self.assertEqual(j["actor"], "autolearn")
        self.assertIsNotNone(db.list_watch()[0]["last_run"])
        st = self.stats()
        self.assertEqual(st["cycles"], 1)
        self.assertEqual(st["collect_jobs"], [j["id"]])
        self.assertEqual(self.al.status()["phase"], "collecting")
        # 수집 작업이 아직 돌고 있으면 새로 내지 않는다
        self.assertEqual(self.al.tick(self.now + timedelta(minutes=30))["actions"], ["collect_busy"])
        self.finish_jobs()
        # 간격(10분)이 지나지 않았으면 내지 않는다
        self.assertEqual(self.al.tick(self.now + timedelta(minutes=5))["actions"], [])
        self.assertEqual(len(self.app.jobs.submitted), 1)
        self.assertEqual(self.al.status()["phase"], "cooldown")
        out = self.al.tick(self.now + timedelta(minutes=11))
        self.assertTrue(any(a.startswith("collect_search") for a in out["actions"]), out)
        self.assertEqual(self.stats()["cycles"], 2)

    def test_link_watch_submits_collect_links(self):
        self.app.db.add_watch("link", "https://example.com/a", depth=1)
        s = self.al.start_session("admin", {"pages_per_cycle": 3})
        self.al.tick(self.now)
        j = self.app.jobs.submitted[-1]
        self.assertEqual(j["kind"], "collect_links")
        self.assertEqual(j["params"], {"urls": ["https://example.com/a"], "depth": 1, "same_domain": True,
                                       "max_pages": 3, "session_id": s["id"]})

    def test_window_closed_no_collect(self):
        self.app.db.add_watch("search", "인공지능")
        self.app.cfg.data["allowed_windows"] = []
        self.al.start_session("admin")
        out = self.al.tick(self.now)
        self.assertIn("waiting_window", out["actions"])
        self.assertEqual(self.app.jobs.submitted, [])
        self.app.allowed = False
        st = self.al.status()
        self.assertEqual(st["phase"], "waiting_window")
        self.assertIsNotNone(st["next_collect_at"])

    def test_window_only_weekday_hours(self):
        self.app.db.add_watch("search", "인공지능")
        self.app.cfg.data["timezone"] = "UTC"
        self.app.cfg.data["allowed_windows"] = [{"days": [0, 1, 2, 3, 4], "start": "09:00", "end": "18:00"}]
        self.al.start_session("admin")
        sat = datetime(2026, 11, 14, 10, 0, tzinfo=timezone.utc)  # 토요일
        self.assertIn("waiting_window", self.al.tick(sat)["actions"])
        self.assertEqual(self.app.jobs.submitted, [])
        mon = datetime(2026, 11, 16, 10, 0, tzinfo=timezone.utc)
        self.assertTrue(any(a.startswith("collect_search") for a in self.al.tick(mon)["actions"]))

    def test_hour_cap(self):
        self.app.db.add_watch("search", "인공지능")
        s = self.al.start_session("admin", {"max_pages_per_hour": 10, "pages_per_cycle": 5,
                                            "collect_interval_min": 0})
        record_collect(self.app, s["id"], {"visited": 8, "new": 2, "updated": 1, "errors": ["x"]}, now=self.now)
        self.al.tick(self.now)
        self.assertEqual(self.app.jobs.submitted[-1]["params"]["limit"], 2)  # 남은 2쪽만
        self.finish_jobs()
        record_collect(self.app, s["id"], {"visited": 2}, now=self.now)
        out = self.al.tick(self.now + timedelta(minutes=1))
        self.assertEqual(out["actions"], ["hour_cap"])
        self.assertEqual(len(self.app.jobs.submitted), 1)
        nxt = datetime.fromisoformat(self.al.status()["next_collect_at"])
        self.assertGreater(nxt, datetime.now(timezone.utc) + timedelta(minutes=50))
        # 1시간이 지나면 다시 수집
        out = self.al.tick(self.now + timedelta(minutes=61))
        self.assertTrue(any(a.startswith("collect_search") for a in out["actions"]), out)

    def test_discovery_when_watchlist_empty(self):
        add_docs(self.app.db)
        s = self.al.start_session("admin")
        out = self.al.tick(self.now)
        self.assertTrue(any(a.startswith("discover:") for a in out["actions"]), out)
        auto = [w for w in self.app.db.list_watch() if w["origin"] == "auto"]
        self.assertTrue(auto)
        self.assertLessEqual(len(auto), autolearn.DISCOVER_PER_CYCLE)
        j = self.app.jobs.submitted[-1]
        self.assertEqual(j["kind"], "collect_search")
        self.assertIn(j["params"]["query"], {w["value"] for w in auto})
        self.assertEqual(self.stats(s["id"])["discovered_topics"], len(auto))

    def test_discovery_after_full_cycle_and_cap(self):
        add_docs(self.app.db)
        self.app.db.add_watch("search", "사용자주제")
        self.al.start_session("admin", {"collect_interval_min": 0, "max_discovered_topics": 2})
        out = self.al.tick(self.now)  # 사용자가 등록한 항목 먼저
        self.assertIn("collect_search:사용자주제", out["actions"])
        self.finish_jobs()
        out = self.al.tick(self.now + timedelta(minutes=1))  # 한 바퀴 돌았으니 새 주제를 찾는다 (최대 2개)
        self.assertIn("discover:2", out["actions"])
        self.assertEqual(self.app.db.count_watch("auto"), 2)
        self.finish_jobs()
        self.al.tick(self.now + timedelta(minutes=2))
        self.finish_jobs()
        out = self.al.tick(self.now + timedelta(minutes=3))
        self.assertFalse(any(a.startswith("discover") for a in out["actions"]), out)
        self.assertEqual(self.app.db.count_watch("auto"), 2)

    def test_seed_from_titles_when_discovery_off(self):
        add_docs(self.app.db)
        self.al.start_session("admin", {"discover_topics": False})
        out = self.al.tick(self.now)
        self.assertTrue(any(a.startswith("seed:") for a in out["actions"]), out)
        seeds = {w["value"] for w in self.app.db.list_watch()}
        self.assertTrue(seeds & {"인공지능", "기계학습", "김치찌개", "입문", "만들기"}, seeds)
        self.assertNotIn("위키백과", seeds)

    def test_no_topic(self):
        self.al.start_session("admin")
        out = self.al.tick(self.now)
        self.assertIn("no_topic", out["actions"])
        self.assertIn("수집할 주제가 없습니다", self.stats()["note"])


# ------------------------------------------------------------------ tick: 학습
class TickTrainTest(Base):
    def setUp(self):
        super().setUp()
        self.app.cfg.data["autolearn"]["discover_topics"] = False

    def test_train_by_new_chars(self):
        s = self.al.start_session("admin", {"train_min_new_chars": 500, "train_interval_min": 600})
        self.assertFalse(any(a.startswith("train") for a in self.al.tick(self.now)["actions"]))  # 자료 없음
        add_docs(self.app.db)
        out = self.al.tick(self.now)
        self.assertTrue(any(a.startswith("train:") for a in out["actions"]), out)
        j = [x for x in self.app.jobs.submitted if x["kind"] == "autolearn_train"]
        self.assertEqual(len(j), 1)
        self.assertEqual(j[0]["params"], {"session_id": s["id"]})
        self.assertEqual(j[0]["actor"], "autolearn")
        self.assertEqual(self.stats()["train_jobs"], [j[0]["id"]])
        self.assertIn("train_busy", self.al.tick(self.now + timedelta(minutes=1))["actions"])
        self.app.db.update_job(j[0]["id"], status="done")
        # 학습이 끝나 last_train_at이 갱신되면 새 자료가 없으므로 다시 하지 않는다
        backdate_docs(self.app.db)
        autolearn._mutate_stats(self.app.db, s["id"], lambda st: st.update(last_train_at=autolearn.now_iso()))
        out = self.al.tick(self.now + timedelta(minutes=10))
        self.assertFalse(any(a.startswith("train") for a in out["actions"]), out)

    def test_train_by_interval(self):
        add_docs(self.app.db)
        self.al.start_session("admin", {"train_min_new_chars": 10_000_000, "train_interval_min": 60})
        self.assertFalse(any(a.startswith("train") for a in self.al.tick(self.now)["actions"]))
        out = self.al.tick(self.now + timedelta(minutes=61))
        self.assertTrue(any(a.startswith("train:") for a in out["actions"]), out)

    def test_train_respects_window_and_outside_flag(self):
        add_docs(self.app.db)
        self.app.cfg.data["allowed_windows"] = []
        s = self.al.start_session("admin", {"train_min_new_chars": 100})
        self.assertEqual(self.al.tick(self.now)["actions"], ["waiting_window"])
        self.app.db.update_session(s["id"], status="ended")
        self.al.start_session("admin", {"train_min_new_chars": 100, "train_outside_window": True})
        out = self.al.tick(self.now)
        self.assertIn("waiting_window", out["actions"])
        self.assertTrue(any(a.startswith("train:") for a in out["actions"]), out)
        self.assertNotIn("collect_search", self.app.jobs.kinds())

    def test_heavy_job_blocks_training(self):
        add_docs(self.app.db)
        self.app.jobs.submit("train_tiny", {"preset": "nano"}, "admin")  # 사람이 낸 학습이 돌고 있음
        self.al.start_session("admin", {"train_min_new_chars": 100})
        self.assertIn("train_busy", self.al.tick(self.now)["actions"])
        self.assertNotIn("autolearn_train", self.app.jobs.kinds())

    def test_no_targets_no_training(self):
        add_docs(self.app.db)
        self.al.start_session("admin", {"train_min_new_chars": 1, "targets": {"tiny": False, "embeddings": False}})
        self.assertFalse(any(a.startswith("train") for a in self.al.tick(self.now)["actions"]))


# ------------------------------------------------------------------ 상태 / 수집 통계 / 백그라운드
class StatusTest(Base):
    def test_phases_and_fields(self):
        st = self.al.status()
        self.assertEqual((st["active"], st["phase"], st["session"]), (False, "idle", None))
        self.assertIn("window", st)
        add_docs(self.app.db)
        s = self.al.start_session("admin")
        st = self.al.status()
        self.assertTrue(st["active"])
        self.assertEqual(st["phase"], "cooldown")
        self.assertGreater(st["new_chars_since_train"], 0)
        self.assertIsNotNone(st["next_collect_at"])
        self.assertIsNotNone(st["next_train_check_at"])
        self.assertEqual(st["recent_sessions"][0]["id"], s["id"])
        jid = self.app.jobs.submit("autolearn_train", {"session_id": s["id"]}, "autolearn")
        self.assertEqual(self.al.status()["phase"], "training")
        self.assertEqual(self.al.status()["jobs"]["train"], jid)
        self.app.db.update_job(jid, status="done")
        self.app.allowed = False
        self.assertEqual(self.al.status()["phase"], "waiting_window")

    def test_record_collect_merges(self):
        add_docs(self.app.db)
        s = self.al.start_session("admin")
        self.app.db.add_document("https://ex.com/new", "새 문서", "새로 수집한 문서 내용입니다. " * 20, "search")
        record_collect(self.app, s["id"], {"new": 2, "updated": 1, "unchanged": 3, "skipped": 1, "errors": ["e1"],
                                           "stopped": None, "visited": 7})
        record_collect(self.app, s["id"], {"new": 1, "visited": 2})
        record_collect(self.app, 9999, {"new": 1})  # 없는 세션은 무시
        st = self.stats()
        self.assertEqual(st["pages_collected"], 9)
        self.assertEqual(st["docs_new"], 3)
        self.assertEqual(st["docs_updated"], 1)
        self.assertEqual(st["collect_errors"], 1)
        self.assertGreaterEqual(st["chars_new"], len("새로 수집한 문서 내용입니다. " * 20))
        self.assertEqual(sum(h["n"] for h in st["hour_pages"]), 9)
        self.assertEqual(st["last_collect_result"]["pages"], 2)

    def test_background_thread_ticks_after_start(self):
        self.app.db.add_watch("search", "인공지능")
        self.app.cfg.data["autolearn"]["targets"] = {"tiny": False, "lora": False, "embeddings": False, "qa": False}
        self.al.start()
        self.al.start()  # 두 번 불러도 스레드는 하나
        self.al.start_session("admin")
        deadline = time.time() + 8
        while time.time() < deadline and "collect_search" not in self.app.jobs.kinds():
            time.sleep(0.05)
        self.assertIn("collect_search", self.app.jobs.kinds())
        self.al.stop()


# ------------------------------------------------------------------ 주제 찾기
class DiscoverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = DB(self.tmp / "t.db")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_korean_particles_stripped(self):
        add_docs(self.db)
        topics = discover_topics(self.db, limit=10)
        self.assertIn("인공지능", topics)
        self.assertIn("기계학습", topics)
        for t in topics:
            self.assertFalse(t.isdigit(), t)
            self.assertTrue(2 <= len(t) <= 12, t)
            self.assertNotIn(t, {"인공지능은", "인공지능의", "인공지능을", "기계학습은", "김치찌개는", "있다", "이다"})
            self.assertFalse(t.endswith(("은", "는", "을", "를", "에서")), t)
        self.assertNotIn("3000개", topics)
        self.assertNotIn("2024년", topics)

    def test_exclude_and_limit(self):
        add_docs(self.db)
        topics = discover_topics(self.db, limit=3, exclude={"인공지능", "기계학습"})
        self.assertLessEqual(len(topics), 3)
        self.assertNotIn("인공지능", topics)
        self.assertNotIn("기계학습", topics)
        self.assertEqual(discover_topics(self.db, limit=0), [])

    def test_empty_db_and_english(self):
        self.assertEqual(discover_topics(self.db), [])
        self.db.add_document("https://e/1", "Transformers", "Transformers use attention. Attention layers "
                             "and transformers changed language models. The model uses attention.", "link")
        topics = discover_topics(self.db, limit=5)
        self.assertIn("attention", topics)
        self.assertNotIn("the", topics)


# ------------------------------------------------------------------ 학습 작업 / 종료 작업
class TrainJobTest(Base):
    targets = {"tiny": True, "lora": True, "embeddings": True, "qa": True}

    def test_train_registers_models_and_survives_failure(self):
        add_docs(self.app.db)
        s = self.al.start_session("admin", {"steps_per_cycle": 40, "threads": 2})
        prog = []
        r = run_autolearn_train(self.app, s["id"], progress=lambda p, m: prog.append(p))
        self.assertEqual(len(r["models"]), 1)
        self.assertIn("lora", r["errors"])
        self.assertIn("LoRA 실패", r["errors"]["lora"])
        self.assertEqual(r["steps"]["tiny"], 40)
        self.assertEqual(r["eval"]["tiny"]["before"], 5.5)
        self.assertEqual(r["eval"]["tiny"]["after"], 3.25)
        self.assertEqual(r["embeddings"]["indexed"], 7)
        self.assertIn("qa", r["skipped"])  # 온라인 모델 사용 불가 -> 건너뜀
        self.assertEqual(self.calls["qa"], 0)
        self.assertEqual(prog[-1], 1.0)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in prog))
        call = self.calls["tiny"][0]
        self.assertIsNone(call["resume_from"])
        self.assertEqual((call["steps"], call["batch_size"], call["threads"]), (40, 8, 2))
        self.assertEqual(call["arch"]["d_model"], 128)
        self.assertEqual(self.calls["lora"][0]["base_model"], self.app.cfg["lora_training"]["base_model"])
        m = self.app.db.get_model(r["models"][0])
        self.assertEqual(m["kind"], "tiny")
        self.assertTrue(Path(m["path"], "model.onnx").is_file())
        self.assertEqual(m["info"]["session_id"], s["id"])
        # 실패한 LoRA 폴더는 지운다
        self.assertEqual([p.name for p in self.app.cfg.models_dir.iterdir()], [m["name"]])
        st = self.stats()
        self.assertEqual(st["trainings"], 1)
        self.assertEqual(st["models"], r["models"])
        self.assertEqual(st["train_history"][0]["kind"], "tiny")
        self.assertEqual(st["train_history"][0]["eval_loss_after"], 3.25)
        self.assertIsNotNone(st["last_train_at"])
        notes = self.app.db.list_notifications()
        warn = [n for n in notes if n["level"] == "warning"]
        self.assertEqual(len(warn), 1)
        self.assertIn("LoRA", warn[0]["title"])
        ok = [n for n in notes if n["level"] == "success"]
        self.assertEqual(len(ok), 1)
        self.assertIn("3.250", ok[0]["body"])

        # 두 번째 주기: 같은 구조면 이 세션의 최신 소형 모델에서 이어 학습, 같은 경고는 다시 보내지 않는다
        old = self.app.cfg.datasets_dir / "corpus-20200101-000000"  # 이전 주기 말뭉치 (같은 초 충돌 방지용 이름)
        old.mkdir()
        (old / "train.jsonl").write_text("{}\n", encoding="utf-8")
        autolearn._mutate_stats(self.app.db, s["id"], lambda x: x.update(corpus_dir=str(old)))
        r2 = run_autolearn_train(self.app, s["id"])
        self.assertEqual(self.calls["tiny"][1]["resume_from"], m["path"])
        self.assertEqual(self.app.db.get_model(r2["models"][0])["info"]["total_steps"], 80)
        self.assertEqual(len([n for n in self.app.db.list_notifications() if n["level"] == "warning"]), 1)
        self.assertEqual(self.stats()["trainings"], 2)
        self.assertEqual(len(self.stats()["models"]), 2)
        self.assertFalse(old.exists())  # 이전 주기 말뭉치는 정리
        self.assertTrue(Path(self.stats()["corpus_dir"]).exists())

        # 구조가 바뀌면 이어 학습하지 않는다
        conf = self.app.db.get_session(s["id"])["config"]
        conf["tiny_arch"] = {**conf["tiny_arch"], "d_model": 256, "preset": "custom"}
        self.app.db.update_session(s["id"], config=conf)
        run_autolearn_train(self.app, s["id"])
        self.assertIsNone(self.calls["tiny"][2]["resume_from"])

    def test_cancel_propagates(self):
        add_docs(self.app.db)
        s = self.al.start_session("admin")

        class Cancelled(Exception):
            pass
        n = {"c": 0}

        def check():
            n["c"] += 1
            if n["c"] >= 3:
                raise Cancelled()
        with self.assertRaises(Cancelled):
            run_autolearn_train(self.app, s["id"], check_cancel=check)
        self.assertEqual(list(self.app.cfg.models_dir.iterdir()), [])  # 반쯤 만든 모델 폴더 없음

    def test_no_data_records_errors(self):
        s = self.al.start_session("admin")
        r = run_autolearn_train(self.app, s["id"])
        self.assertEqual(r["models"], [])
        self.assertIn("tiny", r["errors"])
        self.assertIn("embeddings", r)  # 다른 대상은 계속


class TrainQaTest(Base):
    targets = {"tiny": False, "lora": True, "embeddings": False, "qa": True}

    def module_opts(self):
        return {"online_available": True, "lora_error": None}

    def test_qa_before_lora_and_lora_resume(self):
        add_docs(self.app.db)
        s = self.al.start_session("admin", {"lora_steps_per_cycle": 7})
        r = run_autolearn_train(self.app, s["id"])
        self.assertEqual(self.calls["qa"], 1)
        self.assertEqual(r["qa"]["added"], 1)
        lora = self.calls["lora"][0]
        self.assertEqual(lora["max_steps"], 7)
        self.assertTrue(lora["qa_path"].endswith("qa.jsonl"))  # Q&A가 LoRA보다 먼저 만들어졌다
        self.assertIsNone(lora["resume_from"])
        self.assertEqual(self.app.db.get_model(r["models"][0])["kind"], "lora")
        run_autolearn_train(self.app, s["id"])
        self.assertEqual(self.calls["lora"][1]["resume_from"], self.app.db.get_model(r["models"][0])["path"])


class FinalizeTest(Base):
    targets = {"tiny": True, "lora": False, "embeddings": False, "qa": False}

    def test_finalize_trains_exports_notifies_and_ends(self):
        add_docs(self.app.db)
        usb = self.tmp / "usb"
        usb.mkdir()
        s = self.al.start_session("admin", {"results_export_target": str(usb), "formats": ["jsonl", "md"]})
        run_autolearn_train(self.app, s["id"])
        self.app.db.add_document("https://ex.com/late", "늦게 온 문서", "학습 뒤에 들어온 새 자료입니다. " * 30, "search")
        r = self.al.stop_session("admin")
        prog = []
        res = run_finalize(self.app, r["session_id"], progress=lambda p, m: prog.append(p))
        self.assertEqual(len(self.calls["tiny"]), 2)  # 새 자료가 있어 마지막 학습을 했다
        self.assertEqual(self.calls["tiny"][1]["resume_from"], self.app.db.get_model(
            self.stats(s["id"])["models"][0])["path"])
        self.assertTrue(Path(res["zip"]).is_file())
        self.assertTrue(Path(res["dir"], "documents", "documents.jsonl").is_file())
        self.assertFalse(Path(res["dir"], "documents", "documents.csv").exists())  # 세션의 formats
        self.assertEqual(len(list(Path(res["dir"], "models").iterdir())), 2)
        self.assertTrue(verify(res["dir"])["ok"])
        self.assertEqual(res["summary"]["session_id"], s["id"])
        self.assertEqual(res["summary"]["session_documents"], 4)
        self.assertEqual(res["copied_to"], str(usb / res["name"]))
        self.assertTrue((usb / res["name"] / "manifest.json").is_file())
        self.assertTrue((usb / f"{res['name']}.zip").is_file())
        self.assertEqual(prog[-1], 1.0)
        sess = self.app.db.get_session(s["id"])
        self.assertEqual(sess["status"], "ended")
        self.assertIsNotNone(sess["ended_at"])
        self.assertEqual(sess["result"]["name"], res["name"])
        self.assertEqual(sess["result"]["file_count"], len(res["files"]))
        self.assertTrue(any(f.get("dir") for f in sess["result"]["files"]))  # 마크다운은 폴더 한 줄
        self.assertIsNone(self.app.db.active_session())
        done = [n for n in self.app.db.list_notifications() if n["title"] == "학습이 종료되었습니다"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["level"], "success")
        self.assertEqual(done[0]["data"]["result"], res["name"])
        self.assertIn(res["dir"], done[0]["body"])
        self.assertIn("ZIP", done[0]["body"])
        self.assertIn(f"파일 {len(res['files'])}개", done[0]["body"])
        self.assertTrue(any(a["action"] == "autolearn_finalize" for a in self.app.db.list_audit()))
        self.assertEqual(self.al.status()["phase"], "idle")
        self.al.start_session("admin")  # 끝났으니 새 세션 시작 가능
        with self.assertRaises(ValueError):
            run_finalize(self.app, s["id"])  # 이미 끝난 세션

    def test_finalize_without_new_chars_skips_training_and_uses_latest_model(self):
        add_docs(self.app.db)
        s = self.al.start_session("admin", {"final_train": True})
        run_autolearn_train(self.app, s["id"])
        mid = self.stats(s["id"])["models"][0]
        backdate_docs(self.app.db)
        self.app.db.update_session(s["id"], status="ended")
        s2 = self.al.start_session("admin")
        self.al.stop_session("admin")
        res = run_finalize(self.app, s2["id"])
        self.assertEqual(len(self.calls["tiny"]), 1)
        self.assertIn("skipped", res["final_train"])
        self.assertEqual([m["id"] for m in res["summary"]["models"]], [mid])  # 세션 모델이 없으면 최신 모델

    def test_finalize_bad_target_still_ends(self):
        add_docs(self.app.db)
        s = self.al.start_session("admin", {"results_export_target": str(self.tmp / "no-such-drive"),
                                            "final_train": False})
        self.al.stop_session("admin")
        res = run_finalize(self.app, s["id"])
        self.assertIsNone(res["copied_to"])
        self.assertIn("대상 폴더가 없습니다", res["copy_error"])
        self.assertEqual(self.app.db.get_session(s["id"])["status"], "ended")
        self.assertTrue(any(n["level"] == "warning" for n in self.app.db.list_notifications()))

    def test_finalize_failure_marks_session_failed(self):
        add_docs(self.app.db)
        s = self.al.start_session("admin", {"final_train": False})
        self.al.stop_session("admin")
        with mock.patch("learnbox.results.export_results", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                run_finalize(self.app, s["id"])
        sess = self.app.db.get_session(s["id"])
        self.assertEqual(sess["status"], "failed")
        self.assertIn("disk full", sess["result"]["error"])
        self.assertTrue(any(n["level"] == "error" for n in self.app.db.list_notifications()))
        self.assertIsNone(self.app.db.active_session())


# ------------------------------------------------------------------ 실제 tinylm (설치되어 있을 때만)
@unittest.skipUnless(HAS_TORCH, "torch가 없어 실제 소형 모델 통합 테스트를 건너뜁니다")
class RealTinyIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.app = FakeApp(self.tmp / "data")
        self.app.cfg.data["autolearn"]["targets"] = {"tiny": True, "lora": False, "embeddings": False, "qa": False}
        # 다른 테스트의 가짜 모듈이 남아 있지 않게 실제 모듈을 쓴다
        for k in ("learnbox.tinylm", "learnbox.devices"):
            if k in sys.modules and not getattr(sys.modules[k], "__file__", None):
                del sys.modules[k]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real_train_and_finalize(self):
        try:
            import learnbox.tinylm  # noqa: F401
        except Exception as e:  # pragma: no cover
            self.skipTest(f"tinylm을 불러올 수 없습니다: {e}")
        docs = [(f"문서 {i}", " ".join(t for _, t in KO_DOCS)) for i in range(6)]
        add_docs(self.app.db, docs)
        al = AutoLearner(self.app)
        s = al.start_session("admin", {"tiny_preset": "custom",
                                       "tiny_arch": {"d_model": 64, "n_layer": 1, "n_head": 4, "seq_len": 32,
                                                     "vocab_size": 200},
                                       "tiny_batch_size": 8, "steps_per_cycle": 8, "threads": 1})
        r = run_autolearn_train(self.app, s["id"])
        self.assertEqual(r["errors"], {}, r)
        m = self.app.db.get_model(r["models"][0])
        self.assertTrue(Path(m["path"], "model.pt").is_file())
        self.assertEqual(m["info"]["arch"]["d_model"], 64)
        al.stop_session("admin")
        res = run_finalize(self.app, s["id"])
        self.assertTrue(any(f["path"].endswith("model.pt") and f["kind"] == "model" for f in res["files"]))
        self.assertTrue(verify(res["dir"])["ok"])


if __name__ == "__main__":
    unittest.main()
