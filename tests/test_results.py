"""학습 결과 파일 내보내기(results) 테스트. 네트워크/torch 없이 돈다.

실행: python -m unittest tests.test_results -v
"""
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path

from learnbox import results
from learnbox.config import Config
from learnbox.db import DB, now_iso
from learnbox.export import verify

TEXT = "딥 러닝은 여러 층의 인공 신경망을 사용하는 기계 학습 방법이다. 데이터에서 규칙을 배운다.\n" * 8
BAD_TITLE = 'CON: a/b\\c*d?"e"<f>|g. '
DOCS = [
    ("https://ex.com/a", "딥 러닝 - 위키백과", "link"),
    ("https://ex.com/b", BAD_TITLE, "link"),
    ("https://search.ex/c", "검색으로 찾은 글: 인공지능?", "search"),
    ("file://1/notes.md", "내 메모" + "가" * 200, "file"),
]
WIN_BAD = set('<>:"/\\|?*')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = Config(self.tmp / "data")
        self.db = DB(self.cfg.db_path)
        for url, title, src in DOCS:
            self.assertEqual(self.db.add_document(url, title, TEXT + title, src, file_id=1 if src == "file" else None),
                             "new")
        self.db.add_file("notes.md", "0" * 64, 1234, "text/markdown", str(self.tmp / "x"), "user")
        self.db.set_file_docs(1, 1)
        # 모델 두 개 (가짜 소형 AI 모델 폴더)
        self.model_ids = []
        for name in ("tiny-20261114-100000", "tiny-20261114-110000"):
            d = self.cfg.models_dir / name
            d.mkdir()
            (d / "model.onnx").write_bytes(os.urandom(2048))
            (d / "model.int8.onnx").write_bytes(os.urandom(512))
            (d / "vocab.json").write_text('{"itos": []}', encoding="utf-8")
            info = {"kind": "tiny", "params": 98765, "steps": 60, "total_steps": 60, "eval_loss_before": 5.1,
                    "eval_loss_after": 2.75, "preset": "nano"}
            (d / "info.json").write_text(json.dumps(info), encoding="utf-8")
            self.model_ids.append(self.db.add_model("tiny", name, str(d), info))
        (self.cfg.datasets_dir / "qa.jsonl").write_text(
            json.dumps({"question": "딥 러닝이란?", "answer": "신경망 학습", "source_url": "https://ex.com/a"},
                       ensure_ascii=False) + "\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class ExportLayoutTest(Base):
    def test_full_layout_manifest_zip_and_report(self):
        prog = []
        res = results.export_results(self.cfg, self.db, progress=lambda p, m: prog.append((p, m)))
        out = Path(res["dir"])
        self.assertRegex(res["name"], r"^learnbox-results-\d{8}-\d{6}$")
        self.assertEqual(out, (self.cfg.data_dir / "results" / res["name"]))
        self.assertEqual(Path(res["zip"]), out.parent / f"{res['name']}.zip")
        self.assertEqual(res["zip_bytes"], Path(res["zip"]).stat().st_size)
        self.assertEqual(prog[-1][0], 1.0)
        self.assertTrue(all(0 <= p <= 1 for p, _ in prog))

        # documents.jsonl
        rows = [json.loads(line) for line in (out / "documents/documents.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), len(DOCS))
        self.assertEqual(set(rows[0]), {"id", "url", "title", "source", "query", "fetched_at", "chars", "text"})
        self.assertEqual(rows[1]["title"], BAD_TITLE)
        self.assertEqual(rows[0]["chars"], len(rows[0]["text"]))

        # CSV: UTF-8 BOM, text 열 포함, 줄바꿈 있는 본문도 한 행
        raw = (out / "documents/documents.csv").read_bytes()
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        recs = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
        self.assertEqual(recs[0], ["id", "url", "title", "source", "query", "fetched_at", "chars", "text"])
        self.assertEqual(len(recs), len(DOCS) + 1)
        self.assertEqual(recs[1][7], rows[0]["text"])

        # Markdown: Windows에서도 안전한 이름 + front matter
        md = sorted((out / "documents/markdown").iterdir())
        self.assertEqual(len(md), len(DOCS))
        for f in md:
            self.assertRegex(f.name, r"^\d{5}-.+\.md$")
            stem = f.name[:-3]
            self.assertFalse(set(stem) & WIN_BAD, f.name)
            self.assertFalse(stem.endswith((".", " ")), f.name)
            self.assertLessEqual(len(f.name), 6 + results.SLUG_MAX + 3)
        first = md[0].read_text(encoding="utf-8")
        self.assertTrue(first.startswith("---\ntitle: \"딥 러닝 - 위키백과\"\nurl: \"https://ex.com/a\"\n"
                                         "source: \"link\"\nfetched_at: \""), first[:200])
        self.assertIn("\n---\n\n# 딥 러닝 - 위키백과\n", first)
        bad = next(f for f in md if f.name.startswith(f"{rows[1]['id']:05d}-"))
        self.assertIn(json.dumps(BAD_TITLE, ensure_ascii=False), bad.read_text(encoding="utf-8"))

        # all.txt
        txt = (out / "documents/all.txt").read_text(encoding="utf-8")
        self.assertIn(f"### 딥 러닝 - 위키백과\nhttps://ex.com/a\n\n{rows[0]['text']}\n\n", txt)

        # 말뭉치 / Q&A / DB 사본 / 모델
        tr = (out / "corpus/train.jsonl").read_text(encoding="utf-8").splitlines()
        ev = (out / "corpus/eval.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(tr) + len(ev), self.db.counts()["chunks"])
        self.assertIn("text", json.loads(tr[0]))
        self.assertEqual((out / "corpus/qa.jsonl").read_text(encoding="utf-8"),
                         (self.cfg.datasets_dir / "qa.jsonl").read_text(encoding="utf-8"))
        self.assertEqual([p.name for p in out.iterdir() if p.name.startswith("corpus-")], [])  # 임시 폴더 없음
        con = sqlite3.connect(out / "index/learnbox.db")
        self.assertEqual(con.execute("SELECT count(*) FROM documents").fetchone()[0], len(DOCS))
        self.assertGreater(con.execute("SELECT count(*) FROM chunks WHERE chunks MATCH '\"신경망\"'").fetchone()[0], 0)
        con.close()
        for name in ("tiny-20261114-100000", "tiny-20261114-110000"):  # session/model_ids 없으면 전체 모델
            self.assertEqual(sha(out / "models" / name / "model.onnx"),
                             sha(self.cfg.models_dir / name / "model.onnx"))

        # manifest: 모든 파일의 sha256, export.verify와 호환
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        on_disk = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        self.assertEqual(sorted(list(man["files"]) + ["manifest.json"]), on_disk)
        for rel, meta in man["files"].items():
            self.assertEqual(meta["sha256"], sha(out / rel), rel)
            self.assertEqual(meta["size"], (out / rel).stat().st_size)
        self.assertTrue(verify(str(out))["ok"])

        # 반환 files: 모든 파일 (manifest 포함), sha256/kind/bytes
        self.assertEqual(sorted(f["path"] for f in res["files"]), on_disk)
        for f in res["files"]:
            self.assertEqual(f["sha256"], sha(out / f["path"]))
            self.assertEqual(f["bytes"], (out / f["path"]).stat().st_size)
        kinds = {f["path"]: f["kind"] for f in res["files"]}
        self.assertEqual(kinds["documents/documents.csv"], "documents")
        self.assertEqual(kinds["corpus/qa.jsonl"], "qa")
        self.assertEqual(kinds["index/learnbox.db"], "index")
        self.assertEqual(kinds["REPORT.md"], "report")
        self.assertEqual(kinds["manifest.json"], "manifest")
        self.assertEqual(kinds["models/tiny-20261114-100000/model.onnx"], "model")
        self.assertTrue(all(k == "markdown" for p, k in kinds.items() if p.startswith("documents/markdown/")))
        self.assertEqual(res["total_bytes"], sum(f["bytes"] for f in res["files"]))

        # ZIP: 폴더 이름 아래 같은 파일들
        with zipfile.ZipFile(res["zip"]) as zf:
            self.assertIsNone(zf.testzip())
            self.assertEqual(sorted(zf.namelist()), sorted(f"{res['name']}/{p}" for p in on_disk))
            self.assertEqual(zf.read(f"{res['name']}/manifest.json"), (out / "manifest.json").read_bytes())
            self.assertEqual(zf.read(f"{res['name']}/models/tiny-20261114-100000/model.onnx"),
                             (out / "models/tiny-20261114-100000/model.onnx").read_bytes())

        # 보고서 숫자
        s = res["summary"]
        self.assertEqual(s["documents"], len(DOCS))
        self.assertEqual(s["chars"], sum(r["chars"] for r in rows))
        self.assertEqual(s["by_source"]["link"]["documents"], 2)
        self.assertEqual(s["by_source"]["search"]["documents"], 1)
        self.assertEqual(s["by_source"]["file"]["documents"], 1)
        self.assertEqual(s["files_ingested"], 1)
        self.assertEqual(s["qa_pairs"], 1)
        self.assertEqual(len(s["models"]), 2)
        self.assertIsNone(s["session_documents"])
        rep = json.loads((out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep["summary"]["documents"], len(DOCS))
        md_report = (out / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("# LearnBox 학습 결과 보고서", md_report)
        self.assertIn(f"| **합계** | **{len(DOCS)}** | **{s['chars']:,}** |", md_report)
        self.assertIn("| 웹 링크 (link) | 2 |", md_report)
        self.assertIn("5.100 → 2.750", md_report)
        self.assertIn("98,765", md_report)
        self.assertIn("notes.md", md_report)
        self.assertIn("documents/documents.csv", md_report)
        self.assertIn("python -m learnbox verify", md_report)

    def test_session_scope_formats_and_models(self):
        with self.db.conn() as c:  # 앞의 두 문서는 세션 전에 수집됨
            c.execute("UPDATE documents SET fetched_at='2020-01-01T00:00:00+00:00' WHERE url IN (?, ?)",
                      (DOCS[0][0], DOCS[1][0]))
        sid = self.db.create_session("admin", {"targets": {"tiny": True}})
        session = self.db.get_session(sid)
        session["started_at"] = "2025-01-01T00:00:00+00:00"
        session["stats"] = {"models": [self.model_ids[1]], "pages_collected": 12, "docs_new": 2, "cycles": 3,
                            "train_history": [{"ts": now_iso(), "model_id": self.model_ids[1], "kind": "tiny",
                                               "eval_loss_before": 4.0, "eval_loss_after": 3.0}]}
        res = results.export_results(self.cfg, self.db, session=session, formats=["jsonl"])
        out = Path(res["dir"])
        self.assertTrue((out / "documents/documents.jsonl").is_file())
        for missing in ("documents/documents.csv", "documents/all.txt", "documents/markdown"):
            self.assertFalse((out / missing).exists(), missing)
        # 문서는 전체(지식 베이스), 세션 중 추가분 표시
        self.assertEqual(len((out / "documents/documents.jsonl").read_text(encoding="utf-8").splitlines()), len(DOCS))
        s = res["summary"]
        self.assertEqual(s["session_documents"], 2)
        self.assertEqual(s["by_source"]["link"]["session_documents"], 0)
        self.assertEqual(s["by_source"]["search"]["session_documents"], 1)
        self.assertEqual(s["session_id"], sid)
        self.assertEqual(s["trainings"], 1)
        # 세션 모델만 복사
        self.assertEqual([p.name for p in (out / "models").iterdir()], ["tiny-20261114-110000"])
        rep = (out / "REPORT.md").read_text(encoding="utf-8")
        self.assertIn("이번 세션에 추가·갱신된 문서: 2개", rep)
        self.assertIn("방문한 웹 페이지: 12개", rep)
        self.assertIn(f"자율 학습 세션: #{sid}", rep)
        self.assertIn("4.000 → 3.000", rep)
        self.assertTrue(verify(str(out))["ok"])

        # model_ids가 세션보다 우선, 빈 목록이면 모델 없음
        res2 = results.export_results(self.cfg, self.db, session=session, model_ids=[])
        self.assertEqual(list(Path(res2["dir"], "models").iterdir()), [])
        self.assertNotEqual(res2["name"], res["name"])  # 같은 초에 만들어도 이름이 겹치지 않는다
        res3 = results.export_results(self.cfg, self.db, model_ids=[self.model_ids[0], 9999])
        self.assertEqual([p.name for p in Path(res3["dir"], "models").iterdir()], ["tiny-20261114-100000"])

    def test_missing_model_dir_and_empty_db(self):
        shutil.rmtree(self.cfg.models_dir / "tiny-20261114-100000")
        res = results.export_results(self.cfg, self.db)
        self.assertTrue(any("모델 폴더가 없어" in n for n in res["summary"]["notes"]))
        self.assertTrue(verify(res["dir"])["ok"])
        # 문서가 없어도 내보내기는 된다 (말뭉치는 비어 있음)
        cfg2 = Config(self.tmp / "empty")
        db2 = DB(cfg2.db_path)
        res2 = results.export_results(cfg2, db2)
        self.assertEqual(res2["summary"]["documents"], 0)
        self.assertIsNone(res2["summary"]["corpus"])
        self.assertTrue(Path(res2["zip"]).is_file())

    def test_cancel_cleans_up(self):
        class Cancelled(Exception):
            pass
        n = {"c": 0}

        def check():
            n["c"] += 1
            if n["c"] >= 4:
                raise Cancelled()
        with self.assertRaises(Cancelled):
            results.export_results(self.cfg, self.db, check_cancel=check)
        base = self.cfg.data_dir / "results"
        self.assertEqual(list(base.iterdir()), [])
        self.assertEqual(results.list_results(self.cfg), [])


class ListAndPathTest(Base):
    def test_list_results_newest_first(self):
        self.assertEqual(results.list_results(self.cfg), [])
        a = results.export_results(self.cfg, self.db)
        b = results.export_results(self.cfg, self.db, formats=["csv"])
        (self.cfg.data_dir / "results" / "learnbox-results-20990101-000000").mkdir()  # 미완성(매니페스트 없음) 무시
        (self.cfg.data_dir / "results" / "other-folder").mkdir()
        lst = results.list_results(self.cfg)
        self.assertEqual([r["name"] for r in lst], sorted([a["name"], b["name"]], reverse=True))
        r = next(x for x in lst if x["name"] == a["name"])
        self.assertEqual(set(r) >= {"name", "dir", "zip", "created_at", "total_bytes", "files"}, True)
        self.assertEqual(r["total_bytes"], a["total_bytes"])
        self.assertEqual(r["zip"], a["zip"])
        self.assertEqual(r["file_count"], len(a["files"]))
        md = [f for f in r["files"] if f["kind"] == "markdown"]
        self.assertEqual(len(md), 1)  # 문서별 마크다운은 폴더 한 줄로 묶음
        self.assertEqual(md[0]["count"], len(DOCS))
        self.assertTrue(md[0]["dir"])
        self.assertIn("REPORT.md", {f["path"] for f in r["files"]})

    def test_result_file_path_and_traversal(self):
        res = results.export_results(self.cfg, self.db)
        name = res["name"]
        self.assertEqual(results.result_file_path(self.cfg, name), Path(res["zip"]).resolve())
        p = results.result_file_path(self.cfg, name, "documents/documents.jsonl")
        self.assertEqual(p, (Path(res["dir"]) / "documents" / "documents.jsonl").resolve())
        self.assertTrue(results.result_file_path(self.cfg, name, "models/tiny-20261114-100000/model.onnx").is_file())
        (self.tmp / "secret.txt").write_text("secret", encoding="utf-8")
        bad_rels = ["../secret.txt", "../../secret.txt", "documents/../../" + name + ".zip", "..\\..\\secret.txt",
                    "/etc/passwd", "\\windows\\win.ini", "C:/Windows/win.ini", "documents//documents.jsonl",
                    "documents/./documents.jsonl", "", "documents", "nope.txt", "documents/\x00.jsonl"]
        for rel in bad_rels:
            with self.assertRaises(ValueError, msg=rel):
                results.result_file_path(self.cfg, name, rel)
        for bad_name in ["../results", "..", "learnbox-results-x", "learnbox-results-20990101-000000",
                         f"{name}/../{name}", "", f"{name}.zip"]:
            with self.assertRaises(ValueError, msg=bad_name):
                results.result_file_path(self.cfg, bad_name)
        # 결과 폴더 안의 심볼릭 링크가 밖을 가리키면 거부
        try:
            os.symlink(self.tmp / "secret.txt", Path(res["dir"]) / "link.txt")
        except (OSError, NotImplementedError):
            return
        with self.assertRaises(ValueError):
            results.result_file_path(self.cfg, name, "link.txt")
        # ZIP이 없으면 오류
        os.remove(res["zip"])
        with self.assertRaises(ValueError):
            results.result_file_path(self.cfg, name)


class SlugTest(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(results.slugify("딥 러닝 - 위키백과"), "딥-러닝-위키백과")
        s = results.slugify(BAD_TITLE)
        self.assertFalse(set(s) & WIN_BAD)
        self.assertFalse(s.endswith((".", " ", "-")))
        self.assertEqual(results.slugify(""), "문서")
        self.assertEqual(results.slugify(None), "문서")
        self.assertEqual(results.slugify("???***"), "문서")
        self.assertEqual(len(results.slugify("가" * 300)), results.SLUG_MAX)
        self.assertEqual(results.slugify("Hello World.txt..."), "hello-world.txt")
        self.assertNotIn("\n", results.slugify("a\nb\tc\x00d"))
        self.assertTrue(re.fullmatch(r"[\w.\-]+", results.slugify("CON|AUX<1>%&#@!")))

    def test_fmt_bytes(self):
        self.assertEqual(results.fmt_bytes(512), "512 B")
        self.assertEqual(results.fmt_bytes(2048), "2.0 KB")
        self.assertEqual(results.fmt_bytes(5 * 1024 * 1024), "5.0 MB")


if __name__ == "__main__":
    unittest.main()
