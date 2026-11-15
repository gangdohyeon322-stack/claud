"""학습 결과 파일 내보내기: 자율 학습 '학습 종료' 시(또는 수동으로) 배운 모든 것을 파일로 남긴다.

결과 폴더 (cfg.data_dir/results/learnbox-results-YYYYmmdd-HHMMSS, 로컬 시각):
  documents/documents.jsonl   문서 전체 {id,url,title,source,query,fetched_at,chars,text}
  documents/documents.csv     같은 내용 (엑셀용 UTF-8 BOM)
  documents/markdown/         문서별 마크다운 {id:05d}-{slug}.md (front matter 포함)
  documents/all.txt           모든 문서를 이어 붙인 텍스트
  corpus/train.jsonl, eval.jsonl, qa.jsonl   학습용 말뭉치 (+ Q&A 학습 데이터가 있으면)
  index/learnbox.db           SQLite 사본 (FTS5 검색 인덱스 + 임베딩 포함)
  models/{이름}/...           모델 폴더 사본 (ONNX fp32/int8, model.pt, vocab.json, LoRA 어댑터 등)
  REPORT.md, report.json      한국어 보고서 / 같은 내용의 기계용 요약
  manifest.json               파일별 SHA-256 (python -m learnbox verify <폴더> 로 확인 가능)
그리고 폴더 옆에 같은 이름의 ZIP.

표준 라이브러리만 사용한다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

FORMATS = ("jsonl", "csv", "md", "txt")
RESULT_PREFIX = "learnbox-results-"
_NAME_RE = re.compile(r"^learnbox-results-\d{8}-\d{6}(?:-\d{1,3})?$")
_WIN_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_SLUG_DROP = re.compile(r"[^\w\s.\-]", re.UNICODE)
SLUG_MAX = 50

SOURCE_LABELS = {"link": "웹 링크", "search": "검색 수집", "file": "가져온 파일", "auto": "자율 수집",
                 "unknown": "기타"}
KIND_LABELS = {"tiny": "소형 AI 모델", "lora": "LoRA 모델", "mobile": "소형 AI 모델", "pc": "LoRA 모델"}


def _noop_progress(p: float, m: str = "") -> None:
    return None


def _noop_cancel() -> None:
    return None


# ---------------------------------------------------------------------------------------------------------------
# 작은 도우미
# ---------------------------------------------------------------------------------------------------------------
def results_dir(cfg) -> Path:
    p = Path(cfg.data_dir) / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def slugify(title: str | None, max_len: int = SLUG_MAX) -> str:
    """Windows/macOS/Linux 어디서나 안전한 파일 이름 조각. 한글은 그대로 둔다."""
    s = unicodedata.normalize("NFKC", str(title or ""))
    s = _WIN_BAD.sub(" ", s)
    s = _SLUG_DROP.sub(" ", s)
    s = re.sub(r"[\s_]+", "-", s.strip())
    s = re.sub(r"-{2,}", "-", s).strip("-.")
    s = s[:max_len].rstrip("-. ")
    return s.lower() or "문서"


def _safe_component(name: str) -> str:
    """폴더 이름 한 조각 (모델 폴더 이름 등). 경로 구분자/상위 경로를 없앤다."""
    s = _WIN_BAD.sub("_", unicodedata.normalize("NFKC", str(name or ""))).strip().strip(".")
    return s[:100] or "model"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else 0


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "-"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "-"


def _fmt_loss(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "-"


def _tz(cfg):
    try:
        from zoneinfo import ZoneInfo

        return ZoneInfo(str(cfg["timezone"]))
    except Exception:
        return None


def _local(iso: str | None, cfg=None) -> str:
    """UTC ISO 시각 -> 설정 시간대(없으면 이 PC 시간대)의 'YYYY-mm-dd HH:MM'."""
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(str(iso))
    except ValueError:
        return str(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    tz = _tz(cfg) if cfg is not None else None
    dt = dt.astimezone(tz) if tz else dt.astimezone()
    return dt.strftime("%Y-%m-%d %H:%M")


def _duration(start: str | None, end: str | None) -> str:
    try:
        a, b = datetime.fromisoformat(str(start)), datetime.fromisoformat(str(end))
    except (TypeError, ValueError):
        return "-"
    sec = max(0, int((b - a).total_seconds()))
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    parts = ([f"{d}일"] if d else []) + ([f"{h}시간"] if h else []) + [f"{m}분"]
    return " ".join(parts)


def file_kind(rel: str) -> str:
    """결과 폴더 안 상대 경로 -> 파일 종류."""
    if rel.startswith("documents/markdown/"):
        return "markdown"
    if rel.startswith("documents/"):
        return "documents"
    if rel == "corpus/qa.jsonl":
        return "qa"
    if rel.startswith("corpus/"):
        return "corpus"
    if rel.startswith("index/"):
        return "index"
    if rel.startswith("models/"):
        return "model"
    if rel in ("REPORT.md", "report.json"):
        return "report"
    if rel == "manifest.json":
        return "manifest"
    return "other"


def display_files(files: list[dict]) -> list[dict]:
    """화면 표시용 파일 목록: 문서별 마크다운(수천 개일 수 있음)은 폴더 한 줄로 묶는다."""
    out, md = [], None
    for f in files:
        if f.get("kind") == "markdown":
            if md is None:
                md = {"path": "documents/markdown/", "bytes": 0, "kind": "markdown", "dir": True, "count": 0}
                out.append(md)
            md["bytes"] += int(f.get("bytes") or 0)
            md["count"] += 1
        else:
            out.append(dict(f))
    return out


def _new_out_dir(cfg) -> Path:
    base = results_dir(cfg)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for i in range(1, 1000):
        name = f"{RESULT_PREFIX}{stamp}" + (f"-{i}" if i > 1 else "")
        out = base / name
        if out.exists() or (base / f"{name}.zip").exists():
            continue
        try:
            out.mkdir(parents=True)
        except FileExistsError:
            continue
        return out
    raise RuntimeError("결과 폴더 이름을 만들 수 없습니다.")


def _model_rows(db, session: dict | None, model_ids: list[int] | None) -> list[dict]:
    if model_ids is not None:
        ids = [int(i) for i in model_ids]
    elif session is not None:
        ids = [int(i) for i in ((session.get("stats") or {}).get("models") or [])]
    else:
        return list(reversed(db.list_models()))  # 오래된 것부터
    rows, seen = [], set()
    by_id = {m["id"]: m for m in db.list_models()}
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        if i in by_id:
            rows.append(by_id[i])
    return rows


# ---------------------------------------------------------------------------------------------------------------
# 내보내기
# ---------------------------------------------------------------------------------------------------------------
def export_results(cfg, db, session: dict | None = None, model_ids: list[int] | None = None,
                   formats: list[str] | None = None, progress=_noop_progress, check_cancel=_noop_cancel) -> dict:
    """학습 결과를 파일로 내보낸다. 반환:
    {"name","dir","zip","zip_bytes","files":[{"path","bytes","sha256","kind"}],"total_bytes","summary":{...}}"""
    if formats is None:
        try:
            formats = list(cfg["autolearn"].get("formats") or FORMATS)
        except (KeyError, TypeError, AttributeError):
            formats = list(FORMATS)
    if isinstance(formats, str):
        formats = [f.strip() for f in formats.split(",")]
    fmts = [f for f in FORMATS if f in {str(x).strip().lower() for x in formats}] or list(FORMATS)

    out = _new_out_dir(cfg)
    name = out.name
    zip_path = out.parent / f"{name}.zip"
    try:
        result = _export_into(cfg, db, out, zip_path, session, model_ids, fmts, progress, check_cancel)
    except BaseException:
        # 취소/오류: 반쯤 만든 결과는 지운다 (목록에 불완전한 결과가 나타나지 않게)
        shutil.rmtree(out, ignore_errors=True)
        for p in (zip_path, zip_path.with_name(zip_path.name + ".part")):
            try:
                p.unlink()
            except OSError:
                pass
        raise
    return result


def _export_into(cfg, db, out: Path, zip_path: Path, session, model_ids, fmts, progress, check_cancel) -> dict:
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    started_at = (session or {}).get("started_at")
    notes: list[str] = []
    progress(0.0, "학습 결과 내보내기 시작")

    # ---- 1. 문서 ----
    counts = db.counts()
    n_total = max(1, int(counts.get("documents") or 0))
    docs_dir = out / "documents"
    docs_dir.mkdir()
    md_dir = docs_dir / "markdown"
    by_source: dict[str, dict] = {}
    n_docs = chars_total = sess_docs = sess_chars = 0
    f_jsonl = f_csv = f_txt = None
    writer = None
    try:
        if "jsonl" in fmts:
            f_jsonl = open(docs_dir / "documents.jsonl", "w", encoding="utf-8", newline="\n")
        if "csv" in fmts:
            f_csv = open(docs_dir / "documents.csv", "w", encoding="utf-8-sig", newline="")
            writer = csv.writer(f_csv)
            writer.writerow(["id", "url", "title", "source", "query", "fetched_at", "chars", "text"])
        if "txt" in fmts:
            f_txt = open(docs_dir / "all.txt", "w", encoding="utf-8", newline="\n")
        if "md" in fmts:
            md_dir.mkdir()
        for d in db.iter_documents():
            if n_docs % 100 == 0:
                check_cancel()
                progress(0.02 + 0.33 * n_docs / n_total, f"문서 쓰는 중 {n_docs}/{n_total}")
            text = d.get("text") or ""
            src = d.get("source") or "unknown"
            row = {"id": d["id"], "url": d.get("url"), "title": d.get("title") or "", "source": src,
                   "query": d.get("query"), "fetched_at": d.get("fetched_at"), "chars": len(text), "text": text}
            n_docs += 1
            chars_total += len(text)
            bs = by_source.setdefault(src, {"documents": 0, "chars": 0, "session_documents": 0, "session_chars": 0})
            bs["documents"] += 1
            bs["chars"] += len(text)
            if started_at and (d.get("fetched_at") or "") >= started_at:
                sess_docs += 1
                sess_chars += len(text)
                bs["session_documents"] += 1
                bs["session_chars"] += len(text)
            if f_jsonl:
                f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            if writer:
                writer.writerow([row[k] if row[k] is not None else "" for k in
                                 ("id", "url", "title", "source", "query", "fetched_at", "chars", "text")])
            if f_txt:
                f_txt.write(f"### {row['title']}\n{row['url']}\n\n{text}\n\n")
            if "md" in fmts:
                fm = "\n".join(f"{k}: {json.dumps(row[k] if row[k] is not None else '', ensure_ascii=False)}"
                               for k in ("title", "url", "source", "fetched_at"))
                body = f"---\n{fm}\n---\n\n# {row['title'] or '(제목 없음)'}\n\n{text}\n"
                (md_dir / f"{int(d['id']):05d}-{slugify(row['title'])}.md").write_text(body, encoding="utf-8")
    finally:
        for f in (f_jsonl, f_csv, f_txt):
            if f:
                f.close()

    # ---- 2. 학습용 말뭉치 ----
    check_cancel()
    progress(0.36, "학습용 말뭉치 만드는 중")
    corpus_dir = out / "corpus"
    corpus_dir.mkdir()
    corpus_info = None
    if n_docs:
        from .dataset import build_corpus

        tmp = Path(tempfile.mkdtemp(prefix="corpus-", dir=out))
        try:
            c = build_corpus(db, tmp)
            for fn in ("train.jsonl", "eval.jsonl"):
                shutil.move(str(Path(c["dir"]) / fn), str(corpus_dir / fn))
            corpus_info = {"train": c["train"], "eval": c["eval"], "chars": c["chars"]}
        except ValueError as e:
            notes.append(f"학습용 말뭉치를 만들지 못했습니다: {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    qa_pairs = 0
    qa_src = Path(cfg.data_dir) / "datasets" / "qa.jsonl"
    if qa_src.is_file():
        shutil.copy2(qa_src, corpus_dir / "qa.jsonl")
        with open(corpus_dir / "qa.jsonl", encoding="utf-8") as f:
            qa_pairs = sum(1 for line in f if line.strip())

    # ---- 3. 검색 인덱스 (SQLite 사본) ----
    check_cancel()
    progress(0.45, "검색 인덱스(DB) 복사 중")
    (out / "index").mkdir()
    db.backup_to(out / "index" / "learnbox.db")

    # ---- 4. 모델 ----
    models_out = out / "models"
    models_out.mkdir()
    rows = _model_rows(db, session, model_ids)
    models: list[dict] = []
    used_names: set[str] = set()
    for i, m in enumerate(rows):
        check_cancel()
        progress(0.55 + 0.2 * i / max(1, len(rows)), f"모델 복사 중: {m.get('name')}")
        src = Path(m.get("path") or "")
        info = m.get("info") or {}
        entry = {"id": m["id"], "name": m.get("name"), "kind": m.get("kind"), "created_at": m.get("created_at"),
                 "params": info.get("params") or info.get("total_params"),
                 "trainable_params": info.get("trainable_params"),
                 "preset": info.get("preset"), "base_model": info.get("base_model"),
                 "steps": info.get("steps"), "total_steps": info.get("total_steps"), "parent": info.get("parent"),
                 "eval_loss_before": info.get("eval_loss_before"), "eval_loss_after": info.get("eval_loss_after"),
                 "onnx": info.get("onnx"), "onnx_int8": info.get("onnx_int8"),
                 "dir": None, "bytes": 0, "files": [], "missing": False}
        if not src.is_dir():
            entry["missing"] = True
            notes.append(f"모델 폴더가 없어 건너뛰었습니다: {m.get('name')} ({src})")
            models.append(entry)
            continue
        dname = _safe_component(m.get("name") or f"model-{m['id']}")
        base, k = dname, 2
        while dname.lower() in used_names:
            dname = f"{base}-{k}"
            k += 1
        used_names.add(dname.lower())
        shutil.copytree(src, models_out / dname)
        entry["dir"] = f"models/{dname}"
        entry["bytes"] = _dir_size(models_out / dname)
        entry["files"] = sorted(f.relative_to(models_out / dname).as_posix()
                                for f in (models_out / dname).rglob("*") if f.is_file())
        models.append(entry)

    # ---- 5. 보고서 ----
    check_cancel()
    progress(0.76, "보고서 쓰는 중")
    stats = (session or {}).get("stats") or {}
    files_list = db.list_files(100000) if hasattr(db, "list_files") else []
    sess_files = [f for f in files_list if started_at and (f.get("created_at") or "") >= started_at]
    train_history = list(stats.get("train_history") or [])
    if not session:
        train_history = [{"ts": m.get("created_at"), "model_id": m["id"], "kind": m.get("kind"),
                          "eval_loss_before": m.get("eval_loss_before"), "eval_loss_after": m.get("eval_loss_after")}
                         for m in models]
    ended_at = (session or {}).get("ended_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = {
        "name": out.name,
        "created_at": created_at,
        "session_id": (session or {}).get("id"),
        "period": {"start": started_at, "end": ended_at if session else None},
        "documents": n_docs,
        "chars": chars_total,
        "session_documents": sess_docs if session else None,
        "session_chars": sess_chars if session else None,
        "by_source": by_source,
        "files_ingested": len(files_list),
        "session_files_ingested": len(sess_files) if session else None,
        "pages_collected": stats.get("pages_collected"),
        "discovered_topics": stats.get("discovered_topics"),
        "trainings": len(train_history),
        "models": [{k: m[k] for k in ("id", "name", "kind", "params", "bytes", "steps", "total_steps",
                                      "eval_loss_before", "eval_loss_after", "dir")} for m in models],
        "formats": list(fmts),
        "corpus": corpus_info,
        "qa_pairs": qa_pairs,
        "notes": notes,
    }
    report = {"created_at": created_at, "generator": "LearnBox", "summary": summary, "counts": counts,
              "session": ({k: session.get(k) for k in ("id", "status", "actor", "started_at", "ended_at", "config",
                                                       "stats")} if session else None),
              "train_history": train_history, "models": models,
              "files_ingested": [{k: f.get(k) for k in ("id", "name", "size", "docs", "created_at")}
                                 for f in files_list[:1000]]}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "REPORT.md").write_text(_report_md(cfg, summary, session, models, files_list, sess_files,
                                              train_history, fmts), encoding="utf-8")

    # ---- 6. 체크섬 (manifest.json: export.verify 와 같은 형식) ----
    check_cancel()
    progress(0.8, "체크섬 계산 중")
    files: list[dict] = []
    manifest_files: dict[str, dict] = {}
    for f in sorted(out.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out).as_posix()
            entry = {"path": rel, "bytes": f.stat().st_size, "sha256": _sha256(f), "kind": file_kind(rel)}
            files.append(entry)
            manifest_files[rel] = {"size": entry["bytes"], "sha256": entry["sha256"], "kind": entry["kind"]}
    manifest = {"created_at": created_at, "generator": "LearnBox", "name": out.name,
                "session_id": summary["session_id"], "file_count": len(manifest_files),
                "total_bytes": sum(v["size"] for v in manifest_files.values()), "files": manifest_files}
    mpath = out / "manifest.json"
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append({"path": "manifest.json", "bytes": mpath.stat().st_size, "sha256": _sha256(mpath),
                  "kind": "manifest"})
    total_bytes = sum(f["bytes"] for f in files)

    # ---- 7. ZIP ----
    part = zip_path.with_name(zip_path.name + ".part")
    done = 0
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            check_cancel()
            done += f["bytes"]
            progress(0.85 + 0.14 * done / max(1, total_bytes), f"ZIP 압축 중: {f['path']}")
            zf.write(out / f["path"], f"{out.name}/{f['path']}")
    os.replace(part, zip_path)
    zip_bytes = zip_path.stat().st_size
    progress(1.0, f"학습 결과 내보내기 완료 · 파일 {len(files)}개 · {fmt_bytes(total_bytes)}")
    return {"name": out.name, "dir": str(out), "zip": str(zip_path), "zip_bytes": zip_bytes, "files": files,
            "total_bytes": total_bytes, "summary": summary}


def _report_md(cfg, summary: dict, session: dict | None, models: list[dict], files_list: list[dict],
               sess_files: list[dict], train_history: list[dict], fmts: list[str]) -> str:
    L: list[str] = ["# LearnBox 학습 결과 보고서", ""]
    L.append(f"- 만든 시각: {_local(summary['created_at'], cfg)}")
    if session:
        st, en = session.get("started_at"), summary["period"]["end"]
        L.append(f"- 학습 기간: {_local(st, cfg)} ~ {_local(en, cfg)} ({_duration(st, en)})")
        L.append(f"- 자율 학습 세션: #{session.get('id')} (시작: {session.get('actor') or '-'})")
    else:
        L.append("- 범위: 지금까지 학습한 전체 자료 (자율 학습 세션과 무관한 수동 내보내기)")
    L.append(f"- 결과 폴더: `{summary['name']}` (같은 이름의 ZIP 파일도 함께 만들어졌습니다)")
    L.append("")

    # 1. 자료
    L += ["## 1. 수집·가져온 자료", ""]
    if session:
        L += ["| 출처 | 문서 수 | 글자 수 | 이번 세션에 추가·갱신 |", "|---|---:|---:|---:|"]
    else:
        L += ["| 출처 | 문서 수 | 글자 수 |", "|---|---:|---:|"]
    for src, v in sorted(summary["by_source"].items(), key=lambda kv: -kv[1]["documents"]):
        label = f"{SOURCE_LABELS.get(src, src)} ({src})"
        row = f"| {label} | {_fmt_int(v['documents'])} | {_fmt_int(v['chars'])} |"
        if session:
            row += f" {_fmt_int(v['session_documents'])}개 · {_fmt_int(v['session_chars'])}자 |"
        L.append(row)
    total = f"| **합계** | **{_fmt_int(summary['documents'])}** | **{_fmt_int(summary['chars'])}** |"
    if session:
        total += f" **{_fmt_int(summary['session_documents'])}개 · {_fmt_int(summary['session_chars'])}자** |"
    L += [total, ""]
    if session:
        st = session.get("stats") or {}
        L.append(f"- 이번 세션에 추가·갱신된 문서: {_fmt_int(summary['session_documents'])}개 "
                 f"({_fmt_int(summary['session_chars'])}자)")
        L.append(f"- 이번 세션 동안 방문한 웹 페이지: {_fmt_int(st.get('pages_collected') or 0)}개 "
                 f"(새 문서 {_fmt_int(st.get('docs_new') or 0)}, 갱신 {_fmt_int(st.get('docs_updated') or 0)}), "
                 f"수집 주기 {_fmt_int(st.get('cycles') or 0)}회")
        L.append(f"- 스스로 찾은 주제: {_fmt_int(st.get('discovered_topics') or 0)}개")
        L.append("- 문서 파일에는 이번 세션 것만이 아니라 지금까지 학습한 전체 자료(지식 베이스)가 들어 있습니다.")
        L.append("")

    # 2. 파일
    L += ["## 2. 가져온 파일", ""]
    if files_list:
        L.append(f"- 전체 {len(files_list)}개" + (f" (이번 세션 {len(sess_files)}개)" if session else ""))
        for f in files_list[:30]:
            L.append(f"  - {f.get('name')} · {fmt_bytes(f.get('size'))} · 문서 {_fmt_int(f.get('docs') or 0)}개")
        if len(files_list) > 30:
            L.append(f"  - … 외 {len(files_list) - 30}개")
    else:
        L.append("- 가져온 파일이 없습니다.")
    L.append("")

    # 3. 학습 기록
    L += ["## 3. 학습 기록", ""]
    if train_history:
        L += ["| 시각 | 종류 | 모델 | 평가 loss (전 → 후) |", "|---|---|---|---|"]
        names = {m["id"]: m.get("name") for m in models}
        for h in train_history:
            L.append(f"| {_local(h.get('ts'), cfg)} | {KIND_LABELS.get(h.get('kind'), h.get('kind') or '-')} | "
                     f"{names.get(h.get('model_id'), h.get('model_id') or '-')} | "
                     f"{_fmt_loss(h.get('eval_loss_before'))} → {_fmt_loss(h.get('eval_loss_after'))} |")
        L.append("")
        L.append("평가 loss는 학습에 쓰지 않은 자료로 잰 값이며 낮을수록 자료를 잘 배운 것입니다.")
    else:
        L.append("- 이번에 학습한 기록이 없습니다.")
    L.append("")

    # 4. 모델
    L += ["## 4. 모델", ""]
    if models:
        L += ["| 이름 | 종류 | 파라미터 | 크기 | 단계 (누적) | 평가 loss (전 → 후) |", "|---|---|---:|---:|---|---|"]
        for m in models:
            if m.get("missing"):
                L.append(f"| {m.get('name')} | {KIND_LABELS.get(m.get('kind'), m.get('kind'))} | - | (폴더 없음) | - | - |")
                continue
            params = _fmt_int(m.get("params")) if m.get("params") else "-"
            if m.get("kind") in ("lora", "pc") and m.get("trainable_params"):
                params = f"{_fmt_int(m.get('trainable_params'))} (학습) / {params}"
            L.append(f"| {m.get('name')} | {KIND_LABELS.get(m.get('kind'), m.get('kind'))} | {params} | "
                     f"{fmt_bytes(m.get('bytes'))} | {_fmt_int(m.get('steps'))} ({_fmt_int(m.get('total_steps'))}) | "
                     f"{_fmt_loss(m.get('eval_loss_before'))} → {_fmt_loss(m.get('eval_loss_after'))} |")
    else:
        L.append("- 포함된 모델이 없습니다.")
    L.append("")

    # 5. 사용법
    L += ["## 5. 파일 사용법", ""]
    if "jsonl" in fmts:
        L.append("- `documents/documents.jsonl`: 문서 한 줄에 하나 (JSON: id, url, title, source, query, fetched_at, "
                 "chars, text). 다른 프로그램이나 학습 도구에 그대로 넣을 수 있습니다.")
    if "csv" in fmts:
        L.append("- `documents/documents.csv`: 엑셀/구글 시트에서 바로 열 수 있습니다 (UTF-8 BOM). "
                 "엑셀은 한 칸에 32,767자까지만 보여 줍니다.")
    if "md" in fmts:
        L.append("- `documents/markdown/`: 문서마다 마크다운 파일 하나. 옵시디언 같은 노트 앱 폴더에 넣어 쓰세요.")
    if "txt" in fmts:
        L.append("- `documents/all.txt`: 모든 문서를 이어 붙인 텍스트 (`### 제목`, 주소, 본문 순).")
    L.append("- `corpus/train.jsonl`, `corpus/eval.jsonl`: 모델 학습용 말뭉치 ({\"text\",\"title\",\"url\"}).")
    if summary.get("qa_pairs"):
        L.append(f"- `corpus/qa.jsonl`: 온라인 모델로 만든 질문·답변 학습 데이터 {summary['qa_pairs']}개.")
    L.append("- `index/learnbox.db`: SQLite 데이터베이스 사본 (문서 + 키워드 검색 인덱스 + 의미 검색 임베딩). "
             "DB Browser for SQLite 등으로 열 수 있습니다.")
    if any(m.get("kind") in ("tiny", "mobile") for m in models):
        L.append("- `models/tiny-*/model.onnx`: 소형 AI 모델 (onnxruntime / 브라우저 onnxruntime-web에서 실행). "
                 "`model.int8.onnx`는 휴대폰용 경량 버전, `model.pt`는 PyTorch 가중치, `vocab.json`은 글자 사전, "
                 "`info.json`은 학습 정보입니다.")
    if any(m.get("kind") in ("lora", "pc") for m in models):
        L.append("- `models/lora-*/`: LoRA 어댑터 (`adapter_model.safetensors` + `adapter_config.json`). "
                 "PEFT로 기반 모델(info.json의 base_model)에 붙여 사용합니다.")
    L.append("- `report.json`: 이 보고서의 숫자를 담은 JSON.")
    L.append("- `manifest.json`: 파일별 SHA-256. 무결성 확인: `python -m learnbox verify <이 폴더>`")
    if summary.get("notes"):
        L += ["", "## 참고", ""] + [f"- {n}" for n in summary["notes"]]
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------
# 목록 / 다운로드 경로
# ---------------------------------------------------------------------------------------------------------------
def list_results(cfg) -> list[dict]:
    """[{"name","dir","zip","zip_bytes","created_at","total_bytes","file_count","files","session_id"}] 최신순.
    files는 화면 표시용 목록(문서별 마크다운은 폴더 한 줄로 묶음). 완성된(manifest.json이 있는) 결과만."""
    base = Path(cfg.data_dir) / "results"
    if not base.is_dir():
        return []
    out = []
    for d in base.iterdir():
        if not d.is_dir() or not _NAME_RE.match(d.name):
            continue
        mpath = d / "manifest.json"
        if not mpath.is_file():
            continue
        try:
            man = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        mf = man.get("files") or {}
        files = [{"path": rel, "bytes": int(v.get("size") or 0), "sha256": v.get("sha256"),
                  "kind": v.get("kind") or file_kind(rel)} for rel, v in sorted(mf.items())]
        files.append({"path": "manifest.json", "bytes": mpath.stat().st_size, "kind": "manifest"})
        z = base / f"{d.name}.zip"
        out.append({
            "name": d.name, "dir": str(d), "zip": str(z) if z.is_file() else None,
            "zip_bytes": z.stat().st_size if z.is_file() else None,
            "created_at": man.get("created_at") or datetime.fromtimestamp(d.stat().st_mtime).astimezone()
            .isoformat(timespec="seconds"),
            "total_bytes": sum(f["bytes"] for f in files), "file_count": len(files),
            "files": display_files(files), "session_id": man.get("session_id"),
        })
    out.sort(key=lambda r: r["name"], reverse=True)
    return out


def result_file_path(cfg, name: str, rel: str | None = None) -> Path:
    """다운로드용 안전한 경로. rel이 None이면 ZIP. 경로 탈출/없는 결과면 ValueError."""
    name = str(name or "")
    if not _NAME_RE.match(name):
        raise ValueError("학습 결과를 찾을 수 없습니다.")
    base = (Path(cfg.data_dir) / "results").resolve()
    root = base / name
    if not root.is_dir():
        raise ValueError("학습 결과를 찾을 수 없습니다.")
    if rel is None:
        z = base / f"{name}.zip"
        if not z.is_file():
            raise ValueError("학습 결과 ZIP 파일이 없습니다.")
        return z
    rel = str(rel)
    if not rel or "\x00" in rel:
        raise ValueError("잘못된 파일 경로입니다.")
    parts = rel.replace("\\", "/").split("/")
    if rel.startswith(("/", "\\")) or any(p in ("", ".", "..") or ":" in p for p in parts):
        raise ValueError("잘못된 파일 경로입니다.")
    root_r = root.resolve()
    target = (root_r / Path(*parts)).resolve()
    if root_r not in target.parents:
        raise ValueError("잘못된 파일 경로입니다.")
    if not target.is_file():
        raise ValueError("파일을 찾을 수 없습니다.")
    return target
