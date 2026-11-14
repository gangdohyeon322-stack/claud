"""사용자 파일 가져오기: txt/md/csv/tsv/json/jsonl/html/xml/pdf/docx/srt/log -> 문서(검색·학습 자료).

- extract_file: 파일 바이트에서 문서 목록을 뽑는다 (파일 하나 = 문서 하나, JSON/JSONL 레코드는 레코드마다 문서).
- ingest_bytes: 크기 제한, sha256 중복 확인, 원본 보관(data/files), 문서 저장(source="file"), 감사 기록.
- ingest_path: 파일 또는 폴더(하위 폴더 포함, 숨김 파일 제외)를 한꺼번에 가져온다.

pypdf 는 PDF를 읽을 때만 import 한다. DOCX는 표준 라이브러리(zipfile + XML)로 읽는다.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import mimetypes
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import quote

from .db import HL_END, HL_START
from .extract import decode as _decode_html_bytes
from .extract import html_to_text

SUPPORTED = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm", ".xml", ".pdf",
             ".docx", ".srt", ".log"}
TEXT_KEYS = ("text", "content", "body", "article")
TITLE_KEYS = ("title", "name", "subject", "headline")
MIN_DOC_CHARS = 20
_MIME = {".md": "text/markdown", ".markdown": "text/markdown", ".jsonl": "application/x-ndjson",
         ".srt": "application/x-subrip", ".log": "text/plain", ".tsv": "text/tab-separated-values",
         ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?[A-Za-z0-9_\-]+""", re.I)
_SRT_TIME = re.compile(r"^\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{1,3}.*$")


def _noop_progress(p: float, m: str = "") -> None:
    return None


def _noop_cancel() -> None:
    return None


# ---------------------------------------------------------------------------------------------------------------
# 글자 처리
# ---------------------------------------------------------------------------------------------------------------
def decode_text(data: bytes) -> str:
    """utf-8(BOM 포함) -> (BOM이 있으면 utf-16) -> cp949(한국어 윈도우/엑셀) -> utf-8(깨진 글자 대체)."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def clean_text(text: str) -> str:
    """NFC 정규화(맥에서 만든 한글 자모 분리 방지), 줄바꿈 통일, 제어문자/검색 강조 기호 제거, 빈 줄 정리."""
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(HL_START, "").replace(HL_END, "")
    text = _CTRL.sub("", text).replace("﻿", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_title(t, fallback: str) -> str:
    t = re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(t or ""))).strip()
    t = _CTRL.sub("", t)
    return (t or fallback)[:300]


def _base_name(name: str) -> str:
    name = unicodedata.normalize("NFC", str(name or "")).replace("\\", "/").split("/")[-1]
    name = _CTRL.sub("", name).strip()
    return name or "file"


def _doc(title: str, text: str, part: str | None = None) -> dict:
    return {"title": title, "text": clean_text(text), "part": part}


# ---------------------------------------------------------------------------------------------------------------
# 형식별 추출
# ---------------------------------------------------------------------------------------------------------------
def _plain(name: str, stem: str, ext: str, data: bytes) -> list[dict]:
    text = decode_text(data)
    title = stem
    if ext in (".md", ".markdown"):
        for line in text.lstrip("﻿").splitlines()[:8]:
            m = re.match(r"^#\s+(.+?)\s*#*\s*$", line)
            if m:
                title = m.group(1)
                break
    elif ext == ".srt":
        lines = []
        for line in text.splitlines():
            s = line.strip()
            if not s or s.isdigit() or _SRT_TIME.match(s) or s.upper() == "WEBVTT":
                continue
            lines.append(re.sub(r"</?[a-zA-Z][^>]*>", "", s))
        text = "\n".join(lines)
    return [_doc(_clean_title(title, stem), text)]


def _table(stem: str, ext: str, data: bytes) -> list[dict]:
    text = decode_text(data)
    if ext == ".tsv":
        delim = "\t"
    else:
        try:
            delim = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delim = ","
    try:
        csv.field_size_limit(16 * 1024 * 1024)
        rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    except csv.Error:
        return [_doc(stem, text)]
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        return [_doc(stem, "")]
    header = [(h or "").strip() or f"열{i + 1}" for i, h in enumerate(rows[0])]
    if len(rows) == 1:
        return [_doc(stem, " | ".join(header))]
    blocks = []
    for r in rows[1:]:
        lines = []
        for i, v in enumerate(r):
            v = (v or "").strip()
            if not v:
                continue
            h = header[i] if i < len(header) else f"열{i + 1}"
            lines.append(f"{h}: {v}")
        if lines:
            blocks.append("\n".join(lines))
    return [_doc(stem, "\n\n".join(blocks))]


def _flatten(obj, prefix: str = "", out: list | None = None, depth: int = 0) -> list[str]:
    """JSON 값을 '경로: 값' 줄들로 (검색되도록)."""
    out = [] if out is None else out
    if depth > 20 or len(out) > 200_000:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten(v, f"{prefix}[{i}]" if prefix else f"[{i}]", out, depth + 1)
    elif obj is not None and str(obj).strip():
        out.append(f"{prefix}: {obj}" if prefix else str(obj))
    return out


def _as_text(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join(_as_text(x) for x in v if x is not None)
    if isinstance(v, dict):
        return "\n".join(_flatten(v))
    return "" if v is None else str(v)


def _record_text(rec: dict) -> str | None:
    """레코드의 본문. text/content/body/article, 없으면 질문·답변 쌍, 지시·출력 쌍."""
    for k in TEXT_KEYS:
        if k in rec and _as_text(rec[k]).strip():
            return _as_text(rec[k])
    lower = {str(k).lower(): v for k, v in rec.items()}
    for k in TEXT_KEYS:
        if k in lower and _as_text(lower[k]).strip():
            return _as_text(lower[k])
    if lower.get("question") and lower.get("answer"):
        return f"질문: {_as_text(lower['question'])}\n답변: {_as_text(lower['answer'])}"
    if lower.get("instruction") and lower.get("output"):
        inp = f"\n입력: {_as_text(lower['input'])}" if lower.get("input") else ""
        return f"지시: {_as_text(lower['instruction'])}{inp}\n출력: {_as_text(lower['output'])}"
    return None


def _record_title(rec: dict) -> str | None:
    for k in TITLE_KEYS:
        v = rec.get(k)
        if isinstance(v, (str, int, float)) and str(v).strip():
            return str(v)
    return None


def _records_to_docs(name: str, stem: str, records: list, max_docs: int, numbers: list[int] | None = None) -> list[dict]:
    """레코드마다 문서 하나. part = 레코드 번호(JSONL은 줄 번호, 1부터)."""
    docs, rest = [], []
    for i, rec in zip(numbers or range(1, len(records) + 1), records):
        if len(docs) >= max_docs:
            break
        if isinstance(rec, dict):
            body = _record_text(rec)
            if body is not None:
                docs.append(_doc(_clean_title(_record_title(rec), f"{name} #{i}"), body, str(i)))
                continue
            rest.extend(_flatten(rec, f"[{i}]"))
        elif isinstance(rec, str):
            if rec.strip():
                docs.append(_doc(f"{name} #{i}", rec, str(i)))
        elif rec is not None:
            rest.extend(_flatten(rec, f"[{i}]"))
    if rest and len(docs) < max_docs:  # 본문 필드가 없는 레코드들은 모아서 문서 하나로
        docs.append(_doc(stem, "\n".join(rest), "rest"))
    if len(docs) == 1:  # 문서가 하나뿐이면 파일 하나 = 문서 하나
        docs[0]["part"] = None
    return docs


def _json(name: str, stem: str, data: bytes, max_docs: int) -> list[dict]:
    text = decode_text(data)
    try:
        obj = json.loads(text)
    except ValueError:
        return [_doc(stem, text)]  # 형식이 깨졌어도 글자는 검색되게
    if isinstance(obj, list):
        return _records_to_docs(name, stem, obj, max_docs)
    if isinstance(obj, dict):
        body = _record_text(obj)
        if body is not None:
            return [_doc(_clean_title(_record_title(obj), stem), body)]
        # {"data": [ {...}, ... ]} 처럼 레코드 목록을 감싼 경우
        for v in obj.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v[:20]) \
                    and any(_record_text(x) is not None for x in v[:20]):
                return _records_to_docs(name, stem, v, max_docs)
    return [_doc(stem, "\n".join(_flatten(obj)))]


def _jsonl(name: str, stem: str, data: bytes, max_docs: int) -> list[dict]:
    text = decode_text(data)
    records, numbers = [], []
    for no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:  # 깨진 줄은 건너뛴다
            continue
        numbers.append(no)
    if not records:
        return [_doc(stem, text)]
    return _records_to_docs(name, stem, records, max_docs, numbers)


def _html(stem: str, data: bytes) -> list[dict]:
    html = _decode_html_bytes(data, "") if _META_CHARSET.search(data[:4096]) else decode_text(data)
    title, text, _links = html_to_text(html)
    return [_doc(_clean_title(title, stem), text)]


def _xml(stem: str, data: bytes) -> list[dict]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        title, text, _ = html_to_text(decode_text(data))  # 깨진 XML: 태그만 걷어낸다
        return [_doc(_clean_title(title, stem), text)]
    title = None
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1].lower() if isinstance(el.tag, str) else ""
        if tag == "title" and (el.text or "").strip():
            title = el.text
            break
    parts = [t.strip() for t in root.itertext() if t and t.strip()]
    return [_doc(_clean_title(title, stem), "\n".join(parts))]


def _pdf(stem: str, data: bytes) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ValueError("PDF를 읽으려면 pypdf가 필요합니다: pip install pypdf") from e
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except Exception:
                ok = 0
            if not ok:
                raise ValueError("암호가 걸린 PDF는 읽을 수 없습니다. 암호를 푼 파일을 올려 주세요.")
        pages = []
        for page in reader.pages:
            try:
                pages.append((page.extract_text() or "").strip())
            except Exception:  # 한 쪽이 깨져도 나머지는 읽는다
                pages.append("")
        title = None
        try:
            meta = reader.metadata
            title = meta.title if meta else None
        except Exception:
            pass
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"PDF 파일을 읽지 못했습니다: {e}") from e
    return [_doc(_clean_title(title, stem), "\n\n".join(p for p in pages if p))]


def _docx_text(node, out: list[str]) -> None:
    for child in node:
        tag = child.tag
        if tag == f"{_W}t":
            out.append(child.text or "")
        elif tag == f"{_W}tab":
            out.append("\t")
        elif tag in (f"{_W}br", f"{_W}cr"):
            out.append("\n")
        elif tag in (f"{_W}delText", f"{_W}instrText"):
            continue  # 지운 글자(변경 추적), 필드 코드
        else:
            _docx_text(child, out)
            if tag in (f"{_W}p", f"{_W}tr"):
                out.append("\n")
            elif tag == f"{_W}tc":
                out.append("\t")


def _docx(stem: str, data: bytes) -> list[dict]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = set(z.namelist())
            if "word/document.xml" not in names:
                raise ValueError("DOCX 파일에 본문(word/document.xml)이 없습니다.")
            if z.getinfo("word/document.xml").file_size > 200 * 1024 * 1024:
                raise ValueError("DOCX 본문이 너무 큽니다.")
            body_xml = z.read("word/document.xml")
            core = z.read("docProps/core.xml") if "docProps/core.xml" in names else None
    except zipfile.BadZipFile as e:
        raise ValueError("DOCX 파일을 열 수 없습니다 (손상되었거나 DOCX 형식이 아닙니다).") from e
    try:
        root = ET.fromstring(body_xml)
    except ET.ParseError as e:
        raise ValueError(f"DOCX 본문을 읽지 못했습니다: {e}") from e
    out: list[str] = []
    _docx_text(root, out)
    text = "".join(out)
    text = "\n".join(line.strip("\t ").replace("\t\t", "\t") for line in text.split("\n"))
    title = None
    if core:
        try:
            for el in ET.fromstring(core).iter():
                if isinstance(el.tag, str) and el.tag.endswith("}title") and (el.text or "").strip():
                    title = el.text
                    break
        except ET.ParseError:
            pass
    return [_doc(_clean_title(title, stem), text)]


def extract_file(name: str, data: bytes, max_docs: int = 5000) -> list[dict]:
    """파일 이름(확장자)과 내용으로 문서 목록 [{"title","text","part"}] 을 만든다. 지원하지 않는 형식은 ValueError."""
    base = _base_name(name)
    ext = Path(base).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext or '(확장자 없음)'} "
                         f"(지원 형식: {', '.join(sorted(SUPPORTED))})")
    stem = Path(base).stem or base
    max_docs = max(1, int(max_docs or 5000))
    data = data or b""
    if ext in (".txt", ".md", ".markdown", ".srt", ".log"):
        docs = _plain(base, stem, ext, data)
    elif ext in (".csv", ".tsv"):
        docs = _table(stem, ext, data)
    elif ext == ".json":
        docs = _json(base, stem, data, max_docs)
    elif ext == ".jsonl":
        docs = _jsonl(base, stem, data, max_docs)
    elif ext in (".html", ".htm"):
        docs = _html(stem, data)
    elif ext == ".xml":
        docs = _xml(stem, data)
    elif ext == ".pdf":
        docs = _pdf(stem, data)
    else:  # .docx
        docs = _docx(stem, data)
    return docs[:max_docs]


# ---------------------------------------------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------------------------------------------
def _safe_disk_name(name: str) -> str:
    p = Path(name)
    stem = re.sub(r"[^\w.\-]+", "_", p.stem).strip("._") or "file"
    suffix = re.sub(r"[^\w.]+", "", p.suffix.lower())
    return stem[:100] + suffix


def files_dir(cfg) -> Path:
    d = Path(cfg.data_dir) / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ingest_bytes(cfg, db, name: str, data: bytes, actor: str = "user") -> dict:
    """파일 하나를 가져온다. 반환 {"status":"ok"|"duplicate","file_id","name","docs","chars","skipped", ...}."""
    name = _base_name(name)
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {ext or '(확장자 없음)'} "
                         f"(지원 형식: {', '.join(sorted(SUPPORTED))})")
    data = bytes(data or b"")
    icfg = cfg["ingest"] if "ingest" in getattr(cfg, "data", cfg) else {}
    max_mb = float(icfg.get("max_file_mb", 50) or 50)
    if len(data) > max_mb * 1024 * 1024:
        raise ValueError(f"파일이 너무 큽니다 ({len(data) / 1024 / 1024:.1f}MB). "
                         f"최대 {max_mb:g}MB까지 가져올 수 있습니다.")
    if not data.strip():
        raise ValueError("빈 파일입니다.")
    sha = hashlib.sha256(data).hexdigest()
    dup = db.find_file(sha)
    if dup:
        return {"status": "duplicate", "file_id": dup["id"], "name": name, "docs": dup.get("docs") or 0,
                "chars": 0, "skipped": 0, "existing_name": dup.get("name"),
                "message": f"이미 가져온 파일입니다 ({dup.get('name')})."}

    max_docs = int(icfg.get("max_docs_per_file", 5000) or 5000)
    docs = extract_file(name, data, max_docs=max_docs + 1)  # 형식 오류는 파일 기록 전에 드러난다
    truncated = len(docs) > max_docs
    docs = docs[:max_docs]

    mime = _MIME.get(ext) or mimetypes.guess_type(name)[0] or "application/octet-stream"
    try:
        file_id = db.add_file(name, sha, len(data), mime, "", actor)
    except Exception as e:  # 동시에 같은 파일을 올린 경우 (sha256 UNIQUE)
        dup = db.find_file(sha)
        if dup:
            return {"status": "duplicate", "file_id": dup["id"], "name": name, "docs": dup.get("docs") or 0,
                    "chars": 0, "skipped": 0, "existing_name": dup.get("name"),
                    "message": f"이미 가져온 파일입니다 ({dup.get('name')})."}
        raise e
    path = files_dir(cfg) / f"{file_id}_{_safe_disk_name(name)}"
    try:
        path.write_bytes(data)
        with db.conn() as c:
            c.execute("UPDATE files SET path=? WHERE id=?", (str(path), file_id))
    except Exception:
        with db.conn() as c:
            c.execute("DELETE FROM files WHERE id=?", (file_id,))
        path.unlink(missing_ok=True)
        raise

    multi = len(docs) > 1 or any(d.get("part") for d in docs)
    base_url = f"file://{file_id}/{quote(name)}"
    n_docs = chars = skipped = 0
    try:
        for i, d in enumerate(docs, 1):
            text = d["text"]
            if len(text) < MIN_DOC_CHARS:
                skipped += 1
                continue
            url = base_url + (f"#{d.get('part') or i}" if multi else "")
            db.add_document(url, d["title"] or name, text, source="file", query=None, file_id=file_id)
            n_docs += 1
            chars += len(text)
    finally:
        db.set_file_docs(file_id, n_docs)
    db.audit(actor, "ingest", {"file_id": file_id, "name": name, "size": len(data), "docs": n_docs,
                               "chars": chars, "skipped": skipped})
    out = {"status": "ok", "file_id": file_id, "name": name, "docs": n_docs, "chars": chars, "skipped": skipped,
           "size": len(data), "truncated": truncated}
    if n_docs == 0:
        out["message"] = ("글자를 찾지 못했습니다. 스캔한 이미지 PDF는 지원하지 않습니다." if ext == ".pdf"
                          else f"{MIN_DOC_CHARS}자 이상인 내용이 없어 저장한 문서가 없습니다.")
    elif truncated:
        out["message"] = f"문서가 너무 많아 처음 {max_docs}개만 가져왔습니다."
    return out


def _iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            if fn.startswith(".") or Path(fn).suffix.lower() not in SUPPORTED:
                continue
            yield Path(dirpath) / fn


def ingest_path(cfg, db, path: str, actor: str = "cli", progress=_noop_progress, check_cancel=_noop_cancel) -> dict:
    """파일 또는 폴더(하위 폴더 포함)를 가져온다. 숨김 파일/폴더와 지원하지 않는 형식은 건너뛴다.

    반환 {"files","docs","duplicates","errors":[{"path","error"}], "chars","skipped","total"}
    """
    p = Path(str(path)).expanduser()
    if not p.exists():
        raise ValueError(f"경로를 찾을 수 없습니다: {path}")
    progress(0.0, "가져올 파일 찾는 중")
    targets = [p] if p.is_file() else list(_iter_files(p))
    icfg = cfg["ingest"] if "ingest" in getattr(cfg, "data", cfg) else {}
    max_bytes = float(icfg.get("max_file_mb", 50) or 50) * 1024 * 1024
    stats = {"files": 0, "docs": 0, "duplicates": 0, "errors": [], "chars": 0, "skipped": 0, "total": len(targets)}
    n = len(targets) or 1
    for i, f in enumerate(targets):
        check_cancel()
        progress(i / n, f"[{i + 1}/{len(targets)}] {f.name}")
        try:
            size = f.stat().st_size
            if size > max_bytes:
                raise ValueError(f"파일이 너무 큽니다 ({size / 1024 / 1024:.1f}MB). "
                                 f"최대 {max_bytes / 1024 / 1024:g}MB까지 가져올 수 있습니다.")
            r = ingest_bytes(cfg, db, f.name, f.read_bytes(), actor=actor)
        except Exception as e:
            if len(stats["errors"]) < 200:
                stats["errors"].append({"path": str(f), "error": str(e)})
            continue
        if r["status"] == "duplicate":
            stats["duplicates"] += 1
        else:
            stats["files"] += 1
            stats["docs"] += r["docs"]
            stats["chars"] += r["chars"]
            stats["skipped"] += r["skipped"]
    progress(1.0, f"가져오기 완료: 파일 {stats['files']}개, 문서 {stats['docs']}개")
    return stats
