"""SQLite 저장소: 문서, 검색 인덱스(FTS5 trigram), 작업, 모델, 감사 기록."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    source TEXT,
    query TEXT,
    content_hash TEXT,
    fetched_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(title, content, doc_id UNINDEXED, tokenize='trigram');
CREATE TABLE IF NOT EXISTS jobs(
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    params TEXT,
    status TEXT NOT NULL,
    progress REAL DEFAULT 0,
    message TEXT DEFAULT '',
    result TEXT,
    actor TEXT,
    created_at TEXT,
    started_at TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS models(
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT,
    path TEXT NOT NULL,
    info TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS watchlist(
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    value TEXT NOT NULL,
    depth INTEGER DEFAULT 0,
    last_run TEXT,
    created_at TEXT,
    UNIQUE(kind, value)
);
CREATE TABLE IF NOT EXISTS audit(
    id INTEGER PRIMARY KEY,
    ts TEXT,
    actor TEXT,
    action TEXT,
    detail TEXT
);
"""

CHUNK_CHARS = 800
# 검색 결과 스니펫의 강조 구분자 (UI에서 <mark>로 바꾼다)
HL_START, HL_END = "\x02", "\x03"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def chunk_text(text: str, size: int = CHUNK_CHARS) -> list[str]:
    """문단 경계를 살려 약 size자 단위로 자른다."""
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:  # 아주 긴 문단은 강제로 자름
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:size])
            p = p[size:]
        if len(cur) + len(p) + 1 > size and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


class DB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    # ---------------- 문서 ----------------
    def add_document(self, url: str, title: str, text: str, source: str, query: str | None = None) -> str:
        """문서를 저장하고 인덱싱한다. 반환: 'new' | 'updated' | 'unchanged'."""
        h = hashlib.sha256(text.encode()).hexdigest()
        with self.conn() as c:
            row = c.execute("SELECT id, content_hash FROM documents WHERE url=?", (url,)).fetchone()
            if row and row["content_hash"] == h:
                return "unchanged"
            if row:
                doc_id = row["id"]
                c.execute(
                    "UPDATE documents SET title=?, text=?, source=?, query=?, content_hash=?, fetched_at=? WHERE id=?",
                    (title, text, source, query, h, now_iso(), doc_id),
                )
                c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
                status = "updated"
            else:
                cur = c.execute(
                    "INSERT INTO documents(url,title,text,source,query,content_hash,fetched_at) VALUES(?,?,?,?,?,?,?)",
                    (url, title, text, source, query, h, now_iso()),
                )
                doc_id = cur.lastrowid
                status = "new"
            c.executemany(
                "INSERT INTO chunks(title, content, doc_id) VALUES(?,?,?)",
                [(title, ch, doc_id) for ch in chunk_text(text)],
            )
            return status

    def delete_document(self, doc_id: int) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def list_documents(self, limit: int = 50, offset: int = 0) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT id,url,title,source,query,fetched_at,length(text) AS chars FROM documents ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [dict(r) for r in rows]

    def counts(self) -> dict:
        with self.conn() as c:
            d = c.execute("SELECT count(*), coalesce(sum(length(text)),0) FROM documents").fetchone()
            ch = c.execute("SELECT count(*) FROM chunks").fetchone()[0]
            return {"documents": d[0], "chars": d[1], "chunks": ch}

    def iter_chunks(self):
        with self.conn() as c:
            for r in c.execute("SELECT chunks.title, chunks.content, d.url FROM chunks JOIN documents d ON d.id = chunks.doc_id"):
                yield dict(r)

    def snapshot(self, max_chars: int = 4000) -> list[dict]:
        """휴대폰 오프라인 검색용 사본."""
        with self.conn() as c:
            rows = c.execute(
                "SELECT id,url,title,substr(text,1,?) AS text,fetched_at FROM documents ORDER BY id", (max_chars,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ---------------- 검색 ----------------
    def search(self, q: str, limit: int = 10) -> list[dict]:
        terms = [t for t in q.split() if t]
        if not terms:
            return []
        long_terms = [t for t in terms if len(t) >= 3]
        short_terms = [t for t in terms if len(t) < 3]  # trigram은 3글자 미만을 못 찾음 -> LIKE
        like_params = [f"%{t}%" for t in short_terms]
        with self.conn() as c:
            if long_terms:
                match = " ".join('"' + t.replace('"', '""') + '"' for t in long_terms)
                sql = (
                    "SELECT chunks.doc_id AS doc_id, d.url, d.title, "
                    f"snippet(chunks, 1, '{HL_START}', '{HL_END}', '…', 24) AS snippet, "
                    "bm25(chunks) AS score, chunks.content AS content "
                    "FROM chunks JOIN documents d ON d.id = chunks.doc_id WHERE chunks MATCH ?"
                    + "".join(" AND chunks.content LIKE ?" for _ in short_terms)
                    + " ORDER BY score LIMIT ?"
                )
                rows = c.execute(sql, [match, *like_params, limit * 4]).fetchall()
            else:
                sql = (
                    "SELECT chunks.doc_id AS doc_id, d.url, d.title, substr(chunks.content,1,200) AS snippet, "
                    "0 AS score, chunks.content AS content FROM chunks JOIN documents d ON d.id = chunks.doc_id WHERE "
                    + " AND ".join("chunks.content LIKE ?" for _ in short_terms)
                    + " LIMIT ?"
                )
                rows = c.execute(sql, [*like_params, limit * 4]).fetchall()
        out, seen = [], set()
        for r in rows:  # 문서당 가장 잘 맞는 조각 하나
            if r["doc_id"] in seen:
                continue
            seen.add(r["doc_id"])
            out.append(dict(r))
            if len(out) >= limit:
                break
        return out

    # ---------------- 작업 ----------------
    def create_job(self, kind: str, params: dict, actor: str) -> int:
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO jobs(kind,params,status,actor,created_at) VALUES(?,?,?,?,?)",
                (kind, json.dumps(params, ensure_ascii=False), "queued", actor, now_iso()),
            )
            return cur.lastrowid

    def update_job(self, job_id: int, **fields) -> None:
        if "result" in fields and not isinstance(fields["result"], str):
            fields["result"] = json.dumps(fields["result"], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))

    def get_job(self, job_id: int) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return self._job(r) if r else None

    def list_jobs(self, limit: int = 30) -> list[dict]:
        with self.conn() as c:
            return [self._job(r) for r in c.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))]

    def running_jobs(self, kinds: tuple[str, ...]) -> int:
        q = ",".join("?" * len(kinds))
        with self.conn() as c:
            return c.execute(
                f"SELECT count(*) FROM jobs WHERE status IN ('queued','running') AND kind IN ({q})", kinds
            ).fetchone()[0]

    def mark_interrupted(self) -> None:
        with self.conn() as c:
            c.execute(
                "UPDATE jobs SET status='interrupted', message='엔진이 재시작되어 중단됨', finished_at=? "
                "WHERE status IN ('queued','running')",
                (now_iso(),),
            )

    @staticmethod
    def _job(r) -> dict:
        d = dict(r)
        for k in ("params", "result"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except ValueError:
                    pass
        return d

    # ---------------- 모델 ----------------
    def add_model(self, kind: str, name: str, path: str, info: dict) -> int:
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO models(kind,name,path,info,created_at) VALUES(?,?,?,?,?)",
                (kind, name, path, json.dumps(info, ensure_ascii=False), now_iso()),
            )
            return cur.lastrowid

    def list_models(self) -> list[dict]:
        with self.conn() as c:
            out = []
            for r in c.execute("SELECT * FROM models ORDER BY id DESC"):
                d = dict(r)
                d["info"] = json.loads(d["info"] or "{}")
                out.append(d)
            return out

    def get_model(self, model_id: int) -> dict | None:
        return next((m for m in self.list_models() if m["id"] == model_id), None)

    # ---------------- 예약 수집 목록 ----------------
    def add_watch(self, kind: str, value: str, depth: int = 0) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO watchlist(kind,value,depth,created_at) VALUES(?,?,?,?)",
                (kind, value, depth, now_iso()),
            )

    def remove_watch(self, watch_id: int) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM watchlist WHERE id=?", (watch_id,))

    def list_watch(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM watchlist ORDER BY id")]

    def next_watch(self) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM watchlist ORDER BY coalesce(last_run,'') ASC, id ASC LIMIT 1").fetchone()
            return dict(r) if r else None

    def touch_watch(self, watch_id: int) -> None:
        with self.conn() as c:
            c.execute("UPDATE watchlist SET last_run=? WHERE id=?", (now_iso(), watch_id))

    # ---------------- 감사 기록 ----------------
    def audit(self, actor: str, action: str, detail: dict | str = "") -> None:
        if not isinstance(detail, str):
            detail = json.dumps(detail, ensure_ascii=False)
        with self.conn() as c:
            c.execute("INSERT INTO audit(ts,actor,action,detail) VALUES(?,?,?,?)", (now_iso(), actor, action, detail))

    def list_audit(self, limit: int = 50) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))]

    def backup_to(self, dest: str | Path) -> None:
        """일관된 DB 사본 (쓰기 중에도 안전)."""
        src = sqlite3.connect(self.path)
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
            src.close()
