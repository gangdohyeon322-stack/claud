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
CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    sha256 TEXT UNIQUE NOT NULL,
    size INTEGER,
    mime TEXT,
    path TEXT,
    docs INTEGER DEFAULT 0,
    actor TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS embeddings(
    chunk_id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_doc ON embeddings(doc_id);
CREATE TABLE IF NOT EXISTS notifications(
    id INTEGER PRIMARY KEY,
    ts TEXT,
    level TEXT,
    title TEXT,
    body TEXT,
    data TEXT,
    read INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS learn_sessions(
    id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    actor TEXT,
    started_at TEXT,
    ended_at TEXT,
    config TEXT,
    stats TEXT,
    result TEXT
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
            self._migrate(c)

    @staticmethod
    def _migrate(c) -> None:
        """이전 버전 DB에 없는 컬럼을 추가한다."""
        def cols(table):
            return {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
        if "origin" not in cols("watchlist"):
            c.execute("ALTER TABLE watchlist ADD COLUMN origin TEXT DEFAULT 'user'")
        if "file_id" not in cols("documents"):
            c.execute("ALTER TABLE documents ADD COLUMN file_id INTEGER")
        # 예전 모델 종류 이름 -> 새 이름
        c.execute("UPDATE models SET kind='tiny' WHERE kind='mobile'")
        c.execute("UPDATE models SET kind='lora' WHERE kind='pc'")

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
    def add_document(self, url: str, title: str, text: str, source: str, query: str | None = None,
                     file_id: int | None = None) -> str:
        """문서를 저장하고 인덱싱한다. 반환: 'new' | 'updated' | 'unchanged'."""
        h = hashlib.sha256(text.encode()).hexdigest()
        with self.conn() as c:
            row = c.execute("SELECT id, content_hash FROM documents WHERE url=?", (url,)).fetchone()
            if row and row["content_hash"] == h:
                return "unchanged"
            if row:
                doc_id = row["id"]
                c.execute(
                    "UPDATE documents SET title=?, text=?, source=?, query=?, content_hash=?, fetched_at=?, "
                    "file_id=coalesce(?, file_id) WHERE id=?",
                    (title, text, source, query, h, now_iso(), file_id, doc_id),
                )
                c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
                c.execute("DELETE FROM embeddings WHERE doc_id=?", (doc_id,))
                status = "updated"
            else:
                cur = c.execute(
                    "INSERT INTO documents(url,title,text,source,query,content_hash,fetched_at,file_id) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (url, title, text, source, query, h, now_iso(), file_id),
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
            c.execute("DELETE FROM embeddings WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def list_documents(self, limit: int = 50, offset: int = 0, source: str | None = None) -> list[dict]:
        where, params = ("WHERE source=?", [source]) if source else ("", [])
        with self.conn() as c:
            rows = c.execute(
                f"SELECT id,url,title,source,query,fetched_at,file_id,length(text) AS chars FROM documents {where} "
                "ORDER BY id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_document(self, doc_id: int) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT *, length(text) AS chars FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(r) if r else None

    def iter_documents(self, since: str | None = None):
        """전체 문서 (내보내기용). since(ISO 시각) 이후 수집분만 고를 수 있다."""
        sql, params = "SELECT * FROM documents", []
        if since:
            sql, params = sql + " WHERE fetched_at >= ?", [since]
        with self.conn() as c:
            for r in c.execute(sql + " ORDER BY id", params):
                yield dict(r)

    def chars_since(self, since: str | None) -> int:
        """since 이후 새로 저장/갱신된 문서의 글자 수 (자율 학습의 학습 시점 판단용)."""
        with self.conn() as c:
            if not since:
                return c.execute("SELECT coalesce(sum(length(text)),0) FROM documents").fetchone()[0]
            return c.execute(
                "SELECT coalesce(sum(length(text)),0) FROM documents WHERE fetched_at >= ?", (since,)
            ).fetchone()[0]

    def source_counts(self) -> dict:
        with self.conn() as c:
            return {r[0] or "unknown": r[1] for r in c.execute("SELECT source, count(*) FROM documents GROUP BY source")}

    def counts(self) -> dict:
        with self.conn() as c:
            d = c.execute("SELECT count(*), coalesce(sum(length(text)),0) FROM documents").fetchone()
            ch = c.execute("SELECT count(*) FROM chunks").fetchone()[0]
            emb = c.execute("SELECT count(*) FROM embeddings").fetchone()[0]
            files = c.execute("SELECT count(*) FROM files").fetchone()[0]
            return {"documents": d[0], "chars": d[1], "chunks": ch, "embeddings": emb, "files": files}

    def iter_chunks(self, doc_ids: list[int] | None = None):
        """검색 조각. 각 행: chunk_id(=FTS rowid), doc_id, title, content, url, source."""
        sql = ("SELECT chunks.rowid AS chunk_id, chunks.doc_id AS doc_id, chunks.title AS title, "
               "chunks.content AS content, d.url AS url, d.source AS source "
               "FROM chunks JOIN documents d ON d.id = chunks.doc_id")
        params: list = []
        if doc_ids:
            sql += f" WHERE chunks.doc_id IN ({','.join('?' * len(doc_ids))})"
            params = list(doc_ids)
        with self.conn() as c:
            for r in c.execute(sql, params):
                yield dict(r)

    def get_chunks(self, chunk_ids: list[int]) -> list[dict]:
        if not chunk_ids:
            return []
        q = ",".join("?" * len(chunk_ids))
        with self.conn() as c:
            rows = c.execute(
                "SELECT chunks.rowid AS chunk_id, chunks.doc_id AS doc_id, chunks.title AS title, "
                "chunks.content AS content, d.url AS url, d.source AS source FROM chunks "
                f"JOIN documents d ON d.id = chunks.doc_id WHERE chunks.rowid IN ({q})", chunk_ids).fetchall()
        by_id = {r["chunk_id"]: dict(r) for r in rows}
        return [by_id[i] for i in chunk_ids if i in by_id]

    def chunk_snapshot(self, max_chunks: int | None = None) -> list[dict]:
        """휴대폰 오프라인 검색용: 조각 단위 사본 (최근 문서 우선)."""
        sql = ("SELECT chunks.rowid AS c, chunks.doc_id AS d, chunks.title AS t, chunks.content AS x, doc.url AS u, "
               "doc.source AS s FROM chunks JOIN documents doc ON doc.id = chunks.doc_id ORDER BY chunks.doc_id DESC")
        params: list = []
        if max_chunks:
            sql += " LIMIT ?"
            params = [int(max_chunks)]
        with self.conn() as c:
            return [dict(r) for r in c.execute(sql, params)]

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
    def add_watch(self, kind: str, value: str, depth: int = 0, origin: str = "user") -> bool:
        """origin: user(사람이 등록) | auto(자율 학습이 발견). 새로 추가되면 True."""
        with self.conn() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO watchlist(kind,value,depth,created_at,origin) VALUES(?,?,?,?,?)",
                (kind, value, depth, now_iso(), origin),
            )
            return cur.rowcount > 0

    def count_watch(self, origin: str | None = None) -> int:
        with self.conn() as c:
            if origin:
                return c.execute("SELECT count(*) FROM watchlist WHERE origin=?", (origin,)).fetchone()[0]
            return c.execute("SELECT count(*) FROM watchlist").fetchone()[0]

    def remove_watch(self, watch_id: int) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM watchlist WHERE id=?", (watch_id,))

    def list_watch(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM watchlist ORDER BY id")]

    def next_watch(self, origin: str | None = None) -> dict | None:
        """가장 오래전에 수집한(또는 한 번도 안 한) 항목. 사람이 등록한 항목이 같은 조건이면 먼저."""
        where, params = ("WHERE origin=?", [origin]) if origin else ("", [])
        with self.conn() as c:
            r = c.execute(
                f"SELECT * FROM watchlist {where} ORDER BY coalesce(last_run,'') ASC, "
                "CASE origin WHEN 'user' THEN 0 ELSE 1 END, id ASC LIMIT 1", params).fetchone()
            return dict(r) if r else None

    def touch_watch(self, watch_id: int) -> None:
        with self.conn() as c:
            c.execute("UPDATE watchlist SET last_run=? WHERE id=?", (now_iso(), watch_id))

    # ---------------- 가져온 파일 ----------------
    def find_file(self, sha256: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM files WHERE sha256=?", (sha256,)).fetchone()
            return dict(r) if r else None

    def add_file(self, name: str, sha256: str, size: int, mime: str, path: str, actor: str) -> int:
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO files(name,sha256,size,mime,path,actor,created_at) VALUES(?,?,?,?,?,?,?)",
                (name, sha256, size, mime, path, actor, now_iso()),
            )
            return cur.lastrowid

    def set_file_docs(self, file_id: int, docs: int) -> None:
        with self.conn() as c:
            c.execute("UPDATE files SET docs=? WHERE id=?", (docs, file_id))

    def list_files(self, limit: int = 200) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM files ORDER BY id DESC LIMIT ?", (limit,))]

    # ---------------- 임베딩 (의미 검색) ----------------
    def missing_embedding_chunks(self, model: str, limit: int = 256) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT chunks.rowid AS chunk_id, chunks.doc_id AS doc_id, chunks.title AS title, chunks.content AS content "
                "FROM chunks LEFT JOIN embeddings e ON e.chunk_id = chunks.rowid AND e.model = ? "
                "WHERE e.chunk_id IS NULL LIMIT ?", (model, limit)).fetchall()
            return [dict(r) for r in rows]

    def upsert_embeddings(self, rows: list[tuple[int, int, str, int, bytes]]) -> None:
        """rows: (chunk_id, doc_id, model, dim, vec_bytes[float32])"""
        with self.conn() as c:
            c.executemany("INSERT OR REPLACE INTO embeddings(chunk_id,doc_id,model,dim,vec) VALUES(?,?,?,?,?)", rows)

    def iter_embeddings(self, model: str):
        with self.conn() as c:
            for r in c.execute("SELECT chunk_id, doc_id, dim, vec FROM embeddings WHERE model=?", (model,)):
                yield r["chunk_id"], r["doc_id"], r["dim"], r["vec"]

    def embedding_count(self, model: str) -> int:
        with self.conn() as c:
            return c.execute("SELECT count(*) FROM embeddings WHERE model=?", (model,)).fetchone()[0]

    # ---------------- 알림 ----------------
    def notify(self, level: str, title: str, body: str = "", data: dict | None = None) -> int:
        """level: info | success | warning | error"""
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO notifications(ts,level,title,body,data) VALUES(?,?,?,?,?)",
                (now_iso(), level, title, body, json.dumps(data or {}, ensure_ascii=False)),
            )
            return cur.lastrowid

    def list_notifications(self, limit: int = 50, unread_only: bool = False) -> list[dict]:
        where = "WHERE read=0" if unread_only else ""
        with self.conn() as c:
            out = []
            for r in c.execute(f"SELECT * FROM notifications {where} ORDER BY id DESC LIMIT ?", (limit,)):
                d = dict(r)
                d["data"] = json.loads(d["data"] or "{}")
                out.append(d)
            return out

    def unread_notifications(self) -> int:
        with self.conn() as c:
            return c.execute("SELECT count(*) FROM notifications WHERE read=0").fetchone()[0]

    def mark_notifications_read(self, ids: list[int] | None = None) -> None:
        with self.conn() as c:
            if ids:
                c.execute(f"UPDATE notifications SET read=1 WHERE id IN ({','.join('?' * len(ids))})", ids)
            else:
                c.execute("UPDATE notifications SET read=1")

    # ---------------- 자율 학습 세션 ----------------
    def create_session(self, actor: str, config: dict) -> int:
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO learn_sessions(status,actor,started_at,config,stats) VALUES('running',?,?,?,?)",
                (actor, now_iso(), json.dumps(config, ensure_ascii=False), "{}"),
            )
            return cur.lastrowid

    def update_session(self, session_id: int, **fields) -> None:
        for k in ("config", "stats", "result"):
            if k in fields and not isinstance(fields[k], str):
                fields[k] = json.dumps(fields[k], ensure_ascii=False)
        cols = ", ".join(f"{k}=?" for k in fields)
        with self.conn() as c:
            c.execute(f"UPDATE learn_sessions SET {cols} WHERE id=?", (*fields.values(), session_id))

    def get_session(self, session_id: int) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM learn_sessions WHERE id=?", (session_id,)).fetchone()
            return self._session(r) if r else None

    def active_session(self) -> dict | None:
        """진행 중(running) 또는 마무리 중(finishing)인 세션."""
        with self.conn() as c:
            r = c.execute(
                "SELECT * FROM learn_sessions WHERE status IN ('running','finishing') ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return self._session(r) if r else None

    def list_sessions(self, limit: int = 20) -> list[dict]:
        with self.conn() as c:
            return [self._session(r) for r in c.execute("SELECT * FROM learn_sessions ORDER BY id DESC LIMIT ?", (limit,))]

    @staticmethod
    def _session(r) -> dict:
        d = dict(r)
        for k in ("config", "stats", "result"):
            d[k] = json.loads(d[k]) if d.get(k) else ({} if k != "result" else None)
        return d

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
