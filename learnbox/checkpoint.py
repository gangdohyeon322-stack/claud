"""학습 중간 저장(체크포인트) 공용 도우미.

체크포인트 폴더(root) 구조:
    root/current/   마지막으로 끝까지 저장된 체크포인트 (checkpoint.json + 학습 상태 파일들)
    root/prev/      바꿔 끼우는 동안 잠깐만 존재 (current가 깨졌을 때 대신 읽음)
    root/.tmp/      쓰는 중인 체크포인트 (commit 전에는 절대 읽지 않음)

쓰기 순서: begin() → 호출한 쪽이 tmp 폴더에 상태 파일을 쓴다 → commit(meta).
commit은 checkpoint.json(complete=True)을 마지막에 쓰고 디렉터리를 바꿔 끼운다.
어느 순간에 전원이 나가거나 저장장치가 빠져도 current 또는 prev 중 하나는 온전하다.
(Windows는 이미 있는 폴더 위로 os.replace를 못 하므로 rename 두 번으로 바꿔 끼운다.)
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

FORMAT = 1
META = "checkpoint.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fsync_file(path: Path) -> None:
    try:
        with open(path, "rb+") as f:
            os.fsync(f.fileno())
    except OSError:
        pass


def fsync_tree(path: Path) -> None:
    """폴더 안 파일들을 디스크에 확실히 기록한다 (가능한 경우만, 실패는 무시)."""
    for p in Path(path).rglob("*"):
        if p.is_file():
            _fsync_file(p)


def dir_size(path: str | os.PathLike) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def begin(root: str | os.PathLike) -> Path:
    """새 체크포인트를 쓸 빈 임시 폴더를 만들어 돌려준다 (이전에 쓰다 만 임시 폴더는 지운다)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / ".tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    return tmp


def commit(root: str | os.PathLike, tmp: str | os.PathLike, meta: dict) -> dict:
    """tmp에 쓴 상태 파일들을 확정한다. meta는 checkpoint.json에 들어간다 (step 키 권장).

    반환: {"path", "step", "saved_at", "bytes"}"""
    root, tmp = Path(root), Path(tmp)
    meta = dict(meta)
    meta["format"] = FORMAT
    meta["saved_at"] = _now()
    meta["complete"] = True
    mp = tmp / META
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    fsync_tree(tmp)
    cur, prev = root / "current", root / "prev"
    if prev.exists():
        shutil.rmtree(prev, ignore_errors=True)
    if cur.exists():
        os.rename(cur, prev)
    os.rename(tmp, cur)
    if prev.exists():
        shutil.rmtree(prev, ignore_errors=True)
    return {"path": str(root), "step": meta.get("step"), "saved_at": meta["saved_at"], "bytes": dir_size(cur)}


def _read_meta(d: Path) -> dict | None:
    try:
        meta = json.loads((d / META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not meta.get("complete") or meta.get("format") != FORMAT:
        return None
    return meta


def load(root: str | os.PathLike | None) -> tuple[dict, Path] | None:
    """가장 최근의 온전한 체크포인트 -> (meta, 상태 파일이 든 폴더). 없으면 None."""
    if not root:
        return None
    root = Path(root)
    for name in ("current", "prev"):
        d = root / name
        if d.is_dir():
            meta = _read_meta(d)
            if meta is not None:
                return meta, d
    return None


def summary(root: str | os.PathLike | None) -> dict | None:
    """작업 목록/화면에 보여 줄 짧은 정보: {"path","kind","step","steps","saved_at","bytes","device"} 또는 None.

    device: 저장한 학습 장치의 정식 이름(tpu 등). 예전 체크포인트에는 없어서 None."""
    got = load(root)
    if not got:
        return None
    meta, d = got
    return {"path": str(Path(root)), "kind": meta.get("kind"), "step": meta.get("step"), "steps": meta.get("steps"),
            "saved_at": meta.get("saved_at"), "bytes": dir_size(d), "device": meta.get("device")}


def remove(root: str | os.PathLike | None) -> None:
    if root and Path(root).exists():
        shutil.rmtree(root, ignore_errors=True)
