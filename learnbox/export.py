"""관리자 전용: 외장 저장장치(USB/외장 SSD/HDD)로 내보내기.

내보낸 폴더에는 manifest.json(파일별 SHA-256)이 들어가며, verify()로 무결성을 확인할 수 있다.
설정 파일은 PIN 해시와 API 키를 뺀 사본만 저장한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import string
from datetime import datetime
from pathlib import Path

INCLUDE_CHOICES = ("database", "models", "datasets")


def list_drives() -> list[dict]:
    """연결된 외장 저장장치 후보 (OS별 마운트 위치)."""
    cands: list[Path] = []
    system = platform.system()
    if system == "Windows":
        cands = [Path(f"{c}:\\") for c in string.ascii_uppercase if c not in "AC" and Path(f"{c}:\\").exists()]
    elif system == "Darwin":
        cands = [p for p in Path("/Volumes").glob("*") if p.is_dir() and not p.is_symlink()]
    else:
        for base in (Path("/media"), Path("/run/media"), Path("/mnt")):
            if base.exists():
                for p in base.glob("*"):
                    if p.is_dir():
                        sub = [q for q in p.glob("*") if q.is_dir() and os.path.ismount(q)]
                        cands.extend(sub if sub else ([p] if os.path.ismount(p) else []))
    out = []
    for p in cands:
        try:
            u = shutil.disk_usage(p)
            if u.total == 0 or not os.access(p, os.W_OK):
                continue  # 읽기 전용/가상 마운트 제외
            out.append({"path": str(p), "free_gb": round(u.free / 1e9, 1), "total_gb": round(u.total / 1e9, 1)})
        except OSError:
            continue
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def validate_target(cfg, target: str) -> Path:
    target_dir = Path(target).expanduser()
    if not target_dir.is_dir():
        raise ValueError(f"대상 폴더가 없습니다 (저장장치가 연결되어 있는지 확인하세요): {target_dir}")
    if not os.access(target_dir, os.W_OK):
        raise ValueError(f"대상에 쓸 수 없습니다: {target_dir}")
    resolved = target_dir.resolve()
    if resolved == cfg.data_dir or cfg.data_dir in resolved.parents:
        raise ValueError("데이터 폴더 안으로는 내보낼 수 없습니다.")
    return target_dir


def export(cfg, db, target: str, include: list[str], actor: str, progress=lambda p, m: None,
           check_cancel=lambda: None) -> dict:
    target_dir = validate_target(cfg, target)
    include = [i for i in include if i in INCLUDE_CHOICES] or list(INCLUDE_CHOICES)

    need = 0
    if "database" in include:
        need += cfg.db_path.stat().st_size if cfg.db_path.exists() else 0
    if "models" in include:
        need += _dir_size(cfg.models_dir)
    if "datasets" in include:
        need += _dir_size(cfg.datasets_dir)
    free = shutil.disk_usage(target_dir).free
    if need * 1.1 > free:
        raise ValueError(f"공간 부족: 필요 약 {need/1e9:.2f}GB, 남은 공간 {free/1e9:.2f}GB")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = target_dir / f"LearnBox-export-{stamp}"
    out.mkdir()
    progress(0.02, "내보내기 시작")

    if "database" in include:
        db.backup_to(out / "learnbox.db")
    (out / "config.public.json").write_text(json.dumps(cfg.public(), ensure_ascii=False, indent=2), encoding="utf-8")

    copy_jobs = []
    if "models" in include:
        copy_jobs.append((cfg.models_dir, out / "models"))
    if "datasets" in include:
        copy_jobs.append((cfg.datasets_dir, out / "datasets"))
    files = [(f, dst / f.relative_to(src)) for src, dst in copy_jobs for f in src.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f, _ in files) or 1
    done = 0
    for f, dst in files:
        check_cancel()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        done += f.stat().st_size
        progress(0.05 + 0.8 * done / total, f"복사 중: {f.name}")

    progress(0.9, "체크섬 계산 중")
    manifest_files = {}
    for f in sorted(out.rglob("*")):
        if f.is_file():
            manifest_files[str(f.relative_to(out)).replace(os.sep, "/")] = {"size": f.stat().st_size, "sha256": _sha256(f)}
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "exported_by": actor,
        "include": include,
        "counts": db.counts(),
        "models": [{k: m[k] for k in ("id", "kind", "name", "created_at")} for m in db.list_models()],
        "files": manifest_files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.txt").write_text(
        "LearnBox 내보내기\n\n"
        "- learnbox.db: 수집 자료와 검색 인덱스 (SQLite)\n"
        "- models/: 학습된 모델 (tiny-*: 소형 AI 모델과 ONNX 파일, lora-*: LoRA 어댑터)\n"
        "- datasets/: 학습에 사용한 말뭉치\n"
        "- manifest.json: 파일별 SHA-256. 확인: python -m learnbox verify <이 폴더>\n"
        "- 다른 PC에서 쓰기: python -m learnbox --data <새 데이터 폴더> import <이 폴더>\n",
        encoding="utf-8",
    )
    size = sum(v["size"] for v in manifest_files.values())
    result = {"path": str(out), "files": len(manifest_files), "bytes": size}
    db.audit(actor, "export", result)
    return result


def verify(export_dir: str) -> dict:
    root = Path(export_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bad, missing = [], []
    for rel, meta in manifest["files"].items():
        f = root / rel
        if not f.exists():
            missing.append(rel)
        elif _sha256(f) != meta["sha256"]:
            bad.append(rel)
    return {"ok": not bad and not missing, "checked": len(manifest["files"]), "corrupted": bad, "missing": missing}


def import_export(export_dir: str, cfg, db) -> dict:
    """내보낸 폴더를 다른 PC의 데이터 폴더로 가져온다 (모델/말뭉치 복사 + 문서 병합)."""
    import sqlite3

    root = Path(export_dir)
    v = verify(export_dir)
    if not v["ok"]:
        raise ValueError(f"무결성 검사 실패: {v}")
    merged = 0
    src_db = root / "learnbox.db"
    if src_db.exists():
        s = sqlite3.connect(str(src_db))
        for url, title, text, source, query in s.execute("SELECT url,title,text,source,query FROM documents"):
            if db.add_document(url, title, text, source, query) != "unchanged":
                merged += 1
        s.close()
    for sub, dst in (("models", cfg.models_dir), ("datasets", cfg.datasets_dir)):
        src = root / sub
        if src.exists():
            for d in src.iterdir():
                if d.is_dir() and not (dst / d.name).exists():
                    shutil.copytree(d, dst / d.name)
                    if sub == "models":
                        info = json.loads((d / "info.json").read_text(encoding="utf-8")) if (d / "info.json").exists() else {}
                        kind = {"mobile": "tiny", "pc": "lora"}.get(info.get("kind"), info.get("kind", "unknown"))
                        db.add_model(kind, d.name, str(dst / d.name), info)
    return {"documents_merged": merged}
