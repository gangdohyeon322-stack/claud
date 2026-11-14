"""설정 파일 로드/저장과 PIN 인증 해시."""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import secrets
import threading
from pathlib import Path

DEFAULTS: dict = {
    # 수집(인터넷 접속)이 허용되는 시간대. days: 0=월 ... 6=일. end가 start보다 작으면 자정을 넘기는 구간.
    "timezone": "Asia/Seoul",
    "allowed_windows": [{"days": [0, 1, 2, 3, 4, 5, 6], "start": "00:00", "end": "24:00"}],
    # 비어 있으면 모든 도메인 허용(차단 목록은 항상 적용). 예: ["wikipedia.org", "go.kr"]
    "allowed_domains": [],
    "blocked_domains": [],
    "allow_private_network": False,
    "request_interval_sec": 3.0,
    "max_pages_per_job": 20,
    "max_page_bytes": 3_000_000,
    "min_text_chars": 200,
    "user_agent": "LearnBox/0.1 (personal research crawler)",
    "search": {
        # none | wikipedia | searxng | brave
        "provider": "wikipedia",
        "wikipedia_lang": "ko",
        "searxng_url": "",
        "brave_api_key": "",
        "results_to_fetch": 5,
    },
    "auto_collect": {"enabled": False, "interval_min": 60},
    "pc_training": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_steps": 200,
        "lr": 2e-4,
        "seq_len": 512,
        "batch_size": 2,
        "grad_accum": 4,
        "lora_r": 8,
    },
    "mobile_training": {
        "preset": "small",  # tiny | small
        "steps": 2000,
        "batch_size": 32,
        "lr": 3e-3,
        "vocab_size": 3000,
    },
    "auth": {"salt": "", "user_pin_hash": "", "admin_pin_hash": ""},
}

SECRET_KEYS = [("auth",), ("search", "brave_api_key")]


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def hash_pin(pin: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), 200_000).hex()


class Config:
    def __init__(self, data_dir: str | os.PathLike):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "config.json"
        self._lock = threading.Lock()
        self.data = copy.deepcopy(DEFAULTS)
        if self.path.exists():
            self.data = _merge(DEFAULTS, json.loads(self.path.read_text(encoding="utf-8")))

    # 경로
    @property
    def db_path(self) -> Path:
        return self.data_dir / "learnbox.db"

    @property
    def models_dir(self) -> Path:
        p = self.data_dir / "models"
        p.mkdir(exist_ok=True)
        return p

    @property
    def datasets_dir(self) -> Path:
        p = self.data_dir / "datasets"
        p.mkdir(exist_ok=True)
        return p

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)

    def update(self, changes: dict) -> None:
        """사용자 입력으로 설정 변경. 인증 정보는 이 경로로 바꿀 수 없다."""
        changes = {k: v for k, v in changes.items() if k in DEFAULTS and k != "auth"}
        self.data = _merge(self.data, changes)
        self.save()

    def public(self) -> dict:
        """비밀값을 가린 사본 (UI 표시, 외장 저장장치 내보내기용)."""
        d = copy.deepcopy(self.data)
        d.pop("auth", None)
        if d.get("search", {}).get("brave_api_key"):
            d["search"]["brave_api_key"] = "********"
        return d

    # 인증
    @property
    def initialized(self) -> bool:
        a = self.data["auth"]
        return bool(a.get("salt") and a.get("user_pin_hash") and a.get("admin_pin_hash"))

    def set_pins(self, user_pin: str, admin_pin: str) -> None:
        if len(user_pin) < 4 or len(admin_pin) < 6:
            raise ValueError("사용자 PIN은 4자 이상, 관리자 PIN은 6자 이상이어야 합니다.")
        if user_pin == admin_pin:
            raise ValueError("사용자 PIN과 관리자 PIN은 달라야 합니다.")
        salt = secrets.token_hex(16)
        self.data["auth"] = {
            "salt": salt,
            "user_pin_hash": hash_pin(user_pin, salt),
            "admin_pin_hash": hash_pin(admin_pin, salt),
        }
        self.save()

    def check_pin(self, pin: str) -> str | None:
        """PIN이 맞으면 역할('admin' | 'user')을 돌려준다."""
        a = self.data["auth"]
        if not self.initialized:
            return None
        h = hash_pin(pin, a["salt"])
        if hmac.compare_digest(h, a["admin_pin_hash"]):
            return "admin"
        if hmac.compare_digest(h, a["user_pin_hash"]):
            return "user"
        return None
