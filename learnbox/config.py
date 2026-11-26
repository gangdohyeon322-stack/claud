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
    # 휴대폰/태블릿 브라우저에서 직접 학습(온디바이스). 올리기 상한과 기본 학습 설정.
    "mobile_training": {"max_upload_mb": 64, "max_params": 2_000_000, "default_steps": 100, "default_batch": 8,
                        "default_seq_len": 64, "checkpoint_every": 20, "thermal_pause": True},
    # 소형 AI 모델(처음부터 학습하는 글자 단위 GPT). PC와 휴대폰(온보드) 공용. preset 또는 직접 지정한 크기.
    "tiny_training": {
        "preset": "auto",  # auto | nano | mini | small | base | large | custom
        "d_model": 0, "n_layer": 0, "n_head": 0, "seq_len": 0, "vocab_size": 0,  # 0이면 preset 값 사용
        "steps": 1500,
        "batch_size": 32,
        "lr": 3e-3,
    },
    # PC 학습 중간 저장(체크포인트)과 일시정지. 끊겨도 마지막 저장 지점부터 이어서 할 수 있다.
    "training_checkpoint": {
        "enabled": True,
        "every_minutes": 10,   # 이 시간마다 저장 (0이면 시간 기준 저장 안 함)
        "every_steps": 0,      # 이 단계마다 저장 (0이면 단계 기준 저장 안 함)
        "dir": "",             # 비우면 데이터 폴더/checkpoints. 외장 하드에서 학습할 때는 내장 드라이브 경로를 권장
    },
    # 학습 장치: auto | cuda(NVIDIA·AMD ROCm) | tpu | xpu(Intel Arc) | mps(Mac) | cpu.
    # 우선순위: 학습 요청의 device(웹 폼, --device) → 환경변수 LEARNBOX_DEVICE → 이 값 → auto (docs/PRD-tpu-training.md 6.1)
    "training_device": "auto",
    # 공개 소형 언어모델에 LoRA 추가 학습 (PC)
    "lora_training": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "max_steps": 200,
        "lr": 2e-4,
        "seq_len": 512,
        "batch_size": 2,
        "grad_accum": 4,
        "lora_r": 8,
        "lora_alpha": 16,
        "use_qa": True,
        # 정밀도: auto(bf16 가능하면 bf16, 아니면 fp16 혼합 정밀도) | fp16 | off(예전 방식) | fp32.
        # 우선순위: 요청의 amp → 환경변수 LEARNBOX_AMP → 이 값 → auto
        "amp": "auto",
    },
    # 온라인 모델 (Claude API). 검색된 자료 일부가 Anthropic API로 전송된다.
    "online": {
        "enabled": True,
        "model": "claude-opus-5-5",
        "api_key": "",  # 비우면 ANTHROPIC_API_KEY 환경변수 / ant auth 프로필 사용
        "effort": "medium",  # low | medium | high | xhigh | max
        "max_context_chunks": 8,
        "fallbacks": True,  # 거절 시 서버 측 대체 모델로 자동 재시도
    },
    # 의미 검색(임베딩). transformers가 있으면 사용.
    "embeddings": {"enabled": True, "model": "intfloat/multilingual-e5-small", "batch_size": 32},
    # 파일 가져오기
    "ingest": {"max_file_mb": 50, "max_docs_per_file": 5000},
    # 자율 학습: 허용 시간대에 스스로 수집하고 주기적으로 학습한다.
    "autolearn": {
        "targets": {"tiny": True, "lora": False, "embeddings": True, "qa": False},
        "collect_interval_min": 10,
        "pages_per_cycle": 5,
        "max_pages_per_hour": 60,
        "discover_topics": True,
        "max_discovered_topics": 50,
        "train_min_new_chars": 20000,
        "train_interval_min": 60,
        "steps_per_cycle": 300,
        "train_outside_window": False,
        "threads": 0,  # 0이면 자동 (PC 코어 수 - 1)
        "final_train": True,
        "formats": ["jsonl", "csv", "md", "txt"],
        "results_export_target": "",  # 지정하면 학습 종료 시 이 외장 저장장치에도 복사
    },
    "auth": {"salt": "", "user_pin_hash": "", "admin_pin_hash": ""},
}

SECRET_KEYS = [("auth",), ("search", "brave_api_key"), ("online", "api_key")]
MASK = "********"


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def device_value(value, name: str = "device") -> str:
    """학습 장치 값 검사 -> 정식 이름 (auto|cuda|tpu|xpu|mps|cpu, 번호 붙은 cuda:1·xpu:0도 허용).

    별칭(gpu·rocm→cuda, xla→tpu, arc→xpu, mac→mps)은 accel.normalize가 바꾼다. 빈 값은 auto.
    잘못된 값이면 한국어 ValueError (서버는 400)."""
    from . import accel   # torch를 불러오지 않는 가벼운 모듈

    if value is None or (isinstance(value, str) and not value.strip()):
        return "auto"
    if not isinstance(value, str):
        raise ValueError(f"'{name}' 값이 올바르지 않습니다: {value!r} (auto, cuda, tpu, xpu, mps, cpu 중 하나)")
    if value.strip().lower().startswith(("tpu:", "xla:")):  # TPU는 칩 1개(v5e-1)만 쓰므로 번호를 받지 않는다
        raise ValueError(f"'{name}' 값이 올바르지 않습니다: {value!r} (auto, cuda, tpu, xpu, mps, cpu 중 하나)")
    n = accel.normalize(value)
    base, sep, idx = n.partition(":")
    if base in accel.DEVICES and (not sep or (idx.isdigit() and base in ("cuda", "xpu"))):
        return n
    raise ValueError(f"'{name}' 값이 올바르지 않습니다: {value!r} (auto, cuda, tpu, xpu, mps, cpu 중 하나)")


def amp_value(value, name: str = "amp") -> str:
    """LoRA 정밀도 모드 값 검사 -> auto|fp16|off|fp32 (float16·half→fp16, 0·false→off 등). 빈 값은 auto.
    잘못된 값이면 한국어 ValueError."""
    from . import accel

    if value is None:
        return "auto"
    m = accel.normalize_amp(value) if isinstance(value, (str, int)) else None   # bool도 int (true→fp16, false→off)
    if m is None:
        raise ValueError(f"'{name}' 값이 올바르지 않습니다: {value!r} (auto, fp16, off, fp32 중 하나)")
    return m


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
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.data = _merge(DEFAULTS, saved)
            self._migrate_v1(saved)

    def _migrate_v1(self, saved: dict) -> None:
        """v1 설정 이름(pc_training, mobile_training)의 값을 새 이름(lora_training, tiny_training)으로 옮긴다."""
        old = saved.get("pc_training")
        if "lora_training" not in saved and isinstance(old, dict):
            self.data["lora_training"] = _merge(DEFAULTS["lora_training"],
                                                {k: v for k, v in old.items() if k in DEFAULTS["lora_training"]})
        old = saved.get("mobile_training")
        if "tiny_training" not in saved and isinstance(old, dict):  # 크기 이름(tiny/small)은 뜻이 달라져 옮기지 않는다
            self.data["tiny_training"] = _merge(DEFAULTS["tiny_training"],
                                                {k: v for k, v in old.items() if k in ("steps", "batch_size", "lr")})

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
        for path in SECRET_KEYS[1:]:  # 화면에 가려 보낸 값(********)이 돌아오면 기존 값 유지
            sec, key = path
            if isinstance(changes.get(sec), dict) and changes[sec].get(key) == MASK:
                changes[sec].pop(key)
        self.data = _merge(self.data, changes)
        self.save()

    def public(self) -> dict:
        """비밀값을 가린 사본 (UI 표시, 외장 저장장치 내보내기용)."""
        d = copy.deepcopy(self.data)
        d.pop("auth", None)
        for sec, key in SECRET_KEYS[1:]:
            if d.get(sec, {}).get(key):
                d[sec][key] = MASK
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
