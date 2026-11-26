"""PC 성능 프로필(CPU/RAM/GPU, 설치된 라이브러리)과 그에 맞는 학습 설정 추천.

psutil 없이 표준 라이브러리로 RAM을 읽는다 (Linux /proc/meminfo, macOS sysctl, Windows GlobalMemoryStatusEx).
GPU는 torch가 설치돼 있을 때만 확인한다. 결과는 60초 동안 캐시한다.
휴대폰/브라우저 쪽 성능 판단은 web/onboard.js(Onboard.detectDevice)가 맡는다.
"""
from __future__ import annotations

import copy
import importlib.util
import os
import platform
import subprocess
import sys
import threading
import time

CACHE_SEC = 60.0
_LOCK = threading.Lock()
_CACHE: dict = {"t": 0.0, "v": None}

MODULES = ("torch", "transformers", "peft", "onnx", "onnxruntime", "anthropic", "pypdf", "torch_xla")

LORA_MODELS = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
    "7B": "Qwen/Qwen2.5-7B-Instruct",
}
TPU_HBM_DEFAULT_GB = 16.0     # 코랩 TPU v5e-1 칩 1개 (메모리를 알 수 없을 때)


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:  # 잘못 설치된 패키지 등
        return False


def _cpu_count() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except Exception:
        return max(1, os.cpu_count() or 1)


def _cgroup_limit_bytes() -> int | None:
    """컨테이너(cgroup) 메모리 제한이 있으면 그 값."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path) as f:
                s = f.read().strip()
        except OSError:
            continue
        if s.isdigit():
            v = int(s)
            if 0 < v < (1 << 60):
                return v
    return None


def _ram_linux() -> tuple[float | None, float | None]:
    total = avail = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, rest = line.partition(":")
                parts = rest.split()
                if not parts or not parts[0].isdigit():
                    continue
                if key == "MemTotal":
                    total = int(parts[0]) * 1024
                elif key == "MemAvailable":
                    avail = int(parts[0]) * 1024
    except OSError:
        return None, None
    lim = _cgroup_limit_bytes()
    if total and lim and lim < total:
        total = lim
        if avail:
            avail = min(avail, lim)
    return total, avail


def _ram_macos() -> tuple[float | None, float | None]:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3)
        v = int(out.stdout.strip())
        return (v if v > 0 else None), None
    except Exception:
        return None, None


def _ram_windows() -> tuple[float | None, float | None]:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
            return None, None
        return int(st.ullTotalPhys) or None, int(st.ullAvailPhys) or None
    except Exception:
        return None, None


def _ram_bytes() -> tuple[float | None, float | None]:
    sysname = platform.system() or ""
    if sysname == "Darwin":
        return _ram_macos()
    if sysname == "Windows":
        return _ram_windows()
    if sysname == "Linux" or (not sysname and sys.platform.startswith("linux")):
        return _ram_linux()
    try:  # 기타 POSIX
        pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        return (pages * size if pages > 0 and size > 0 else None), None
    except (ValueError, OSError, AttributeError):
        return None, None


def _gpu_info(ram_gb: float | None) -> tuple[dict, str | None]:
    gpu = {"type": None, "name": None, "vram_gb": None}
    if not _has("torch"):
        return gpu, None
    try:
        import torch

        version = getattr(torch, "__version__", None)
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = {"type": "cuda", "name": torch.cuda.get_device_name(0),
                   "vram_gb": round(props.total_memory / 1024 ** 3, 1), "count": torch.cuda.device_count()}
            hip = getattr(torch.version, "hip", None)
            if hip:   # AMD ROCm용 PyTorch도 장치 이름은 cuda (PRD 6.1)
                gpu.update(vendor="amd", rocm=str(hip))
        else:
            xpu = _xpu_info(torch)
            mps = getattr(torch.backends, "mps", None)
            if xpu is not None:
                gpu = xpu
            elif mps is not None and mps.is_available():
                # Apple Silicon은 CPU와 메모리를 함께 쓴다: 전체 RAM의 약 60%를 GPU용으로 본다.
                gpu = {"type": "mps", "name": "Apple Silicon GPU",
                       "vram_gb": round(ram_gb * 0.6, 1) if ram_gb else None, "unified_memory": True}
        return gpu, version
    except Exception as e:  # GPU 드라이버 문제 등으로 실패해도 CPU 프로필은 돌려준다
        gpu["error"] = f"{type(e).__name__}: {e}"[:200]
        return gpu, None


def _xpu_info(torch) -> dict | None:
    """Intel GPU(xpu) 정보. xpu용 PyTorch가 아니거나 확인 중 오류가 나면 None (다른 장치 확인을 막지 않는다)."""
    try:
        x = getattr(torch, "xpu", None)
        if x is None or not x.is_available():
            return None
        out = {"type": "xpu", "name": None, "vram_gb": None, "count": None}
        try:
            out["name"] = x.get_device_name(0)
            out["vram_gb"] = round(x.get_device_properties(0).total_memory / 1024 ** 3, 1)
            out["count"] = x.device_count()
        except Exception:
            pass
        out["name"] = out["name"] or "Intel GPU"
        return out
    except Exception:
        return None


def tier_of(gpu: dict | None) -> str:
    g = gpu or {}
    vram = g.get("vram_gb") or 0
    if g.get("type") == "tpu":       # pc["gpu"]에는 TPU가 들어가지 않지만, 넘겨받은 값이 TPU면 TPU 등급
        return "tpu"
    if g.get("type") in ("cuda", "mps", "xpu") and vram >= 16:
        return "pc-gpu-high"
    if g.get("type") in ("cuda", "mps", "xpu") and vram >= 6:
        return "pc-gpu"
    return "pc-cpu"


def _tier_label(tier: str) -> str:
    return {"pc-cpu": "PC (CPU)", "pc-gpu": "PC (GPU)", "pc-gpu-high": "PC (고성능 GPU)",
            "tpu": "TPU"}.get(tier, tier)


def _accel_info() -> dict:
    """설치된 가속 패키지와 장치 (accel.info). TPU는 실제로 열지 않고 설치 여부와 런타임 신호만 본다."""
    try:
        from . import accel

        return accel.info(probe_tpu=False)
    except Exception as e:  # 감지 실패는 조용히 넘어간다 (PRD 6.11)
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def tpu_usable(profile: dict | None) -> tuple[bool, dict]:
    """프로필의 accel 정보로 TPU 학습이 가능해 보이는가 -> (가능, TPU 정보). torch_xla가 설치돼 있고
    TPU 런타임 신호가 있거나(짐작) 이미 열어 본 경우만 참. torch_xla를 불러오지 않는다."""
    acc = (profile or {}).get("accel")
    acc = acc if isinstance(acc, dict) else {}
    t = acc.get("tpu") if isinstance(acc.get("tpu"), dict) else {}
    avail = acc.get("available") if isinstance(acc.get("available"), dict) else {}
    return bool(avail.get("tpu") or t.get("ok")), t


def pc_profile(refresh: bool = False) -> dict:
    """이 PC의 성능 프로필. 60초 동안 캐시한다 (refresh=True면 다시 측정)."""
    now = time.monotonic()
    with _LOCK:
        if not refresh and _CACHE["v"] is not None and now - _CACHE["t"] < CACHE_SEC:
            return copy.deepcopy(_CACHE["v"])
    total, avail = _ram_bytes()
    ram_gb = round(total / 1024 ** 3, 1) if total else None
    gpu, torch_version = _gpu_info(ram_gb)
    prof = {
        "os": platform.system() or sys.platform,
        "os_version": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": _cpu_count(),
        "ram_gb": ram_gb,
        "ram_available_gb": round(avail / 1024 ** 3, 1) if avail else None,
        "gpu": gpu,
    }
    for m in MODULES:
        prof[m] = _has(m)
    prof["torch_version"] = torch_version
    prof["accel"] = _accel_info()   # torch_xla·ROCm·xpu·TPU 런타임 신호 (PRD 6.11). gpu·tier의 뜻은 그대로
    prof["tier"] = tier_of(gpu)
    prof["tier_label"] = _tier_label(prof["tier"])
    prof["checked_at"] = time.time()
    with _LOCK:
        _CACHE["v"], _CACHE["t"] = prof, time.monotonic()
    return copy.deepcopy(prof)


def _fit_batch(arch: dict, budget_mb: float, sizes=(32, 24, 16, 12, 8, 4, 2)) -> int:
    from .tinylm import estimate

    for b in sizes:
        if estimate(arch, b)["train_ram_mb"] <= budget_mb:
            return b
    return sizes[-1]


def recommend(profile: dict | None = None) -> dict:
    """성능 프로필 -> 소형 AI 모델 크기 / 배치 / 스레드, LoRA 기본 모델 추천과 한국어 안내."""
    from .tinylm import PRESETS, estimate, resolve_arch

    p = profile if profile is not None else pc_profile()
    gpu = p.get("gpu") or {}
    tier = p.get("tier") or tier_of(gpu)
    if tier not in ("pc-cpu", "pc-gpu", "pc-gpu-high", "tpu"):
        tier = tier_of(gpu)
    cpu = int(p.get("cpu_count") or 1)
    threads = max(1, cpu - 1)
    ram = p.get("ram_gb")
    vram = gpu.get("vram_gb")
    notes: list[str] = []
    tpu_ok, tpu = tpu_usable(p)
    if not gpu.get("type") and tier == "pc-cpu":
        if tpu_ok:         # GPU가 없고 TPU를 쓸 수 있어 보이면 TPU 등급 (PRD 6.8). accel 정보가 없으면 예전 그대로
            tier = "tpu"
        elif tpu.get("likely") and not tpu.get("installed"):
            notes.append("TPU 런타임으로 보이지만 TPU 패키지(torch_xla)가 없어 CPU 기준으로 추천합니다. "
                         "python -m learnbox.setup_accel 로 설치하세요.")

    if tier == "tpu":
        try:
            hbm = float(tpu.get("hbm_gb") or tpu.get("total_gb") or 0) or None
        except (TypeError, ValueError):
            hbm = None
        mem = hbm or TPU_HBM_DEFAULT_GB
        preset, steps = ("large", 1000) if mem >= 16 else ("base", 500)
        lora, lora_steps = LORA_MODELS["1.5B"], 300
        kind = tpu.get("accelerator_type") or "TPU"
        notes.append(f"TPU({kind}, 메모리 {f'{hbm:g}GB' if hbm else f'약 {mem:g}GB로 가정'})로 학습합니다.")
        notes.append("TPU는 처음 학습할 때 컴파일하느라 첫 단계가 몇 분 걸릴 수 있습니다 (그 뒤로는 빠릅니다).")
        notes.append("LoRA는 측정 전까지 1.5B 모델을 권장합니다. 3B는 메모리가 빠듯할 수 있습니다.")
        budget = mem * 1024 * 0.75
    elif tier == "pc-gpu-high":
        preset, steps, lora, lora_steps = "large", 1000, LORA_MODELS["7B"], 300
        notes.append(f"고성능 GPU({gpu.get('name') or 'GPU'}, VRAM {vram or '?'}GB)가 있어 큰 모델 학습을 권장합니다.")
        if vram and vram < 20:
            notes.append("7B LoRA는 VRAM을 많이 씁니다. 메모리가 부족하면 1.5B 모델을 고르세요.")
        budget = (vram or 16) * 1024 * 0.75
    elif tier == "pc-gpu":
        preset, steps, lora, lora_steps = "base", 500, LORA_MODELS["1.5B"], 300
        notes.append(f"GPU({gpu.get('name') or 'GPU'}, VRAM {vram or '?'}GB)로 학습합니다.")
        budget = (vram or 6) * 1024 * 0.75
    else:
        lora_steps = 50
        if gpu.get("type") and vram is not None:
            notes.append(f"GPU 메모리(VRAM {vram}GB)가 작아 CPU 기준으로 추천합니다.")
        else:
            notes.append("GPU가 없어 CPU로 학습합니다. 학습은 느리지만 허용 시간 동안 조금씩 이어서 학습할 수 있습니다.")
        if ram is not None and ram < 6:
            preset, steps = "mini", 300
            notes.append(f"RAM이 {ram}GB로 6GB 미만이라 '미니' 크기를 권장합니다.")
        else:
            preset, steps = "small", 200
            if ram is None:
                notes.append("RAM 크기를 확인하지 못했습니다. 메모리가 부족하면 '미니' 크기를 고르세요.")
        if ram is not None and ram >= 8:
            lora = LORA_MODELS["0.5B"]
            notes.append("LoRA는 CPU에서 매우 느리므로 0.5B 모델과 적은 단계 수를 권장합니다.")
        else:
            lora = None
            notes.append("RAM이 8GB 미만이라 LoRA 학습은 권장하지 않습니다. 소형 AI 모델과 검색을 사용하세요."
                         if ram is not None else "RAM을 확인하지 못해 LoRA 학습 모델을 추천하지 않습니다.")
        budget = min((ram or 4) * 1024 * 0.5, 6000)

    arch = resolve_arch(preset)
    batch = _fit_batch(arch, budget)
    est = estimate(arch, batch)
    notes.append(f"소형 AI 모델: '{PRESETS[preset]['label']}' · {est['label']} (배치 {batch}).")
    if batch < 32:
        notes.append(f"메모리에 맞추어 배치 크기를 {batch}(으)로 줄였습니다.")
    notes.append(f"학습에 CPU 스레드 {threads}개를 사용합니다 (서버 응답용으로 1개를 남깁니다)."
                 if cpu > 1 else "CPU 코어가 1개라 학습 중에는 화면 응답이 느릴 수 있습니다.")
    if PRESETS[preset]["target"].startswith("pc"):
        notes.append("휴대폰에서 직접 실행(온보드)하려면 '나노'(저사양) 또는 '미니'(고사양) 크기로 따로 학습하세요.")

    if p.get("torch") is False:
        notes.append("PyTorch가 설치되지 않아 모델 학습을 할 수 없습니다: pip install -r requirements-train.txt")
    if lora and (p.get("transformers") is False or p.get("peft") is False):
        notes.append("transformers/peft가 없어 LoRA 학습을 할 수 없습니다: pip install -r requirements-train.txt")
        lora = None
    if p.get("onnxruntime") is False:
        notes.append("onnxruntime이 없어 ONNX 검증과 int8(휴대폰용 경량) 변환을 건너뜁니다: pip install onnxruntime")
    if p.get("anthropic") is False:
        notes.append("anthropic 패키지가 없어 온라인 모델을 쓸 수 없습니다: pip install anthropic")

    return {
        "tier": tier,
        "tiny_preset": preset,
        "tiny_steps_per_cycle": steps,
        "tiny_batch_size": batch,
        "threads": threads,
        "lora_base_model": lora,
        "lora_max_steps": lora_steps,
        "phone_presets": {"phone-low": "nano", "phone-high": "mini"},
        "notes": notes,
    }
