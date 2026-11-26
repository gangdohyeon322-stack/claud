"""장치 계층: 학습 장치 고르기와 장치별 차이를 한곳에 모은다 (docs/PRD-tpu-training.md 8.2).

지원 장치(정식 이름): cuda(NVIDIA, AMD ROCm) · tpu(torch_xla) · xpu(Intel Arc) · mps(Apple) · cpu.

원칙 (PRD 7장):
- cuda/mps/cpu에서는 예전 코드와 똑같이 동작한다. 새 동작은 tpu/xpu일 때만 실행된다.
- 이 모듈을 import해도 torch나 torch_xla를 불러오지 않는다. 모두 함수 안에서 필요할 때만 불러온다.
- torch_xla가 없거나 깨져 있어도 예외를 밖으로 내보내지 않는다. 감지에서 빠지고 안내 문구만 남긴다.

장치 값의 우선순위(학습 요청 기준): 요청의 device(웹 폼, --device) → 환경변수 LEARNBOX_DEVICE
→ 설정 training_device → auto. auto 순서: cuda → tpu → xpu → mps → cpu.
LoRA 정밀도(LEARNBOX_AMP, 설정 lora_training.amp): auto(기본) · fp16 · off · fp32. lora_precision 참고.
"""
from __future__ import annotations

import contextlib
import glob
import importlib.util
import os
import threading

DEVICES = ("auto", "cuda", "tpu", "xpu", "mps", "cpu")
LABELS = {"cuda": "GPU", "tpu": "TPU", "xpu": "Intel GPU", "mps": "Mac GPU", "cpu": "CPU"}
_ALIASES = {"": "auto", "none": "auto", "auto": "auto", "gpu": "cuda", "rocm": "cuda", "nvidia": "cuda",
            "xla": "tpu", "tpu": "tpu", "arc": "xpu", "intel": "xpu", "xpu": "xpu",
            "mac": "mps", "apple": "mps", "mps": "mps", "cpu": "cpu", "cuda": "cuda"}

AMP_MODES = ("auto", "fp16", "off", "fp32")
_AMP_ALIASES = {"": "auto", "auto": "auto", "default": "auto",
                "fp16": "fp16", "float16": "fp16", "half": "fp16", "1": "fp16", "on": "fp16", "true": "fp16",
                "yes": "fp16", "off": "off", "0": "off", "false": "off", "no": "off",
                "fp32": "fp32", "float32": "fp32"}

# TPU 종류별 장치(칩) 1개 메모리(GB). xm.get_memory_info를 못 읽을 때만 쓰는 대략값.
_TPU_HBM_GB = {"v5litepod": 16, "v5lite": 16, "v5e": 16, "v6e": 32, "v5p": 95, "v4": 32, "v3": 16, "v2": 8}

_TPU_LOCK = threading.Lock()
_TPU: dict = {"probed": False, "device": None, "reason": None, "backend": None}
# TPU 백엔드: torch_tpu(TorchTPU, 코랩 v5e 기본 설치, 즉시 실행) 우선, torch_xla(지원 종료, 지연 실행)는 예전 런타임용.
TPU_BACKENDS = ("torch_tpu", "torch_xla")

GB = 1024 ** 3


# ---------------------------------------------------------------- 이름과 요청 해석

def normalize(device) -> str:
    """장치 값을 정식 이름으로. auto/빈 값은 'auto'. 'cuda:1' 같은 번호 붙은 값과 모르는 값은 소문자로 그대로."""
    if device is None:
        return "auto"
    if hasattr(device, "type") and not isinstance(device, str):  # torch.device
        return name(device)
    s = str(device).strip().lower()
    if s.startswith(("xla:", "tpu:")):  # 번호 붙은 TPU 장치도 TPU 감지를 거친다
        return "tpu"
    return _ALIASES.get(s, s)


def requested_device(explicit=None, cfg_value=None) -> str:
    """요청 → LEARNBOX_DEVICE → 설정 training_device → auto. 'auto'나 빈 값은 '지정 안 함'으로 보고 다음으로 넘어간다."""
    for v in (explicit, os.environ.get("LEARNBOX_DEVICE"), cfg_value):
        n = normalize(v)
        if n != "auto":
            return n
    return "auto"


def normalize_amp(value) -> str | None:
    """정밀도 모드 값을 auto/fp16/off/fp32로. 모르는 값은 None."""
    if value is None:
        return None
    return _AMP_ALIASES.get(str(value).strip().lower())


def requested_amp(explicit=None, cfg_value=None) -> str:
    """요청 → LEARNBOX_AMP → 설정 lora_training.amp → auto. 'auto'나 모르는 값은 다음으로 넘어간다."""
    for v in (explicit, os.environ.get("LEARNBOX_AMP"), cfg_value):
        n = normalize_amp(v)
        if n and n != "auto":
            return n
    return "auto"


def name(dev) -> str:
    """torch.device나 문자열 → 정식 이름 (xla → tpu)."""
    if dev is None:
        return "cpu"
    t = getattr(dev, "type", None) if not isinstance(dev, str) else str(dev).split(":")[0].lower()
    t = (t or "cpu").lower()
    return "tpu" if t == "xla" else t


def label(dev) -> str:
    n = name(dev) if not isinstance(dev, str) else normalize(dev).split(":")[0]
    return LABELS.get(n, n.upper())


def is_tpu(dev) -> bool:
    return name(dev) == "tpu"


def tpu_backend(dev=None) -> str | None:
    """이 TPU 장치를 다루는 백엔드: 'torch_tpu'(torch.device 'tpu') 또는 'torch_xla'(torch.device 'xla'). TPU가 아니면 None."""
    if dev is not None and not is_tpu(dev):
        return None
    t = getattr(dev, "type", None) if dev is not None and not isinstance(dev, str) else None
    if t == "xla":
        return "torch_xla"
    if t == "tpu":
        return "torch_tpu"
    return _TPU["backend"] or (tpu_packages() or [None])[0]


def fixed_shapes(dev) -> bool:
    """입력 모양을 고정해야 하는 장치인가 (TPU: 모양이 바뀌면 다시 컴파일한다)."""
    return is_tpu(dev)


# ---------------------------------------------------------------- 감지

def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


# Google TPU PCI 장치 번호 (torch_xla 2.9 _internal/tpu.py와 같은 목록). 공급자 번호 0x1ae0.
_TPU_PCI = {"0x0027": "v2/v3", "0x005e": "v4", "0x0063": "v5e", "0x0062": "v5p", "0x006f": "v6e",
            "0x0076": "7x", "0x0056": "test"}
_TPU_PCI_HBM = {"v5e": 16, "v6e": 32, "v4": 32, "v5p": 95}


def pci_tpu_chips() -> tuple[int, str | None]:
    """/sys/bus/pci를 읽어 TPU 칩 수와 종류를 센다. 아무것도 불러오지 않고 멈추지도 않는다. (Linux 외: 0)"""
    n, kind = 0, None
    try:
        for v in glob.glob("/sys/bus/pci/devices/*/vendor"):
            try:
                with open(v, encoding="ascii") as f:
                    if f.read().strip().lower() != "0x1ae0":
                        continue
                with open(os.path.join(os.path.dirname(v), "device"), encoding="ascii") as f:
                    dev = f.read().strip().lower()
            except OSError:
                continue
            if dev in _TPU_PCI:
                n += 1
                kind = kind or _TPU_PCI[dev]
    except Exception:
        return 0, None
    return n, kind


def tpu_hint() -> dict:
    """TPU 패키지를 불러오지 않고 TPU 런타임인지 짐작한다 (설치 스크립트·장치 카드·자동 선택 전 확인용).

    확실한 감지는 pick('tpu')가 한다. 가장 믿을 만한 신호는 PCI의 TPU 칩(v5e는 /dev/accel 대신 /dev/vfio를 쓴다).
    """
    signals = []
    chips, kind = pci_tpu_chips()
    if chips:
        signals.append(f"pci:{kind}x{chips}")
    for key in ("TPU_ACCELERATOR_TYPE", "TPU_NAME", "COLAB_TPU_ADDR", "TPU_WORKER_ID"):
        if os.environ.get(key):
            signals.append(key)
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        signals.append("PJRT_DEVICE=TPU")
    try:
        if glob.glob("/dev/accel[0-9]*"):
            signals.append("/dev/accel*")
        if glob.glob("/dev/vfio/[0-9]*"):
            # vfio 그룹은 다른 장치 통과(passthrough)에도 쓰이므로 이것만으로는 TPU라고 보지 않는다(아래 likely 참고).
            signals.append("/dev/vfio/*")
    except Exception:
        pass
    acc = os.environ.get("TPU_ACCELERATOR_TYPE") or None
    strong = [s for s in signals if s != "/dev/vfio/*"]
    hbm = _hbm_from_type(acc)
    if hbm is None and kind in _TPU_PCI_HBM:
        hbm = float(_TPU_PCI_HBM[kind])
    return {"likely": bool(strong) or (bool(signals) and bool(acc)), "signals": signals,
            "accelerator_type": acc or kind, "chips": chips, "hbm_gb": hbm}


def _hbm_from_type(acc: str | None) -> float | None:
    a = (acc or "").lower()
    for k, v in _TPU_HBM_GB.items():
        if a.startswith(k):
            return float(v)
    return None


def _xla_modules():
    """(torch_xla, xla_model 또는 None, runtime 또는 None). 없으면 ImportError."""
    import torch_xla  # noqa: F401

    xm = xr = None
    try:
        import torch_xla.core.xla_model as xm  # type: ignore[no-redef]
    except Exception:
        xm = None
    try:
        import torch_xla.runtime as xr  # type: ignore[no-redef]
    except Exception:
        xr = None
    return torch_xla, xm, xr


def _mm(v: str) -> tuple:
    return tuple(v.split("+")[0].split(".")[:2])


def _xla_version_mismatch() -> str | None:
    """설치된 torch와 torch_xla의 major.minor가 다르면 안내 문구. 버전을 알 수 없으면 None(확인 생략).

    torch_xla는 torch 의존성을 선언하지 않아서 pip가 짝이 안 맞는 버전을 그냥 설치한다. import 전에 막는다.
    """
    t, x = _version("torch"), _version("torch_xla")
    if t and x and _mm(t) != _mm(x):
        return (f"torch {t}와 torch_xla {x}의 버전이 맞지 않습니다. "
                f"python -m learnbox.setup_accel 로 맞는 짝을 설치하세요.")
    return None


def _undo_xla_patches() -> None:
    """torch_xla import가 프로세스 전체에 덮어쓰는 함수를 원래대로 돌린다(같은 서버의 다른 장치 학습을 예전과 같게).

    torch_xla 2.9는 torch.nn.utils.clip_grad_norm_과 nn.GRU를 바꾸고 원본을 ._orig에 둔다.
    torch의 원래 clip_grad_norm_은 XLA에서도 동작한다.
    """
    try:
        import torch.nn as nn

        for owner, attr in ((nn.utils, "clip_grad_norm_"), (nn, "GRU")):
            orig = getattr(getattr(owner, attr, None), "_orig", None)
            if orig is not None:
                setattr(owner, attr, orig)
    except Exception:
        pass


def tpu_packages() -> list[str]:
    """설치된 TPU 백엔드 패키지 (우선순위 순)."""
    return [m for m in TPU_BACKENDS if _has(m)]


def _busy_note(msg: str) -> str:
    return ("다른 프로세스가 TPU를 쓰고 있습니다(코랩 노트북 셀이 TPU를 열었거나 다른 학습 명령이 돌고 있는지 "
            "확인하세요). " + msg[:160])


def _open_torch_tpu():
    """torch_tpu로 TPU를 연다. 실제로 작은 연산을 해서 칩을 잡아 본다(다른 프로세스가 쓰고 있으면 여기서 오류)."""
    import torch

    accel_mod = getattr(torch, "accelerator", None)
    cur = accel_mod.current_accelerator() if accel_mod is not None else None
    if cur is None or cur.type != "tpu":
        # TORCH_DEVICE_BACKEND_AUTOLOAD=0 등으로 백엔드가 안 올라왔으면 직접 올려 본다.
        with contextlib.suppress(Exception):
            from torch_tpu import _loader  # type: ignore

            _loader.load()
        cur = accel_mod.current_accelerator() if accel_mod is not None else None
    if cur is None or cur.type != "tpu":
        raise RuntimeError("torch_tpu 백엔드가 올라오지 않았습니다(TPU 칩이 없거나 TORCH_DEVICE_BACKEND_AUTOLOAD=0)")
    if not accel_mod.is_available():
        raise RuntimeError("torch_tpu가 TPU 칩을 찾지 못했습니다")
    d = torch.device("tpu")
    (torch.ones(1, device=d) + 1).cpu()  # 런타임 시작 + 칩 확보 확인
    return d


def _probe_tpu():
    """TPU 장치를 연다. 성공하면 torch.device(tpu 또는 xla), 실패하면 None과 이유. 한 번만 시도하고 결과를 기억한다.

    torch_tpu(즉시 실행)를 먼저, 안 되면 torch_xla(지연 실행)를 쓴다. 둘 다 없으면 이유는 'not_installed'.
    """
    with _TPU_LOCK:
        if _TPU["probed"]:
            return _TPU["device"], _TPU["reason"]
        dev, reason, backend = None, None, None
        if _has("torch_tpu"):
            try:
                dev, backend = _open_torch_tpu(), "torch_tpu"
            except Exception as e:
                msg = str(e)
                reason = (_busy_note(msg) if "already opened by another process" in msg or "already in use" in msg
                          else f"torch_tpu: {type(e).__name__}: {msg}"[:240])
        if dev is None and not _has("torch_xla"):
            reason = reason or "not_installed"
        elif dev is None and _xla_version_mismatch():
            reason = _xla_version_mismatch() if reason is None else f"{reason} / {_xla_version_mismatch()}"
        elif dev is None:
            reason = None
            try:
                if not os.environ.get("PJRT_DEVICE") and tpu_hint()["likely"]:
                    os.environ["PJRT_DEVICE"] = "TPU"
                txla, xm, xr = _xla_modules()
                _undo_xla_patches()
                if hasattr(txla, "device"):
                    d = txla.device()
                elif xm is not None:
                    d = xm.xla_device()
                else:
                    raise RuntimeError("torch_xla에서 장치를 여는 함수를 찾지 못했습니다")
                kind = None
                if xm is not None and hasattr(xm, "xla_device_hw"):
                    with contextlib.suppress(Exception):
                        kind = xm.xla_device_hw(d)
                if kind is None and xr is not None and hasattr(xr, "device_type"):
                    with contextlib.suppress(Exception):
                        kind = xr.device_type()
                if str(kind or "").upper() == "TPU":
                    dev, backend = d, "torch_xla"
                else:
                    reason = f"XLA 장치가 TPU가 아닙니다 ({kind or '알 수 없음'})"
            except Exception as e:  # 버전 불일치, 런타임 초기화 실패 등
                msg = str(e)
                if "already in use" in msg or "libtpu_lockfile" in msg:
                    reason = _busy_note(msg)
                else:
                    reason = f"{type(e).__name__}: {msg}"[:200]
        _TPU.update(probed=True, device=dev, reason=None if dev is not None else reason, backend=backend)
        return _TPU["device"], _TPU["reason"]


def _reset_tpu_probe() -> None:  # 테스트용
    with _TPU_LOCK:
        _TPU.update(probed=False, device=None, reason=None, backend=None)


def _xpu_ok() -> bool:
    try:
        import torch

        x = getattr(torch, "xpu", None)
        return bool(x is not None and x.is_available())
    except Exception:
        return False


def _mps_ok() -> bool:
    import torch

    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _tpu_missing_note(reason: str | None) -> str:
    if reason == "not_installed":
        return ("TPU 패키지(torch_tpu 또는 torch_xla)가 없어 CPU로 학습했습니다. "
                "TPU 런타임에서 python -m learnbox.setup_accel 을 실행해 설치하세요.")
    return f"TPU를 찾지 못해 CPU로 학습했습니다 ({reason or '알 수 없음'})."


def pick(device=None, *, allow_tpu: bool = True):
    """학습 장치 고르기 → (정식 이름, torch.device, 안내 문구 또는 None).

    고른 장치를 쓸 수 없으면 CPU와 이유를 돌려준다. cuda/mps/cpu 결과와 문구는 예전 tinylm._pick_device와 같다.
    """
    import torch

    want = normalize(device)
    if want == "auto":
        if torch.cuda.is_available():
            return "cuda", torch.device("cuda"), None
        # 자동 선택은 TPU 신호가 있을 때만 TPU 패키지를 불러온다(아니면 import 부작용 없이 넘어감).
        if allow_tpu and tpu_packages() and (_TPU["probed"] or tpu_hint()["likely"]):
            d, _ = _probe_tpu()
            if d is not None:
                return "tpu", d, None
        if _xpu_ok():
            return "xpu", torch.device("xpu"), None
        if _mps_ok():
            return "mps", torch.device("mps"), None
        return "cpu", torch.device("cpu"), None
    if want.startswith("cuda"):
        if not torch.cuda.is_available():
            return "cpu", torch.device("cpu"), "CUDA GPU를 찾지 못해 CPU로 학습했습니다."
        try:
            return "cuda", torch.device(want), None
        except Exception:
            return "cpu", torch.device("cpu"), f"알 수 없는 장치 '{device}' 대신 CPU로 학습했습니다."
    if want == "tpu":
        if not allow_tpu:
            return "cpu", torch.device("cpu"), "TPU는 학습에만 씁니다. CPU로 실행했습니다."
        d, reason = _probe_tpu()
        if d is not None:
            return "tpu", d, None
        return "cpu", torch.device("cpu"), _tpu_missing_note(reason)
    if want.startswith("xpu"):
        if not _xpu_ok():
            return "cpu", torch.device("cpu"), "Intel GPU(XPU)를 찾지 못해 CPU로 학습했습니다."
        try:
            return "xpu", torch.device(want), None
        except Exception:
            return "cpu", torch.device("cpu"), f"알 수 없는 장치 '{device}' 대신 CPU로 학습했습니다."
    if want == "mps":
        if not _mps_ok():
            return "cpu", torch.device("cpu"), "Apple GPU(MPS)를 찾지 못해 CPU로 학습했습니다."
        return "mps", torch.device("mps"), None
    if want == "cpu":
        return "cpu", torch.device("cpu"), None
    try:
        d = torch.device(want)
        return name(d), d, None
    except Exception:
        return "cpu", torch.device("cpu"), f"알 수 없는 장치 '{device}' 대신 CPU로 학습했습니다."


def available(probe_tpu: bool = False) -> dict:
    """장치별 사용 가능 여부. probe_tpu=False면 TPU는 torch_xla를 불러오지 않고 짐작(설치 여부 + 신호)만 한다."""
    out = {"auto": True, "cuda": False, "tpu": False, "xpu": False, "mps": False, "cpu": True}
    if not _has("torch"):
        return out
    try:
        import torch

        out["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    out["xpu"] = _xpu_ok()
    with contextlib.suppress(Exception):
        out["mps"] = _mps_ok()
    if tpu_packages():
        if probe_tpu or _TPU["probed"]:
            out["tpu"] = _probe_tpu()[0] is not None
        else:
            out["tpu"] = tpu_hint()["likely"]
    return out


# ---------------------------------------------------------------- 정밀도

def native_bf16(dev) -> bool:
    """하드웨어가 bf16을 직접 지원하는가 (에뮬레이션은 지원으로 치지 않는다)."""
    n = name(dev)
    if n == "tpu":
        return True
    try:
        import torch

        if n == "cuda":
            try:
                return bool(torch.cuda.is_bf16_supported(including_emulation=False))
            except TypeError:  # 오래된 torch
                if getattr(torch.version, "hip", None):
                    return True
                return torch.cuda.get_device_properties(0).major >= 8
        if n == "xpu":
            x = getattr(torch, "xpu", None)
            f = getattr(x, "is_bf16_supported", None)
            return bool(f()) if f else False
    except Exception:
        return False
    return False


def tiny_amp_dtype(dev):
    """소형 AI 학습의 자동 혼합 정밀도 dtype. None이면 혼합 정밀도를 쓰지 않는다 (mps/cpu).

    cuda는 예전 그대로(torch.cuda.is_bf16_supported()). LEARNBOX_AMP는 LoRA 전용이라 여기서는 보지 않는다.
    """
    import torch

    n = name(dev)
    if n == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if n == "tpu":
        return torch.bfloat16
    if n == "xpu":
        return torch.bfloat16 if native_bf16(dev) else torch.float16
    return None


def lora_precision(dev, mode: str | None = None, *, requested_dtype: str | None = None,
                   plan_dtype: str = "float32") -> tuple[str, bool, str | None]:
    """LoRA 학습의 (가중치 dtype 이름, fp16 혼합 정밀도 사용 여부, 안내 문구).

    - requested_dtype: 호출자가 직접 준 dtype(또는 재개 시 저장된 dtype). 'auto'/None이면 정책을 따른다.
    - plan_dtype: train_pc._plan이 예전 규칙으로 고른 dtype. mode='off'는 이 값을 그대로 쓴다(예전 동작).
    - mode: auto(기본) · fp16 · off · fp32. None이면 requested_amp()(환경변수 → auto).
    cuda/xpu: auto는 bf16을 하드웨어로 지원하면 bf16, 아니면 fp16 혼합 정밀도(기본). tpu: bf16(fp16 없음).
    mps/cpu: plan_dtype 그대로 (예전과 같음).
    """
    n = name(dev)
    m = normalize_amp(mode) if mode is not None else requested_amp()
    m = m or "auto"
    req = None if requested_dtype in (None, "", "auto") else str(requested_dtype)
    if n == "tpu":
        if req == "float16":
            return "bfloat16", False, "TPU는 fp16 대신 bf16으로 학습합니다."
        if req:
            return req, False, None
        if m == "fp32":
            return "float32", False, None
        note = "TPU에서는 fp16 대신 bf16으로 학습합니다 (LEARNBOX_AMP 무시)." if m == "fp16" else None
        return "bfloat16", False, note
    if n in ("cuda", "xpu"):
        if req:
            return req, req == "float16", None
        if m == "fp16":
            return "float16", True, "혼합 정밀도(fp16)로 학습합니다 (LEARNBOX_AMP)."
        if m == "off":
            return plan_dtype, plan_dtype == "float16", None
        if m == "fp32":
            return "float32", False, None
        if native_bf16(dev):
            return "bfloat16", False, None
        return "float16", True, ("bf16을 하드웨어로 지원하지 않는 GPU라 혼합 정밀도(fp16)로 학습합니다 "
                                 "(끄려면 LEARNBOX_AMP=off).")
    return plan_dtype, False, None


def autocast(dev, dtype):
    """혼합 정밀도 문맥. dtype이 None/float32이면 아무것도 하지 않는 문맥(예전 cpu/mps 경로와 같음)."""
    import torch

    if dtype is None or dtype == torch.float32:
        return contextlib.nullcontext()
    n = name(dev)
    device_type = ("xla" if tpu_backend(dev) == "torch_xla" else "tpu") if n == "tpu" else n
    return torch.autocast(device_type=device_type, dtype=dtype)  # dtype은 항상 지정(torch_tpu 기본은 fp16)


def make_scaler(dev, enabled: bool):
    """기울기 스케일러. 꺼진 스케일러는 step()이 opt.step()만 부른다. TPU는 항상 꺼짐(bf16)."""
    import torch

    n = name(dev)
    on = bool(enabled) and n in ("cuda", "xpu")
    kind = "xpu" if (on and n == "xpu") else "cuda"
    try:
        return torch.amp.GradScaler(kind, enabled=on)
    except (AttributeError, TypeError, RuntimeError):  # torch < 2.3, 또는 xpu 스케일러가 없는 torch
        return torch.cuda.amp.GradScaler(enabled=on and kind == "cuda")


def optimizer_step(opt, scaler, dev=None):
    """매개변수 갱신. 모든 장치에서 scaler.step(opt) (꺼진 스케일러면 opt.step()).

    TPU에서는 이 뒤에 step_end(dev)로 모아 둔 연산을 실행한다(코어 1개라 기울기 합치기가 필요 없다).
    """
    return scaler.step(opt)


def step_end(dev) -> None:
    """한 단계 마무리. torch_xla: 모아 둔 연산을 실행(sync). torch_tpu(즉시 실행)와 다른 장치: 아무것도 안 함."""
    if tpu_backend(dev) != "torch_xla":
        return
    import torch_xla

    if hasattr(torch_xla, "sync"):
        torch_xla.sync()
    else:  # 오래된 torch_xla
        import torch_xla.core.xla_model as xm

        xm.mark_step()


def manual_seed(dev, seed: int) -> None:
    """장치 난수 seed. torch_xla만 해당(torch.manual_seed가 XLA 난수를 정하지 않음).

    torch_tpu는 torch.manual_seed가 TPU 난수도 정하므로 여기서는 아무것도 안 한다. 다른 장치도 마찬가지.
    """
    if tpu_backend(dev) != "torch_xla":
        return
    with contextlib.suppress(Exception):
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(int(seed), device=dev)


def rng_state(dev):
    """체크포인트에 넣을 장치 난수 상태. cuda만 (예전과 같음), 그 밖에는 None."""
    if name(dev) != "cuda":
        return None
    import torch

    return torch.cuda.get_rng_state_all()


def to_cpu(obj, dev=None):
    """저장용 상태를 CPU 텐서로. TPU가 아니면 그대로 돌려준다(예전과 같음).

    dev를 주지 않으면 텐서의 장치를 보고 xla 텐서만 옮긴다.
    """
    if dev is not None and not is_tpu(dev):
        return obj
    import torch

    seen: dict = {}  # 같은 텐서(묶인 가중치 등)는 한 번만 옮겨 같은 CPU 텐서를 공유하게 한다

    def conv(x):
        if torch.is_tensor(x):
            if not (dev is not None or x.device.type in ("xla", "tpu")):
                return x
            key = id(x)
            if key not in seen:
                seen[key] = (x, x.detach().cpu())  # 원본도 잡아 두어 id 재사용을 막는다
            return seen[key][1]
        if isinstance(x, dict):
            out = type(x)((k, conv(v)) for k, v in x.items())
            if hasattr(x, "_metadata"):  # state_dict(OrderedDict)의 버전 정보 유지
                out._metadata = x._metadata
            return out
        if isinstance(x, list):
            return [conv(v) for v in x]
        if isinstance(x, tuple):
            return tuple(conv(v) for v in x)
        return x

    return conv(obj)


def empty_cache(dev=None) -> None:
    """장치 메모리 캐시 비우기. dev가 없으면 cuda가 있을 때만(예전 unload와 같음). TPU는 아무것도 안 함."""
    try:
        import torch
    except Exception:
        return
    n = name(dev) if dev is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    if n == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif n == "xpu":
        with contextlib.suppress(Exception):
            torch.xpu.empty_cache()


def is_oom(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "out of memory" in s or "resource_exhausted" in s or "resource exhausted" in s


def mem_total_gb(dev) -> float | None:
    """장치 메모리 전체(GB). 모르면 None."""
    n = name(dev)
    try:
        import torch

        if n == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / GB
        if n == "xpu":
            return torch.xpu.get_device_properties(0).total_memory / GB
        if n == "tpu":
            if tpu_backend(dev) == "torch_tpu":
                with contextlib.suppress(Exception):  # PCI 정보에서 읽음(런타임을 시작하지 않음)
                    p = torch.tpu.get_device_properties()
                    b = (getattr(p, "hbm_bytes_per_device", None) or getattr(p, "hbm_bytes_per_chip", None)
                         or getattr(p, "total_memory", None))
                    if b:
                        return b / GB
            info = tpu_memory_info(dev)
            if info.get("total_gb"):
                return info["total_gb"]
            return tpu_hint()["hbm_gb"]
    except Exception:
        return None
    return None


def tpu_memory_info(dev=None) -> dict:
    """TPU 메모리 사용량 {'total_gb','used_gb'} (읽지 못하면 빈 dict). torch_xla는 버전마다 키 이름이 달라 둘 다 본다."""
    if tpu_backend(dev) == "torch_tpu":
        try:
            import torch

            free, total = torch.accelerator.get_memory_info()
            return {"total_gb": total / GB, "used_gb": (total - free) / GB} if total else {}
        except Exception:
            return {}
    try:
        import torch_xla.core.xla_model as xm

        d = dev if (dev is not None and not isinstance(dev, str)) else _probe_tpu()[0]
        if d is None:
            return {}
        m = xm.get_memory_info(d)
        if "bytes_limit" in m:
            return {"total_gb": m["bytes_limit"] / GB, "used_gb": m.get("bytes_used", 0) / GB}
        if "kb_total" in m:
            total = m["kb_total"] * 1024 / GB
            return {"total_gb": total, "used_gb": total - m.get("kb_free", 0) * 1024 / GB}
    except Exception:
        return {}
    return {}


def compile_count() -> int | None:
    """지금까지 XLA 컴파일 횟수 (재컴파일 경고용). TPU가 아니거나 읽지 못하면 None."""
    if not _TPU["probed"] or _TPU["device"] is None or _TPU["backend"] != "torch_xla":
        return None  # torch_tpu는 연산마다 모양별로 컴파일해 캐시하므로 이 값을 쓰지 않는다
    try:
        import torch_xla.debug.metrics as met

        d = met.metric_data("CompileTime")
        if d:
            return int(d[0])
        return int(getattr(met, "counter_value", lambda _n: 0)("UncachedCompile") or 0)
    except Exception:
        return None


def _version(pkg: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return None


def info(probe_tpu: bool = False) -> dict:
    """장치 정보 (명령줄 device, 웹 기기 카드용). TPU는 probe_tpu=True일 때만 실제로 연다."""
    out = {"available": available(probe_tpu=probe_tpu), "torch_version": _version("torch"),
           "torch_xla_version": _version("torch_xla"), "rocm": False, "xpu_name": None,
           "torch_tpu_version": _version("torch_tpu"),
           "tpu": {"installed": bool(tpu_packages()), "packages": tpu_packages(), **tpu_hint(),
                   "probed": _TPU["probed"], "ok": _TPU["device"] is not None, "reason": _TPU["reason"],
                   "backend": _TPU["backend"] or (tpu_packages() or [None])[0]}}
    if _has("torch"):
        try:
            import torch

            out["rocm"] = bool(getattr(torch.version, "hip", None))
            if out["available"]["xpu"]:
                with contextlib.suppress(Exception):
                    out["xpu_name"] = torch.xpu.get_device_name(0)
        except Exception:
            pass
    return out
