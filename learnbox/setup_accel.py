"""가속 패키지 설치 도우미: TPU 런타임이면 TPU 백엔드를 설치하고, GPU별 PyTorch 빌드를 안내한다.

    python -m learnbox.setup_accel            # TPU 런타임이면 필요한 패키지 설치, 아니면 건너뜀
    python -m learnbox.setup_accel --check    # 설치하지 않고 상태만 JSON으로 (종료 코드 0=준비됨, 2=설치 필요, 1=TPU 아님)
    python -m learnbox.setup_accel --dry-run  # 실행할 명령만 보여 줌
    python -m learnbox.setup_accel --torch-build  # 이 PC의 GPU에 맞는 PyTorch 설치 명령 안내

결정 ② (docs/PRD-tpu-training.md): TPU 패키지는 기본 구성에 포함하되 **TPU 런타임에서만** 설치한다.
pip의 조건 표시로는 TPU를 알 수 없으므로 requirements-train.txt가 아니라 이 도우미가 설치한다.

TPU 백엔드 선택:
- torch_tpu (TorchTPU): 코랩 TPU v5e 런타임에 기본 설치. torch>=2.11, Python 3.11 이상. 이미 있으면 아무것도 안 한다.
- torch_xla 2.9 (지원 종료): Python 3.10~3.12에서 torch_tpu를 쓸 수 없을 때만. torch를 2.9로 맞춘다.
  torch_tpu가 설치돼 있으면 torch_xla를 설치하지 않는다(torch를 내리면 torch_tpu와 import torch가 깨진다).

표준 라이브러리만 쓴다(torch를 불러오지 않는다).
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from . import accel

ROOT = Path(__file__).resolve().parent.parent
REQ_TPU = ROOT / "requirements-tpu.txt"
REQ_TPU_XLA = ROOT / "requirements-tpu-xla.txt"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_BUILDS = {  # GPU 종류 → PyTorch 설치 위치 (pytorch.org 안내 기준)
    "nvidia": "https://download.pytorch.org/whl/cu126",
    "intel": "https://download.pytorch.org/whl/xpu",
    "amd": "https://download.pytorch.org/whl/rocm6.4",
    "cpu": CPU_INDEX,
}


def _vt(v: str | None) -> tuple:
    out = []
    for part in (v or "").split("+")[0].split(".")[:3]:
        num = "".join(ch for ch in part if ch.isdigit())
        out.append(int(num) if num else 0)
    return tuple(out)


def status() -> dict:
    """설치 판단에 필요한 정보 (아무것도 불러오지 않고 PCI·환경변수·패키지 메타데이터만 읽는다)."""
    return {
        "os": platform.system(), "machine": platform.machine(),
        "python": "%d.%d" % sys.version_info[:2],
        "hint": accel.tpu_hint(),
        "torch": accel._version("torch"),
        "torch_tpu": accel._version("torch_tpu"),
        "torch_xla": accel._version("torch_xla"),
    }


def plan(st: dict, force: bool = False, backend: str = "auto") -> dict:
    """status() → 할 일 {'action': skip|ready|install|unsupported, 'backend', 'message', 'requirements'}."""
    linux_x86 = st["os"] == "Linux" and st["machine"] in ("x86_64", "AMD64")
    tpu = bool(st["hint"].get("likely"))
    if not linux_x86 and not force:
        return {"action": "skip", "message": "TPU 런타임이 아니라 건너뜁니다 (Linux x86_64가 아님)."}
    if not tpu and not force:
        return {"action": "skip", "message": "TPU 런타임이 아니라 건너뜁니다."}
    py = _vt(st["python"])
    torch_v = _vt(st["torch"])
    if st["torch_tpu"] and backend in ("auto", "torch_tpu"):
        if st["torch"] and torch_v < (2, 11):
            return {"action": "unsupported", "backend": "torch_tpu",
                    "message": f"torch_tpu {st['torch_tpu']}는 torch 2.11 이상이 필요한데 torch {st['torch']}가 설치돼 "
                               "있습니다. pip install -r requirements-tpu.txt 로 맞추세요."}
        return {"action": "ready", "backend": "torch_tpu",
                "message": f"TPU 준비됨: torch_tpu {st['torch_tpu']} · torch {st['torch'] or '?'}"}
    if st["torch_xla"] and backend in ("auto", "torch_xla"):
        if st["torch"] and _vt(st["torch"])[:2] != _vt(st["torch_xla"])[:2]:
            return {"action": "unsupported", "backend": "torch_xla",
                    "message": f"torch {st['torch']}와 torch_xla {st['torch_xla']}의 버전이 맞지 않습니다. "
                               "pip install -r requirements-tpu.txt 로 torch_tpu를 설치하세요(Python 3.11 이상)."}
        return {"action": "ready", "backend": "torch_xla",
                "message": f"TPU 준비됨(지원 종료된 torch_xla): torch_xla {st['torch_xla']} · torch {st['torch']}"}
    if backend in ("auto", "torch_tpu") and py >= (3, 11):
        return {"action": "install", "backend": "torch_tpu", "requirements": str(REQ_TPU),
                "message": "TPU 백엔드 torch_tpu를 설치합니다 (torch 2.11 이상, CPU용 torch 빌드)."}
    if backend in ("auto", "torch_xla") and (3, 10) <= py < (3, 13):
        return {"action": "install", "backend": "torch_xla", "requirements": str(REQ_TPU_XLA),
                "message": "torch_tpu를 쓸 수 없는 Python이라 지원 종료된 torch_xla 2.9를 설치합니다 "
                           "(torch가 2.9로 바뀝니다)."}
    return {"action": "unsupported", "backend": None,
            "message": f"Python {st['python']}에서 설치할 수 있는 TPU 백엔드가 없습니다 "
                       "(torch_tpu: 3.11 이상, torch_xla 2.9: 3.10~3.12)."}


def pip_command(req: str) -> list[str]:
    return [sys.executable, "-m", "pip", "install", "-r", req, "--retries", "10", "--timeout", "120"]


def detect_gpu_vendor() -> str:
    """이 PC의 GPU 종류(nvidia/intel/amd/cpu) 짐작. torch를 불러오지 않는다."""
    try:
        if shutil.which("nvidia-smi"):
            return "nvidia"
        if platform.system() == "Linux":
            if Path("/dev/kfd").exists():
                return "amd"
            for v in Path("/sys/class/drm").glob("card*/device/vendor"):
                vendor = v.read_text().strip().lower()
                if vendor == "0x10de":
                    return "nvidia"
                if vendor == "0x1002":
                    return "amd"
            return "cpu"
        if platform.system() == "Windows":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_VideoController).Name"],
                                 capture_output=True, text=True, timeout=30).stdout.lower()
            if "nvidia" in out:
                return "nvidia"
            if "arc" in out and "intel" in out:
                return "intel"
            if "radeon" in out or "amd" in out:
                return "amd"
    except Exception:
        pass
    return "cpu"


def torch_build_advice(vendor: str | None = None) -> dict:
    vendor = vendor or detect_gpu_vendor()
    sysname = platform.system()
    if vendor == "amd" and sysname != "Linux":
        return {"vendor": vendor, "index": CPU_INDEX,
                "message": "AMD GPU용 PyTorch(ROCm)는 Linux에서만 공식 지원합니다. CPU용 torch를 그대로 씁니다."}
    if sysname == "Darwin":
        return {"vendor": "apple", "index": None,
                "message": "Mac은 기본 torch가 Apple GPU(MPS)를 지원합니다. 따로 설치할 필요가 없습니다."}
    index = TORCH_BUILDS.get(vendor, CPU_INDEX)
    label = {"nvidia": "NVIDIA GPU(CUDA)", "intel": "Intel Arc GPU(XPU, 미검증)", "amd": "AMD GPU(ROCm, 미검증)",
             "cpu": "CPU"}.get(vendor, vendor)
    return {"vendor": vendor, "index": index,
            "command": f"{sys.executable} -m pip install torch --index-url {index}",
            "message": f"{label}용 PyTorch: pip install torch --index-url {index}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learnbox.setup_accel", description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="설치하지 않고 상태만 보기 (JSON)")
    ap.add_argument("--dry-run", action="store_true", help="실행할 명령만 보여 주기")
    ap.add_argument("--force", action="store_true", help="TPU 신호가 없어도 TPU 백엔드 설치 시도")
    ap.add_argument("--backend", choices=("auto", "torch_tpu", "torch_xla"), default="auto")
    ap.add_argument("--torch-build", action="store_true", help="이 PC의 GPU에 맞는 PyTorch 설치 명령 안내")
    args = ap.parse_args(argv)

    if args.torch_build:
        adv = torch_build_advice()
        print(adv["message"])
        return 0
    st = status()
    p = plan(st, force=args.force, backend=args.backend)
    if args.check:
        print(json.dumps({"status": st, "plan": p}, ensure_ascii=False, indent=2))
        return {"ready": 0, "install": 2, "unsupported": 2}.get(p["action"], 1)
    print(p["message"])
    if p["action"] != "install":
        return 0 if p["action"] in ("skip", "ready") else 1
    cmd = pip_command(p["requirements"])
    print("실행: " + " ".join(cmd))
    if args.dry_run:
        return 0
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("TPU 패키지 설치에 실패했습니다. 위 오류를 확인하세요. 설치하지 않아도 GPU·CPU로는 학습할 수 있습니다.")
        return 1
    print("설치가 끝났습니다. 서버를 다시 켜면 TPU로 학습합니다 (python -m learnbox device 로 확인).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
