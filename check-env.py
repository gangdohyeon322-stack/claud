# 학습 환경 설치 확인: 필요한 패키지를 실제로 불러와 보고 빠진 것이 있으면 종료 코드 1을 돌려준다.
# setup-gpu-windows.bat 의 마지막 단계에서 실행된다.
import importlib
import sys

MODULES = ["torch", "transformers", "peft", "onnx", "onnxruntime", "onnxscript", "anthropic", "pypdf"]

bad = []
for name in MODULES:
    try:
        importlib.import_module(name)
        print(f"  ok   {name}")
    except Exception as exc:  # 설치는 됐지만 DLL 등으로 불러오지 못하는 경우도 잡는다
        bad.append(name)
        print(f"  실패 {name}: {type(exc).__name__}: {exc}")

if not bad:
    import torch
    print("PyTorch", torch.__version__)
    print("GPU 사용 가능:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("GPU를 찾지 못했습니다 (CPU로 학습됩니다)")

print("빠진 패키지:", ", ".join(bad) if bad else "없음")
sys.exit(1 if bad else 0)
