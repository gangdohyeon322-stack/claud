@echo off
rem 새 폴더에서 학습 환경(가상환경, 기본 패키지, GPU용 PyTorch)을 한 번에 다시 만든다.
rem data 폴더(자료, 모델, 설정, PIN)는 건드리지 않는다. 이미 있는 .venv는 그대로 두고 빠진 것만 채운다.
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo [1/5] 파이썬 확인
py -3 --version >nul 2>&1
if errorlevel 1 (
  python --version >nul 2>&1
  if errorlevel 1 (
    echo 파이썬이 없습니다. python.org 에서 3.10 이상을 설치한 뒤 다시 실행하세요.
    pause
    exit /b 1
  )
  set PY=python
) else (
  set PY=py -3
)

echo [2/5] 가상환경 확인
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv
  if errorlevel 1 (
    echo 가상환경을 만들지 못했습니다. 위의 오류를 확인하세요.
    pause
    exit /b 1
  )
)
.venv\Scripts\python.exe -m pip install -q --upgrade pip --retries 10 --timeout 120

echo [3/5] 기본 패키지 설치
.venv\Scripts\python.exe -m pip install -r requirements.txt --retries 10 --resume-retries 10 --timeout 120
if errorlevel 1 (
  echo 기본 패키지 설치에 실패했습니다. 위의 오류 문구를 확인하세요.
  pause
  exit /b 1
)

echo [4/5] GPU용 PyTorch 설치 (CUDA 12.6, 실패하면 CUDA 12.1로 다시 시도)
.venv\Scripts\python.exe -m pip uninstall -y torch >nul 2>&1
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --retries 10 --resume-retries 10 --timeout 120
if errorlevel 1 (
  echo CUDA 12.6 설치에 실패해서 CUDA 12.1로 다시 시도합니다.
  .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --retries 10 --resume-retries 10 --timeout 120
)

echo [5/5] 학습 패키지 설치와 확인
.venv\Scripts\python.exe -m pip install -r requirements-train.txt --retries 10 --resume-retries 10 --timeout 120
.venv\Scripts\python.exe -c "import torch; print('PyTorch', torch.__version__); print('GPU 사용 가능:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU를 찾지 못했습니다 (CPU로 학습됩니다)')"

echo.
echo 설치가 끝났습니다. 위에서 "GPU 사용 가능: True"가 나오면 정상입니다.
echo 이제 start-windows.bat 을 실행하면 서버가 켜집니다.
pause
