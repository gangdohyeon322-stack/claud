@echo off
rem 새 폴더에서 학습 환경(가상환경, 기본 패키지, PyTorch, 학습 패키지)을 한 번에 다시 만든다.
rem data 폴더(자료, 모델, 설정, PIN)는 건드리지 않는다. 이미 있는 .venv는 그대로 두고 빠진 것만 채운다.
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "VPY=.venv\Scripts\python.exe"

echo [1/6] 파이썬 확인
py -3 --version >nul 2>&1
if not errorlevel 1 (
  set "PY=py -3"
) else (
  python --version >nul 2>&1
  if errorlevel 1 (
    echo 파이썬이 없습니다. python.org 에서 3.10 이상을 설치한 뒤 다시 실행하세요.
    goto fail
  )
  set "PY=python"
)
%PY% --version

echo [2/6] 가상환경 확인
if exist %VPY% (
  %VPY% -m pip --version >nul 2>&1
  if errorlevel 1 (
    echo 기존 .venv가 다른 컴퓨터용이라 깨져 있습니다. 다시 만듭니다. ^(data 폴더는 그대로입니다^)
    rmdir /s /q .venv
  )
)
if not exist %VPY% (
  %PY% -m venv .venv
  if errorlevel 1 (
    echo 가상환경을 만들지 못했습니다. 위의 오류를 확인하세요.
    goto fail
  )
)
%VPY% -m pip install -q --upgrade pip --retries 10 --timeout 120

echo [3/6] 기본 패키지 설치
%VPY% -m pip install -r requirements.txt --retries 10 --resume-retries 10 --timeout 120
if errorlevel 1 (
  echo 기본 패키지 설치에 실패했습니다. 위의 오류 문구를 확인하세요.
  goto fail
)

echo [4/6] PyTorch 설치
where nvidia-smi >nul 2>&1
if errorlevel 1 goto cpu_only

%VPY% -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
  echo GPU용 PyTorch가 이미 있습니다. 건너뜁니다.
  goto torch_ok
)
%VPY% -m pip uninstall -y torch >nul 2>&1
echo NVIDIA GPU를 찾았습니다. CUDA 12.6용 PyTorch를 설치합니다 (용량이 커서 시간이 걸립니다).
%VPY% -m pip install torch --index-url https://download.pytorch.org/whl/cu126 --retries 10 --resume-retries 10 --timeout 120
if not errorlevel 1 goto torch_ok
echo CUDA 12.6 설치에 실패해서 CUDA 12.1로 다시 시도합니다.
%VPY% -m pip install torch --index-url https://download.pytorch.org/whl/cu121 --retries 10 --resume-retries 10 --timeout 120
if not errorlevel 1 goto torch_ok
echo GPU용 PyTorch 설치에 실패했습니다. CPU용으로 대신 설치합니다 (CPU로 학습됩니다).

:cpu_only
%VPY% -c "import torch" >nul 2>&1
if not errorlevel 1 goto torch_ok
echo CPU용 PyTorch를 설치합니다.
%VPY% -m pip install torch --retries 10 --resume-retries 10 --timeout 120
if errorlevel 1 (
  echo PyTorch 설치에 실패했습니다. 위의 오류 문구를 확인하세요.
  goto fail
)

:torch_ok
echo [5/6] 학습 패키지 설치
%VPY% -m pip install -r requirements-train.txt --retries 10 --resume-retries 10 --timeout 120
if errorlevel 1 (
  echo 학습 패키지 설치에 실패했습니다. 위의 오류 문구를 확인하세요.
  goto fail
)

echo [6/6] 설치 확인
%VPY% check-env.py
if errorlevel 1 (
  echo 일부 패키지를 불러오지 못했습니다. 위의 목록을 확인하세요.
  goto fail
)

echo.
echo 설치가 끝났습니다. 위에서 "GPU 사용 가능: True"가 나오면 GPU로 학습합니다. False면 CPU로 학습합니다.
echo 이제 start-windows.bat 을 실행하면 서버가 켜집니다.
pause
exit /b 0

:fail
echo.
echo 설치가 중단되었습니다. 위 오류를 확인한 뒤 이 파일을 다시 실행하세요 (끝난 단계는 건너뜁니다).
pause
exit /b 1
