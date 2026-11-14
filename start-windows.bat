@echo off
rem Windows 실행 스크립트: 처음 실행하면 설치와 PIN 설정을 하고, 이후에는 서버만 켭니다.
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\pip install -q -r requirements.txt
  set /p YN="학습 기능도 설치할까요? (용량 수 GB) [y/N] "
  if /i "!YN!"=="y" .venv\Scripts\pip install -r requirements-train.txt
)
.venv\Scripts\python -c "import sys; from learnbox.config import Config; sys.exit(0 if Config('data').initialized else 1)" || .venv\Scripts\python -m learnbox init
.venv\Scripts\python -m learnbox serve --lan %*
pause
