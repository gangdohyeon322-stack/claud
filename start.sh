#!/usr/bin/env bash
# Mac / Linux 실행 스크립트: 처음 실행하면 설치와 PIN 설정을 하고, 이후에는 서버만 켭니다.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
  read -r -p "학습 기능도 설치할까요? (용량 수 GB) [y/N] " yn
  if [ "$yn" = "y" ] || [ "$yn" = "Y" ]; then .venv/bin/pip install -r requirements-train.txt; fi
fi
if ! .venv/bin/python -c "import sys; from learnbox.config import Config; sys.exit(0 if Config('data').initialized else 1)"; then
  .venv/bin/python -m learnbox init
fi
exec .venv/bin/python -m learnbox serve --lan "$@"
