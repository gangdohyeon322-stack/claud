# LearnBox 코랩(Google Colab) 매뉴얼

코랩에서 LearnBox 서버를 켜고, 웹 화면으로 자료 수집·검색·학습을 하는 방법을 처음부터 끝까지 정리했습니다.
명령은 모두 코랩 **코드 셀**에 붙여 넣고 실행합니다(셀 왼쪽 ▶ 또는 `Shift + Enter`).

---

## 목차

1. [먼저 알아둘 것](#1-먼저-알아둘-것)
2. [드라이브 준비 (PC에서, 한 번만)](#2-드라이브-준비-pc에서-한-번만)
3. [런타임 고르기 (GPU / TPU / CPU)](#3-런타임-고르기-gpu--tpu--cpu)
4. [처음 시작하기](#4-처음-시작하기)
5. [다음날 이어서 시작하기](#5-다음날-이어서-시작하기)
6. [웹 화면 접속과 사용](#6-웹-화면-접속과-사용)
7. [학습하기](#7-학습하기)
8. [일시정지 · 중간 저장 · 이어서 재개](#8-일시정지--중간-저장--이어서-재개)
9. [부가 기능](#9-부가-기능)
10. [백업](#10-백업)
11. [종료와 요금 절약](#11-종료와-요금-절약)
12. [문제 해결](#12-문제-해결)
13. [보안과 주의 사항](#13-보안과-주의-사항)
14. [자주 쓰는 셀 모음](#14-자주-쓰는-셀-모음)

---

## 1. 먼저 알아둘 것

- **코랩 디스크(`/content`)는 임시입니다.** 세션이 끝나거나 런타임을 바꾸면 모두 지워집니다. 오래 남길 것은 반드시 **구글 드라이브**에 백업합니다.
- **서버는 코랩 안에서 돌고, 웹 화면은 외부 주소(cloudflared)로 엽니다.** 이 주소는 인터넷에 공개되므로 PIN을 길게 정하고, 다 쓰면 끕니다.
- **데이터베이스(`learnbox.db`)는 드라이브에서 직접 쓰지 않습니다.** 드라이브에서 쓰면 불안정할 수 있어서, 코랩 디스크로 복사해 쓰고 끝날 때 드라이브로 백업합니다.
- **셀은 위에서부터 하나씩** 실행하고, 각 셀의 "확인" 결과가 나온 뒤 다음으로 넘어갑니다.

이 매뉴얼에서 쓰는 경로:

| 이름 | 경로 |
|---|---|
| 프로그램 코드 | `/content/claud` |
| 작업 데이터 | `/content/data` |
| 드라이브 폴더 | `/content/drive/MyDrive/LearnBox` |
| 백업 | `/content/drive/MyDrive/LearnBox/backup` |
| 체크포인트 | `/content/drive/MyDrive/LearnBox/ckpt` |

---

## 2. 드라이브 준비 (PC에서, 한 번만)

### 2-1. 데이터 압축

1. PC의 LearnBox 프로젝트 폴더에서 **`data` 폴더**를 찾습니다(`learnbox`, `web` 폴더가 있는 곳).
2. `data` 폴더를 **우클릭 → 압축(zip)** 합니다. 결과 파일 이름은 `data.zip`입니다.
3. 용량이 너무 크면(1GB 이상) 새 폴더에 `learnbox.db`와 `config.json`만 복사해서 그 폴더를 압축해도 됩니다.

### 2-2. 드라이브에 올리기

1. https://drive.google.com 에서 **내 드라이브 → 새로 만들기 → 새 폴더 → `LearnBox`**
2. `LearnBox` 폴더 안에 `data.zip`을 업로드합니다. 100%가 될 때까지 기다립니다.

최종 모습:

```
내 드라이브/
  LearnBox/
    data.zip
```

> 폴더 이름은 **대소문자까지** 정확히 `LearnBox`여야 합니다. 위치가 헷갈리면 4-4의 `find` 셀로 찾을 수 있습니다.

---

## 3. 런타임 고르기 (GPU / TPU / CPU)

메뉴 **런타임 → 런타임 유형 변경**에서 고릅니다.

| 목적 | 런타임 | 이유 |
|---|---|---|
| 학습(소형 AI, LoRA), 의미 검색 색인을 빨리 | **T4 GPU** | 빠르지만 요금(컴퓨팅 단위)이 더 듦 |
| 학습(소형 AI, LoRA)을 TPU로 | **v5e-1 TPU** | TPU 칩 1개(메모리 16GB), bf16으로 학습. TPU 패키지(`torch_tpu`)가 이미 깔려 있음. 첫 학습은 컴파일 때문에 첫 단계가 몇 분 걸림 |
| 서버·검색·자료 수집만 | **CPU** | 요금이 적음, 학습은 느림 |

- TPU는 **학습에만** 씁니다. TPU 런타임에서 답변(LoRA·소형 AI)과 의미 검색 색인은 TPU를 쓰지 않고 CPU로 합니다.
- TPU에서 끊긴 학습을 T4에서, T4에서 끊긴 학습을 TPU에서 이어서 할 수 있습니다([8-4](#8-4-다른-런타임에서-이어서-재개-tpu--t4)).

> **런타임을 바꾸면 `/content`가 모두 지워집니다.** 바꾸기 전에 반드시 [10. 백업](#10-백업)을 하세요.

---

## 4. 처음 시작하기

### 4-1. GPU 확인 (GPU 런타임일 때만)

```
!nvidia-smi
```
**확인:** `Tesla T4` 같은 GPU 이름이 보이면 성공. CPU 런타임이면 건너뜁니다.
TPU 런타임이면 이 셀 대신 4-4의 **TPU 확인** 셀을 실행합니다(프로그램을 받은 뒤에 확인할 수 있음).

### 4-2. 드라이브 연결

```
from google.colab import drive
drive.mount('/content/drive')
```
**확인:** 권한 창에서 허용 → `Mounted at /content/drive`

### 4-3. 프로그램 받기

```
!git clone -b claude/vibe-scraping-ai-engine-din8fn https://github.com/gangdohyeon322-stack/claud.git /content/claud
!ls /content/claud
```
**확인:** `learnbox`, `web`, `requirements.txt`가 보이면 성공.
`already exists` 오류는 이미 받아 둔 것이므로 무시해도 됩니다.

### 4-4. 패키지 설치 (몇 분 걸림)

```
!pip install -q -r /content/claud/requirements.txt -r /content/claud/requirements-train.txt
```
- 셀 아이콘이 돌고 있으면 **끝날 때까지 기다립니다.**
- CPU로 서버·검색만 할 때는 `requirements.txt`만 설치해도 됩니다.

**torchao 충돌 해결 (`Found an incompatible version of torchao` 오류가 날 때)**
코랩에 미리 깔린 `torchao`가 `transformers`·`peft`와 맞지 않으면 학습 시작 때 위 오류가 납니다. `torchao`는 LearnBox가 쓰지 않으므로 지우면 됩니다.
```
!pip uninstall -y torchao
```
```
!pip install -q -r /content/claud/requirements.txt -r /content/claud/requirements-train.txt
```
- 지운 뒤에는 **런타임 → 세션 다시 시작**을 누르고, 4-4 설치 셀을 다시 실행하세요.
- 또는 최신 `torchao`로 맞출 수도 있습니다: `!pip install -U torchao` (다시 충돌하면 위의 `uninstall` 방법을 쓰세요).
- 설치 뒤 확인: `!cd /content/claud && python -c "import torch, transformers, peft; print('OK', transformers.__version__, peft.__version__)"`

TPU 패키지 확인과 설치 (모든 런타임에서 실행해도 됨):
```
!cd /content/claud && python -m learnbox.setup_accel
```
- TPU 런타임: 코랩에 기본으로 깔린 `torch_tpu`가 있으면 `TPU 준비됨: torch_tpu … · torch …`만 출력하고 아무것도 설치하지 않습니다. 없으면 TPU 패키지를 설치합니다.
- GPU·CPU 런타임: `TPU 런타임이 아니라 건너뜁니다.`가 나오고 아무것도 하지 않습니다.
- 위의 `pip install`은 `torch>=2.2`라서 코랩 TPU 런타임에 깔린 torch를 바꾸지 않습니다(그래야 `torch_tpu`가 유지됨). **torch를 따로 다시 설치하지 마세요.**

설치 확인:
```
!cd /content/claud && python -c "import learnbox; print('learnbox OK')"
```

TPU 확인 (TPU 런타임일 때만, 서버를 켜기 전에):
```
!cd /content/claud && python -m learnbox device
```
**확인:** `가속:` 줄에 `TPU v5e` 같은 TPU 이름과 `자동 선택 tpu`가 보이면 성공.
`TPU … 감지 (TPU 패키지 필요: python -m learnbox.setup_accel)`가 나오면 위의 `setup_accel` 셀을 다시 실행합니다. 자세한 상태는 `!cd /content/claud && python -m learnbox.setup_accel --check`로 봅니다(설치하지 않고 JSON으로 보여 줌).

> **TPU는 한 프로세스만 쓸 수 있습니다.** 서버(4-8)가 TPU로 학습하므로, 노트북 셀에서 `import torch`로 TPU를 쓰지 마세요.

### 4-5. 데이터 풀기

`data.zip` 위치 확인:
```
!find /content/drive/MyDrive -maxdepth 3 -name "data.zip"
```

압축 풀기 (항상 `/content/data` 안에 풀기):
```
!mkdir -p /content/data
!unzip -q -o "/content/drive/MyDrive/LearnBox/data.zip" -d /content/data
!ls /content/data
```
**확인:** `config.json`, `learnbox.db` (그리고 `datasets`, `models`)가 보이면 성공.

> `-o`는 질문 없이 덮어쓰기입니다. `replace …? [y]es, [n]o, [A]ll, [N]one` 질문이 나오면 `N`을 입력하고, 위 명령을 다시 실행하세요.

### 4-6. PIN 설정 (데이터가 없을 때만)

`data.zip`을 풀었다면 PIN이 이미 들어 있으므로 **건너뜁니다.** 빈 상태로 시작할 때만:
```
!cd /content/claud && python -m learnbox --data /content/data init --user-pin 사용자PIN --admin-pin 관리자PIN
```
- 사용자 PIN 4자 이상, 관리자 PIN 6자 이상, 서로 달라야 합니다.

### 4-7. 체크포인트를 드라이브에 저장하도록 설정 (권장)

```
!cd /content/claud && python -c "from learnbox.config import Config; c=Config('/content/data'); c.update({'training_checkpoint': {'dir': '/content/drive/MyDrive/LearnBox/ckpt'}}); print('저장 위치 변경 완료')"
```
학습이 끊겨도 드라이브에 중간 저장분이 남습니다.

### 4-8. 서버 켜기

```
!cd /content/claud && setsid nohup python -m learnbox --data /content/data serve --port 8750 > /content/server.log 2>&1 < /dev/null &
```
> `setsid … < /dev/null`이 꼭 있어야 셀이 끝나도 서버가 꺼지지 않습니다.

**정밀도와 장치 (환경변수, 선택)**
- 이제 fp16이 기본이라 예전처럼 `LEARNBOX_AMP=fp16`을 붙이지 않아도 T4에서 LoRA가 fp16으로 학습됩니다(붙여도 같은 동작).
- 예전 방식으로 되돌리려면 `LEARNBOX_AMP=off`, 장치를 정해 두려면 `LEARNBOX_DEVICE=tpu`(또는 `cuda`, `cpu`)를 씁니다.
- 웹에서 시작하는 학습은 **서버가** 정밀도와 장치를 정하므로, 환경변수는 다른 셀이 아니라 **서버 명령 앞에** 붙여야 합니다:
```
!cd /content/claud && LEARNBOX_AMP=off setsid nohup python -m learnbox --data /content/data serve --port 8750 > /content/server.log 2>&1 < /dev/null &
```
- 환경변수 대신 웹 [학습] 탭 폼의 **장치**·**정밀도** 선택을 써도 됩니다. 폼에서 고른 값이 가장 먼저 쓰이고, '자동'이면 환경변수 → 설정(`training_device`, `lora_training.amp`) 순서로 정합니다. 설정은 4-7처럼 서버를 켜기 전에 바꿀 수 있습니다:
```
!cd /content/claud && python -c "from learnbox.config import Config; c=Config('/content/data'); c.update({'lora_training': {'amp': 'off'}}); print('설정 변경 완료')"
```

### 4-9. 서버 확인 (다음 셀에서)

```
!sleep 15 && timeout 10 curl -s -o /dev/null -w "서버 응답 코드: %{http_code}\n" http://127.0.0.1:8750/api/info
```
**확인:** `서버 응답 코드: 200` → 성공. `000`이면 [12. 문제 해결](#12-문제-해결) 참고.
처음 켤 때는 1분쯤 걸릴 수 있습니다. 로그로 상태 보기:
```
!tail -n 10 /content/server.log
```
`LearnBox 실행 중: http://127.0.0.1:8750`이 보이면 켜진 것입니다.

### 4-10. 외부 주소 도구(cloudflared) 설치 (세션마다 한 번)

```
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null
!which cloudflared
```
**확인:** `/usr/local/bin/cloudflared` 같은 경로가 한 줄 나오면 성공.

### 4-11. 외부 주소 만들기

```
!setsid nohup cloudflared tunnel --url http://127.0.0.1:8750 > /content/tunnel.log 2>&1 < /dev/null &
```
다음 셀에서 주소 확인:
```
!sleep 15 && grep -o "https://[a-z0-9.-]*trycloudflare.com" /content/tunnel.log | head -1
```
**확인:** `https://…trycloudflare.com` 주소가 나오면 성공. 이 주소를 브라우저에서 엽니다.

---

## 5. 다음날 이어서 시작하기

새 세션은 `/content`가 비어 있으므로 다음 순서로 다시 준비합니다. **PIN 설정(4-6)은 하지 않습니다.**

1. 런타임 선택 (3장)
2. 드라이브 연결 (4-2)
3. 프로그램 받기 (4-3)
4. 패키지 설치 (4-4)
5. **백업에서 복원** (아래)
6. 체크포인트 위치 설정 (4-7) — 백업한 `config.json`에 이미 들어 있으면 생략 가능
7. 서버 켜기·확인 (4-8, 4-9)
8. cloudflared 설치·주소 (4-10, 4-11)

### 백업에서 복원

```
!mkdir -p /content/data/models
!cp /content/drive/MyDrive/LearnBox/backup/learnbox.db /content/drive/MyDrive/LearnBox/backup/config.json /content/data/
!cp -r /content/drive/MyDrive/LearnBox/backup/models/* /content/data/models/ 2>/dev/null
!ls /content/data /content/data/models
```
**확인:** `config.json`, `learnbox.db`와 모델 폴더(`tiny-…`)가 보이면 성공.

> 예전 백업에서 모델 폴더(`tiny-…`)가 `backup/models/`가 아니라 `backup/` 바로 아래에 있다면:
> ```
> !cp -r /content/drive/MyDrive/LearnBox/backup/tiny-* /content/data/models/ 2>/dev/null
> ```

---

## 6. 웹 화면 접속과 사용

1. 4-11의 주소를 PC나 휴대폰 브라우저에서 엽니다.
2. **사용자 PIN** 또는 **관리자 PIN**으로 로그인합니다.
   - 관리자: 학습, 일시정지, 설정, 내보내기 가능
3. 처음 몇 초는 `502`가 날 수 있습니다. 20~30초 뒤 새로고침하세요.

| 탭 | 할 수 있는 일 |
|---|---|
| 검색 | 자료 검색, 질문하기(검색만·소형 AI·LoRA·온라인·온보드) |
| 자료 | 링크 수집, 실시간 검색 수집, 파일 올리기, 예약 목록, 문서 삭제(관리자) |
| 학습 | 자율 학습, 소형 AI 학습, LoRA 학습, 의미 검색 색인, 이 기기에서 학습 |
| 모델 | 학습한 모델 보기, 이 기기로 받기, 생성 시험 |
| 작업 | 진행 상황, **일시정지 · 재개 · 지금까지 저장하고 끝내기 · 취소 · 이어서 재개** |
| 관리 | 설정(수집 허용 시간, 검색 제공자, 학습 중간 저장 위치 등), 내보내기, 감사 기록 |

---

## 7. 학습하기

### 7-1. 어떤 크기를 고를까 (소형 AI)

| 크기 | 차원·층·헤드 | 대략 파라미터 | 언제 |
|---|---|---|---|
| 나노 | 128·2·4 | 약 67만 | 시험용, 휴대폰용 |
| 미니 | 256·4·4 | 약 400만 | 자료 수백만 글자 이하에서 안전한 시작 |
| 스몰 | 384·6·6 | 약 1,200만 | 자료를 정리한 뒤 |
| 베이스 | 512·8·8 | 약 2,600만 | 자료가 많을 때, 학습률 5e-4~1e-3 |
| 라지 | 768·12·12 | 약 9,000만 | 자료가 아주 많을 때만, T4에서 배치 8 이하 |

- 자료가 정리되지 않았으면(영어 조각, 표, 수식 섞임) 크기를 키워도 좋아지지 않습니다. **자료 정리가 먼저**입니다.
- 평가 손실(전 → 후)이 **내려가는지**로 판단합니다. 학습 손실만 내려가고 평가 손실이 오르면 외우는 중이니 멈춥니다.
- 직접 설정할 때 **차원은 헤드 수로 나누어떨어져야** 합니다(예: 192는 헤드 8·12 가능, 193은 불가).

### 7-2. 웹 화면에서 학습 (권장: 일시정지 가능)

[학습] 탭 → **소형 AI 모델 학습** → 크기·단계·배치 입력 → 시작 → [작업] 탭에서 진행 확인.

### 7-3. 셀 명령으로 학습

> **TPU 런타임에서는 웹 화면(7-2)으로 학습하세요.** TPU는 한 프로세스만 쓸 수 있어서, 서버가 켜져 있을 때 셀 명령(`train-tiny`, `train-lora`, `resume-job`)을 따로 실행하면 "다른 프로세스가 TPU를 쓰고 있습니다" 안내와 함께 CPU로 학습하거나 오류가 납니다. 셀 명령을 쓰려면 먼저 서버를 끕니다(`!pkill -f "[l]earnbox --data"`).

소형 AI (베이스, 1000단계 예):
```
!cd /content/claud && setsid nohup python -m learnbox --data /content/data train-tiny --preset base --steps 1000 --batch-size 16 --lr 0.0005 --checkpoint-minutes 5 > /content/train.log 2>&1 < /dev/null &
```

직접 크기 지정 예:
```
!cd /content/claud && setsid nohup python -m learnbox --data /content/data train-tiny --preset custom --d-model 640 --n-layer 10 --n-head 10 --seq-len 512 --vocab-size 8000 --steps 1000 --batch-size 16 --lr 0.0005 --checkpoint-minutes 5 > /content/train.log 2>&1 < /dev/null &
```

LoRA (지시형 모델, T4 또는 TPU 권장):
```
!cd /content/claud && setsid nohup python -m learnbox --data /content/data train-lora --base-model Qwen/Qwen2.5-0.5B-Instruct --max-steps 300 --checkpoint-minutes 5 > /content/lora.log 2>&1 < /dev/null &
```

T4에서 3B LoRA (환경변수 없이 fp16으로 학습):
```
!cd /content/claud && setsid nohup python -m learnbox --data /content/data train-lora --base-model Qwen/Qwen2.5-3B-Instruct --max-steps 300 --checkpoint-minutes 5 > /content/lora.log 2>&1 < /dev/null &
```
- T4는 bf16을 하드웨어로 지원하지 않아서 정밀도 `auto`(기본)가 fp16 혼합 정밀도를 고릅니다. 예전처럼 `LEARNBOX_AMP=fp16`을 붙일 필요가 없습니다.
- 끄려면 `--amp off`(예전 방식) 또는 `--amp fp32`를 붙입니다.
- Qwen2.5-3B는 비상업 라이선스입니다.

TPU(v5e-1)에서 LoRA (서버를 먼저 끈 뒤):
```
!cd /content/claud && setsid nohup python -m learnbox --data /content/data train-lora --base-model Qwen/Qwen2.5-1.5B-Instruct --max-steps 300 --device tpu --checkpoint-minutes 5 > /content/lora.log 2>&1 < /dev/null &
```
- TPU는 bf16으로 학습합니다(`--amp`는 쓰지 않음). 첫 단계는 컴파일 때문에 몇 분 걸릴 수 있습니다.
- 측정 전까지는 1.5B를 권장합니다. 3B는 메모리(16GB)가 빠듯할 수 있습니다.
- `--device`를 빼도 TPU 런타임에서는 자동으로 TPU를 고릅니다.

CPU에서 학습할 때는 작게:
```
!cd /content/claud && setsid nohup python -m learnbox --data /content/data train-tiny --preset mini --steps 500 --batch-size 8 --lr 0.001 --checkpoint-minutes 5 > /content/train.log 2>&1 < /dev/null &
```

진행 확인:
```
!tail -n 5 /content/train.log
!nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv
```
- GPU 사용률이 높고 메모리가 올라가 있으면 정상입니다. 메모리를 다 쓰지 않아도(예: 16GB 중 5GB) 사용률이 100%면 GPU를 꽉 쓰고 있는 것입니다.
- TPU 런타임에는 `nvidia-smi`가 없습니다. 로그(`tail`)나 [작업] 탭으로 확인하고, `tpu-info`가 있으면 `!tpu-info`로 TPU 사용량을 봅니다.
- 셀 명령으로 시작한 학습에는 **일시정지 버튼이 없습니다.** 대신 5분마다 드라이브에 중간 저장됩니다. 일시정지가 필요하면 웹 화면(7-2)으로 시작하세요.

### 7-4. 학습 결과 보관

완료된 모델은 `/content/data/models`에 저장됩니다. [10. 백업](#10-백업)으로 드라이브에 옮기세요.

---

## 8. 일시정지 · 중간 저장 · 이어서 재개

### 8-1. 일시정지와 재개 (웹 화면)

[작업] 탭의 학습 카드에서:
- **일시정지**: 그 시점까지 체크포인트를 저장하고 멈춤
- **재개**: 이어서 학습
- **지금까지 저장하고 끝내기**: 남은 단계를 건너뛰고 지금 상태로 모델 저장
- **취소**: 버림 (체크포인트도 삭제)

> 일시정지해도 서버는 저장장치를 쓰고 있습니다. 세션을 끝내려면 백업 후 종료하세요.

### 8-2. 다음날 이어서 재개

1. 오늘: 일시정지(또는 그냥 두기, 5분마다 자동 저장) → **[10. 백업](#10-백업)** (작업 기록이 든 `learnbox.db` 필수)
2. 내일: [5장](#5-다음날-이어서-시작하기) 순서로 준비 → 서버·주소 열기
3. 관리자 로그인 → [작업] 탭 → 해당 학습 카드의 **이어서 재개 (N/M단계부터)**

셀 명령으로 이어서 할 때:
```
!cd /content/claud && python -m learnbox --data /content/data jobs --limit 5
!cd /content/claud && setsid nohup python -m learnbox --data /content/data resume-job 작업번호 > /content/resume.log 2>&1 < /dev/null &
```
TPU 런타임에서 셀 명령으로 이어서 할 때는 서버를 먼저 끕니다(7-3의 주의). 웹 화면의 **이어서 재개**를 쓰면 그럴 필요가 없습니다.

### 8-3. 꼭 지킬 것

- 체크포인트 저장 위치가 **드라이브**(`/content/drive/MyDrive/LearnBox/ckpt`)여야 다음날 남아 있습니다(4-7).
- 데이터 폴더 경로(`/content/data`)를 바꾸지 마세요. 체크포인트는 학습 자료 경로를 기억합니다.

### 8-4. 다른 런타임에서 이어서 재개 (TPU ↔ T4)

TPU 세션 시간이 끝나 끊긴 학습을 다음날 T4에서, T4에서 끊긴 학습을 TPU에서 이어서 할 수 있습니다. CPU로도 됩니다.

1. 학습을 시작한 뒤 [10. 백업](#10-백업)을 해 둡니다(작업 기록이 든 `learnbox.db`가 있어야 [작업] 탭에 이어서 재개가 나옴). 체크포인트는 드라이브에 있어야 합니다(4-7).
2. 새 런타임(T4 또는 v5e-1 TPU)을 고르고 [5장](#5-다음날-이어서-시작하기) 순서로 준비합니다.
3. [작업] 탭에서 **이어서 재개**를 누릅니다.

- 체크포인트는 TPU에서도 CPU 텐서로 저장되므로 어느 장치에서나 읽을 수 있습니다.
- 이어서 재개는 원래 장치를 이어받지 않고 **지금 런타임의 장치**(자동 선택)에서 합니다. 셀 명령으로 장치를 정하려면 `resume-job 작업번호 --device tpu`(또는 `cuda`, `cpu`)를 씁니다.
- 장치가 바뀌면 LoRA 정밀도는 새 장치 기준으로 다시 고릅니다(TPU는 bf16, T4는 fp16). 안내 문구로 알려 줍니다.
- 결과는 끊기지 않은 학습과 비트 단위로 같지는 않습니다(장치가 바뀐 재개라서).

---

## 9. 부가 기능

### 9-1. 실시간 검색 수집 (GPU 불필요)

웹: [자료] 탭 → **실시간 검색 후 수집** → 낱말 입력 → 검색 · 수집 → [작업] 탭에서 "새 문서 ○개" 확인.

셀 명령:
```
!cd /content/claud && python -m learnbox --data /content/data websearch "고구려" --limit 5
```

### 9-2. 지정 링크 수집

```
!cd /content/claud && python -m learnbox --data /content/data collect "https://ko.wikipedia.org/wiki/고구려" --depth 1
```
- `--depth`는 같은 사이트 안에서 따라갈 링크 단계(0~2)입니다.

### 9-3. 파일 가져오기

웹: [자료] 탭 → 파일 선택(끌어다 놓기). txt, md, csv, json, jsonl, html, xml, pdf, docx 지원.

### 9-4. 의미 검색 색인 (뜻이 비슷한 글 찾기)

```
!cd /content/claud && python -m learnbox --data /content/data index-embeddings
```
- 학습과 별개입니다. 검색 품질을 높입니다.
- 중간에 멈춰도 다시 실행하면 이어서 만듭니다.
- 자료를 추가했으면 다시 실행하세요. 색인은 `learnbox.db`에 들어가므로 백업하면 같이 보관됩니다.

### 9-5. 질문하기

웹: [검색] 탭 → 질문 입력 → 답변 모델 선택 → 질문하기.

| 답변 모델 | 특징 |
|---|---|
| 검색만 | 자료와 핵심 문장 그대로 |
| 소형 AI | 자료의 말투 흉내, 사실 답변은 어려움 |
| LoRA | 지시형 모델 기반, 자연스러움, GPU 권장 |
| 온라인 | Claude API, API 키 필요, 자료 일부가 외부로 전송 |

### 9-6. 상태 확인

```
!cd /content/claud && python -m learnbox --data /content/data status
!cd /content/claud && python -m learnbox --data /content/data models
```

### 9-7. 모델을 휴대폰에서 쓰기

[모델] 탭 → **이 기기로 받기** → 휴대폰 브라우저에서 생성. (외부 주소로 휴대폰에서 접속)

---

### 9-8. 챗봇과 대화하기 (LoRA)

웹: [대화] 탭에서 이름을 정하고(관리자, '챗봇 이름·말투') 말을 걸면 됩니다. "안녕 넌 누구야" → "저는 ○○입니다".

이름을 모델에 학습시키려면 (학습 전에도 지시문으로 이름은 적용됨):
1. [학습] 탭 → **대화 학습 데이터 만들기** (외부 전송 없음)
2. LoRA 학습 ('대화 학습 데이터도 사용'이 켜져 있으면 함께 학습)
3. 작업 카드의 **정체성 시험 N%** 확인 (시스템 지시문 없이도 이름을 말하는 비율)

셀 명령 (서버를 끈 상태에서):
```
!cd /content/claud && python -m learnbox --data /content/data build-chat
!cd /content/claud && python -m learnbox --data /content/data chat -m "안녕 넌 누구야" -m "고구려 수도는 어디였어?"
```
- 이름을 바꿨으면 대화 학습 데이터를 다시 만들고 다시 학습하세요.
- 같은 문장이 반복되는 적은 자료로 단계를 많이 돌리면 모델이 자료를 그대로 외워 되풀이합니다. 평가 손실이 0에 가깝게 떨어지면 단계를 줄이세요.

## 10. 백업

작업이 끝날 때, 런타임을 바꾸기 전, 자기 전에 **항상** 실행합니다.

```
!mkdir -p /content/drive/MyDrive/LearnBox/backup/models
!cp /content/data/learnbox.db /content/data/config.json /content/drive/MyDrive/LearnBox/backup/
!cp -r /content/data/models/* /content/drive/MyDrive/LearnBox/backup/models/ 2>/dev/null; echo "백업 끝"
```

확인:
```
!ls -la /content/drive/MyDrive/LearnBox/backup /content/drive/MyDrive/LearnBox/backup/models
```

- `learnbox.db`: 자료, 작업 기록, 의미 검색 색인
- `config.json`: 설정과 PIN(해시)
- `models/`: 학습한 모델

> 학습이나 색인이 **끝난 뒤** 백업해야 결과가 모두 들어갑니다.

---

## 11. 종료와 요금 절약

### 11-1. 끄는 순서 (셀을 각각 따로 실행)

```
!pkill cloudflared
```
```
!pkill -f "[l]earnbox --data"
```
그다음 **런타임 → 세션 관리 → 종료**.

> `pkill -f "learnbox --data"`처럼 대괄호 없이 쓰면 **셀 자신까지 꺼져서** `^C`만 나옵니다. 반드시 `[l]earnbox`로 쓰세요.

### 11-2. 요금 줄이기

- 학습이 끝나면 바로 **CPU로 바꾸거나 세션을 종료**합니다(바꾸기 전 백업).
- 서버·검색·수집은 CPU로 충분합니다. TPU도 학습에만 쓰이므로 학습이 없을 때는 CPU 런타임을 씁니다.
- 큰 모델은 자료가 충분할 때만 돌립니다.
- 자리를 오래 비우면 연결이 끊깁니다. 끊김을 억지로 막는 자동 클릭 스크립트는 이용 약관 위반이 될 수 있어 권하지 않습니다. 대신 중간 저장(8장)을 믿고 이어서 하세요.

---

## 12. 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `No such file or directory: /content/claud` | 프로그램을 안 받음 (새 세션) | 4-3 실행 |
| `No such file or directory: /content/data` | 데이터를 안 풂 | 4-5 또는 5장 복원 |
| `No module named learnbox` | `cd /content/claud` 없이 실행 / 설치 안 됨 | 명령 앞에 `cd /content/claud &&` 붙이기, 4-4 설치 |
| `fatal: destination path ... already exists` | 이미 받아 둠 | 무시하고 진행 |
| 서버 응답 `000` | 서버가 꺼져 있음 | `!tail -n 30 /content/server.log`로 원인 확인 후 4-8 다시 |
| 로그에 `종료합니다.` | 셀이 끝나며 서버가 같이 꺼짐 | `setsid … < /dev/null`을 붙인 4-8 명령 사용 |
| 로그에 `먼저 PIN을 설정하세요` | 빈 데이터 폴더 | 4-6 PIN 설정 또는 데이터 복원 |
| `Address already in use` | 서버가 이미 떠 있음 | 그대로 쓰거나 `!pkill -f "[l]earnbox --data"` 후 4-8 |
| `^C`만 나오고 다음 출력 없음 | 이전 셀이 아직 돌고 있음 / `pkill`이 자기 자신을 끔 | **런타임 → 실행 중단**(Ctrl+M, I) 후 그 셀만 다시. `pkill`은 `[l]earnbox`로 |
| 주소는 나오는데 `502` | 터널이 서버에 못 붙음 | 4-9가 200인지 확인 → 20~30초 뒤 새로고침 |
| 주소가 안 나옴 | 터널이 아직 준비 중 / 설치 실패 | `!tail -n 10 /content/tunnel.log`, `!which cloudflared` 확인 |
| `replace /content/config.json? [y]es, [n]o…` | zip을 `/content`에 바로 풂 | `N` 입력 → 4-5처럼 `-d /content/data`, `-o` 사용 |
| `cp: … backup/: Not a directory` | `backup`이 파일로 생김 | `!mv …/LearnBox/backup …/LearnBox/backup_old.db` 후 `mkdir -p …/backup` |
| `find`에 `data.zip`이 두 개 | 다른 폴더에도 올림 | `LearnBox/data.zip` 쪽을 사용 |
| 모델 목록이 비어 있음 | 모델 복원 누락 | 5장 복원의 models 복사 → 서버 재시작 |
| 학습 중인데 GPU 메모리 0, RAM도 안 씀 | 학습이 실제로 안 돎 / CPU 런타임 | `!ps aux \| grep "[t]rain"`, 로그 확인, 런타임이 T4/TPU인지 확인 (TPU는 `nvidia-smi` 대신 [작업] 탭이나 `!tpu-info`) |
| 학습 패키지가 없다는 메시지 | `requirements-train.txt` 미설치 | 4-4 설치 |
| `TPU 패키지(torch_tpu 또는 torch_xla)가 없어 CPU로 학습했습니다` | TPU 런타임인데 TPU 패키지가 없음 | `!cd /content/claud && python -m learnbox.setup_accel` 실행 → 서버 끄고(`!pkill -f "[l]earnbox --data"`) 4-8로 다시 켜기 |
| TPU 학습이 첫 단계에서 몇 분째 그대로 | 처음 한 번 컴파일 중 (`TPU 컴파일 중(처음 한 번, 몇 분 걸릴 수 있음)`) | 기다리기. 그 뒤로는 빠름 |
| `다른 프로세스가 TPU를 쓰고 있습니다` | 노트북 셀이나 다른 학습 명령이 TPU를 먼저 잡음 (TPU는 한 프로세스만 씀) | 셀 명령 학습 끄기(`!ps aux \| grep "[t]rain"`로 확인), 노트북에서 `import torch`를 했다면 **런타임 → 세션 다시 시작**(파일은 남음). 그다음 서버를 다시 켜고(4-8) 웹 화면에서 학습 |
| `torch …와 torch_xla …의 버전이 맞지 않습니다` / `torch_tpu …는 torch 2.11 이상이 필요한데…` | torch를 따로 다시 설치했거나 `torch_xla`를 직접 설치해 짝이 깨짐 | torch·torch_xla를 따로 설치하지 않기. 새 세션에서 4장 순서대로(코랩 기본 `torch_tpu` 사용), 필요하면 `setup_accel` |
| `fp16 학습에서 손실이 20단계 연속 nan/inf입니다` | T4의 fp16(기본)에서 수치가 넘침 | 서버를 `LEARNBOX_AMP=off`로 다시 켜고(4-8) [작업] 탭에서 **이어서 재개**, 또는 학습률 낮추기. 셀 명령이면 `resume-job 작업번호 --amp off`(웹 폼에서 정밀도를 fp16으로 직접 골랐던 작업은 이 방법만 됨) |
| `ONNX 결과가 PyTorch와 다릅니다 (최대 오차 1.39e-03)` | 큰 모델의 계산 오차가 검사 기준(0.001)을 조금 넘음 | 모델은 정상, 그대로 사용 가능 (온보드 결과가 PC와 약간 다를 수 있음) |
| 생성 문장이 뒤섞이고 엉뚱함 | 평가 손실이 높음 / 자료에 영어·표·수식 섞임 | 자료 정리, 주제 집중, 미니·스몰부터 |
| 평가 손실이 7 근처에서 안 내려감 | 학습률이 높거나 자료 부족 | `--lr 0.0005`, 크기 낮추기 |
| 다음날 [작업] 탭에 이어서 재개가 없음 | `learnbox.db` 백업·복원 안 함 / 체크포인트가 `/content`에 있었음 | 10장 백업과 4-7 설정을 다음부터 지키기 |
| 연결이 자주 끊김 | 자리 비움, 무료 시간 제한 | 중간 저장을 믿고 8-2로 이어서 재개 |

로그 한 번에 보기:
```
!echo "--- 프로세스 ---"; ps aux | grep -E "[l]earnbox|[c]loudflared" | head -5
!echo "--- 서버 로그 ---"; tail -n 20 /content/server.log
!echo "--- 터널 로그 ---"; tail -n 5 /content/tunnel.log
```

---

## 13. 보안과 주의 사항

- **외부 주소는 누구나 열 수 있습니다.** 주소를 공유하지 말고, 관리자 PIN은 길고 복잡하게 정하세요. 다 쓰면 `!pkill cloudflared`.
- **온라인 모델**을 쓰면 질문과 검색된 자료 일부가 외부(Anthropic)로 전송됩니다.
- **수집**은 robots.txt와 수집 허용 시간대를 지킵니다. 사이트 약관과 저작권(위키백과 CC BY-SA, 나무위키 CC BY-NC-SA 등)을 확인하세요.
- **드라이브에 올린 자료는 구글 계정에 저장됩니다.** 민감한 자료는 올리지 마세요.
- 코랩에서 코드를 직접 고치면 세션이 끝날 때 사라집니다. 수정은 PC에서 하고 저장소에 반영하세요.

---

## 14. 자주 쓰는 셀 모음

**새 세션 준비 (다음날용, 순서대로 각각 실행)**
```
from google.colab import drive
drive.mount('/content/drive')
```
```
!git clone -b claude/vibe-scraping-ai-engine-din8fn https://github.com/gangdohyeon322-stack/claud.git /content/claud
!pip install -q -r /content/claud/requirements.txt -r /content/claud/requirements-train.txt
!cd /content/claud && python -m learnbox.setup_accel
```
```
!mkdir -p /content/data/models
!cp /content/drive/MyDrive/LearnBox/backup/learnbox.db /content/drive/MyDrive/LearnBox/backup/config.json /content/data/
!cp -r /content/drive/MyDrive/LearnBox/backup/models/* /content/data/models/ 2>/dev/null
!ls /content/data
```
```
!cd /content/claud && setsid nohup python -m learnbox --data /content/data serve --port 8750 > /content/server.log 2>&1 < /dev/null &
```
```
!sleep 15 && timeout 10 curl -s -o /dev/null -w "서버 응답 코드: %{http_code}\n" http://127.0.0.1:8750/api/info
```
```
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null
!setsid nohup cloudflared tunnel --url http://127.0.0.1:8750 > /content/tunnel.log 2>&1 < /dev/null &
```
```
!sleep 15 && grep -o "https://[a-z0-9.-]*trycloudflare.com" /content/tunnel.log | head -1
```

**끝낼 때 (각각 실행)**
```
!mkdir -p /content/drive/MyDrive/LearnBox/backup/models
!cp /content/data/learnbox.db /content/data/config.json /content/drive/MyDrive/LearnBox/backup/
!cp -r /content/data/models/* /content/drive/MyDrive/LearnBox/backup/models/ 2>/dev/null; echo "백업 끝"
```
```
!pkill cloudflared
```
```
!pkill -f "[l]earnbox --data"
```
