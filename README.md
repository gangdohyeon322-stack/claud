# LearnBox

인터넷 자료를 **허용된 시간에만** 조금씩 모으고, 검색하고, 그 자료로 **PC용 모델**과 **휴대폰용 초소형 모델**을 학습해, 결과를 **관리자가 외장 저장장치(USB·외장 SSD·HDD)** 에 저장하는 개인용 도구입니다.
PC와 휴대폰이 **같은 화면, 같은 기능, 같은 데이터**를 씁니다.

```
[PC: LearnBox 엔진] ── 수집(허용 시간대만) → 정제 → 검색 인덱스(SQLite)
        │                                   └→ 학습: PC용 LoRA / 휴대폰용 ONNX
        │                                   └→ 관리자 내보내기 → 외장 저장장치
        ├── PC 브라우저        ┐
        └── 휴대폰 브라우저    ┘ 같은 웹 화면 (같은 Wi-Fi)
              └ 기기에 저장: 자료 사본(오프라인 검색) + 휴대폰용 모델(온디바이스 실행)
```

## 기능

| 기능 | 설명 |
|---|---|
| 지정 링크 학습 | 링크를 넣으면 본문을 가져와 저장·색인. 같은 사이트 안 링크를 1~2단계 따라갈 수 있음 |
| 실시간 검색 수집 | 검색어로 웹을 검색하고 결과 페이지를 가져와 저장. 위키백과(키 불필요), SearXNG, Brave Search API 지원 |
| 예약 자동 수집 | 등록한 검색어·링크를 허용 시간대에 정해진 간격으로 하나씩 조금씩 수집 |
| 수집 시간 제한 | 설정한 요일·시간대에만 인터넷 접속. **요청 하나하나마다** 시각을 확인해 시간이 끝나면 즉시 멈춤 |
| 수집 예절·안전 | robots.txt 준수, 사이트별 요청 간격, 허용/차단 도메인, 내부망 주소 차단, 페이지 크기 제한 |
| 검색 | 한국어 부분 일치 검색(FTS5 trigram), 강조 표시 |
| 질문하기 | 검색된 자료를 근거로 PC 모델이 답변 (출처 번호 표시) |
| PC 모델 학습 | 공개 소형 모델(기본: Qwen2.5-0.5B-Instruct)에 수집 자료로 LoRA 추가 학습. GPU 자동 사용 |
| 휴대폰용 모델 학습 | CPU로 학습되는 글자 단위 초소형 GPT(약 0.5M~3M 파라미터) → ONNX 변환 → 휴대폰 브라우저에서 실행 |
| 외장 저장 내보내기 | **관리자만**. 자료·인덱스·모델·말뭉치를 복사하고 파일별 SHA-256(manifest.json) 기록, 감사 기록 남김 |
| 무결성 검사 / 가져오기 | 내보낸 폴더 검증, 다른 PC로 가져오기 |
| 오프라인 | 휴대폰에 자료 사본과 모델을 저장해 인터넷 없이 검색·생성 |

## 설치와 실행 (PC)

Python 3.10 이상이 필요합니다.

**간단 실행**
- Windows: `start-windows.bat` 더블클릭
- Mac/Linux: `./start.sh`

처음 실행할 때 설치를 하고 PIN 두 개를 정합니다.
- **사용자 PIN** (4자 이상): 검색, 질문, 수집, 진행 상황 보기
- **관리자 PIN** (6자 이상): 위 기능 + 학습, 외장 저장 내보내기, 설정, 문서 삭제, 감사 기록

**직접 실행**
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt            # Windows: .venv\Scripts\pip
.venv/bin/pip install -r requirements-train.txt      # 학습 기능 (선택, 수 GB)
.venv/bin/python -m learnbox init                    # PIN 설정
.venv/bin/python -m learnbox serve --lan             # 서버 실행
```

실행하면 접속 주소가 나옵니다.
```
LearnBox 실행 중: http://127.0.0.1:8750
  휴대폰(같은 Wi-Fi)에서 접속: http://192.168.0.12:8750
```

## 휴대폰에서 쓰기

1. PC에서 `serve --lan`으로 실행합니다.
2. 휴대폰을 PC와 **같은 Wi-Fi**에 연결하고, 출력된 주소로 접속해 로그인합니다.
3. [검색] 탭 → **사본 받기**: 자료를 휴대폰에 저장해 인터넷 없이 검색합니다.
4. [모델] 탭 → 휴대폰용 모델 **이 기기로 받기** → **생성**: 휴대폰에서 직접 모델을 실행합니다. 처음 한 번은 실행 엔진(onnxruntime-web)을 인터넷에서 받습니다.

**HTTPS가 필요한 경우**: 홈 화면에 앱으로 설치하거나, PC가 꺼진 상태에서도 화면을 여는 기능은 브라우저 정책상 HTTPS에서만 됩니다.
인증서가 있으면 `serve --lan --cert cert.pem --key key.pem`으로 실행하세요. 예를 들어 Tailscale을 쓰면 `tailscale cert`로 인증서를 받을 수 있고, 밖에서도 PC에 접속할 수 있습니다.
HTTP로 접속해도 자료 사본과 모델은 휴대폰에 저장되고 동작합니다.

## 학습

### PC 모델 (LoRA)
- [학습] 탭 또는 `python -m learnbox train-pc --max-steps 200`
- 수집 자료를 말뭉치(`data/datasets/corpus-*`)로 만들고, 기반 모델에 LoRA 어댑터를 학습해 `data/models/pc-*`에 저장합니다.
- 학습 전후 **평가 손실**을 기록합니다. 값이 내려가면 수집 자료의 문체·용어에 더 맞춰졌다는 뜻입니다.
- 학습이 끝나면 [검색] 탭의 **답변 모델**에서 고르고 **질문하기**를 누릅니다.
- 처음 실행 시 기반 모델을 Hugging Face에서 내려받습니다 (0.5B 모델 약 1GB).

| 장비 | 권장 |
|---|---|
| NVIDIA GPU (VRAM 8GB 이상) | 0.5B~1.5B 모델, 수백~수천 단계 |
| Apple Silicon (M1 이상) | 0.5B 모델, MPS 자동 사용 |
| CPU만 | 0.5B 모델, 적은 단계로만. 매우 느림 |

다른 기반 모델은 설정 파일(`data/config.json`의 `pc_training.base_model`) 또는 [학습] 탭에서 바꿉니다. **모델마다 라이선스가 다르니** 배포 전에 확인하세요.

### 휴대폰용 모델 (저사양)
- [학습] 탭 또는 `python -m learnbox train-mobile --preset small --steps 2000`
- 글자 단위 초소형 모델을 처음부터 학습해 `model.onnx`(약 3~15MB)로 내보냅니다. CPU로 수 분 걸립니다.
- **한계**: 수집 자료의 문장을 흉내 내는 수준이며, 사실을 정확히 답하지 못합니다. 사실 확인은 검색 결과(출처)를 보세요.

## 외장 저장장치로 내보내기 (관리자)

- [관리] 탭 → 연결된 장치를 고르거나 경로 입력 → **내보내기**
- 또는 `python -m learnbox export E:\` (관리자 PIN 확인)

```
LearnBox-export-20261114-123000/
├─ learnbox.db          수집 자료 + 검색 인덱스
├─ models/              pc-* (LoRA 어댑터), mobile-* (ONNX)
├─ datasets/            학습 말뭉치
├─ config.public.json   설정 사본 (PIN·API 키 제외)
├─ manifest.json        파일별 SHA-256
└─ README.txt
```

- 무결성 검사: `python -m learnbox verify <내보낸 폴더>`
- 다른 PC로 가져오기: `python -m learnbox --data <데이터 폴더> import <내보낸 폴더>`
- 저장장치는 **exFAT 또는 NTFS** 포맷을 권장합니다 (FAT32는 4GB 넘는 파일 불가).
- 자료에 개인정보가 섞일 수 있으면 BitLocker/FileVault로 **드라이브를 암호화**하세요.

## 명령줄

```
python -m learnbox init | serve | status
python -m learnbox collect <URL...> [--depth 1]
python -m learnbox websearch "검색어"
python -m learnbox search "검색어"
python -m learnbox train-pc [--base-model ...] [--max-steps N]
python -m learnbox train-mobile [--preset tiny|small] [--steps N]
python -m learnbox drives | export <경로> | verify <폴더> | import <폴더>
```
데이터 폴더는 기본 `./data`이며 `--data` 또는 `LEARNBOX_DATA` 환경변수로 바꿉니다.

## 주요 설정 (`data/config.json`, [관리] 탭에서 수정)

| 항목 | 기본값 | 설명 |
|---|---|---|
| `allowed_windows` | 매일 00:00~24:00 | 수집 허용 시간대. 시작이 끝보다 늦으면 자정을 넘기는 구간 (예: 22:00~06:00) |
| `timezone` | Asia/Seoul | |
| `allowed_domains` | 비어 있음(전체) | 넣으면 그 도메인만 수집 |
| `request_interval_sec` | 3 | 같은 사이트에 보내는 요청 사이 최소 간격 |
| `max_pages_per_job` | 20 | 작업 하나가 가져오는 최대 페이지 |
| `search.provider` | wikipedia | wikipedia / searxng / brave / none |
| `auto_collect` | 꺼짐, 60분 | 예약 목록 자동 수집 |

## 법적·윤리적 주의

- 사이트 **이용약관**을 확인하세요. robots.txt를 지켜도 약관상 수집이 금지된 곳이 있습니다.
- 수집한 글의 **저작권**은 원저작자에게 있습니다. 개인 학습·연구 용도로 쓰고, 원문을 재배포하지 마세요. 위키백과 글은 CC BY-SA입니다.
- **개인정보**가 포함된 페이지는 수집하지 않도록 허용 도메인을 설정하세요.

## 테스트

```bash
.venv/bin/python -m unittest discover -s tests
```
