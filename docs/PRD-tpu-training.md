# PRD: TPU 학습 지원 (기존 기능 유지)

- 상태: **구현 완료 (코드, P0~P4)**. 코랩 실기기 검증 남음 (아래 [구현 상태](#구현-상태-2026-11-26))
- 관련 문서: `docs/PRD.md`(제품 전체), `docs/SPEC.md`(구현 계약), `docs/SPEC-pause-resume.md`(중간 저장·재개), `docs/COLAB_MANUAL.md`(코랩 사용법)
- 대상 장치: **코랩 TPU v5e-1** (TPU v5e 칩 1개, 칩당 HBM 16GB, bf16 기본)
- 한 줄 요약: 코랩 등에서 **TPU로 소형 AI 모델과 LoRA를 학습**할 수 있게 하되, **지금 되는 GPU·CPU·Mac 학습과 모든 부가 기능은 그대로** 둔다.
- 기본 정책 (개정): **TPU 백엔드와 다른 가속 기능(Intel Arc `xpu`, AMD ROCm 감지)을 기본 구성에 포함**하고, **fp16 혼합 정밀도를 기본값**으로 한다. 따로 옵션을 켜지 않아도 쓸 수 있는 가속기를 자동으로 쓴다.
  - TPU 백엔드는 **TorchTPU(`torch_tpu`)가 우선**이다. 코랩 v5e-1 런타임에 기본 설치되어 있다. 지원이 끝난 PyTorch/XLA(`torch_xla`)는 `torch_tpu`를 쓸 수 없을 때만 대체로 쓴다.
  - TPU 패키지는 **TPU 런타임에서만** `python -m learnbox.setup_accel`이 설치한다(결정 ②).

---

## 구현 상태 (2026-11-26)

| 단계 | 상태 | 내용 |
|---|---|---|
| P0 | 코드 완료 | `learnbox/accel.py`(장치 계층) 추가, 소형 AI·LoRA 학습 루프를 장치 계층 호출로 교체 |
| P0.5 | 코드 완료 | LoRA fp16 기본값(`accel.lora_precision`), `LEARNBOX_AMP`·설정 `lora_training.amp`·웹 정밀도 선택, `learnbox/setup_accel.py` + `requirements-tpu.txt`·`requirements-tpu-xla.txt`, `start.sh`에서 `setup_accel` 실행, `setup-gpu-windows.bat`의 Intel Arc(XPU) 분기, `xpu`·ROCm 감지, `device`의 '가속:' 줄 |
| P1 | 코드 완료 | TPU 감지·선택(`torch_tpu` 우선, `torch_xla` 대체), 소형 AI TPU 학습 |
| P2 | 코드 완료 | 체크포인트를 CPU 텐서로 저장, 장치가 바뀐 재개(장치가 바뀌면 LoRA 정밀도 다시 고름) |
| P3 | 코드 완료 | LoRA TPU 학습(bf16, `seq_len`까지 패딩, TorchTPU에서 체크포인팅 시 dropout 끔) |
| P4 | 코드 완료 | 웹 장치·정밀도 선택, `--device`·`--amp`, `device`·`estimate` 표시, `README.md`·`COLAB_MANUAL` 갱신 |

- 테스트: 전체 410개 통과. 기존 284개는 **고치지 않고** 통과. 새 테스트: `test_accel`, `test_tpu_tinylm`, `test_tpu_lora`, `test_device_plumbing`, `test_setup_accel` (TPU는 가짜 모듈로 검사).
- **아직 실기기로 확인하지 못한 것** (코랩에서 확인 필요):
  - 코랩 **TPU v5e-1**: `setup_accel`이 "TPU 준비됨"을 내는지, 소형 AI(나노·미니)와 LoRA(0.5B → 1.5B → 3B) 학습 완료, 첫 컴파일 시간과 단계당 속도, 웹에서 학습 시작·일시정지·재개, 서버 작업 스레드에서 TPU 사용.
  - 장치 간 재개: TPU → T4, T4 → TPU, TPU → CPU.
  - 코랩 **T4**: 옵션 없이 3B LoRA가 fp16(기본)으로 학습되는지, `LEARNBOX_AMP=off`로 예전 동작이 재현되는지.
  - **Intel Arc(`xpu`)·AMD(ROCm)** 실기기 학습, Windows·Mac에서 설치 스크립트가 깨지지 않는지.
  - 5장 성능 표의 숫자.

## 1. 배경

구현 전 PC 학습은 세 가지 장치만 썼다.

| 장치 | 감지 위치 | 비고 |
|---|---|---|
| NVIDIA GPU (`cuda`) | `tinylm._pick_device`, `train_pc._device`, `devices._gpu_info` | 소형 AI는 bf16/fp16 자동 혼합 정밀도, LoRA는 `LEARNBOX_AMP=fp16`일 때 fp16 |
| Apple GPU (`mps`) | 같은 곳 | |
| CPU | 같은 곳 | 스레드 수 조절 |

코랩 무료·저가 환경의 T4 GPU(16GB)는 3B LoRA를 겨우 돌린다(fp16 옵션으로 해결). 코랩은 TPU 런타임도 제공하지만, LearnBox는 TPU를 인식하지 못해 **CPU로 떨어져** 사실상 학습할 수 없었다.

PyTorch에서 TPU를 쓰는 방법은 두 가지다.

| 백엔드 | 상태 | 실행 방식 | 비고 |
|---|---|---|---|
| **TorchTPU (`torch_tpu`)** | 코랩 TPU v5e-1 런타임에 **기본 설치** (`torch_tpu` 0.2.0, torch 2.14.1, Python 3.13. 코랩 공지 googlecolab/colabtools#6147) | 즉시 실행(eager). 장치 이름 `tpu` | torch 2.11 이상, Python 3.11 이상. LearnBox가 **먼저** 쓴다 |
| PyTorch/XLA (`torch_xla`) | **지원 종료**: 2.9.0이 마지막, 저장소 2026-11-15 보관 | 지연 실행. 장치 이름 `xla`, PJRT | torch 2.9가 필요. `torch_xla[tpu]`는 Python 3.13에서 설치 불가(libtpu 0.0.21에 cp313 휠 없음). `torch_tpu`를 쓸 수 없을 때만 대체로 쓴다 |

TPU는 GPU와 달리 다음 성질이 있어서 코드를 조금 바꿔야 한다.
- `torch_xla`는 연산을 모아 두었다가 **한 번에 컴파일해서 실행**한다(지연 실행). 단계마다 "여기까지 실행" 신호(sync)가 필요하다. `torch_tpu`는 즉시 실행이라 필요 없다.
- 텐서 모양(배치 크기, 길이)이 바뀌면 **다시 컴파일**한다. 모양이 자주 바뀌면 GPU보다 훨씬 느려진다.
- 기본 정밀도는 **bf16**이다. fp16 기울기 스케일러가 필요 없다.
- TPU 위의 텐서를 그대로 `torch.save`하면 TPU가 없는 곳에서 읽을 수 없다.
- TPU는 **한 프로세스만** 열 수 있다. `torch_tpu` 0.2.0은 `torch.manual_seed`만 불러도 칩을 잡는다.

## 2. 목표와 비목표

**목표**
1. TPU에서 **소형 AI 모델 학습**(`train-tiny`, 웹 [학습] 탭)이 된다.
2. TPU에서 **LoRA 학습**(`train-lora`, 웹 [학습] 탭)이 된다. 0.5B부터 시작해 3B까지 단계적으로 넓힌다.
3. **기존 기능은 바뀌지 않는다.** TPU가 없거나 TPU를 고르지 않으면 지금과 똑같이 동작한다(7장의 유지 원칙).
4. TPU에서 저장한 체크포인트를 **GPU·CPU에서 이어서** 할 수 있고, 그 반대도 된다.
5. 장치는 **자동 감지**가 기본이고, 사용자가 **직접 고를 수도** 있다.
6. **가속 패키지를 기본 구성**에 넣는다. 기본 설치(`requirements-train.txt` + 설치 스크립트)만으로 그 PC·런타임에서 쓸 수 있는 가속기(NVIDIA·TPU·Intel Arc·AMD·Mac)를 쓴다. TPU는 코랩 v5e-1에 기본 설치된 `torch_tpu`를 쓰고, 없으면 TPU 런타임에서만 `setup_accel`이 설치한다(6.11).
7. **fp16을 기본값**으로 한다. bf16을 지원하지 않는 GPU(T4 등)에서 LoRA를 학습하면 `LEARNBOX_AMP`를 주지 않아도 fp16 혼합 정밀도로 학습한다(6.4).

**비목표 (이번 범위 밖)**
- 여러 TPU 칩을 동시에 쓰는 분산 학습(SPMD/FSDP). 대상인 v5e-1은 **칩이 1개**라서 분산 학습이 필요 없다. 더 큰 TPU(v5e-4 이상 등)를 쓰게 되면 별도 PRD.
- TPU에서 답변 생성(LoRA 답변, 소형 AI 이어 쓰기). 답변은 지금처럼 GPU/CPU에서 한다.
- JAX/Flax로 다시 쓰기.
- 휴대폰(브라우저) 학습. TPU와 관계없다.
- Intel Arc(`xpu`)·AMD Radeon(ROCm)의 **성능 최적화와 실기기 보증**. 이번에는 **감지와 기본 학습 경로**까지만 넣고(6.1, 6.11), 실기기 검증은 장비가 있을 때 한다.
- 대화형(다턴) 학습 데이터와 대화 화면. 별도 작업.

## 3. 사용 시나리오

1. **코랩 TPU로 LoRA**: 코랩에서 TPU 런타임을 고르고 매뉴얼대로 서버를 켠다. 웹 [학습] 탭에서 LoRA 학습을 시작하면 장치가 자동으로 `TPU`로 잡힌다. [작업] 탭에 "TPU에서 학습 중"과 진행률이 보인다.
2. **TPU에서 시작, GPU에서 마무리**: TPU 세션 시간이 끝나 학습이 끊긴다. 다음날 T4 런타임에서 [작업] 탭의 **이어서 재개**를 누르면 같은 체크포인트에서 이어 간다.
3. **장치 강제 선택**: TPU 런타임인데 비교를 위해 CPU로 돌려 보고 싶다. 학습 폼의 장치를 `CPU`로 고르거나 `--device cpu`를 준다.
4. **기존 사용자**: TPU를 쓰지 않는 사용자는 아무 변화도 느끼지 않는다. 설치할 패키지도 늘지 않는다.

## 4. 지원 범위

| 항목 | 1단계 값 | 비고 |
|---|---|---|
| 장치 | **TPU v5e-1** 칩 1개, HBM 16GB. 백엔드 `torch_tpu` 우선, 안 되면 `torch_xla`(PJRT) | 다중 칩은 비목표 |
| 학습 종류 | 소형 AI(처음부터·이어서), LoRA(새로·이어서·재개) | 자율 학습도 장치 설정을 따름 |
| 정밀도 | bf16 (가중치 bf16, LoRA 어댑터 fp32 유지) | fp16 스케일러 없음 |
| LoRA 기반 모델 | 0.5B → 1.5B → 3B 순서로 검증 | 3B는 bf16 가중치 약 6GB라 16GB에 들어갈 것으로 예상, 측정으로 확정 |
| 설치 | 코랩 v5e-1은 `torch_tpu` 기본 설치라 설치 없음. 그 밖의 TPU 런타임은 `python -m learnbox.setup_accel`이 `requirements-tpu.txt`(또는 `requirements-tpu-xla.txt`)를 설치. `requirements-train.txt`에는 안내 주석만 | 6.11 |
| 기본 정밀도 | CUDA: bf16 가능하면 bf16, 아니면 **fp16(기본)** / TPU: bf16 / Arc: bf16 가능하면 bf16, 아니면 fp16 | 6.4 |
| 환경 | 코랩 TPU 런타임 **v5e-1** (`torch_tpu` 0.2.0, torch 2.14.1, Python 3.13) | 다른 TPU 종류·Cloud TPU VM은 문서만 |

## 5. 예상 성능 (추정치, 측정으로 확인 필요)

| 항목 | 예상 | 근거 |
|---|---|---|
| 첫 단계 소요 | 수십 초 ~ 수 분 | 첫 컴파일 시간 |
| 2단계 이후 | 모양이 고정되면 재컴파일 없음 | 입력 모양 고정 요구사항(6.5) |
| 속도 | T4 fp16보다 빠를 것으로 예상 | v5e는 bf16 행렬 연산에 최적화돼 있음. 모델 크기·재컴파일 여부에 따라 크게 다름 |
| 3B LoRA 메모리 | 가중치 약 6GB + 어댑터·옵티마이저 + 활성값 | T4에서 fp16·배치 2·문맥 512로 약 14.9GB를 썼음. v5e-1(16GB)에서도 비슷한 수준으로 예상 |

숫자는 구현 후 코랩에서 같은 자료·설정으로 T4와 비교해 이 표를 갱신한다.

## 6. 기능 요구사항

### 6.1 장치 감지와 선택
- 선택 값: `auto`(기본) · `cuda` · `tpu` · `xpu`(Intel Arc) · `mps` · `cpu`. AMD ROCm용 PyTorch는 `cuda`로 잡히므로 따로 값을 두지 않고, 장치 정보에 "AMD(ROCm)"로 표시한다. 별칭(`gpu`·`rocm`→`cuda`, `xla`→`tpu`, `arc`→`xpu`, `mac`→`mps`)도 받는다.
- `auto` 순서: `cuda`(NVIDIA·AMD ROCm) → `tpu` → `xpu` → `mps` → `cpu`. TPU는 TPU 패키지(`torch_tpu` 또는 `torch_xla`)가 **설치돼 있고**, TPU 런타임 신호(PCI의 구글 TPU 칩 번호, 환경변수)가 있고, 실제 장치를 열 수 있을 때만 고른다. 감지 중 오류가 나면 조용히 다음 후보로 넘어간다.
- 지정 방법 (우선순위 높은 것부터): 학습 요청의 `device` 값(웹 폼, `--device`) → 환경변수 `LEARNBOX_DEVICE` → 설정 `training_device` → `auto`.
- 고른 장치를 쓸 수 없으면 CPU로 학습하고 이유를 안내 문구(`notes`)에 남긴다. 지금 `tinylm._pick_device`의 동작("CUDA GPU를 찾지 못해 CPU로 학습했습니다")과 같은 방식.
- `python -m learnbox device`와 [학습] 탭 기기 카드에 TPU 정보(종류, 메모리를 알 수 있으면)를 보여 준다. 등급(`tier`)에 `tpu`를 추가한다.

### 6.2 소형 AI 모델 학습 (TPU)
- 지금 학습 루프(`tinylm.train`)를 그대로 쓰고, 장치별 차이는 장치 계층(8장)이 처리한다.
- `torch_xla`에서는 단계마다 실행 신호(sync)를 보낸다. `torch_tpu`(즉시 실행)는 보내지 않는다. 손실 값은 진행 표시 간격에서만 읽어 불필요한 동기화를 줄인다.
- 옵티마이저 갱신은 모든 장치에서 `accel.optimizer_step`(스케일러 `step`)을 거친다.
- TPU 메모리가 부족하면 GPU처럼 배치를 줄여 다시 하지 않고 "TPU 메모리가 부족해 학습을 멈췄습니다. 배치 크기·문맥 길이·모델 크기를 줄이세요."로 멈춘다(체크포인트가 있으면 재개 가능).
- 학습 후 ONNX 변환과 검증은 지금처럼 **CPU로 옮긴 뒤** 한다(이미 `model.to("cpu")` 뒤에 수행).

### 6.3 LoRA 학습 (TPU)
- 지금 학습 루프(`train_pc.train_lora`)를 그대로 쓴다.
- 기반 모델은 bf16으로 불러오고, LoRA 어댑터는 fp32로 둔다.
- CUDA에서 bf16을 지원하지 않으면 **fp16 혼합 정밀도가 기본**이다. 끄려면 `LEARNBOX_AMP=off`(예전 동작)나 `fp32`(float32)를 준다. `LEARNBOX_AMP=fp16`은 지금처럼 동작한다(기본과 같음). TPU에서는 이 값을 무시하고 안내 문구에 남긴다.
- TorchTPU 알려진 문제(#6966): 활성값 체크포인팅과 dropout을 함께 쓰면 기울기가 틀린다. TPU(`torch_tpu`)에서 체크포인팅을 켤 때는 dropout을 끄고 안내 문구에 남긴다.
- 메모리 부족 시 지금처럼 오류로 끝내고, 안내 문구에 "배치·문맥 길이·모델 크기를 줄이세요"를 보여 준다.

### 6.4 정밀도
| 장치 | 소형 AI | LoRA |
|---|---|---|
| cuda | 지금 그대로 (bf16 가능하면 bf16, 아니면 fp16 + 스케일러) | bf16 가능하면 지금 그대로 bf16, 아니면 **fp16 + 스케일러가 기본** (`off`면 예전 동작, `fp32`면 float32) |
| tpu | bf16 자동 혼합 정밀도, 스케일러 없음 | 기반 모델 bf16, 어댑터 fp32 |
| xpu | bf16 가능하면 bf16, 아니면 fp16 + 스케일러 | 같은 규칙 |
| mps / cpu | 지금 그대로 | 지금 그대로 |

- LoRA 정밀도 값: `auto`(기본) · `fp16` · `off` · `fp32`.
  - `off`: **예전 동작 그대로**. 최신 torch에서 T4는 bf16 에뮬레이션, 아주 오래된 torch에서는 float32.
  - `fp32`: float32 강제.
- 기본값을 바꿔도 설정·명령으로 되돌릴 수 있어야 한다. 우선순위: 학습 요청(웹 LoRA 폼의 "정밀도" 선택, `--amp`) → 환경변수 `LEARNBOX_AMP` → 설정 `lora_training.amp` → `auto`.
- 소형 AI는 정밀도 설정을 받지 않는다(GPU에서 예전처럼 자동 혼합 정밀도).
- 자동 혼합 정밀도 문맥에는 항상 dtype을 지정한다(`torch_tpu`의 `autocast("tpu")` 기본값이 fp16이라서).
- fp16에서 손실이 `nan`이면 스케일러가 그 단계를 건너뛴다. 같은 일이 연속 20번 일어나면 체크포인트를 저장하고 "fp16 학습에서 손실이 20단계 연속 nan/inf입니다. LEARNBOX_AMP=off 로 끄거나 학습률을 낮추세요."로 멈춘다. 이어서 재개할 때 `off`/`fp32`를 주면 정밀도를 다시 고른다.
- 모델 정보(`info.json`)에 실제 쓴 정밀도를 기록한다.

### 6.5 입력 모양 고정 (재컴파일 방지)
- 학습 배치는 항상 `[batch_size, seq_len]` 모양이어야 한다.
  - 소형 AI: 지금도 고정 길이 창을 뽑으므로 그대로 만족.
  - LoRA 말뭉치 블록: 고정 길이. 마지막 짧은 블록은 TPU에서 **패딩으로 `seq_len`까지 채운다**(손실에서는 제외).
  - LoRA Q&A 예시: 길이가 제각각이므로 TPU에서는 **`seq_len`까지 패딩**한다.
  - 평가: 같은 규칙으로 고정 모양.
- GPU·CPU에서는 지금처럼 패딩하지 않는다(결과와 속도 유지).
- 학습 중 재컴파일 횟수를 세어, 첫 몇 단계 이후 늘어나면 경고를 남긴다(디버그용, `LEARNBOX_XLA_DEBUG=1`로 켬, `torch_xla` 전용).

### 6.6 체크포인트와 이어서 재개
- 저장 전에 모델·옵티마이저·스케일러 상태를 **CPU 텐서로 옮겨** 저장한다. 그래야 TPU가 없는 곳에서도 읽힌다.
- 저장 형식(`state.pt`, LoRA 어댑터 폴더, `meta`)은 바꾸지 않는다. 장치 이름만 `meta`에 기록한다.
- 재개 시 상태를 현재 장치로 옮긴다(지금 `opt.state`를 매개변수 장치로 옮기는 코드와 같은 방식). 장치가 바뀌면 LoRA 정밀도를 새 장치 기준으로 다시 고르고 안내한다.
- 난수 상태: CPU 난수는 지금처럼 저장·복원한다. TPU 난수는 저장하지 않으며, 장치가 바뀐 재개처럼 "결과가 비트 단위로 같지는 않음"으로 취급한다.
- 일시정지·지금까지 저장하고 끝내기·취소·Ctrl+C 중단은 지금과 같은 시점(단계 경계)에서 동작한다. `torch_xla`에서는 실행 신호를 보낸 뒤 저장한다.

### 6.7 진행 표시와 작업 기록
- 진행 문구에 장치를 보여 준다: `LoRA 학습 12/300 · TPU · loss 2.10 · 남은 시간 약 9분`.
- 첫 컴파일 중에는 "TPU 컴파일 중(처음 한 번, 몇 분 걸릴 수 있음)"을 보여 준다.
- 모델 정보(`info.json`)에 `device`(예: `tpu`)와 `dtype`을 기록한다. 기존 키는 그대로.

### 6.8 메모리 예측과 추천
- `estimate --kind lora|tiny`에 장치 `tpu`를 받을 수 있게 하고, TPU 메모리를 알 수 있으면 "들어감/안 들어감"을 표시한다. 모르면 "측정 필요"로 표시한다.
- 추천(`devices.recommend`)에 TPU 등급 `tpu`를 추가한다. v5e-1의 추천 LoRA 기반 모델은 측정 전까지 1.5B, 측정에서 3B가 배치 2·문맥 512로 들어가면 3B로 올린다.

### 6.9 화면과 명령줄
- 웹 [학습] 탭의 소형 AI·LoRA·자율 학습 폼에 **장치** 선택(자동/GPU/TPU/Intel Arc/Mac/CPU)을 추가한다. 쓸 수 없는 장치는 비활성으로 보여 주고, '자동'에는 실제로 쓸 장치를 보여 준다. LoRA 폼에는 **정밀도** 선택(자동/fp16/끄기(예전 방식)/fp32)도 둔다.
- `train-tiny`, `train-lora`, `autolearn start`·`run`, `estimate`에 `--device`를, `train-lora`에 `--amp`를 추가한다. 넣지 않으면 지금과 같다(`auto`).
- `resume-job`은 기본으로 **지금 장치**에서 재개하고(원래 장치를 이어받지 않음), `--device`로 바꿀 수 있다.

### 6.10 오류와 안내
| 상황 | 동작 |
|---|---|
| `tpu`를 골랐는데 TPU 패키지 없음 | CPU로 학습, 안내: "TPU 패키지(torch_tpu 또는 torch_xla)가 없어 CPU로 학습했습니다. TPU 런타임에서 python -m learnbox.setup_accel 을 실행해 설치하세요." |
| `torch_xla`와 `torch` 버전 불일치 (`torch_tpu` 없음) | 감지 단계에서 `torch_xla`를 불러오지 않고 실패로 처리, 안내: "torch …와 torch_xla …의 버전이 맞지 않습니다. python -m learnbox.setup_accel 로 맞는 짝을 설치하세요." |
| TPU 메모리 부족 | 학습 실패, 체크포인트가 있으면 재개 가능 표시 유지 (소형 AI는 배치를 줄이지 않고 멈춤, 6.2) |
| 같은 TPU를 다른 작업이 쓰는 중 | 작업 대기열(학습은 한 번에 하나)로 이미 막힘. 다른 프로세스(노트북 셀, 따로 실행한 학습 명령)가 쓰는 경우 "다른 프로세스가 TPU를 쓰고 있습니다" 안내와 함께 CPU로 학습하거나 실패 |
| TorchTPU에서 활성값 체크포인팅 + dropout | dropout을 끄고 학습, 안내 문구 (6.3) |

### 6.11 기본 패키지 구성

| 패키지 | 들어가는 곳 | 설치 조건 | 없을 때 |
|---|---|---|---|
| `torch`, `transformers`, `peft` 등 | `requirements-train.txt` (지금과 같음) | 모든 환경 | 학습 불가 (지금과 같음) |
| `torch_tpu` (TorchTPU) | `requirements-tpu.txt` (`torch_tpu>=0.2.0`, CPU용 torch 인덱스) | TPU 런타임에서만 `setup_accel`이 설치. 코랩 v5e-1은 이미 있어 아무것도 안 함. Linux x86_64, Python 3.11 이상, torch 2.11 이상 | TPU 대신 CPU로 학습, 안내 문구 |
| `torch_xla[tpu]` 2.9.0 + `torch` 2.9.0 (지원 종료) | `requirements-tpu-xla.txt` | Python 3.10~3.12 TPU 런타임에서 `torch_tpu`를 쓸 수 없을 때만. `torch_tpu`가 있으면 **설치하지 않음** | 위와 같음 |
| Intel Arc(`xpu`) 지원 | PyTorch 본체에 포함(xpu 빌드). 별도 파이썬 패키지는 넣지 않음 | xpu용 PyTorch를 설치한 경우 (`setup-gpu-windows.bat`이 Arc를 찾으면 설치) | CPU로 학습 |
| AMD ROCm 지원 | PyTorch 본체에 포함(ROCm 빌드) | ROCm용 PyTorch를 설치한 경우 (Linux) | CPU로 학습 |

- **TPU 패키지는 `requirements-train.txt`에 넣지 않는다.** pip의 조건 표시(environment marker)는 운영체제·CPU 종류·Python 버전만 알 수 있고 TPU가 있는지는 알 수 없다. `sys_platform == "linux" and platform_machine == "x86_64"` 조건은 코랩 T4를 포함한 모든 Linux PC에 설치된다. 그래서 `requirements-train.txt`에는 안내 주석만 두고, `python -m learnbox.setup_accel`이 TPU 런타임을 확인한 뒤 설치한다.
- `setup_accel`의 판단 (표준 라이브러리만 씀, torch를 불러오지 않음):
  1. Linux x86_64가 아니거나 TPU 런타임 신호(PCI의 구글 TPU 칩 번호, 환경변수)가 없으면 건너뛴다.
  2. `torch_tpu`가 있으면 "TPU 준비됨"만 출력한다(코랩 v5e 기본).
  3. 없고 Python 3.11 이상이면 `requirements-tpu.txt`를 설치한다.
  4. Python 3.10~3.12에서 `torch_tpu`를 쓸 수 없을 때만 `requirements-tpu-xla.txt`를 설치한다.
  - 옵션: `--check`(JSON, 종료 코드 0 준비됨 · 2 설치 필요/불가 · 1 TPU 아님), `--dry-run`, `--force`, `--backend auto|torch_tpu|torch_xla`, `--torch-build`(이 PC GPU에 맞는 PyTorch 설치 명령 안내: NVIDIA CUDA 12.6, Intel Arc XPU, AMD ROCm(Linux), Mac은 기본 torch).
- **`torch_xla`와 `torch`는 버전이 짝이 맞아야 한다.** `torch_xla`는 torch 의존성을 선언하지 않아서 pip가 짝이 안 맞는 버전을 그냥 설치한다. 그래서 `requirements-tpu-xla.txt`가 `torch==2.9.0`을 직접 고정한다. 설치하면 torch가 2.9로 내려가 `torch_tpu`(torch 2.11 이상 필요)가 깨지므로, `torch_tpu`가 있으면 `torch_xla`를 절대 설치하지 않는다.
- Intel Arc·AMD는 파이썬 패키지 하나로 해결되지 않고 **PyTorch를 어느 빌드로 설치하느냐**로 정해진다.
  - `setup-gpu-windows.bat`: NVIDIA가 없고 Intel Arc가 있으면 XPU용 PyTorch, 그 밖에는 예전처럼 CUDA용.
  - `start.sh`: 학습 패키지 설치 뒤 `setup_accel`을 실행한다(TPU가 아니면 건너뜀). GPU 빌드는 고르지 않는다.
  - `start-windows.bat`: 바꾸지 않았다(`torch_xla`·`torch_tpu`는 Windows 휠이 없음).
  - 그 밖의 경우는 `setup_accel --torch-build`가 맞는 설치 명령을 알려 준다.
- 어떤 가속 패키지가 없거나 깨져 있어도 **import 단계에서 오류로 멈추지 않는다.** 장치 감지에서 빠지고 안내 문구만 남긴다.
- `python -m learnbox device`의 '가속:' 줄에 설치된 가속 패키지와 버전(`torch`, `torch_tpu`, `torch_xla`, ROCm, xpu, TPU, 자동 선택)을 보여 준다.

## 7. 기존 기능 유지 원칙 (회귀 금지)

이 PRD에서 가장 중요한 조건이다.

1. **TPU 경로는 장치가 `tpu`일 때만 실행된다.** 그 밖의 장치에서는 학습 루프의 연산 순서, 정밀도, 난수 사용이 지금과 같아야 한다.
2. **TPU 패키지(`torch_tpu`, `torch_xla`)는 필요할 때만 불러온다(지연 import).** 설치돼 있지 않거나 깨져 있어도 import 오류로 멈추지 않는다. `requirements.txt`(학습 없는 기본 설치)는 바뀌지 않는다.
3. 다음 기능은 동작과 결과가 바뀌지 않아야 한다.
   - CUDA: 소형 AI 자동 혼합 정밀도, bf16 지원 GPU의 LoRA, `LEARNBOX_AMP=fp16`, 자동 활성값 체크포인팅
   - **의도한 변경 (예외)**: bf16을 지원하지 않는 CUDA GPU의 LoRA 기본 정밀도가 **fp16**으로 바뀐다. `LEARNBOX_AMP=off`로 이전 동작을 그대로 얻을 수 있어야 한다.
   - MPS, CPU 학습과 스레드 설정
   - 중간 저장·일시정지·재개·지금까지 저장하고 끝내기·취소, 장치가 바뀐 재개
   - 자율 학습, 학습 종료 결과 파일, 외장 저장 내보내기
   - ONNX 변환·검증·int8 변환, `.lbw` 업로드·다운로드, 휴대폰 학습
   - `estimate`, `device`, `jobs`, `resume-job` 명령의 기존 출력
   - 저장된 기존 체크포인트와 모델을 그대로 읽을 수 있음
4. **기존 테스트(`tests/`의 파이썬 테스트와 `tests/js`)를 고치지 않고 모두 통과**해야 한다. 기존 테스트를 바꿔야 한다면 그 이유를 따로 검토한다.
5. 같은 seed·같은 자료로 CPU에서 학습했을 때 변경 전후 손실 기록이 같아야 한다(CPU는 비트 단위 비교).

## 8. 기술 접근

### 8.1 후보
| 방식 | 장점 | 단점 | 판단 |
|---|---|---|---|
| **A. 장치 계층 모듈 + 기존 루프 유지** | 변경 범위가 작고 기존 동작을 그대로 지킬 수 있음 | 장치별 분기를 직접 관리 | **채택** |
| B. Hugging Face Accelerate로 루프 교체 | 여러 장치를 한 번에 지원 | 학습 루프·체크포인트·재개 로직을 크게 바꿔야 해서 회귀 위험이 큼, 의존성 추가 | 보류 |
| C. JAX/Flax로 다시 쓰기 | TPU에 가장 잘 맞음 | 사실상 새로 만들기, 기존 모델·체크포인트와 호환 안 됨 | 제외 |

### 8.2 구조 (구현됨)
새 모듈 `learnbox/accel.py`(장치 계층)를 만들고, 두 학습 루프의 장치 전용 코드를 이 모듈 호출로 바꾼다. CUDA·MPS·CPU에서의 동작은 지금 코드를 그대로 옮긴다. 이 모듈을 import해도 torch나 TPU 패키지를 불러오지 않는다.

| 함수 | 역할 | cuda/mps/cpu에서 | tpu에서 |
|---|---|---|---|
| `pick(device)` | 장치 고르기, 안내 문구 | 지금 `_pick_device`/`_device`와 같음 (+`xpu`) | `torch_tpu` 장치(`tpu`), 안 되면 `torch_xla` 장치(`xla`) |
| `requested_device` / `requested_amp` | 요청 → 환경변수 → 설정 → auto | | |
| `lora_precision(dev, mode)` | LoRA 정밀도 결정 | bf16 가능하면 bf16, 아니면 fp16 기본, `off`면 예전 규칙, `fp32`면 float32 | bf16 |
| `tiny_amp_dtype(dev)` | 소형 AI 혼합 정밀도 dtype | 지금과 같음 | bf16 |
| `autocast(dev, dtype)` | 혼합 정밀도 문맥 (dtype 항상 지정) | 지금과 같음 | bf16 |
| `make_scaler(dev, enabled)` | 기울기 스케일러 | 지금과 같음 | 꺼진 스케일러 |
| `optimizer_step(opt, scaler)` | 갱신 | `scaler.step(opt)` | 같음 |
| `step_end(dev)` | 단계 마무리 | 아무것도 안 함 | `torch_xla`만 실행 신호(sync), `torch_tpu`는 아무것도 안 함 |
| `to_cpu(state)` | 저장용 상태 | 지금과 같음 | CPU 텐서로 복사 (묶인 가중치는 공유 유지) |
| `rng_state(dev)` / `manual_seed` | 난수 | 지금과 같음 | CPU 난수만 저장 |
| `empty_cache(dev)` | 메모리 정리 | cuda만 지금처럼 (+xpu) | 아무것도 안 함 |
| `info()` | 장치 정보 | `devices._gpu_info`와 같음 | TPU 종류·메모리·백엔드 |

- TPU 백엔드는 `torch_tpu`를 먼저 연다. 작은 연산으로 칩을 실제로 잡아 보고, 다른 프로세스가 쓰고 있으면 이유를 안내한다. `torch_tpu`는 `torch.accelerator` API와 `torch.amp.autocast("tpu")`를 쓴다.
- `torch_xla` API는 버전마다 이름이 다르다(예: `torch_xla.sync()`와 이전의 `xm.mark_step()`, `torch_xla.device()`와 이전의 `xm.xla_device()`). `accel.py` 안에서만 버전 차이를 흡수한다. `torch_xla` import가 프로세스 전체에 덮어쓰는 함수(`clip_grad_norm_`, `nn.GRU`)는 원래대로 돌린다.
- 다중 코어용 `xmp.spawn`은 서버의 작업 스레드 구조와 맞지 않으므로 1단계에서 쓰지 않는다.
- 이 계층이 Intel Arc(`xpu`)와 AMD(ROCm) 감지도 맡는다. `xpu`는 `torch.xpu`가 있으면 쓰고, ROCm은 `torch.version.hip`으로 구분한다.
- TPU 런타임 짐작(`tpu_hint`)은 패키지를 불러오지 않고 PCI 장치(공급자 0x1ae0 + TPU 장치 번호), 환경변수, `/dev/accel*`·`/dev/vfio`를 본다. 설치 도우미와 기기 카드가 같이 쓴다.

## 9. 단계별 진행

| 단계 | 내용 | 완료 조건 |
|---|---|---|
| P0 | `accel.py` 추가, 두 학습 루프를 장치 계층 호출로 교체 (동작 변화 없음) | 기존 테스트 전부 통과, CPU 손실 기록 비트 단위 동일 |
| P0.5 | fp16 기본값(`lora_precision`), `LEARNBOX_AMP=off`·설정·웹 정밀도 선택, `setup_accel` + `requirements-tpu*.txt`(결정 ②), `xpu`·ROCm 감지, `device` 출력 | T4에서 옵션 없이 3B LoRA가 fp16으로 학습됨, `off`로 이전 동작 재현, Windows·Mac에서 기본 설치가 깨지지 않음 |
| P1 | TPU 장치 감지·선택, 소형 AI TPU 학습 | 코랩 TPU에서 나노·미니 200단계 완료, 3단계 이후 재컴파일 0회 |
| P2 | 체크포인트 CPU 저장, 장치 간 재개 | TPU→CPU, TPU→T4, T4→TPU 재개 성공 |
| P3 | LoRA TPU 학습 (0.5B → 1.5B → 3B), 고정 모양 패딩 | v5e-1에서 0.5B·1.5B·3B(r 16, 문맥 512, 배치 2) 300단계 완료 |
| P4 | 웹 장치 선택, `--device`, `device`·`estimate` 표시, 매뉴얼 갱신 | 웹에서 TPU 학습 시작·일시정지·재개 가능, `COLAB_MANUAL`에 TPU 장 추가 |
| P5 (선택) | 다중 칩 학습 (v5e-1보다 큰 TPU를 쓸 때) | 별도 PRD |

P0~P4는 코드와 테스트까지 끝났고, 실기기가 필요한 완료 조건은 [구현 상태](#구현-상태-2026-11-26)의 확인 목록으로 남아 있다.

## 10. 테스트 전략

- **기존 테스트**: 단계마다 전부 실행. 고치지 않고 통과해야 한다.
- **장치 계층 단위 테스트**: TPU가 없는 환경에서도 돌도록 `torch_tpu`·`torch_xla`를 흉내 내는 가짜 모듈로 `pick`, `optimizer_step`, `step_end`, `to_cpu` 호출 순서를 검사한다.
- **CPU 동일성 테스트**: 같은 seed로 변경 전후 손실 기록이 같은지 비교한다.
- **정밀도 정책 테스트**: 가짜 장치 정보로 `lora_precision`이 (bf16 가능 GPU → bf16), (불가 GPU → fp16), (`off` → 예전 dtype), (`fp32` → float32), (TPU → bf16)을 고르는지 검사한다.
- **설치 검증**: `setup_accel`의 판단(TPU 아님 → 건너뜀, `torch_tpu` 있음 → 준비됨, Python 버전별 설치 파일)을 가짜 상태로 검사한다. 깨끗한 가상환경에서 Linux·Windows·Mac 각각 `requirements-train.txt` 설치가 끝나는지는 실기기에서 확인한다.
- **TPU 실측 (코랩)**: 이 저장소의 CI에는 TPU가 없으므로, 코랩에서 돌리는 짧은 확인 셀(10단계 학습 → 저장 → 재개 → 재컴파일 횟수 출력)을 `docs/`에 남기고 결과를 기록한다.

## 11. 성공 지표

1. 기존 테스트 통과율 100%, 기존 기능 회귀 보고 0건.
2. 코랩 TPU에서 소형 AI·LoRA(1.5B 이상) 학습이 끝까지 완료된다.
3. 첫 컴파일 이후 단계당 재컴파일 0회.
4. TPU에서 저장한 체크포인트를 GPU·CPU에서 재개할 수 있다(그 반대도).
5. 같은 자료·설정에서 TPU 학습 결과의 평가 손실이 GPU 결과와 비슷한 범위(차이 기준은 측정 후 정함).

## 12. 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| TPU 패키지 설치가 GPU 사용자의 torch를 바꿈 | GPU 사용자의 torch가 바뀌고 설치가 실패할 수 있음 | **해결(결정 ②)**: TPU 패키지는 `requirements-train.txt`에 넣지 않고 `setup_accel`이 TPU 런타임에서만 설치. `torch_tpu`가 있으면 `torch_xla`를 설치하지 않음 |
| `torch_xla` 지원 종료(2.9.0이 마지막, 저장소 보관), Python 3.13에서 설치 불가 | 새 런타임에서 `torch_xla`를 쓸 수 없음 | `torch_tpu`를 우선 백엔드로 쓰고 `torch_xla`는 예전 런타임 대체로만 유지 |
| 코랩 이미지의 `torch_tpu`·torch 버전이 바뀜 | TPU를 열지 못하거나 동작이 달라짐 | torch를 따로 다시 설치하지 말라고 안내, `setup_accel --check`와 `device`로 상태 확인, 열지 못하면 CPU + 안내 |
| TPU는 한 프로세스만 씀 (`torch_tpu` 0.2.0은 `torch.manual_seed`만 불러도 칩을 잡음) | 서버와 노트북 셀·따로 실행한 학습 명령이 부딪혀 CPU로 학습하거나 실패 | "다른 프로세스가 TPU를 쓰고 있습니다" 안내, 매뉴얼에 "서버가 켜져 있을 때 학습 명령을 따로 실행하지 말 것" |
| TorchTPU 알려진 문제(#6966): 체크포인팅 + dropout이면 기울기가 틀림 | 조용히 잘못 학습 | TPU(`torch_tpu`)에서 체크포인팅을 켤 때 dropout을 끄고 안내 |
| fp16 기본값으로 바뀐 뒤 일부 모델에서 `nan` 발생 | 학습 실패·품질 저하 | 연속 `nan` 20단계면 저장 후 멈춤·안내, `LEARNBOX_AMP=off`로 즉시 되돌림 |
| Intel Arc·AMD 실기기 없음 | 감지만 되고 학습 경로 미검증 | 장치 정보·설치 안내에 "미검증" 표시, 장비 확보 시 검증 |
| 입력 모양이 바뀌어 재컴파일 반복 | GPU보다 느려짐 | 6.5 고정 모양 규칙, 재컴파일 수 경고(`torch_xla`) |
| 3B가 v5e-1 메모리(16GB)에 빠듯함 | 3B TPU 학습 불가 또는 배치 축소 | 활성값 체크포인팅을 TPU에서도 자동 적용, 추천은 측정 전까지 1.5B, 그래도 안 되면 배치 1 또는 문맥 384 안내 |
| 서버 작업 스레드에서 TPU 장치 사용 문제 | 웹 학습 실패 | 코랩 실측에서 웹 경로를 먼저 검증, 문제 시 학습만 별도 프로세스로 실행하는 방안 검토 |
| 코랩 TPU 종류·제공 정책 변경 | 사용 불가·성능 변화 | 문서에 확인 날짜 기록, 장치 정보 표시로 사용자가 확인 |
| 장치 계층 리팩터로 인한 기존 동작 변화 | 회귀 | P0를 별도 단계로 두고 비트 단위 비교 테스트 (기존 테스트 284개 수정 없이 통과) |
| 이 저장소 환경에서 TPU 실측 불가 | 검증 지연 | 코랩 확인 셀과 결과 기록 절차를 완료 조건에 포함 |

## 13. 결정 사항

1. ~~대상 TPU 종류~~ → **결정: 코랩 TPU v5e-1** (칩 1개, HBM 16GB).
2. ~~`auto`의 우선순위~~ → **결정: GPU 먼저.** `cuda` → `tpu` → `xpu` → `mps` → `cpu`. TPU는 TPU 런타임 신호가 있을 때만 시도.
3. ~~자율 학습의 장치~~ → **결정: 설정을 따른다.** 세션 설정(`autolearn start --device`, 웹 자율 학습 폼) → `LEARNBOX_DEVICE` → `training_device` → `auto`.
4. **3B의 범위** (열림): v5e-1에서 배치 2·문맥 512가 들어가지 않으면 배치 1로 낮춰 지원할지, 1.5B까지만 지원할지. 측정 전까지 추천은 1.5B.
5. ~~TPU 패키지 기본 포함 방식~~ → **결정: ②** (기본 구성에 포함하되 TPU 런타임에서만 설치). **구현: `python -m learnbox.setup_accel` + `requirements-tpu.txt`·`requirements-tpu-xla.txt`.** `requirements-train.txt`에는 `torch_xla` 줄을 넣지 않고 안내 주석만 둔다. pip 조건 표시로는 TPU를 알 수 없어서, 조건부 줄은 모든 Linux x86_64 PC(코랩 T4 포함)에 설치되기 때문이다. 아래는 검토했던 선택지.
   - ① `requirements-train.txt`에서 Linux일 때 `torch`와 `torch_xla`를 같은 버전(2.9)으로 고정 → 모든 Linux 사용자가 torch 2.9를 씀
   - ② 기본 구성에 포함하되, 설치 스크립트가 TPU 런타임일 때만 설치(코랩 GPU·일반 PC는 영향 없음)
   - ③ 기본 파일에 넣지 않고 설치 스크립트가 자동으로 추가 설치
6. ~~fp16 기본 적용 범위~~ → **결정: LoRA만 바꾸고 웹에 정밀도 선택을 노출.** 소형 AI는 정밀도 설정을 받지 않는다.
7. ~~TPU 백엔드~~ → **결정: `torch_tpu` 우선, `torch_xla`는 대체(지원 종료).** 코랩 v5e-1 런타임에 `torch_tpu`가 기본 설치되어 있고, `torch_xla`는 지원이 끝났으며 Python 3.13에서 설치할 수 없다.
8. ~~TPU 등급 이름~~ → **결정: `tpu`** (`tpu-v5e` 같은 종류별 이름을 쓰지 않음. 종류는 기기 정보에 따로 표시).
9. ~~`LEARNBOX_AMP=off`의 뜻~~ → **결정: `off` = 예전 동작 그대로**(최신 torch에서 T4는 bf16 에뮬레이션, 아주 오래된 torch에서는 float32), **`fp32` = float32 강제.**

## 14. 용어

| 용어 | 뜻 |
|---|---|
| TPU | Google의 기계 학습 전용 가속기 |
| `torch_tpu` (TorchTPU) | PyTorch를 TPU에서 즉시 실행으로 돌리게 해 주는 패키지. 코랩 v5e-1에 기본 설치. LearnBox의 우선 TPU 백엔드 |
| `torch_xla` (PyTorch/XLA) | PyTorch를 TPU에서 지연 실행으로 돌리게 해 주는 패키지. 지원 종료(2.9.0이 마지막) |
| PJRT | `torch_xla`가 TPU에 연결하는 실행 방식 |
| 재컴파일 | 입력 모양이 바뀌어 TPU가 계산 그래프를 다시 만드는 것 |
| 실행 신호(sync) | 모아 둔 연산을 TPU에서 실제로 실행하게 하는 호출 (`torch_xla`만) |
| 장치 계층 | 장치별 차이를 한곳에 모은 모듈(`learnbox/accel.py`) |
| `setup_accel` | TPU 런타임일 때만 TPU 패키지를 설치하는 도우미(`learnbox/setup_accel.py`) |
