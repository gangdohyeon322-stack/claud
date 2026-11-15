# 휴대폰 온디바이스 학습 — 구현 계약 (PRD: `docs/PRD-mobile-training.md`)

이 문서는 구현 계약이다. 아래 파일·함수·키·경로 이름은 그대로 써야 서로 맞물린다. 사용자에게 보이는 문자열은 한국어.

## 0. 결정 사항 (PRD 열린 질문의 답)

- **처음부터 학습 포함**: "이어서 학습"(PC에서 받은 모델)과 "처음부터 학습"(기기에서 새 어휘·새 가중치) 둘 다 지원.
- 학습 시작은 **누구나**(사용자 PIN), PC로 올리기는 **관리자만**.
- 발열 자동 휴식은 **기본 켬**.
- 학습 엔진은 외부 라이브러리 없이 **직접 구현**(순수 JS, Float32Array). WebGPU·ORT 학습 기능은 범위 밖.

## 1. 모델 수식 (PC `learnbox/tinylm.py`의 TinyGPT와 동일해야 한다)

- 토큰 임베딩 `tok[V,d]`, 위치 임베딩 `pos[T,d]`, 출력층은 `tok`와 가중치 공유(`logits = h · tokᵀ`, bias 없음).
- 블록 L개 (pre-LN): `x += proj(attn(ln1(x)))`, `x += mlp(ln2(x))`.
  - `qkv = Linear(d, 3d)` (bias 있음), `proj = Linear(d, d)` (bias), 헤드 h개, `hd = d/h`, 인과 마스크, softmax(q·kᵀ/√hd).
  - `mlp = Linear(d,4d) → GELU(정확한 erf 버전, PyTorch nn.GELU 기본) → Linear(4d,d)` (둘 다 bias).
  - LayerNorm eps = 1e-5, weight·bias 있음.
- 마지막 `ln`, 그다음 head.
- 손실: 다음 글자 예측 cross-entropy (평균), 특수 토큰 PAD=0, UNK=1, BOS=2, EOS=3.
- 옵티마이저: AdamW lr(기본 3e-3), betas (0.9, 0.95), eps 1e-8, weight_decay 0.01은 2차원 이상 파라미터에만, grad clip 1.0, 워밍업 min(100, steps/10) 뒤 코사인 감쇠(min lr = 0.1·lr). PC `train()`과 같은 스케줄.
- 초기화(처음부터 학습): Linear·Embedding 가중치 N(0, 0.02), bias 0, LayerNorm weight 1 / bias 0. 난수는 seed 있는 xorshift/mulberry32로.

### state_dict 키 (PC와 동일, 이 순서로 직렬화)

```
tok.weight [V,d]
pos.weight [T,d]
blocks.{i}.ln1.weight [d], blocks.{i}.ln1.bias [d]
blocks.{i}.qkv.weight [3d,d], blocks.{i}.qkv.bias [3d]
blocks.{i}.proj.weight [d,d], blocks.{i}.proj.bias [d]
blocks.{i}.ln2.weight [d], blocks.{i}.ln2.bias [d]
blocks.{i}.mlp.0.weight [4d,d], blocks.{i}.mlp.0.bias [4d]
blocks.{i}.mlp.2.weight [d,4d], blocks.{i}.mlp.2.bias [d]
ln.weight [d], ln.bias [d]
```
(`head.weight`는 `tok.weight`와 같으므로 저장하지 않는다. PyTorch Linear의 weight는 `[out,in]` 행우선이고, 행렬곱은 `y = x · Wᵀ + b`.)

## 2. 가중치 파일 형식 `.lbw` (LearnBox Weights) — PC ↔ 브라우저 공용

바이너리, little-endian.

```
magic      8 bytes  "LBWEIGHT"
version    uint32   1
hdr_len    uint32   JSON 헤더 길이
header     hdr_len  UTF-8 JSON
payload    float32 배열들을 header.tensors 순서대로 이어 붙임 (각 텐서 행우선)
```

header JSON:
```json
{"format":"lbw","arch":{"d_model":128,"n_layer":2,"n_head":4,"seq_len":128,"vocab_size":2000},
 "itos":["<pad>","<unk>","<bos>","<eos>","가", ...],
 "tensors":[{"name":"tok.weight","shape":[2000,128]}, ...],
 "parent": "tiny-20261115-041044" | null, "parent_model_id": 3 | null,
 "total_steps": 140, "steps": 100, "trained_on": "device", "device": {"label":"고사양 태블릿 · 메모리 12GB · 8코어","ua":"..."},
 "train_loss": 2.31, "eval_loss_before": 3.1, "eval_loss_after": 2.4,
 "train_chars": 123456, "seconds": 310.2, "created_at": "2026-11-15T13:10:00Z"}
```

Python 쪽 `learnbox/lbw.py`:
```python
def write_lbw(path, state: dict[str, "np.ndarray|torch.Tensor"], arch: dict, itos: list[str], meta: dict) -> None
def read_lbw(data: bytes) -> tuple[dict, dict[str, "np.ndarray"]]   # (header, tensors)  — numpy로 읽음
def state_from_torch_dir(model_dir) -> tuple[dict, dict, list[str]]  # (arch, state{name: ndarray}, itos) model.pt에서
def validate_lbw(header: dict, tensors: dict, max_params: int) -> list[str]   # 한국어 오류 목록 (비면 OK)
    # 검사: magic/version, arch 범위(tinylm.LIMITS)와 d_model % n_head, params <= max_params,
    # 모든 텐서 존재·shape 일치, 값이 유한(NaN/Inf 없음), itos 길이 == vocab_size, 특수 토큰 4개 선두.
```

## 3. 서버 API (`learnbox/server.py`, 핸들러는 `learnbox/app.py`)

| 메서드·경로 | 권한 | 설명 |
|---|---|---|
| GET `/api/models/<id>/files/model.lbw` | user | tiny 모델의 가중치를 `.lbw`로 내려줌(없으면 model.pt에서 만들어 캐시, `model.lbw`를 폴더에 저장). MOBILE_FILES에 `model.lbw` 추가. |
| GET `/api/snapshot?max_chunks=N&for=train` | user | 기존 스냅샷 그대로 (학습 자료로도 쓴다). 변경 없음. |
| POST `/api/models/upload` | **admin** | raw body = `.lbw` 파일, 헤더 `X-Model-Name`(URI 인코딩, 선택), `Content-Length` 필수, 상한 `cfg["mobile_training"]["max_upload_mb"]`(기본 64). `validate_lbw` 통과 → 새 폴더 `tiny-<stamp>-device`에 `model.pt`(torch 저장, config = arch + train_seq_len), `vocab.json`, `model.lbw`(원본), ONNX fp32/int8 재생성(`tinylm.export_onnx_files`), `info.json`(kind "tiny", `trained_on: "device"`, parent, total_steps 등 헤더 값 + onnx 결과) → `db.add_model("tiny", name, path, info)` → 알림 `db.notify("success","휴대폰에서 학습한 모델을 등록했습니다",...)` → 감사 기록. 응답 `{"model_id", "name", "info"}`. 검증 실패는 400 + 한국어 사유 목록. torch 없으면 400 (`pip install -r requirements-train.txt`). |
| POST `/api/mobile-training/report` | user | 기기 측정 결과 보고(선택): `{"device": {...}, "arch": {...}, "batch_size", "seq_len", "ms_per_step", "steps"}` → `db.audit(role,"mobile_benchmark",...)`, 응답 `{"ok": true}`. UI 통계용. |

`config.py` DEFAULTS에 추가:
```python
"mobile_training": {"max_upload_mb": 64, "max_params": 2_000_000, "default_steps": 100, "default_batch": 8,
                    "default_seq_len": 64, "checkpoint_every": 20, "thermal_pause": True},
```

## 4. 브라우저 학습 엔진 `web/trainer.js` (Web Worker 스크립트) + `web/onboard.js` 확장

### 4.1 `web/trainer.js` — Worker. `importScripts` 없이 단독. Node에서도 로드 가능(순수 함수 export: `self.LBTrain = {...}` 로도 노출해서 테스트).

메시지 프로토콜 (`postMessage`):

- `{type:"init", arch, weights: {name: Float32Array}|null, itos, corpus: Int32Array(토큰 id 열), evalIds: Int32Array, seed}` → 가중치 없으면 초기화(처음부터). 응답 `{type:"ready", params}`.
- `{type:"bench", steps: 10, batch, seqLen}` → 10단계 학습(가중치는 복사본에서, 원본 유지) 후 `{type:"bench", msPerStep, tokensPerSec}`.
- `{type:"train", steps, batch, seqLen, lr, startStep, totalPlanned, checkpointEvery}` → 매 단계 `{type:"progress", step, loss, msPerStep, eta}`; `checkpointEvery`마다 `{type:"checkpoint", step, state: {weights, opt: {m, v, t}}}` (Transferable); 끝나면 `{type:"done", step, trainLoss}`.
- `{type:"eval", batches: 8, batch: 16, seqLen}` → `{type:"eval", loss}`.
- `{type:"pause"}` / `{type:"resume"}` / `{type:"stop"}` → 단계 사이에서 처리. stop 시 `{type:"stopped", step, state}`.
- `{type:"export"}` → `{type:"weights", weights: {name: Float32Array}}`.
- `{type:"generate", prompt: Int32Array, maxNew, temperature, topK, seed}` → `{type:"token", id}` 스트림 후 `{type:"generated", ids}`. (학습 직후 ONNX 없이 생성하기 위함.)
- 오류는 `{type:"error", message}` (한국어).

구현 요건:
- Float32Array 기반. 행렬곱은 블록/루프 언롤로 최소한의 최적화. 역전파 전부 수동 구현(LN, GELU(erf), softmax-attention, cross-entropy).
- 단계마다 `await`로 양보(`setTimeout(0)` 또는 MessageChannel)해서 pause/stop 메시지가 처리되게 한다.
- 메모리: 활성값 버퍼는 한 번 할당하고 재사용.
- 정확성 기준: PC와 같은 가중치·같은 배치에서 손실 차이 ≤ 1e-3, 기울기 상대 오차 ≤ 1e-2 (Python 테스트가 JSON 픽스처로 검증).
- `erf`는 Abramowitz-Stegun 7.1.26(오차 1.5e-7) 또는 더 정확한 근사.

### 4.2 `web/onboard.js` 추가 API (`window.Onboard`)

```js
Onboard.lbw.parse(arrayBuffer) -> {header, tensors: {name: Float32Array}}
Onboard.lbw.build(header, tensors) -> ArrayBuffer
Onboard.training = {
  // 자료: 스냅샷(조각 x 필드)을 글자 id로 인코딩. itos가 없으면(처음부터) 빈도순으로 새 어휘 생성(vocab_size, 특수 4개 선두).
  prepareCorpus({itos|null, vocabSize, holdout=0.05}) -> {itos, train: Int32Array, eval: Int32Array, chars, unkRate}
  // 기기 측정
  benchmark({arch, itos, batch, seqLen, steps=10, onProgress}) -> {msPerStep, tokensPerSec, est100Sec, est300Sec, grade: "good"|"slow"|"bad", label}
  // 세션 (IndexedDB 키 "train_session": 설정·진행·체크포인트)
  start({mode: "resume"|"scratch", arch, steps, batch, seqLen, lr, source: {modelId, name} | null,
         thermalPause: true, chargingOnly: false, minBattery: 0, maxMinutes: 0, onProgress, onState})
  pause() / resume() / stop()
  status() -> {active, phase: "idle"|"preparing"|"evaluating"|"training"|"paused"|"cooling"|"checkpoint"|"finishing"|"done"|"error",
               step, totalSteps, loss, msPerStep, etaSec, evalBefore, evalAfter, startedAt, message, canResume}
  hasResumable() -> {step, totalSteps, arch, mode, savedAt} | null
  continueSaved() // 저장된 체크포인트에서 이어서
  discard()       // 저장된 세션/체크포인트 삭제
  result() -> {header, tensors, sample, evalBefore, evalAfter} | null   // 마지막 완료 결과
  exportLbw() -> ArrayBuffer                                            // 결과를 .lbw로
  adoptAsLocalModel() -> 결과 가중치로 로컬 생성 모델 등록(ONNX 없이 Worker 생성 사용). Onboard.generate가 이를 우선 사용.
}
```
- Wake Lock: `navigator.wakeLock.request("screen")` 시도, 실패하면 `onState({wakeLock:false})`로 알려 UI가 안내.
- `document.visibilityState === "hidden"` → 자동 pause, 보이면 resume (설정 `autoPauseHidden`, 기본 켬).
- 발열 추정: 최근 10단계 평균 msPerStep이 처음 10단계 평균의 2배 초과 → `cooling` 60초 후 재개 (thermalPause).
- 배터리: `navigator.getBattery` 있으면 chargingOnly/minBattery 검사(매 체크포인트 시점), 없으면 건너뛰고 상태에 `batteryApi:false`.
- 체크포인트: `checkpointEvery` 단계마다 `{weights, opt, step, rngState, config}`를 IndexedDB `train_session`에 저장(기존 kv 사용). 저장 전 `navigator.storage.estimate()`로 여유 확인, 부족하면 오류 메시지.
- 완료: evalAfter 계산 → 결과를 kv `train_result`에 저장, `train_session` 삭제.

## 5. 화면 (`web/index.html`, `web/app.js`, `web/style.css`)

[학습] 탭에 카드 **"이 기기에서 학습 (온디바이스)"** — 모든 사용자에게 보임(관리자 전용 아님). 구성:
1. 기기 요약(Onboard.detectDevice label) + "이 기기에서 측정" 버튼 → 결과: 단계당 시간, 100/300단계 예상, 등급 배지(적합/느림/권장하지 않음) + 서버 보고(`/api/mobile-training/report`, 실패 무시).
2. 폼: 방식(이어서 학습: 서버의 tiny 모델 목록에서 선택 / 처음부터 학습: 프리셋 나노·미니 + 직접 설정 d_model,n_layer,n_head,seq_len,vocab_size), 단계 수(기본 100), 배치(8), 문맥 길이(64), 학습률(3e-3), 옵션(화면 꺼짐 방지, 백그라운드 자동 일시정지, 발열 시 휴식, 충전 중에만, 배터리 n% 이하 중지, 최대 시간 분). 파라미터·메모리 추정(기존 `/api/estimate` 재사용, 상한 `max_params` 초과 시 경고와 시작 막기).
3. 자료 준비 안내: 스냅샷이 없으면 "먼저 [검색] 탭에서 이 기기에 저장" 안내 + 바로가기 버튼.
4. 진행: 진행 막대, 단계/전체, loss, 단계당 ms, 남은 시간, 상태(학습 중/일시정지/발열 휴식/체크포인트 저장), 버튼 일시정지·재개·중지. "이어서 할 수 있는 학습이 있습니다 (N/M단계)" 배너 + 이어서/버리기.
5. 결과 카드: 평가 손실 전→후(나빠졌으면 경고와 "이전 모델 유지" 기본 선택), 샘플 생성, 버튼 **이 기기 모델로 사용**(adoptAsLocalModel), **.lbw 파일로 저장**(다운로드), **PC로 올리기**(관리자일 때만 활성, `/api/models/upload`) → 성공 시 [모델] 탭 목록에 "기기 학습" 배지로 표시.
6. [모델] 탭 카드: `info.trained_on === "device"`면 "📱 기기에서 학습" 배지, parent 표시.

작업(Jobs) 탭에는 넣지 않는다(브라우저 내 작업이므로). 오류/안내는 모두 한국어. 360px 폭에서 가로 스크롤 없음.

## 6. 테스트

- `tests/test_lbw.py`: write/read 왕복, validate 오류(NaN, shape, vocab 불일치, 파라미터 초과), `state_from_torch_dir`.
- `tests/test_mobile_upload.py`: 서버 라우트 — model.lbw 다운로드(캐시 생성), 업로드 성공(ONNX 재생성, 모델 등록, 알림), 사용자 권한 403, 손상 파일 400, 너무 큰 파일 413, report 200.
- `tests/test_trainer_parity.py`: Python이 작은 arch(d 32, L 2, h 4, T 16, V 50)에서 가중치·배치를 만들고 PyTorch로 loss/기울기/1단계 AdamW 후 가중치를 계산 → JSON 픽스처 → `node tests/js/trainer_parity.js <fixture>`로 trainer.js 실행 → 결과 비교(손실 ≤1e-3, 기울기·갱신 가중치 상대 오차 ≤1e-2). node 없으면 skip.
- `tests/js/onboard_training.test.js`(node로 실행, unittest가 subprocess로 호출): prepareCorpus(어휘 생성·특수 토큰·holdout), lbw parse/build 왕복, benchmark 등급 계산, 체크포인트 직렬화.
- Playwright e2e(스크래치 폴더): 태블릿 뷰포트(1280×800, 터치)와 Pixel 7에서 측정 → 처음부터 학습 30단계 → 결과 카드 → 이 기기 모델로 사용 → 생성 → .lbw 다운로드 → 관리자로 PC 올리기 → 모델 탭 배지 → 브라우저 새로고침 후 이어서 학습 배너.
