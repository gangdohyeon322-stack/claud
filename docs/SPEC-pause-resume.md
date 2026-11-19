# PC 학습 일시정지 · 중간 저장 · 재개 — 구현 계약

배경: 사용자가 외장 HDD에서 베이스 크기 소형 AI를 9시간(6000단계 중 4580단계) 학습하다가 HDD를 건드려 연결이 끊겼고, PC 학습에는 중간 저장이 없어 전부 잃었다. 이 기능의 목표는 **끊겨도 마지막 저장 지점부터 이어서 할 수 있게** 하고, **원할 때 멈췄다가 다시 시작**할 수 있게 하는 것이다.

사용자에게 보이는 문자열은 한국어. 아래 이름·키·경로는 계약이므로 그대로 쓴다.

## 0. 이미 있는 것 (수정하지 말고 사용)

- `learnbox/checkpoint.py` — `begin(root) -> tmp`, `commit(root, tmp, meta) -> {"path","step","saved_at","bytes"}`, `load(root) -> (meta, state_dir) | None`, `summary(root) -> dict|None`, `remove(root)`, `dir_size(path)`. root 아래 `current/`(확정), `prev/`, `.tmp/`. 원자적 교체, 어느 순간 끊겨도 current 또는 prev 하나는 온전.
- `cfg["training_checkpoint"] = {"enabled": True, "every_minutes": 10, "every_steps": 0, "dir": ""}` (config.py DEFAULTS에 추가됨). dir가 비면 `cfg.data_dir / "checkpoints"`.

## 1. 학습 함수에 추가할 인자 (tinylm.train, train_pc.train_lora 공통)

모두 키워드 인자, 기본값이면 지금과 똑같이 동작해야 한다(기존 테스트 통과).

```python
checkpoint_dir: str | None = None       # 체크포인트 root. None이면 저장하지 않음(일시정지 시에도 저장 없이 멈춤)
checkpoint_every_steps: int = 0         # N단계마다 저장 (0 = 사용 안 함)
checkpoint_every_sec: float = 600.0     # N초마다 저장 (0 = 사용 안 함). 둘 다 0이면 일시정지/끝내기 때만 저장
resume_checkpoint: str | None = None    # 이전 실행이 남긴 체크포인트 root -> 그 단계 다음부터 이어서
pause_point = None                      # callable(save: Callable[[], dict|None]) -> None. 매 단계 옵티마이저 스텝 직후 1회 호출
stop_requested = None                   # callable() -> bool. True면 '지금까지 저장하고 끝내기'
on_checkpoint = None                    # callable(summary: dict) -> None. 체크포인트를 쓸 때마다 호출 (checkpoint.commit 반환값 + "steps")
```

동작:
- 매 단계 끝: (a) 저장 조건(단계/시간)이 되면 체크포인트 저장 → `on_checkpoint(...)`. (b) `pause_point(save)` 호출 — `save()`는 지금 상태로 체크포인트를 저장하고 요약을 돌려주는 함수(이미 이번 단계에 저장했으면 다시 쓰지 않고 그 요약 반환; checkpoint_dir가 None이면 None 반환). pause_point 안에서 일시정지 동안 블록되며, 취소되면 `Cancelled`(호출자의 예외)가 올라올 수 있다 — 그대로 전파. (c) `stop_requested()`가 True면 남은 단계를 건너뛰고 지금까지의 결과로 평가·저장을 평소처럼 마친다: `info["stopped_early"] = True`, notes에 `f"사용자가 {step}/{steps}단계에서 학습을 끝냈습니다."`.
- 일시정지 중 경과 시간은 학습 시간/ETA에서 빼야 한다(진행 메시지의 남은 시간이 튀지 않게).
- 학습이 정상 종료(완료 또는 일찍 끝내기)되면 함수는 체크포인트를 **지우지 않는다**(지우는 것은 작업 계층의 몫).
- `resume_checkpoint`가 주어지면:
  - `checkpoint.load()`로 읽고, `meta["kind"]`가 맞는지 확인(아니면 한국어 ValueError).
  - 체크포인트에 기록된 값(구조, steps, batch_size, lr, seq_len, grad_accum, lora 설정, base_model, seed, train_path, eval_path, qa_path, parent 등)을 **우선** 사용한다. 호출자가 넘긴 같은 이름의 인자는 무시(다르면 notes에 한 줄 남김).
  - train_path/eval_path 파일이 없으면 `ValueError("체크포인트에 기록된 학습 자료가 없습니다: <경로>. 이어서 할 수 없습니다.")`.
  - 모델 가중치, 옵티마이저 상태, 학습률 스케줄 위치(step), 난수 상태(torch 생성기, CPU/CUDA RNG, Python random), 최근 손실 기록, 학습 전 평가 손실(eval_before), 누적 경과 시간을 복원하고 step+1부터 steps까지 계속.
  - 장치가 바뀌었으면(GPU 저장 → CPU 재개 등) map_location으로 맞춘다. 옵티마이저 상태 텐서도 새 장치로 옮긴다.
  - 결과 info에 `"resumed_from_step": k`, `"resumes": n`(누적 재개 횟수, meta에서 +1) 추가. `seconds`는 누적 시간.
  - 같은 seed·같은 장치(CPU)에서 "끊김 없이 N단계" 와 "k단계에서 저장 → 새 프로세스처럼 재개 → N단계"의 최종 가중치가 같아야 한다(테스트: 최대 절대 오차 1e-5 이하; 데이터 배치 순서까지 같도록 RNG 상태를 저장).

### tinylm (소형 AI) 체크포인트 내용
`state_dir/state.pt`: `{"model": state_dict, "opt": optimizer.state_dict(), "scaler": scaler.state_dict() or None, "gen": batch generator state, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if cuda else None, "recent": 최근 손실 리스트, "B": 실제 배치(자동 축소 반영), "T": 실제 문맥}` + `state_dir/vocab.json`.
meta: `{"kind": "tiny", "step", "steps", "arch", "batch_size", "lr", "seed", "train_path", "eval_path", "parent", "parent_path", "parent_total", "parent_train_seq", "eval_before", "elapsed", "train_seconds", "resumes", "notes", "onnx_int8", "max_seconds"}`.

### train_pc (LoRA) 체크포인트 내용
`state_dir/adapter/` (PEFT `save_pretrained`, 임베딩 저장 안 함) + `state_dir/trainer.pt`: `{"opt": optimizer.state_dict(), "py_rng": random 객체 getstate(), "torch_rng", "cuda_rng", "history": 손실 기록, "qa_used"}`.
meta: `{"kind": "lora", "step", "steps"(=max_steps), "base_model", "lr", "seq_len", "batch_size", "grad_accum", "lora_r", "lora_alpha", "qa_path", "train_path", "eval_path", "seed", "dtype", "max_train_tokens", "gradient_checkpointing", "parent", "parent_path", "parent_total", "eval_before", "elapsed", "resumes", "notes"}`.
재개 시 토큰화는 같은 seed로 다시 해서(같은 순서) 같은 블록을 만든 뒤, Python random 상태를 체크포인트 값으로 덮어쓴다. 어댑터는 `PeftModel.from_pretrained(base, state_dir/"adapter", is_trainable=True)`.

## 2. 작업 계층 (`learnbox/jobs.py`, `learnbox/db.py`)

- 작업 상태에 `"paused"` 추가. `db.mark_interrupted()`는 `('queued','running','paused')`를 `interrupted`로 바꾸고 **result는 그대로 둔다**(체크포인트 정보 보존). 메시지: 체크포인트가 있으면 `"엔진이 재시작되어 중단됨 · 이어서 재개할 수 있습니다 (N/M단계)"`.
- `JobManager`에 `pause_flags: set`, `finish_flags: set`, 그리고:
  - `pause(job_id)` / `resume(job_id)` / `finish(job_id)` ('지금까지 저장하고 끝내기'). 큐에 대기 중인 작업은 pause 불가(ValueError 한국어).
  - `JobContext.pause_point(save)`: pause 요청이 있으면 `save()` 호출(요약 받기) → DB status `paused`, message `"일시정지됨 · {step}/{steps}단계 · {저장 시각} 저장"` (save가 None이면 "체크포인트 없음") → resume/cancel/finish를 0.5초 간격으로 기다림 → resume이면 status `running`으로 되돌리고 진행, cancel이면 `Cancelled`, finish면 플래그 유지한 채 반환(학습 함수가 stop_requested로 처리).
  - `JobContext.stop_requested()` → finish 플래그.
  - `JobContext.on_checkpoint(summary)` → job의 `result`를 `{"checkpoint": {...summary, "steps"}}`로 갱신(학습 중에도 화면에서 '마지막 저장' 표시). 완료 시 최종 result가 덮어씀.
  - cancel이 일시정지 상태의 작업도 깨워서 취소되게.
  - 하위 호환: 기존 `progress`, `check_cancel` 유지.
- 서버 재시작 시(`start(mark_interrupted=True)`): 중단 표시 후, 체크포인트가 있는 학습 작업마다 `db.notify("warning", "학습이 중단되었습니다", "작업 탭에서 '이어서 재개'를 누르면 {step}/{steps}단계부터 계속합니다.", {"kind":"train_interrupted","job_id":...})`.

## 3. 앱 (`learnbox/app.py`)

- `checkpoint_root(job_id) -> Path`: `cfg["training_checkpoint"]["dir"]` 또는 `cfg.data_dir/"checkpoints"` 아래 `job-{job_id}`. 사용 안 함(enabled False)이면 None.
- `_train_tiny`, `_train_lora`: 학습 함수에 `checkpoint_dir`, `checkpoint_every_steps`, `checkpoint_every_sec`(=every_minutes*60), `pause_point=ctx.pause_point`, `stop_requested=ctx.stop_requested`, `on_checkpoint=ctx.on_checkpoint` 전달.
  - params에 `resume_checkpoint`가 있으면: build_corpus 건너뜀(체크포인트에 기록된 자료 사용), 그 root를 checkpoint_dir로도 계속 사용, `resumed_from_job` 기록.
  - 성공(완료/일찍 끝내기): 체크포인트 삭제. 취소(Cancelled): 체크포인트 삭제(사용자가 버린 것). 그 밖의 예외/중단: **보존**하고 job result에 `{"checkpoint": summary, "resumable": True}` 남김(실패 메시지는 그대로).
  - 모델 폴더(out)는 실패 시 지금처럼 지운다(체크포인트와 별개).
- `restart_job(job_id, actor) -> new_job_id`: 원래 작업이 train_tiny/train_lora이고 상태가 interrupted/failed/paused?(paused는 불가)이며 체크포인트가 load 가능해야 함. 새 작업 params = 원래 params + `{"resume_checkpoint": root, "resumed_from_job": job_id}`; 새 작업의 체크포인트 root는 **원래 root를 그대로** 사용(새 job id 폴더로 옮기지 않음 — 원래 root 경로를 params로 넘김). 원래 작업 result에 `"resumed_by": new_id` 기록. 오류는 한국어 ValueError.
- `discard_checkpoint(job_id)`: 체크포인트 삭제, result에서 checkpoint 제거 표시.
- 자율 학습(autolearn_train) 작업도 pause_point/stop_requested를 받을 수 있게 하면 좋지만 이번 범위는 **train_tiny, train_lora 작업만**. autolearn은 손대지 않는다.

## 4. HTTP API (`learnbox/server.py`)

| 메서드·경로 | 권한 | 동작 |
|---|---|---|
| POST `/api/jobs/<id>/pause` | admin | 실행 중인 학습 작업 일시정지 요청. 학습 작업이 아니거나 실행 중이 아니면 400 |
| POST `/api/jobs/<id>/resume` | admin | 일시정지된 작업 재개 |
| POST `/api/jobs/<id>/finish` | admin | 지금까지 저장하고 끝내기 (실행 중/일시정지 중) |
| POST `/api/jobs/<id>/restart` | admin | 체크포인트에서 이어서 재개 → `{"job_id": 새 id}` |
| POST `/api/jobs/<id>/discard-checkpoint` | admin | 체크포인트 삭제 |
| POST `/api/jobs/<id>/cancel` | (기존) | 일시정지 중이어도 취소되게 |

`GET /api/jobs`의 각 작업에 `checkpoint` 요약(result.checkpoint, 없으면 체크포인트 폴더를 summary로 확인)과 `can_pause`, `can_resume`, `can_finish`, `can_restart` 불리언을 붙여 UI가 버튼을 고르게 한다.

## 5. CLI (`learnbox/__main__.py`)

- `python -m learnbox jobs` (최근 작업 목록: id, 종류, 상태, 체크포인트 단계)
- `python -m learnbox resume-job <id>` (체크포인트에서 이어서, 현재 스레드에서 실행하며 진행 출력)
- 학습 명령(train-tiny/train-lora)에 `--checkpoint-minutes N` 옵션(0이면 끔). Ctrl+C 시 체크포인트를 저장하고 "python -m learnbox resume-job <id>로 이어서 할 수 있습니다" 안내.

## 6. 화면 (`web/index.html`, `web/app.js`, `web/style.css`)

- [작업] 탭 학습 작업 카드:
  - 실행 중: **일시정지**, **지금까지 저장하고 끝내기**, 취소 (+ "마지막 저장: 3분 전 · 4500/6000단계")
  - 일시정지: 상태 배지 "일시정지", **재개**, **지금까지 저장하고 끝내기**, 취소. 안내: "일시정지 중에도 서버와 저장장치는 연결되어 있어야 합니다. 저장장치를 빼려면 서버를 끄세요."
  - 중단/실패 + 체크포인트: **이어서 재개 (4500/6000단계부터)**, **체크포인트 삭제** (확인 창)
  - 관리자만 버튼 표시. 진행 막대는 paused에서도 보이게.
- STATUS 라벨에 `paused: "일시정지"` 추가.
- [학습] 탭 소형 AI / LoRA 카드에 한 줄: "학습 중 {N}분마다 중간 저장합니다. 끊겨도 [작업] 탭에서 이어서 재개할 수 있습니다." (설정값 반영, 꺼져 있으면 "중간 저장이 꺼져 있습니다").
- [관리] 탭 설정에 "학습 중간 저장": 사용(체크박스), 간격(분), 단계 간격, 저장 위치(비우면 데이터 폴더/checkpoints; 안내: "외장 하드에서 학습한다면 내장 드라이브 경로를 권장합니다"). `POST /api/settings`로 저장(`training_checkpoint` 섹션).
- 알림 종에 `train_interrupted` 알림이 오면 클릭 시 [작업] 탭으로 이동.
- 360px 폭에서 가로 스크롤 없음, 다크 모드, 모든 문자열 escape.

## 7. 테스트

- `tests/test_tinylm_checkpoint.py`: 작은 구조(d32 L2 h4 T16 V60, CPU, seed 0)로
  - 끊김 없는 30단계 vs (12단계에서 저장 후 예외로 중단 → resume_checkpoint로 재개 → 30단계)의 최종 state_dict 최대 차이 ≤ 1e-5, total_steps/steps 일치, info["resumed_from_step"] == 12.
  - pause_point가 save()를 부르면 체크포인트 생성, stop_requested로 일찍 끝내기(stopped_early, steps 단계 수).
  - 시간 기준 저장(checkpoint_every_sec 아주 작게)과 단계 기준 저장.
  - 손상된 current(파일 일부 삭제) → prev 사용 / 둘 다 없으면 ValueError.
  - train_path가 사라진 체크포인트 → 한국어 ValueError.
- `tests/test_lora_checkpoint.py`: `hf-internal-testing/tiny-random-gpt2`(캐시 없으면 skip)로 같은 동일성 검사(LoRA 가중치 최대 차이 ≤ 1e-5), stop_requested, 재개 횟수.
- `tests/test_pause_resume.py`: 실제 JobManager + in-process 서버로
  - train_tiny 작업 실행 중 pause → status paused + checkpoint 요약 → resume → 완료, 체크포인트 삭제됨.
  - finish → stopped_early 모델 등록.
  - 학습 중 강제 중단(테스트용으로 핸들러에 예외 주입 또는 mark_interrupted 시뮬레이션) → result.checkpoint 남음 → `/restart` → 새 작업이 이어서 완료, total_steps가 목표와 같음.
  - 권한(사용자 403), 잘못된 상태 400, discard-checkpoint.
  - 서버 재시작 시 알림 생성.
