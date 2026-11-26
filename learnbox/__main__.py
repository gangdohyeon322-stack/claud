"""명령줄: python -m learnbox <명령>"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time

ARCH_FLAGS = (("d_model", "--d-model"), ("n_layer", "--n-layer"), ("n_head", "--n-head"), ("seq_len", "--seq-len"),
              ("vocab_size", "--vocab-size"))
FIT_LABELS = (("phone-low", "휴대폰 저사양"), ("phone-high", "휴대폰 고사양"), ("pc-cpu", "PC CPU"),
              ("pc-gpu", "PC GPU"), ("pc-gpu-high", "고성능 GPU"))
DEVICE_CHOICES = ("auto", "cuda", "tpu", "xpu", "mps", "cpu")
AMP_CHOICES = ("auto", "fp16", "off", "fp32")


def _app(args):
    from .app import App

    return App(args.data)


def _j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _print_job(job):
    print(f"[{job['status']}] {job['message']}")
    if job.get("result"):
        print(_j(job["result"])[:4000])
    return 0 if job["status"] in ("done", "blocked") else 1


def _hl(text: str) -> str:
    from .db import HL_END, HL_START

    return (text or "").replace(HL_START, "[").replace(HL_END, "]").replace("\n", " ")


def _fmt_bytes(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return "-"


def _add_arch_flags(s) -> None:
    s.add_argument("--preset", help="크기: auto | nano | mini | small | base | large | custom")
    for key, flag in ARCH_FLAGS:
        s.add_argument(flag, type=int, dest=key, help=f"{key} (직접 설정, 0이면 preset 값)")


def _add_checkpoint_flag(s) -> None:
    s.add_argument("--checkpoint-minutes", type=float, metavar="N",
                   help="학습 중 N분마다 중간 저장 (기본: 설정값, 0이면 시간 기준 저장을 끄고 Ctrl+C 때만 저장)")


def _device_arg(value: str) -> str:
    """--device 값 정리 (GPU→cuda, TPU→tpu 같은 별칭과 대소문자). 고를 수 있는 값은 DEVICE_CHOICES가 검사한다."""
    from . import accel

    return accel.normalize(value)


def _amp_arg(value: str) -> str:
    from . import accel

    return accel.normalize_amp(value) or value


def _add_device_flag(s) -> None:
    s.add_argument("--device", type=_device_arg, choices=DEVICE_CHOICES,
                   help="학습 장치 (기본: 환경변수 LEARNBOX_DEVICE → 설정 training_device → auto). "
                        "auto는 cuda → tpu → xpu → mps → cpu 순서")


def _add_amp_flag(s) -> None:
    s.add_argument("--amp", type=_amp_arg, choices=AMP_CHOICES,
                   help="LoRA 정밀도: auto(bf16 가능하면 bf16, 아니면 fp16 혼합 정밀도) | fp16 | off(예전 방식) | fp32 "
                        "(기본: 환경변수 LEARNBOX_AMP → 설정 lora_training.amp → auto)")


def _add_start_flags(s) -> None:
    s.add_argument("--targets", help="학습 대상 (쉼표로): tiny,lora,embeddings,qa")
    s.add_argument("--tiny-preset", help="소형 AI 모델 크기 (auto | nano | mini | small | base | large | custom)")
    for key, flag in ARCH_FLAGS:
        s.add_argument(flag, type=int, dest=key, help=f"{key} (--tiny-preset custom일 때)")
    s.add_argument("--steps-per-cycle", type=int, help="한 번 학습할 때의 단계 수")
    s.add_argument("--min-new-chars", type=int, help="이만큼 새 글자가 모이면 학습")
    s.add_argument("--train-interval-min", type=int, help="새 자료가 조금이라도 있으면 이 간격(분)마다 학습")
    s.add_argument("--collect-interval-min", type=int, help="수집 간격(분)")
    s.add_argument("--pages-per-cycle", type=int, help="한 번에 수집할 페이지 수")
    s.add_argument("--max-pages-per-hour", type=int, help="1시간에 수집할 최대 페이지 수")
    s.add_argument("--outside-window", action="store_true", help="허용 시간이 아니어도 학습")
    s.add_argument("--results-target", help="학습 종료 시 결과를 복사할 외장 저장장치 경로")
    _add_device_flag(s)


def _start_overrides(args) -> dict:
    ov: dict = {}
    if args.targets is not None:
        names = {t.strip() for t in args.targets.split(",") if t.strip()}
        bad = names - {"tiny", "lora", "embeddings", "qa"}
        if bad:
            raise ValueError(f"알 수 없는 학습 대상: {', '.join(sorted(bad))} (tiny, lora, embeddings, qa)")
        ov["targets"] = {k: k in names for k in ("tiny", "lora", "embeddings", "qa")}
    if args.tiny_preset:
        ov["tiny_preset"] = args.tiny_preset
    arch = {k: getattr(args, k) for k, _ in ARCH_FLAGS if getattr(args, k, None)}
    if arch:
        ov["tiny_arch"] = arch
    for opt, key in (("steps_per_cycle", "steps_per_cycle"), ("min_new_chars", "train_min_new_chars"),
                     ("train_interval_min", "train_interval_min"), ("collect_interval_min", "collect_interval_min"),
                     ("pages_per_cycle", "pages_per_cycle"), ("max_pages_per_hour", "max_pages_per_hour"),
                     ("results_target", "results_export_target")):
        if getattr(args, opt, None) not in (None, ""):
            ov[key] = getattr(args, opt)
    if args.outside_window:
        ov["train_outside_window"] = True
    if getattr(args, "device", None):
        ov["device"] = args.device
    return ov


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="learnbox", description="LearnBox: 수집 · 검색 · 학습 · 외장 저장")
    p.add_argument("--data", default=os.environ.get("LEARNBOX_DATA", "data"), help="데이터 폴더 (기본: ./data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="사용자/관리자 PIN 설정")
    s.add_argument("--user-pin")
    s.add_argument("--admin-pin")

    s = sub.add_parser("serve", help="웹 서버 실행 (PC/휴대폰 공용)")
    s.add_argument("--port", type=int, default=8750)
    s.add_argument("--lan", action="store_true", help="같은 Wi-Fi의 휴대폰에서 접속 허용")
    s.add_argument("--host")
    s.add_argument("--cert", help="HTTPS 인증서 (휴대폰 오프라인/앱 설치에 필요)")
    s.add_argument("--key", help="HTTPS 개인키")

    s = sub.add_parser("collect", help="지정 링크 수집")
    s.add_argument("urls", nargs="+")
    s.add_argument("--depth", type=int, default=0, help="같은 사이트 안에서 따라갈 링크 깊이 (0~2)")

    s = sub.add_parser("websearch", help="실시간 검색 후 결과 페이지 수집")
    s.add_argument("query")
    s.add_argument("--limit", type=int)

    s = sub.add_parser("ingest", help="내 파일/폴더 가져오기 (txt, md, csv, json, jsonl, html, xml, pdf, docx …)")
    s.add_argument("path")

    s = sub.add_parser("search", help="학습한 자료에서 검색 (\"구문\", -제외어 사용 가능)")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--offset", type=int, default=0)
    s.add_argument("--sources", help="자료 종류 (쉼표로): link,search,file,auto")
    s.add_argument("--mode", choices=["hybrid", "keyword", "semantic"], default="hybrid")

    s = sub.add_parser("ask", help="학습한 자료를 근거로 질문하기")
    s.add_argument("question")
    s.add_argument("--engine", choices=["search", "tiny", "lora", "online"], default="search",
                   help="답변 방식: 검색만 | 소형 AI | LoRA | 온라인 모델")
    s.add_argument("--model-id", type=int, help="소형 AI/LoRA 모델 번호 (생략하면 가장 최근 모델)")
    s.add_argument("--sources", help="자료 종류 (쉼표로)")

    s = sub.add_parser("train-tiny", aliases=["train-mobile"], help="소형 AI 모델 학습 (처음부터, 글자 단위 GPT)")
    _add_arch_flags(s)
    s.add_argument("--steps", type=int)
    s.add_argument("--batch-size", type=int)
    s.add_argument("--lr", type=float)
    s.add_argument("--resume", type=int, metavar="MODEL_ID", help="이 소형 AI 모델에서 이어서 학습")
    _add_checkpoint_flag(s)
    _add_device_flag(s)

    s = sub.add_parser("train-lora", aliases=["train-pc"], help="PC용 LoRA 학습")
    s.add_argument("--base-model")
    s.add_argument("--lora-r", type=int)
    s.add_argument("--max-steps", type=int)
    s.add_argument("--seq-len", type=int)
    s.add_argument("--batch-size", type=int)
    s.add_argument("--no-qa", action="store_true", help="Q&A 학습 데이터(qa.jsonl)를 섞지 않기")
    s.add_argument("--resume", type=int, metavar="MODEL_ID", help="이 LoRA 모델에서 이어서 학습")
    _add_checkpoint_flag(s)
    _add_device_flag(s)
    _add_amp_flag(s)

    s = sub.add_parser("jobs", help="최근 작업 목록 (번호, 종류, 상태, 체크포인트 단계)")
    s.add_argument("--limit", type=int, default=20)

    s = sub.add_parser("resume-job", help="중단/실패한 학습 작업을 체크포인트에서 이어서 하기 (Ctrl+C로 저장하고 멈춤)")
    s.add_argument("job_id", type=int, help="이어서 할 작업 번호 (`jobs`로 확인)")
    s.add_argument("--device", type=_device_arg, choices=DEVICE_CHOICES,
                   help="이어서 할 장치 (기본: 원래 작업의 장치가 아니라 지금 장치 = LEARNBOX_DEVICE → 설정 → auto)")

    sub.add_parser("index-embeddings", help="의미 검색 색인 만들기")

    s = sub.add_parser("build-qa", help="Q&A 학습 데이터 만들기 (온라인 모델 사용)")
    s.add_argument("--limit-chunks", type=int)
    s.add_argument("--per-chunk", type=int)

    s = sub.add_parser("autolearn", help="자율 학습: start | stop | status | run")
    asub = s.add_subparsers(dest="action", required=True)
    a = asub.add_parser("start", help="자율 학습 시작 (서버가 켜져 있거나 `autolearn run`을 실행하면 진행됨)")
    _add_start_flags(a)
    asub.add_parser("stop", help="학습 종료: 마지막 학습 + 결과 파일 내보내기 (끝날 때까지 기다림)")
    asub.add_parser("status", help="상태 보기")
    a = asub.add_parser("run", help="이 창에서 자율 학습을 직접 돌리기 (Ctrl+C로 멈춤. 서버와 동시에 쓰지 마세요)")
    _add_start_flags(a)
    a.add_argument("--start", action="store_true", help="진행 중인 세션이 없으면 새로 시작")
    a.add_argument("--interval", type=float, default=20.0, help="판단 간격(초), 기본 20")
    a.add_argument("--until-end", action="store_true", help="세션이 끝나면(학습 종료가 끝나면) 종료")
    a.add_argument("--stop-after", type=float, metavar="SEC", help="이 시간(초)이 지나면 학습 종료를 요청 (테스트용)")

    sub.add_parser("results", help="학습 결과 파일 목록")
    sub.add_parser("models", help="학습한 모델 목록")

    s = sub.add_parser("estimate", help="모델 크기/메모리 예상")
    s.add_argument("--kind", choices=["tiny", "lora"], default="tiny")
    _add_arch_flags(s)
    s.add_argument("--batch-size", type=int)
    s.add_argument("--base-model")
    s.add_argument("--lora-r", type=int)
    _add_device_flag(s)
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("device", help="이 PC의 성능과 학습 추천")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("export", help="(관리자) 외장 저장장치로 내보내기")
    s.add_argument("target")
    s.add_argument("--include", nargs="*", default=[], choices=["database", "models", "datasets"])
    s.add_argument("--admin-pin")

    sub.add_parser("drives", help="연결된 외장 저장장치 목록")
    s = sub.add_parser("verify", help="내보낸 폴더 무결성 검사")
    s.add_argument("export_dir")
    s = sub.add_parser("import", help="내보낸 폴더를 이 데이터 폴더로 가져오기")
    s.add_argument("export_dir")
    sub.add_parser("status", help="현재 상태")
    return p


# ---------------------------------------------------------------------------------------------------------------
# 명령 구현
# ---------------------------------------------------------------------------------------------------------------
def cmd_search(app, args) -> int:
    from . import search

    sources = [x.strip() for x in (args.sources or "").split(",") if x.strip()] or None
    res = search.search(app.cfg, app.db, args.query, limit=args.limit, offset=args.offset, sources=sources,
                        mode=args.mode)
    names = {"hybrid": "통합", "keyword": "키워드", "semantic": "의미"}
    line = f"검색 방식: {names.get(res['mode_used'], res['mode_used'])}"
    if args.mode != "keyword" and res["mode_used"] == "keyword":
        line += " (의미 검색 색인이 없어 키워드로 찾음)"
    print(f"{line} · 문서 약 {res['total']}개 · {res['offset'] + 1}번째부터")
    if res.get("note"):
        print(f"참고: {res['note']}")
    if res["extract"]:
        print("\n[핵심 문장]")
        for e in res["extract"]:
            print(f"  - {e['sentence']}  ({e['title']})")
    print()
    for i, r in enumerate(res["results"], res["offset"] + 1):
        tag = {"keyword": "키워드", "semantic": "의미", "both": "키워드+의미"}.get(r.get("match"), "")
        print(f"{i}. {r['title']}  [{r.get('source') or '-'}{' · ' + tag if tag else ''}]\n   {r['url']}\n   {_hl(r['snippet'])}\n")
    if not res["results"]:
        print("결과가 없습니다.")
    return 0


def cmd_ask(app, args) -> int:
    sources = [x.strip() for x in (args.sources or "").split(",") if x.strip()] or None
    r = app.ask(args.question, args.engine, args.model_id, sources)
    print(f"[{args.engine}]" + (f" 모델: {r['model']}" if r.get("model") else ""))
    print(r["answer"] or "(답변을 만들지 못했습니다)")
    if r.get("note"):
        print(f"\n참고: {r['note']}")
    if r["citations"]:
        print("\n[인용]")
        for c in r["citations"]:
            print(f"  [{c['index']}] {c.get('title') or ''} {c.get('url') or ''}")
    if r["sources"]:
        print("\n[근거 자료]")
        for i, s in enumerate(r["sources"], 1):
            print(f"  [{i}] {s.get('title') or ''} {s.get('url') or ''}")
    return 0


def cmd_estimate(app, args) -> int:
    body = {"kind": args.kind, "batch_size": args.batch_size}
    if getattr(args, "device", None):
        body["device"] = args.device
    if args.kind == "lora":
        body.update(base_model=args.base_model, lora_r=args.lora_r, seq_len=getattr(args, "seq_len", None))
        r = app.estimate(body)
        if args.json:
            print(_j(r))
            return 0
        print(f"기반 모델: {r.get('base_model')} ({r.get('size') or '크기 알 수 없음'}) · 라이선스 {r.get('license') or '-'}")
        print(f"학습 가능 파라미터 약 {r.get('trainable_params_est') or '?'}개 · 어댑터 {r.get('adapter_mb_est')}MB")
        print(f"필요 메모리: RAM 약 {r['ram_gb_est']}GB (bf16 {r.get('ram_gb_est_bf16')}GB) · VRAM 약 {r['vram_gb_est']}GB")
        if r.get("recommendation"):
            print(r["recommendation"])
        if r.get("note"):
            print(f"참고: {r['note']}")
        return 0
    body["preset"] = args.preset or "auto"
    body.update({k: getattr(args, k) for k, _ in ARCH_FLAGS})
    r = app.estimate(body)
    if args.json:
        print(_j(r))
        return 0
    a, e = r["arch"], r["estimate"]
    print(f"크기: {a['preset']} · 차원 {a['d_model']} · 층 {a['n_layer']} · 헤드 {a['n_head']} · "
          f"문맥 {a['seq_len']} · 글자 {a['vocab_size']}")
    print(e["label"])
    print(f"파라미터 {e['params']:,}개 · fp32 {e['fp32_mb']}MB · int8 {e['int8_mb']}MB · "
          f"학습 메모리 {e['train_ram_mb']}MB (배치 {e.get('batch_size')}) · 실행 메모리 {e['infer_ram_mb']}MB")
    print("실행/학습 가능: " + ", ".join(f"{label} {'예' if e['fits'].get(k) else '아니오'}"
                                   for k, label in FIT_LABELS if k in e["fits"]))
    fit = r.get("device_fit")
    if fit:          # --device tpu일 때만 (PRD 6.8)
        mem = f" (메모리 {fit['mem_gb']:g}GB)" if fit.get("mem_gb") else ""
        verdict = {True: "예", False: "아니오"}.get(fit.get("fits"), "측정 필요")
        print(f"TPU 학습 가능: {verdict}{mem}")
    return 0


def _accel_line(acc: dict, pc: dict) -> str:
    """명령줄 device의 '가속:' 줄 (PRD 6.11): torch · torch_xla · ROCm · xpu · TPU 런타임 신호."""
    devs = acc.get("devices") or {}
    tpu = acc.get("tpu") or {}
    tv = acc.get("torch_version") or pc.get("torch_version")
    xv = acc.get("torch_xla_version")
    ttv = acc.get("torch_tpu_version")
    rocm = acc.get("rocm")
    parts = [f"torch {tv or '없음'}"] + ([f"torch_tpu {ttv}"] if ttv else []) + [f"torch_xla {xv or '없음'}",
             f"ROCm {rocm if isinstance(rocm, str) else ('있음' if rocm else '없음')}",
             "xpu " + ("있음" + (f" ({devs['xpu']['name']})" if (devs.get("xpu") or {}).get("name") else "")
                       if (devs.get("xpu") or {}).get("available") else "없음")]
    if (devs.get("tpu") or {}).get("available"):
        t = "TPU " + (tpu.get("accelerator_type") or "사용 가능")
    elif tpu.get("likely") and tpu.get("mismatch"):
        t = (f"TPU {tpu.get('accelerator_type') or ''} 감지 "
             f"(torch·torch_xla 버전 불일치: python -m learnbox.setup_accel)").replace("  ", " ")
    elif tpu.get("likely"):
        t = f"TPU {tpu.get('accelerator_type') or ''} 감지 (TPU 패키지 필요: python -m learnbox.setup_accel)".replace("  ", " ")
    else:
        t = "TPU 없음"
    parts.append(t)
    if acc.get("auto"):
        parts.append(f"자동 선택 {acc['auto']}")
    return " · ".join(parts)


def cmd_device(app, args) -> int:
    info = app.device_info()
    if args.json:
        print(_j({"pc": info["pc"], "recommend": info["recommend"]}))
        return 0
    pc, rec = info["pc"], info["recommend"]
    gpu = pc.get("gpu") or {}
    print(f"운영체제: {pc.get('os')} {pc.get('os_version') or ''} · CPU {pc.get('cpu_count')}코어 · "
          f"메모리 {pc.get('ram_gb') if pc.get('ram_gb') is not None else '알 수 없음'}GB")
    amd = " · AMD ROCm" if gpu.get("vendor") == "amd" else ""
    print("GPU: " + (f"{gpu.get('name') or gpu.get('type')} (VRAM {gpu.get('vram_gb')}GB){amd}" if gpu.get("type")
                     else "없음"))
    pk = ("torch", "transformers", "peft", "onnxruntime", "anthropic", "pypdf")
    print("패키지: " + ", ".join(f"{k} {'있음' if pc.get(k) else '없음'}" for k in pk))
    print("가속: " + _accel_line(info.get("accel") or {}, pc))
    print(f"등급: {pc.get('tier_label') or pc.get('tier')}")
    print(f"\n추천: 소형 AI 모델 '{rec['tiny_preset']}' · 주기당 {rec['tiny_steps_per_cycle']}단계 · 배치 {rec['tiny_batch_size']} · "
          f"스레드 {rec['threads']} · LoRA {rec['lora_base_model'] or '권장 안 함'}")
    for n in rec.get("notes") or []:
        print(f"  - {n}")
    return 0


def cmd_results(app, args) -> int:
    from . import results

    rows = results.list_results(app.cfg)
    if not rows:
        print("학습 결과 파일이 아직 없습니다. 자율 학습을 끝내면(autolearn stop) 만들어집니다.")
        return 0
    for r in rows:
        print(f"{r['name']}  파일 {r['file_count']}개 · {_fmt_bytes(r['total_bytes'])}"
              + (f" · ZIP {_fmt_bytes(r['zip_bytes'])}" if r.get("zip_bytes") else "") + f"\n  {r['dir']}")
        if r.get("zip"):
            print(f"  {r['zip']}")
    return 0


def cmd_models(app, args) -> int:
    rows = app.db.list_models()
    if not rows:
        print("학습한 모델이 없습니다.")
    for m in rows:
        i = m["info"]
        label = {"tiny": "소형 AI", "lora": "LoRA"}.get(m["kind"], m["kind"])
        loss = f" · 평가 loss {i.get('eval_loss_before')} → {i.get('eval_loss_after')}" if i.get("eval_loss_after") is not None else ""
        print(f"#{m['id']} [{label}] {m['name']} · {i.get('total_steps') or i.get('steps') or '?'}단계{loss}")
    return 0


KIND_LABELS = {"collect_links": "링크 수집", "collect_search": "검색 수집", "ingest_path": "파일 가져오기",
               "index_embeddings": "의미 검색 색인", "build_qa": "Q&A 데이터 생성", "train_tiny": "소형 AI 학습",
               "train_lora": "LoRA 학습", "autolearn_train": "자율 학습(학습)", "finalize_learning": "학습 종료·결과 내보내기",
               "export": "외장 저장 내보내기", "train_mobile": "소형 AI 학습", "train_pc": "LoRA 학습"}
STATUS_LABELS = {"queued": "대기", "running": "진행 중", "paused": "일시정지", "done": "완료", "failed": "실패",
                 "cancelled": "취소", "blocked": "시간 밖 중지", "interrupted": "중단"}


def _ago(iso) -> str:
    """ISO 시각 -> '3분 전' (읽지 못하면 원문)."""
    from datetime import datetime, timezone

    try:
        t = datetime.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return str(iso or "-")
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    sec = max(0, int((datetime.now(timezone.utc) - t).total_seconds()))
    if sec < 60:
        return "방금"
    if sec < 3600:
        return f"{sec // 60}분 전"
    if sec < 86400:
        return f"{sec // 3600}시간 전"
    return t.astimezone().strftime("%Y-%m-%d %H:%M")


def _resume_cmd(args, job_id) -> str:
    """이어서 하기 명령 (기본 데이터 폴더가 아니면 --data를 붙인다)."""
    data = getattr(args, "data", None)
    extra = ""
    if data and data != os.environ.get("LEARNBOX_DATA", "data"):
        extra = " --data " + (f'"{data}"' if any(c.isspace() for c in str(data)) else str(data))
    return f"python -m learnbox{extra} resume-job {job_id}"


def cmd_jobs(app, args) -> int:
    rows = app.list_jobs(max(1, min(int(args.limit or 20), 200)))
    if not rows:
        print("작업이 없습니다.")
        return 0
    for j in rows:
        st = j.get("status")
        line = f"#{j['id']} {KIND_LABELS.get(j['kind'], j['kind'])} · {STATUS_LABELS.get(st, st)}"
        if st in ("running", "paused"):
            line += f" · {round(float(j.get('progress') or 0) * 100)}%"
        ck = j.get("checkpoint")
        if ck:
            line += f" · 체크포인트 {ck.get('step')}/{ck.get('steps')}단계 ({_ago(ck.get('saved_at'))} 저장)"
        if j.get("message"):
            line += f" · {j['message']}"
        print(line)
        if j.get("can_restart"):
            print(f"    이어서 하려면: {_resume_cmd(args, j['id'])}")
    return 0


def _run_job_cli(app, args, job_id: int) -> int:
    """작업을 작업 스레드에서 돌리고 진행을 출력한다. Ctrl+C면 지금 단계를 마치고 저장한 뒤 멈춘다(이어서 재개 가능)."""
    import threading

    jobs = app.jobs
    jobs.echo = True
    t = threading.Thread(target=jobs.run_existing, args=(job_id,), name="cli-job", daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(0.2)
    except KeyboardInterrupt:
        return _cli_stop(app, args, job_id, t)
    return _print_job(app.db.get_job(job_id))


def _cli_stop(app, args, job_id: int, t) -> int:
    """Ctrl+C: 일시정지 요청처럼 지금 단계를 마치고 체크포인트를 저장한 뒤 '중단됨'으로 끝낸다."""
    from .db import now_iso

    print("\n멈추는 중입니다: 지금 단계를 마치고 체크포인트를 저장합니다… "
          "(한 번 더 Ctrl+C를 누르면 저장을 기다리지 않고 바로 끝냅니다)", flush=True)
    forced = False
    try:
        while t.is_alive() and not app.jobs.interrupt(job_id, "명령줄에서 중지됨"):
            t.join(0.1)        # 작업이 막 시작하는 중이면 잠깐 기다렸다가 다시 요청
        while t.is_alive():
            t.join(0.2)
    except KeyboardInterrupt:
        forced = True
    job = app.db.get_job(job_id)
    if job and job["status"] in ("queued", "running", "paused"):
        ck = app.job_checkpoint(job)
        msg = "명령줄에서 강제로 중지됨" + (f" · 이어서 재개할 수 있습니다 ({ck['step']}/{ck['steps']}단계)" if ck else "")
        app.db.update_job(job_id, status="interrupted", message=msg, finished_at=now_iso())
        job = app.db.get_job(job_id)
    if job["status"] in ("done", "blocked"):       # 저장하기 전에 학습이 끝났다
        return _print_job(job)
    print(f"[{job['status']}] {job['message']}")
    ck = app.job_checkpoint(job) if job["status"] in ("interrupted", "failed") else None
    if ck:
        print(f"체크포인트: {ck.get('step')}/{ck.get('steps')}단계 저장됨 ({ck.get('path')})")
        print(f"{_resume_cmd(args, job_id)}로 이어서 할 수 있습니다.")
    else:
        off = not (app.cfg["training_checkpoint"] or {}).get("enabled", True)
        print("저장된 체크포인트가 없어 이어서 할 수 없습니다." + (" (설정에서 학습 중간 저장이 꺼져 있습니다)" if off else ""))
    return 130


def cmd_resume_job(app, args) -> int:
    new_id = app.restart_job(args.job_id, "cli", enqueue=False, device=getattr(args, "device", None))
    ck = (app.db.get_job(new_id)["params"] or {}).get("resume_checkpoint")
    print(f"작업 #{args.job_id}의 체크포인트에서 이어서 학습합니다 (새 작업 #{new_id}, 체크포인트 {ck}).")
    print("멈추려면 Ctrl+C (지금 단계를 마치고 저장한 뒤 멈춥니다).", flush=True)
    return _run_job_cli(app, args, new_id)


def _train_cli(app, args, kind: str, params: dict) -> int:
    if getattr(args, "checkpoint_minutes", None) is not None:
        if args.checkpoint_minutes < 0:
            raise ValueError("--checkpoint-minutes는 0 이상이어야 합니다.")
        params["checkpoint_minutes"] = args.checkpoint_minutes
    job_id = app.db.create_job(kind, params, "cli")
    print(f"학습 작업 #{job_id}을(를) 시작합니다. 멈추려면 Ctrl+C (지금 단계를 마치고 저장한 뒤 멈춥니다).", flush=True)
    return _run_job_cli(app, args, job_id)


def _run_loop(app, args) -> int:
    """자율 학습을 이 프로세스에서 직접 돌린다 (작업 스레드를 켜고 tick을 반복)."""
    from .autolearn import TICK_SEC

    al = app.autolearn
    if not app.db.active_session():
        if args.start:
            s = al.start_session("cli", _start_overrides(args))
            print(f"자율 학습 세션 #{s['id']}을 시작했습니다.")
        elif args.until_end:
            print("진행 중인 자율 학습이 없습니다. --start로 새로 시작하세요.")
            return 1
        else:
            print("진행 중인 자율 학습이 없습니다. 다른 창에서 `autolearn start`를 하거나 --start를 붙이세요. 기다립니다.")
    app.jobs.start(mark_interrupted=False)
    app.jobs.echo = True
    interval = max(0.05, float(args.interval or TICK_SEC))
    t0, stopped, last_sid = time.monotonic(), False, None
    print("자율 학습을 돌리는 중입니다. 멈추려면 Ctrl+C (세션은 그대로 남습니다. 끝내려면 `autolearn stop`).")
    try:
        while True:
            res = al.tick()
            if res.get("session_id"):
                last_sid = res["session_id"]
            if res["actions"] and res["actions"] != ["finishing"]:
                print(f"  tick: {', '.join(res['actions'])}", flush=True)
            if args.stop_after is not None and not stopped and time.monotonic() - t0 >= args.stop_after \
                    and app.db.active_session():
                stopped = True
                try:
                    r = al.stop_session("cli")
                    print(f"학습 종료를 요청했습니다 (작업 #{r['job_id']}).")
                except ValueError as e:
                    print(e)
            active = app.db.active_session()
            if args.until_end and not active and (last_sid or stopped):
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n멈춥니다. 진행 중인 세션은 그대로 남아 있습니다.")
        return 0
    s = app.db.get_session(last_sid) if last_sid else None
    if s:
        print(f"세션 #{s['id']} {s['status']}" + (f" · 결과 {s['result'].get('name')}" if s.get("result") and s["result"].get("name") else ""))
        return 0 if s["status"] == "ended" else 1
    return 0


def cmd_autolearn(app, args) -> int:
    al = app.autolearn
    if args.action == "start":
        s = al.start_session("cli", _start_overrides(args))
        c = s["config"]
        print(f"자율 학습 세션 #{s['id']}을 시작했습니다. (소형 AI 크기: {c.get('tiny_preset')})")
        print("서버(`serve`)가 켜져 있거나 `autolearn run`을 실행하면 허용 시간에 수집하고 학습합니다.")
        return 0
    if args.action == "stop":
        r = al.stop_session("cli")
        print(f"학습 종료를 시작합니다 (작업 #{r['job_id']}). 마지막 학습과 결과 파일 내보내기가 끝날 때까지 기다립니다.")
        job = app.jobs.run_existing(r["job_id"])
        return _print_job(job)
    if args.action == "status":
        st = al.status()
        s = st["session"]
        out = {k: st[k] for k in ("active", "phase", "window", "next_collect_at", "next_train_check_at",
                                   "new_chars_since_train")}
        if s:
            out["session"] = {"id": s["id"], "status": s["status"], "started_at": s["started_at"],
                              "tiny_preset": s["config"].get("tiny_preset"), "targets": s["config"].get("targets"),
                              "stats": {k: v for k, v in s["stats"].items()
                                        if k not in ("collect_jobs", "train_jobs", "hour_pages", "train_history", "warned")}}
        out["recent_sessions"] = [{"id": x["id"], "status": x["status"], "ended_at": x["ended_at"],
                                   "result": (x.get("result") or {}).get("name")} for x in st["recent_sessions"]]
        print(_j(out))
        return 0
    if args.action == "run":
        return _run_loop(app, args)
    return 1


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "init":
        from .config import Config

        cfg = Config(args.data)
        up = args.user_pin or getpass.getpass("사용자 PIN (4자 이상): ")
        ap = args.admin_pin or getpass.getpass("관리자 PIN (6자 이상): ")
        try:
            cfg.set_pins(up, ap)
        except ValueError as e:
            print(e)
            return 1
        print(f"PIN을 설정했습니다. 데이터 폴더: {cfg.data_dir}")
        return 0

    if args.cmd == "verify":
        from .export import verify

        r = verify(args.export_dir)
        print(_j(r))
        return 0 if r["ok"] else 1

    if args.cmd == "drives":
        from .export import list_drives

        print(_j(list_drives()))
        return 0

    app = _app(args)
    try:
        return _dispatch(app, args)
    except ValueError as e:
        print(e)
        return 1
    except ImportError as e:
        from .jobs import missing_package_message

        print(missing_package_message(e))
        return 1
    except Exception as e:
        if type(e).__name__ == "OnlineError":
            print(e)
            return 1
        raise


def _dispatch(app, args) -> int:
    cmd = args.cmd
    if cmd == "serve":
        from .server import serve

        host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
        serve(app, host, args.port, args.cert, args.key)
        return 0
    if cmd == "status":
        st = app.status()
        st.pop("jobs")
        print(_j(st))
        return 0
    if cmd == "search":
        return cmd_search(app, args)
    if cmd == "ask":
        return cmd_ask(app, args)
    if cmd == "estimate":
        return cmd_estimate(app, args)
    if cmd == "device":
        return cmd_device(app, args)
    if cmd == "results":
        return cmd_results(app, args)
    if cmd == "models":
        return cmd_models(app, args)
    if cmd == "autolearn":
        return cmd_autolearn(app, args)
    if cmd == "jobs":
        return cmd_jobs(app, args)
    if cmd == "resume-job":
        return cmd_resume_job(app, args)
    if cmd == "import":
        from .export import import_export

        print(json.dumps(import_export(args.export_dir, app.cfg, app.db), ensure_ascii=False))
        return 0

    jobs = app.jobs
    if cmd == "collect":
        return _print_job(jobs.run_inline("collect_links", {"urls": args.urls, "depth": args.depth}))
    if cmd == "websearch":
        return _print_job(jobs.run_inline("collect_search", {"query": args.query, "limit": args.limit}))
    if cmd == "ingest":
        return _print_job(jobs.run_inline("ingest_path", {"path": args.path, "actor": "cli"}))
    if cmd in ("train-tiny", "train-mobile"):
        params = {k: v for k, v in {"preset": args.preset, "steps": args.steps, "batch_size": args.batch_size,
                                    "lr": args.lr, "resume_from_model_id": args.resume, "device": args.device,
                                    **{k: getattr(args, k) for k, _ in ARCH_FLAGS}}.items() if v}
        return _train_cli(app, args, "train_tiny", params)
    if cmd in ("train-lora", "train-pc"):
        params = {k: v for k, v in {"base_model": args.base_model, "lora_r": args.lora_r, "max_steps": args.max_steps,
                                    "seq_len": args.seq_len, "batch_size": args.batch_size,
                                    "resume_from_model_id": args.resume, "device": args.device,
                                    "amp": args.amp}.items() if v}
        if args.no_qa:
            params["use_qa"] = False
        return _train_cli(app, args, "train_lora", params)
    if cmd == "index-embeddings":
        return _print_job(jobs.run_inline("index_embeddings", {}))
    if cmd == "build-qa":
        params = {k: v for k, v in {"limit_chunks": args.limit_chunks, "per_chunk": args.per_chunk}.items() if v}
        return _print_job(jobs.run_inline("build_qa", params))
    if cmd == "export":
        pin = args.admin_pin or getpass.getpass("관리자 PIN: ")
        if app.cfg.check_pin(pin) != "admin":
            print("관리자 PIN이 올바르지 않습니다.")
            return 1
        return _print_job(jobs.run_inline("export", {"target": args.target, "include": args.include, "actor": "admin@cli"}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
