"""명령줄: python -m learnbox <명령>"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import sys


def _app(args):
    from .app import App

    return App(args.data)


def _print_job(job):
    print(f"[{job['status']}] {job['message']}")
    if job.get("result"):
        print(json.dumps(job["result"], ensure_ascii=False, indent=2)[:4000])
    return 0 if job["status"] in ("done", "blocked") else 1


def main(argv=None) -> int:
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

    s = sub.add_parser("search", help="수집한 자료에서 검색")
    s.add_argument("query")

    s = sub.add_parser("train-pc", help="PC용 LoRA 학습")
    s.add_argument("--base-model")
    s.add_argument("--max-steps", type=int)

    s = sub.add_parser("train-mobile", help="휴대폰용 초소형 모델 학습")
    s.add_argument("--preset", choices=["tiny", "small"])
    s.add_argument("--steps", type=int)

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
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if r["ok"] else 1

    if args.cmd == "drives":
        from .export import list_drives

        print(json.dumps(list_drives(), ensure_ascii=False, indent=2))
        return 0

    app = _app(args)

    if args.cmd == "serve":
        from .server import serve

        host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
        serve(app, host, args.port, args.cert, args.key)
        return 0
    if args.cmd == "status":
        st = app.status()
        st.pop("jobs")
        print(json.dumps(st, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "search":
        from .db import HL_END, HL_START

        for r in app.db.search(args.query, 10):
            print(f"- {r['title']}\n  {r['url']}\n  {r['snippet'].replace(HL_START, '[').replace(HL_END, ']')}\n")
        return 0
    if args.cmd == "import":
        from .export import import_export

        print(json.dumps(import_export(args.export_dir, app.cfg, app.db), ensure_ascii=False))
        return 0

    jobs = app.jobs
    if args.cmd == "collect":
        return _print_job(jobs.run_inline("collect_links", {"urls": args.urls, "depth": args.depth}))
    if args.cmd == "websearch":
        return _print_job(jobs.run_inline("collect_search", {"query": args.query, "limit": args.limit}))
    if args.cmd == "train-pc":
        params = {k: v for k, v in {"base_model": args.base_model, "max_steps": args.max_steps}.items() if v}
        return _print_job(jobs.run_inline("train_pc", params))
    if args.cmd == "train-mobile":
        params = {k: v for k, v in {"preset": args.preset, "steps": args.steps}.items() if v}
        return _print_job(jobs.run_inline("train_mobile", params))
    if args.cmd == "export":
        pin = args.admin_pin or getpass.getpass("관리자 PIN: ")
        if app.cfg.check_pin(pin) != "admin":
            print("관리자 PIN이 올바르지 않습니다.")
            return 1
        return _print_job(jobs.run_inline("export", {"target": args.target, "include": args.include, "actor": "admin@cli"}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
