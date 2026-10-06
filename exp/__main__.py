"""명령행 진입점.

기능  : python -m exp <명령> 으로 실험 단계를 실행
구성  : run · check · analyze · llm · freeze · lockbox · export
역할  : 단계별 실행 방법을 하나로 통일. 무엇을 돌릴지는 config 파일이 정함
"""

import argparse
import sys

from . import config as C


def main():
    for stream in (sys.stdout, sys.stderr):      # Windows 콘솔(cp949)에서 Δ·✗ 같은 문자가 깨지지 않게
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="python -m exp", description="Finance 실험 하네스")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="E1~E6: config 의 후보들을 학습·평가")
    r.add_argument("config")
    r.add_argument("--rerun", action="store_true", help="같은 config·코드 결과가 있어도 다시 계산")
    r.add_argument("--force-lockbox", action="store_true", help="lockbox 재사용 허용 (기록이 남음)")

    c = sub.add_parser("check", help="E0 재현성 / G0 실행 가능성")
    c.add_argument("which", choices=["e0", "g0"])
    c.add_argument("--days", type=int, default=8)

    a = sub.add_parser("analyze", help="E4 갭 조건부 표")
    a.add_argument("config")

    lm = sub.add_parser("llm", help="E4-L LLM 스크리닝 (기본 dry_run)")
    lm.add_argument("config")

    fz = sub.add_parser("freeze", help="E6 사전 등록: lockbox 에서 평가할 후보 spec 을 고정")
    fz.add_argument("config")

    sub.add_parser("lockbox", help="lockbox 노출 장부·계획·실행 이력 출력")

    e = sub.add_parser("export", help="E7: 후보 하나를 학습해 artifacts/pipeline.pkl 로 저장")
    e.add_argument("config")
    e.add_argument("--candidate", required=True)
    e.add_argument("--allow-lockbox", action="store_true",
                   help="lockbox 평가 전에 lockbox 기간을 학습에 포함 (노출로 기록)")

    args = ap.parse_args()
    if args.cmd == "run":
        from .runner import run
        run(C.load(args.config), force_lockbox=args.force_lockbox, rerun=args.rerun)
    elif args.cmd == "check":
        from . import checks
        ok = checks.e0(args.days)[0] if args.which == "e0" else checks.g0(args.days) is not None
        return 0 if ok else 1
    elif args.cmd == "analyze":
        from .analysis import run_analysis
        run_analysis(C.load(args.config))
    elif args.cmd == "llm":
        from .llm import run_llm
        run_llm(C.load(args.config))
    elif args.cmd == "freeze":
        from .lockbox import freeze
        cfg = C.load(args.config)
        if not cfg["split"]["use_lockbox"]:
            raise SystemExit("freeze 는 split.use_lockbox = true 인 config 에만 씀")
        freeze(cfg, C.RUNS)
    elif args.cmd == "lockbox":
        import json
        from .lockbox import load_ledger
        print(json.dumps(load_ledger(), indent=2, ensure_ascii=False))
    elif args.cmd == "export":
        from .runner import export
        export(C.load(args.config), args.candidate, allow_lockbox=args.allow_lockbox)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
