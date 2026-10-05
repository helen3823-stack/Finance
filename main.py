#!/usr/bin/env python
"""예측과 채점.

    python main.py predict --date  2026-02-20      하루 예측
    python main.py eval    --split 2026-06-01      구간 채점
    python main.py check   --split 2026-06-01      출력 형식 검사

eval은 predict를 날짜마다 반복해서 정답과 맞춰봄.
check는 점수 없이 형식만 확인함.

[팀 로컬 수정] eval 은 lockbox(대상일 ≥ 2026-06-01, exp/lockbox.py)를 채점하지 않음.
    --until 기본값이 lockbox 시작일이라 평가 구간이 [split, until) 로 잘림.
    lockbox 를 채점하려면 --allow-lockbox 가 필요하고, 그 사실이 exp/lockbox_ledger.json 에 기록됨.
    채점자는 이 파일을 원본으로 덮어쓰므로 제출에는 영향이 없음.

파이썬에서 직접 쓸 때는 이렇게 함.

    from src import load_model, predict

    model = load_model()
    predict(model, "2026-02-20")
"""

import argparse
import json
from pathlib import Path

from src import Dataset, check, evaluate, load_model, predict


def parse_symbols(arg):
    """`--symbols` 를 종목 목록으로 바꿈. 쉼표 나열이나 파일 경로를 받음."""
    if not arg:
        return None
    path = Path(arg)
    if not path.exists():
        return sorted(x.strip() for x in arg.split(",") if x.strip())
    if path.suffix == ".json":
        v = json.loads(path.read_text())
        return sorted(v if isinstance(v, list) else sum(v.values(), []))
    return sorted(x.strip() for x in path.read_text().split() if x.strip())


def eval_days(args, symbols):
    """eval 평가일 목록. 기본은 lockbox 이전까지만. lockbox 를 포함하면 노출로 기록."""
    try:
        from exp import lockbox as LB
    except ImportError:            # exp/ 가 없는 환경(원본 저장소)에서는 원래 동작
        return args.split
    import pandas as pd
    until = pd.Timestamp(args.until) if args.until else LB.LOCKBOX_START
    if pd.Timestamp(args.split) >= LB.LOCKBOX_START and not args.allow_lockbox:
        raise SystemExit(f"split={args.split} 는 lockbox({LB.LOCKBOX_START.date()}~) 안임. lockbox 는 python -m exp 의 "
                         "freeze → run 으로 한 번만 평가함. 형식 확인만 필요하면 python main.py check")
    if until > LB.LOCKBOX_START:
        if not args.allow_lockbox:
            raise SystemExit(f"평가 구간이 lockbox({LB.LOCKBOX_START.date()}~)를 포함함. "
                             "DEV 평가는 --until 을 lockbox 이전으로, lockbox 를 꼭 봐야 하면 --allow-lockbox (기록됨)")
        LB.record_exposure("python main.py eval --allow-lockbox",
                           f"split={args.split}, until={until.date()} 구간 채점", ["artifact"])
    _, test = Dataset(args.data, symbols=symbols).split(args.split)
    days = [d for d in test if d.target < until]
    if not days:
        raise SystemExit(f"평가할 날이 없음: split={args.split} 가 until={until.date()} 이후임")
    return days


def main():
    ap = argparse.ArgumentParser(description="예측과 채점")
    ap.add_argument("command", choices=["predict", "eval", "check"])
    ap.add_argument("--date", help="기준일. 그 다음 거래일을 예측함")
    ap.add_argument("--split", default="2026-06-01",
                    help="평가 시작일. 이 날을 포함해 이후 전부가 평가 구간")
    ap.add_argument("--data", help="데이터 폴더. 생략하면 dataset/")
    ap.add_argument("--symbols", help="대상 종목. 쉼표로 나열하거나 json/txt 파일 경로")
    ap.add_argument("--out", help="결과를 저장할 csv 경로")
    ap.add_argument("--until", help="eval 평가 끝(미포함). 기본은 lockbox 시작일")
    ap.add_argument("--allow-lockbox", action="store_true", help="eval 에 lockbox 구간 포함 (장부에 기록됨)")
    args = ap.parse_args()

    model = load_model()
    symbols = parse_symbols(args.symbols)

    if args.command == "predict":
        if not args.date:
            raise SystemExit("--date 를 지정해야 함")
        out = predict(model, args.date, args.data, symbols)
        print(out.to_string(index=False))
        if args.out:
            out.to_csv(args.out, index=False)
    elif args.command == "eval":
        evaluate(model, eval_days(args, symbols), args.data, symbols, args.out)
    else:
        return check(model, args.split, args.data, symbols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
