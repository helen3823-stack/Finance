#!/usr/bin/env python
"""예측과 채점.

    python main.py predict --date  2026-02-20      하루 예측
    python main.py eval    --split 2026-06-01      구간 채점
    python main.py check   --split 2026-06-01      출력 형식 검사

eval은 predict를 날짜마다 반복해서 정답과 맞춰봄.
check는 점수 없이 형식만 확인함.

파이썬에서 직접 쓸 때는 이렇게 함.

    from src import load_model, predict

    model = load_model()
    predict(model, "2026-02-20")
"""

import argparse
import json
from pathlib import Path

from src import check, evaluate, load_model, predict


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


def main():
    ap = argparse.ArgumentParser(description="예측과 채점")
    ap.add_argument("command", choices=["predict", "eval", "check"])
    ap.add_argument("--date", help="기준일. 그 다음 거래일을 예측함")
    ap.add_argument("--split", default="2026-06-01",
                    help="평가 시작일. 이 날을 포함해 이후 전부가 평가 구간")
    ap.add_argument("--data", help="데이터 폴더. 생략하면 dataset/")
    ap.add_argument("--symbols", help="대상 종목. 쉼표로 나열하거나 json/txt 파일 경로")
    ap.add_argument("--out", help="결과를 저장할 csv 경로")
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
        evaluate(model, args.split, args.data, symbols, args.out)
    else:
        return check(model, args.split, args.data, symbols)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
