"""예측 실행과 채점."""

import time

import pandas as pd

from .data import LABELS, Dataset, Day, score
from .schema import check_output


def predict(model, date, data=None, symbols=None):
    """하루를 예측함. 예측 시점까지의 데이터만 씀.

    Args:
        model: predict(day) 를 가진 객체.
        date: 기준일. Day 객체를 그대로 넘겨도 됨.
        data: 데이터 폴더. 생략하면 dataset/ 을 씀.
        symbols: 예측할 종목. 생략하면 데이터에 있는 전부.

    Returns:
        symbol, label column. 형식 검사를 통과한 것만 돌려줌.
    """
    day = date if isinstance(date, Day) else Dataset(data, symbols=symbols).day(date)
    return check_output(model.predict(day), day.symbols)


def evaluate(model, days, data=None, symbols=None, out=None, verbose=True):
    """하루씩 예측하고 점수를 냄.

    predict를 날짜마다 돌린 뒤 정답과 맞춰봄.

    Args:
        model: predict(day) 를 가진 객체.
        days: 평가 시작일, 또는 Day 목록. 시작일을 주면 그 날부터 끝까지가
            평가 구간이 됨. 노트북처럼 구간을 직접 고른 경우에는 목록을 넘김.
        data / symbols: `days` 가 시작일일 때만 씀.
        out: 예측을 저장할 csv 경로.

    Returns:
        symbol, date, ret_pct, label, label_pred 를 담은 표.
    """
    if isinstance(days, (list, tuple)):
        test = list(days)
    else:
        ds = Dataset(data, symbols=symbols)
        _, test = ds.split(days)
        if verbose:
            print(ds)
    if not test:
        raise ValueError("평가할 날이 없음")
    if verbose:
        print(f"평가 {len(test)}일 "
              f"({test[0].date:%Y-%m-%d} ~ {test[-1].date:%Y-%m-%d})")

    rows, t0 = [], time.time()
    for i, day in enumerate(test, 1):
        try:
            pred = check_output(model.predict(day), day.symbols)
        except (ValueError, TypeError) as e:
            raise SystemExit(f"{day.date:%Y-%m-%d} 예측 규격 위반\n{e}")
        rows.append(day.y[["symbol", "date", "ret_pct", "label"]].merge(
            pred, on="symbol", suffixes=("", "_pred")))
        if verbose and (i % 20 == 0 or i == len(test)):
            print(f"  {i}/{len(test)}일", flush=True)

    r = pd.concat(rows, ignore_index=True)
    if verbose:
        print(f"예측 {time.time() - t0:.1f}초\n")
        report(r)
    if out:
        r.to_csv(out, index=False)
        print(f"저장 {out}")
    return r


def report(r):
    """점수와 혼동행렬을 출력함."""
    s = score(r.label, r.label_pred)

    def num(v):
        """예측한 적이 없어 계산이 안 되는 값은 빗금으로 표시함."""
        return f"{v:.3f}" if v == v else "  -  "

    print(f"{'점수':10} {s['score']:+.4f}   1 = 완벽, 0 = 우연 수준")
    print(f"{'정확도':9} {num(s['accuracy'])}")
    print(f"{'방향 적중':8} {num(s['direction'])}   보합 제외")
    print(f"{'급등락 포착':7} {num(s['big_recall'])}   실제 급등락 중 맞힌 비율")
    print(f"{'급등락 적중':7} {num(s['big_prec'])}   급등락이라 찍은 것 중 맞은 비율")
    print(f"{'건수':10} {s['n']:,}\n")

    name = dict(enumerate(LABELS))
    print(pd.crosstab(r.label.map(name), r.label_pred.map(name),
                      rownames=["실제"], colnames=["예측"], dropna=False).to_string())
    return s


def check(model, days, data=None, symbols=None, n=5):
    """출력 형식만 확인함. 정답은 보지 않음.

    평가 구간에서 며칠을 골라 돌려보고 형식을 지키는지 봄.
    제출 전에 통과시켜 두면 채점 도중에 멈추는 일이 없음.
    """
    import random

    if isinstance(days, (list, tuple)):
        test = list(days)
    else:
        _, test = Dataset(data, symbols=symbols).split(days)
    picked = test if len(test) <= n else random.Random(0).sample(test, n)
    picked = sorted(picked, key=lambda d: d.date)
    print(f"{len(picked)}일 검사\n")

    for day in picked:
        try:
            out = check_output(model.predict(day), day.symbols)
        except (ValueError, TypeError) as e:
            print(f"  {day.date:%Y-%m-%d}  실패\n{e}")
            return 1
        print(f"  {day.date:%Y-%m-%d}  {len(out):>3}종목  "
              f"분포 {out.label.value_counts().sort_index().to_dict()}")

    print("\n형식 확인 완료")
    return 0
