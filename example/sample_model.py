"""완성된 모델 예시.

랜덤 포레스트로 다음 거래일 등락을 예측함.
노트북에서 이걸 불러다 학습하고, 결과를 `src/model.py` 에 옮겨 쓰면 됨.

    from sample_model import FEATURES, build_features, train, SampleModel

    clf = train(train_days)             # 학습
    model = SampleModel(clf)            # 예측기
    model.predict(day)

`src/model.py` 로 옮길 때는 `SampleModel` 을 `Model` 로, `load_model()` 에서
저장해 둔 `clf` 를 읽어 감싸 주면 됨.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ARTIFACTS, FLAT
from src.features import bollinger, macd, rsi

LOOKBACK = 120          # 지표 계산에 쓸 일봉 길이

FEATURES = ["ret1", "ret5", "rsi14", "macd_hist", "pct_b", "band_width",
            "vol_ratio", "pre_ret"]


# ---------------------------------------------------------------- feature

def _one_symbol(g):
    """종목 하나의 일봉에서 feature 한 줄을 뽑음. 마지막 날 기준임."""
    close = g["close"].reset_index(drop=True)
    if len(close) < 30:
        return None

    m = macd(close)              # macd / signal / hist / macd_over_close
    b = bollinger(close, n=20)   # middle / sd / upper / lower / pct_b / band_width
    r = rsi(close, n=14)         # avg_gain / avg_loss / rs / rsi

    return {
        "ret1": close.pct_change().iloc[-1],
        "ret5": close.iloc[-1] / close.iloc[-6] - 1 if len(close) > 5 else np.nan,
        "rsi14": r["rsi"].iloc[-1],
        # 가격으로 나눠야 종목끼리 비교됨
        "macd_hist": m["hist"].iloc[-1] / close.iloc[-1],
        "pct_b": b["pct_b"].iloc[-1],
        "band_width": b["band_width"].iloc[-1],
        "vol_ratio": g["volume"].iloc[-1] / g["volume"].tail(20).mean(),
    }


def build_features(day, lookback=LOOKBACK):
    """하루치 feature. 종목마다 한 줄씩 나옴."""
    d = day.daily(days=lookback,
                  columns=["symbol", "date_et", "close", "volume"])
    d = d.sort_values(["symbol", "date_et"])

    rows = []
    for symbol, g in d.groupby("symbol"):
        x = _one_symbol(g)
        if x is not None:
            rows.append({"symbol": symbol, **x})
    if not rows:                     # 앞쪽 며칠은 일봉이 모자라다
        return pd.DataFrame(columns=["symbol", *FEATURES])

    x = pd.DataFrame(rows)

    # 개장 전 시간외 거래에서의 등락
    p = day.price(days=1, columns=["symbol", "datetime", "close", "session"])
    pre = p[p.session == "pre"].sort_values("datetime").groupby("symbol")["close"]
    x["pre_ret"] = x.symbol.map(
        pre.apply(lambda s: s.iloc[-1] / s.iloc[0] - 1 if len(s) > 1 else np.nan))

    x = x[x.symbol.isin(day.symbols)].reset_index(drop=True)
    return x.replace([np.inf, -np.inf], np.nan)


def build_table(days, with_label=True, verbose=True):
    """여러 날을 쌓아 학습용 표를 만듦."""
    rows = []
    for i, day in enumerate(days, 1):
        x = build_features(day)
        if not len(x):
            continue
        if with_label:
            x = x.merge(day.y[["symbol", "label", "ret_pct"]], on="symbol")
        x["date"] = day.date
        rows.append(x)
        if verbose and i % 100 == 0:
            print(f"  {i}/{len(days)}일", flush=True)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 모델

class SampleModel:
    """학습된 분류기를 감싼다. 예측 형식을 맞추는 것이 이 클래스의 일임."""

    def __init__(self, clf, features=FEATURES):
        self.clf = clf
        self.features = list(features)

    def predict(self, day):
        x = build_features(day)
        out = pd.DataFrame({"symbol": day.symbols})
        x = out.merge(x, on="symbol", how="left")
        ok = x[self.features].notna().all(axis=1)

        # feature 를 만들지 못한 종목은 보합으로 둠. 빠뜨리면 형식 오류가 남
        label = np.full(len(x), FLAT, dtype=int)
        if ok.any():
            label[ok.values] = self.clf.predict(x.loc[ok, self.features])
        return pd.DataFrame({"symbol": x.symbol, "label": label})


def train(days, **kwargs):
    """학습해서 분류기를 돌려줌."""
    from sklearn.ensemble import RandomForestClassifier

    table = build_table(days).dropna(subset=FEATURES)
    params = dict(n_estimators=300, max_depth=8, min_samples_leaf=50,
                  n_jobs=-1, random_state=0)
    params.update(kwargs)

    clf = RandomForestClassifier(**params)
    clf.fit(table[FEATURES], table["label"].astype(int))
    return clf


ARTIFACT = ARTIFACTS / "model.pkl"


def save(clf, path=ARTIFACT):
    """분류기만 저장함. 클래스를 통째로 저장하면 다른 곳에서 못 읽음."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    return path


def load(path=ARTIFACT):
    """저장해 둔 분류기를 읽어 예측기로 감싼다."""
    with open(path, "rb") as f:
        return SampleModel(pickle.load(f))
