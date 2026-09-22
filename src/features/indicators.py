"""Lab 3-2. 네 가지 지표를 직접 구현한 모듈.

각 함수는 입력과 같은 인덱스의 DataFrame을 돌려주고, 입력을 바꾸지 않음.
관례(seed, alpha, ddof)를 함수 인자로 드러내 두었다. 라이브러리와 값이 다를 때
고쳐야 할 곳이 바로 이 인자들이기 때문임.
"""

import numpy as np
import pandas as pd

from .smoothing import ema, sma, span_to_alpha

__all__ = ["wilder", "macd", "rsi", "true_range", "atr", "bollinger",
           "golden_cross"]


def wilder(x, n, seed_index=None, seed=None):
    """Wilder 평활. alpha = 1/n 이며, 보통의 EMA alpha = 2/(n+1)이 아님.

    n = 14에서 1/14 = 0.0714이고 2/15 = 0.1333임. 거의 두 배 차이이므로
    RSI와 ATR을 여기서 틀리면 값이 몇 점씩 어긋난다.
    """
    return ema(x, alpha=1.0 / n, seed=seed, seed_index=seed_index)


def macd(close, fast=12, slow=26, signal=9, seed="first"):
    """MACD. 빠른 EMA에서 느린 EMA를 뺀 값, 그리고 그것을 다시 평활한 신호선.

    seed="first"는 표본의 첫 종가로 두 EMA를 모두 seed하는 관례이며
    pandas의 ewm(adjust=False)와 같음.
    """
    close = pd.Series(close).astype(float)
    f = ema(close, span_to_alpha(fast))
    s = ema(close, span_to_alpha(slow))
    line = f - s
    sig = ema(line, span_to_alpha(signal))
    out = pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig},
                       index=close.index)
    out["macd_over_close"] = out["macd"] / close
    return out


def rsi(close, n=14):
    """RSI. 최근 움직임 중 상승이 차지한 몫을 0에서 100 사이로 적은 값.

    첫 값은 n개 변화량의 단순평균으로 두고, 그 뒤는 Wilder 평활로 갱신함.
    """
    close = pd.Series(close).astype(float)
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)

    g = np.full(len(close), np.nan)
    l = np.full(len(close), np.nan)
    if len(close) > n:
        g[n] = gain.iloc[1:n + 1].mean()
        l[n] = loss.iloc[1:n + 1].mean()
        for t in range(n + 1, len(close)):
            g[t] = (g[t - 1] * (n - 1) + gain.iloc[t]) / n
            l[t] = (l[t - 1] * (n - 1) + loss.iloc[t]) / n

    g = pd.Series(g, index=close.index)
    l = pd.Series(l, index=close.index)
    rs = g / l
    val = 100.0 - 100.0 / (1.0 + rs)
    val = val.where(l > 0, 100.0).where(~g.isna())
    return pd.DataFrame({"avg_gain": g, "avg_loss": l, "rs": rs, "rsi": val},
                        index=close.index)


def true_range(high, low, close):
    """참 범위. 밤사이 갭을 포함한 하루의 이동 거리.

    max( 고가-저가, |고가-전일종가|, |저가-전일종가| )
    """
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)
    prev = pd.Series(close).astype(float).shift(1)
    a = high - low
    b = (high - prev).abs()
    c = (low - prev).abs()
    tr = pd.concat([a, b, c], axis=1).max(axis=1)
    tr.iloc[0] = a.iloc[0]                       # 전일 종가가 없는 첫 행
    return pd.DataFrame({"hl": a, "h_prev": b, "l_prev": c, "tr": tr},
                        index=high.index)


def atr(high, low, close, n=14):
    """ATR. 참 범위의 Wilder 평활. 첫 값은 n개의 단순평균."""
    tr = true_range(high, low, close)["tr"]
    v = tr.to_numpy()
    out = np.full(len(v), np.nan)
    if len(v) > n:
        out[n] = v[1:n + 1].mean()
        for t in range(n + 1, len(v)):
            out[t] = (out[t - 1] * (n - 1) + v[t]) / n
    a = pd.Series(out, index=tr.index)
    return pd.DataFrame({"tr": tr, "atr": a,
                         "atr_over_close": a / pd.Series(close).astype(float)},
                        index=tr.index)


def bollinger(close, n=20, k=2.0, ddof=1):
    """볼린저 밴드. 이동평균과 이동표준편차로 만든 위아래 띠.

    ddof=1이 표본 표준편차, ddof=0이 모표준편차다. 관례가 갈리는 지점이라
    인자로 드러내 두었다. %B는 0.867과 0.881처럼 값이 달라진다.
    """
    close = pd.Series(close).astype(float)
    mid = sma(close, n)
    sd = close.rolling(n).std(ddof=ddof)
    upper = mid + k * sd
    lower = mid - k * sd
    pb = (close - lower) / (upper - lower)
    width = (upper - lower) / mid
    return pd.DataFrame({"middle": mid, "sd": sd, "upper": upper, "lower": lower,
                         "pct_b": pb, "band_width": width}, index=close.index)


def golden_cross(close, fast=50, slow=200):
    """d(t) = SMA(fast) - SMA(slow)의 부호가 바뀐 날을 찾는다.

    골든크로스는 음에서 양, 데드크로스는 양에서 음임. 그 외에는 아무 일도 없음.
    """
    close = pd.Series(close).astype(float)
    d = sma(close, fast) - sma(close, slow)
    sign = np.sign(d)
    prev = sign.shift(1)
    event = pd.Series(np.where((prev < 0) & (sign > 0), "golden",
                      np.where((prev > 0) & (sign < 0), "dead", "")),
                      index=close.index)
    return pd.DataFrame({"d": d, "event": event}, index=close.index)
