"""Lab 3-2. 세 가지 가중 평균을 직접 구현한 모듈.

모든 함수는 pandas Series를 받아 같은 인덱스의 Series를 돌려줌.
인덱스 t의 값은 t까지의 관측만 사용한다 (미래를 보지 않는다).
"""

import numpy as np
import pandas as pd

__all__ = ["sma", "wma", "ema", "span_to_alpha", "alpha_to_span",
           "center_of_mass", "weight_profile"]


def sma(x, n):
    """단순이동평균. 가중치는 모두 1/n.

    앞의 n-1개는 NaN이고, 첫 유효값의 위치는 정확히 n-1임.
    """
    x = pd.Series(x).astype(float)
    v = x.to_numpy()
    out = np.full(len(v), np.nan)
    for t in range(n - 1, len(v)):
        out[t] = v[t - n + 1:t + 1].mean()
    return pd.Series(out, index=x.index, name=f"sma{n}")


def wma(x, n):
    """선형 가중 이동평균. 오늘에 n, 하루 전에 n-1, ..., n-1일 전에 1.

    가중치 합은 n(n+1)/2임. 가중치 순서를 뒤집어도 오류가 나지 않고
    그럴듯한 평균이 나오므로, 단위 테스트로만 잡을 수 있음.
    """
    x = pd.Series(x).astype(float)
    v = x.to_numpy()
    w = np.arange(n, 0, -1, dtype=float)      # [n, n-1, ..., 1] : 맨 앞이 오늘
    w = w / w.sum()
    out = np.full(len(v), np.nan)
    for t in range(n - 1, len(v)):
        window = v[t - n + 1:t + 1][::-1]     # 오늘부터 과거 순으로 뒤집는다
        out[t] = float((window * w).sum())
    return pd.Series(out, index=x.index, name=f"wma{n}")


def ema(x, alpha, seed=None, seed_index=None):
    """지수이동평균. s(t) = alpha * x(t) + (1 - alpha) * s(t-1).

    seed를 명시적인 인자로 받음. 기본값에 숨기지 않는 이유는
    pandas ewm과 값이 달라지는 원인이 거의 항상 seed이기 때문임.

    seed_index : seed를 놓을 위치(정수). 그 앞은 NaN임. 기본은 0.
    """
    x = pd.Series(x).astype(float)
    v = x.to_numpy()
    k = 0 if seed_index is None else int(seed_index)
    s = float(v[k]) if seed is None else float(seed)
    out = np.full(len(v), np.nan)
    out[k] = s
    for t in range(k + 1, len(v)):
        s = alpha * v[t] + (1.0 - alpha) * s
        out[t] = s
    return pd.Series(out, index=x.index, name=f"ema{alpha:.4f}")


def span_to_alpha(n):
    """pandas가 쓰는 변환. span n에 대응하는 alpha는 2/(n+1)임."""
    return 2.0 / (n + 1.0)


def alpha_to_span(alpha):
    """위의 역변환."""
    return 2.0 / alpha - 1.0


def center_of_mass(weights):
    """무게중심. sum(i * w_i), i는 오늘로부터 며칠 전인지.

    이 값이 그 평균이 갖는 지연(lag)의 크기다.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return float((np.arange(len(w)) * w).sum())


def weight_profile(kind, n=5, alpha=None, length=20):
    """세 평활의 가중치를 같은 길이의 벡터로 만들어 비교용으로 돌려줌.

    반환 벡터의 0번 원소가 오늘, 1번이 하루 전임.
    """
    w = np.zeros(length)
    if kind == "sma":
        w[:n] = 1.0 / n
    elif kind == "wma":
        w[:n] = np.arange(n, 0, -1, dtype=float)
        w /= w.sum()
    elif kind == "ema":
        a = span_to_alpha(n) if alpha is None else alpha
        w = a * (1.0 - a) ** np.arange(length)
    else:
        raise ValueError(kind)
    return w
