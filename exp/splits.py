"""분할.

기능  : universe draw (seen 10 + unseen 10), DEV fold 를 만듦 (lockbox 보호는 exp/lockbox.py)
구성  : partition() · universe_draws() · fold_masks()
역할  : 채점 환경(미래 구간 × 처음 보는 종목 10 + 아는 종목 10, 20종목 universe)을 흉내 냄

draw d 의 구성 (시드 = derive_seed(seed, "draw", d))
    unseen  U  : 전체 종목 중 n_unseen 개 → 학습에서 완전히 뺌
    seen    S  : 나머지 종목. 학습 종목
    test universe = U + (S 중 n_seen 개)        ← 평가 행, 교차 feature 도 이 20종목 안에서 계산
    train universes = S 를 20종목 안팎 묶음으로 나눔 ← 학습 행의 교차 feature 도 20종목 안에서 계산
"""

import numpy as np
import pandas as pd

from .config import derive_seed


# -----------------------------------------------------------------------------
# 기능  : 종목 목록을 크기가 고른 묶음으로 나눔
# input : symbols, size (묶음 최대 크기), rng
# output: [[종목, ...], ...]  묶음 수 = ceil(n / size), 크기 차이는 최대 1
# -----------------------------------------------------------------------------
def partition(symbols, size, rng):
    s = list(rng.permutation(sorted(symbols)))
    k = -(-len(s) // size)
    return [sorted(map(str, c)) for c in np.array_split(np.array(s, dtype=object), k)]


# -----------------------------------------------------------------------------
# 기능  : universe draw 목록
# input : symbols  전체 종목,  cfg  config["split"],  seed
# output: [{"draw", "unseen", "seen_test", "test_universe", "train_universes"}, ...]
# -----------------------------------------------------------------------------
def universe_draws(symbols, cfg, seed):
    out = []
    for d in range(int(cfg["n_draws"])):
        rng = np.random.default_rng(derive_seed(seed, "draw", d))
        allsym = sorted(symbols)
        unseen = sorted(map(str, rng.choice(allsym, int(cfg["n_unseen"]), replace=False)))
        seen = [s for s in allsym if s not in unseen]
        seen_test = sorted(map(str, rng.choice(seen, int(cfg["n_seen"]), replace=False)))
        out.append({"draw": d, "unseen": unseen, "seen_test": seen_test,
                    "test_universe": sorted(unseen + seen_test),
                    "train_universes": partition(seen, int(cfg["universe_size"]), rng)})
    return out


# -----------------------------------------------------------------------------
# 기능  : fold 하나의 학습·평가 행 mask
# 수식  : 학습  target < start − embargo
#         평가  start ≤ target < end
# input : train_panel, test_panel, fold {"name", "start", "end"}, embargo_days
# output: (학습 mask, 평가 mask)  numpy bool
# -----------------------------------------------------------------------------
def fold_masks(train_panel, test_panel, fold, embargo_days):
    s, e = pd.Timestamp(fold["start"]), pd.Timestamp(fold["end"])
    tr = (train_panel["target"] < s - pd.Timedelta(days=int(embargo_days))).to_numpy()
    te = ((test_panel["target"] >= s) & (test_panel["target"] < e)).to_numpy()
    return tr, te
