"""통계 평가.

기능  : 기준선 대비 후보의 paired ΔScore 와 신뢰구간, 채택/보류/제외 판정
구성  : day_confusions() · score_from_counts() · paired_bootstrap() · decide()
역할  : 같은 날짜 block 을 두 모델에 동시에 재표본함 (paired)
        실험 단위 = fold × universe draw. 단위마다 따로 재표본하고, 단위 평균을 결합 통계로 씀

수식
    Δ_u      = score_u(cand) − score_u(base)           (단위 u 의 전체 표본)
    Δ̄        = mean_u Δ_u
    재표본 b  : 단위마다 연속 거래일 block(길이 L)을 복원 추출해 원래 일수만큼 이어 붙임
                → Δ̄_b = mean_u [score_u,b(cand) − score_u,b(base)]
    95% CI   = (Δ̄_b 의 2.5%, 97.5% 분위수)
"""

import numpy as np

from src.data import WEIGHT

from .config import derive_seed

_W = WEIGHT.reshape(-1)


# -----------------------------------------------------------------------------
# 기능  : 날짜별 혼동행렬 (25칸 벡터)
# input : dates (행별 날짜), y, p
# output: (D, 25) 배열, D = 날짜 수 (날짜 오름차순)
# -----------------------------------------------------------------------------
def day_confusions(dates, y, p):
    _, idx = np.unique(np.asarray(dates), return_inverse=True)
    out = np.zeros((idx.max() + 1, 25))
    np.add.at(out, (idx, np.asarray(y, int) * 5 + np.asarray(p, int)), 1)
    return out


# -----------------------------------------------------------------------------
# 기능  : 혼동행렬 묶음에서 score 를 한 번에 계산
# 수식  : score = 1 − Σ w·O / Σ w·E,  E = 행합 × 열합 / N
# input : C  (..., 25)
# output: (...) 배열
# -----------------------------------------------------------------------------
def score_from_counts(C):
    O = C.reshape(*C.shape[:-1], 5, 5)
    n = O.sum(axis=(-1, -2))
    E = O.sum(-1)[..., :, None] * O.sum(-2)[..., None, :] / np.maximum(n, 1)[..., None, None]
    den = (E.reshape(*C.shape[:-1], 25) * _W).sum(-1)
    num = (C * _W).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 1 - num / den, np.nan)


# -----------------------------------------------------------------------------
# 기능  : moving-block 재표본 가중치. 각 날짜가 몇 번 뽑혔는지
# input : D (일수), L (block 길이), B (반복 수), rng
# output: (B, D) 정수 배열
# -----------------------------------------------------------------------------
def _block_weights(D, L, B, rng):
    L = max(1, min(int(L), D))
    k = -(-D // L)
    starts = rng.integers(0, D - L + 1, size=(B, k))
    days = (starts[..., None] + np.arange(L)).reshape(B, -1)[:, :D]
    W = np.zeros((B, D), int)
    np.add.at(W, (np.repeat(np.arange(B), D), days.reshape(-1)), 1)
    return W


# -----------------------------------------------------------------------------
# 기능  : paired bootstrap
# input : units  [{"key": (fold, draw), "dates", "y", "base", "cand"}]
#         B, block_days, seed
# output: {"delta": Δ̄, "ci_low", "ci_high", "se", "per_unit": {key: Δ_u}}
# -----------------------------------------------------------------------------
def paired_bootstrap(units, B, block_days, seed):
    per_unit, reps = {}, []
    for i, u in enumerate(units):
        Cb = day_confusions(u["dates"], u["y"], u["base"])
        Cc = day_confusions(u["dates"], u["y"], u["cand"])
        per_unit[u["key"]] = float(score_from_counts(Cc.sum(0)) - score_from_counts(Cb.sum(0)))
        rng = np.random.default_rng(derive_seed(seed, "boot", *u["key"], i))
        W = _block_weights(len(Cb), block_days, int(B), rng)
        reps.append(score_from_counts(W @ Cc) - score_from_counts(W @ Cb))
    rep = np.nanmean(np.vstack(reps), axis=0)
    lo, hi = np.nanpercentile(rep, [2.5, 97.5])
    return {"delta": float(np.mean(list(per_unit.values()))), "ci_low": float(lo), "ci_high": float(hi),
            "se": float(np.nanstd(rep)), "per_unit": per_unit}


# -----------------------------------------------------------------------------
# 기능  : 채택 / 보류 / 제외 판정 (설계 문서 16절)
# input : row  {"fold_delta": {fold: Δ}, "delta", "ci_low", "ci_high", "unseen_delta",
#               "flip_diff", "extreme_ratio"}
#         rule config["adopt"]  (reject_fold_tol: 이만큼 넘게 음수인 fold 만 '악화'로 셈)
# output: ("adopt" | "hold" | "reject", 근거 문자열 목록)
# -----------------------------------------------------------------------------
def decide(row, rule):
    fd = list(row["fold_delta"].values())
    pos, neg = sum(v > 0 for v in fd), sum(v < -rule["reject_fold_tol"] for v in fd)
    why = []
    if row["ci_high"] < 0:
        why.append("CI 상한 < 0")
    if neg >= rule["reject_negative_folds"]:
        why.append(f"Δ < −{rule['reject_fold_tol']} 인 fold {neg}개")
    if why:
        return "reject", why
    checks = {
        f"Δ>0 fold {pos}/{len(fd)} ≥ {rule['min_positive_folds']}": pos >= rule["min_positive_folds"],
        f"CI 하한 {row['ci_low']:+.4f} > 0": row["ci_low"] > 0,
        f"unseen Δ {row['unseen_delta']:+.4f} > 0": row["unseen_delta"] > 0,
        f"최악 fold {min(fd):+.4f} ≥ −{rule['worst_fold_tol']}": min(fd) >= -rule["worst_fold_tol"],
        f"flip 증가 {row['flip_diff']:+.4f} ≤ {rule['flip_tol']}": row["flip_diff"] <= rule["flip_tol"],
        f"급등락 비율 배수 {row['extreme_ratio']:.2f} ∈ 1±{rule['extreme_band']}":
            abs(row["extreme_ratio"] - 1) <= rule["extreme_band"],
        f"평균 Δ {row['delta']:+.4f} ≥ {rule['min_gain']}": row["delta"] >= rule["min_gain"],
    }
    failed = [f"✗ {k}" for k, ok in checks.items() if not ok]
    return ("adopt", list(checks)) if not failed else ("hold", failed)
