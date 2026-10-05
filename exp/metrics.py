"""평가 지표.

기능  : 예측 하나(정답·예측 label 배열)에 대한 지표 묶음
구성  : evaluate()
역할  : 모든 단계가 같은 지표 이름을 씀 → 결과표·판정 코드가 단계와 무관하게 같음
"""

import numpy as np

from src.data import WEIGHT, score

MAG = np.array([2, 1, 0, 1, 2])          # label → 크기 등급 (0 보합, 1 일반, 2 급등락)


# -----------------------------------------------------------------------------
# 기능  : 지표 계산
# input : y, p  정답·예측 label 배열
# output: dict
#   score            과제 점수 1 − Σw·O / Σw·E
#   accuracy         맞힌 비율
#   direction        정답이 보합이 아닌 행에서 부호 적중률 (src.data.score 와 같은 정의)
#   magnitude_acc    크기 등급(보합/일반/급등락) 적중률
#   big_recall       정답 급등락 중 맞힌 비율
#   big_prec         급등락 예측 중 맞은 비율
#   flip_index       P(예측 4 | 정답 0) + P(예측 0 | 정답 4)   (0 ~ 2)
#   flip_rate        (0→4 + 4→0 건수) / n
#   flip_loss_share  64 × flip 건수 / Σw·E   (잃은 점수 중 flip 몫)
#   pred_extreme / true_extreme  예측·정답의 급등락 비율
#   n
# -----------------------------------------------------------------------------
def evaluate(y, p):
    y, p = np.asarray(y, int), np.asarray(p, int)
    base = score(y, p)
    O = np.bincount(y * 5 + p, minlength=25).reshape(5, 5).astype(float)
    den = (WEIGHT * np.outer(O.sum(1), O.sum(0)) / max(len(y), 1)).sum()
    rate = lambda i, j: O[i, j] / O[i].sum() if O[i].sum() else 0.0
    flips = O[0, 4] + O[4, 0]
    return {
        "score": base["score"], "accuracy": base["accuracy"], "direction": base["direction"],
        "magnitude_acc": float((MAG[y] == MAG[p]).mean()) if len(y) else np.nan,
        "big_recall": base["big_recall"], "big_prec": base["big_prec"],
        "flip_index": float(rate(0, 4) + rate(4, 0)),
        "flip_rate": float(flips / max(len(y), 1)),
        "flip_loss_share": float(64 * flips / den) if den else np.nan,
        "pred_extreme": float(np.isin(p, (0, 4)).mean()) if len(p) else np.nan,
        "true_extreme": float(np.isin(y, (0, 4)).mean()) if len(y) else np.nan,
        "n": int(len(y)),
    }
