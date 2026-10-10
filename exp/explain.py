"""설명 가능성 (w6-1).

기능  : DEV 분할에서 학습한 후보 모델을 설명하고, 그 설명 자체를 검증함
구성  : group_of_features()    feature → 묶음 이름 (상관된 feature 는 묶음 단위로 설명, p18)
        block_permute()        묶음 열을 "같은 종목의 다른 날짜 block" 값으로 바꿈 (시계열 구조 보존, p17)
        permutation_importance 묶음별 점수 하락, 반복 n_rep 회 (p15~17)
        faithfulness / stability  설명 검증 (p23)
        pdp_ice()              크기 신호가 feature 값에 따라 어떻게 바뀌는지, 종목별 선(ICE) 포함 (p20~22)
        text_terms()           텍스트 분류기의 단어 계수 상·하위 (intrinsic 설명, p6)
        explain_row()          한 예측의 분해: 갭 → k 경계 → 텍스트 배율 → label
        verify_numbers()       LLM 이 쓴 설명문의 숫자가 프롬프트 안에 있는지 검사 (p35~36)
        run_explain()          위를 묶어 runs/<name>_<hash>_explain_<후보>/ 에 저장
역할  : "모델이 무엇에 의존했는가" 를 보여 줌. 인과("이 feature 가 급등락을 일으킨다")가 아님 (p19, p36)
"""

import json
import re

import numpy as np
import pandas as pd

from src.model import FEATURE_GROUPS, TEXT_COLS, Pipeline, fast_score

from . import config as C
from .panel import RESEARCH_GROUPS, load_tables, universe_panel
from .splits import fold_masks, universe_draws


# -----------------------------------------------------------------------------
# 기능  : feature 이름 → 묶음 이름. 어느 묶음에도 없으면 자기 이름
# -----------------------------------------------------------------------------
def group_of_features(feats):
    groups = {**FEATURE_GROUPS, **RESEARCH_GROUPS}
    out = {}
    for f in feats:
        g = next((g for g, cols in groups.items() if f in cols and g not in ("regime",)), None)
        out.setdefault(g or f, []).append(f)
    return out


# -----------------------------------------------------------------------------
# 기능  : block permutation. 평가 날짜를 block_days 개씩 묶어 block 순서를 섞고,
#         각 행의 묶음 열을 "같은 종목, 섞인 위치의 날짜" 값으로 바꿈. 정답과 다른 열은 그대로
# 수식  : 날짜 d 가 block b 의 i 번째면 → π(b) 의 i 번째 날짜 값 (없으면 NaN)
# input : X (평가 행), cols, block_days, rng
# output: 열만 바뀐 X 사본
# -----------------------------------------------------------------------------
def block_permute(X, cols, block_days, rng):
    days = np.sort(X["target"].unique())
    blocks = [days[i:i + block_days] for i in range(0, len(days), block_days)]
    perm = rng.permutation(len(blocks))
    mapping = {}
    for b, pb in enumerate(perm):
        src, dst = blocks[b], blocks[pb]
        for i, d in enumerate(src):
            mapping[d] = dst[i % len(dst)]
    lookup = X.set_index(["symbol", "target"])[cols]
    key = pd.MultiIndex.from_arrays([X["symbol"], X["target"].map(mapping)])
    Y = X.copy()
    vals = lookup.reindex(key)
    for c in cols:
        Y[c] = vals[c].to_numpy()
    return Y


# -----------------------------------------------------------------------------
# 기능  : 묶음별 permutation importance
# 수식  : imp_g = score(원래) − score(묶음 g 를 block 으로 섞음),  n_rep 회 반복
# output: DataFrame[group, rep, drop]
# -----------------------------------------------------------------------------
def permutation_importance(pipe, X, groups, n_rep, block_days, rng):
    y = X["label"].to_numpy(int)
    base = fast_score(y, pipe.predict(X))
    rows = []
    for g, cols in groups.items():
        for r in range(n_rep):
            Y = block_permute(X, cols, block_days, rng)
            rows.append({"group": g, "rep": r, "drop": base - fast_score(y, pipe.predict(Y))})
    return pd.DataFrame(rows), base


# -----------------------------------------------------------------------------
# 기능  : PDP / ICE. 크기 신호 |s'| (갭 × 텍스트 배율) 과 급등락 예측 비율이 feature 값에 따라 어떻게 바뀌나
#         그리드 = 평가 행의 분위수 10개. 모든 행의 그 열을 그 값으로 바꿔 예측 (모델 재학습 없음)
# output: DataFrame[feature, grid, symbol(전체 = "ALL"), pred_extreme, mean_abs_signal]
# -----------------------------------------------------------------------------
def pdp_ice(pipe, X, feature, n_grid=10):
    v = X[feature].astype(float)
    grid = np.unique(np.nanquantile(v, np.linspace(0.05, 0.95, n_grid)))
    rows = []
    for g in grid:
        Y = X.copy()
        Y[feature] = g
        s = pipe.signal.predict(Y)
        if pipe.regime.var:
            s = pipe.regime.apply(s, pipe.regime.values(Y))
        lab = pipe.decision.predict(s)
        d = pd.DataFrame({"symbol": X["symbol"].to_numpy(), "ext": np.isin(lab, (0, 4)), "abs_s": np.abs(s)})
        rows.append({"feature": feature, "grid": g, "symbol": "ALL", "pred_extreme": d["ext"].mean(),
                     "mean_abs_signal": np.nanmean(d["abs_s"])})
        for sym, h in d.groupby("symbol"):
            rows.append({"feature": feature, "grid": g, "symbol": sym, "pred_extreme": h["ext"].mean(),
                         "mean_abs_signal": np.nanmean(h["abs_s"])})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 기능  : 텍스트 분류기(textclf)의 단어 계수 상·하위 n 개 (급등락 확률을 올리는 / 내리는 표현)
# output: DataFrame[term, coef, direction]  (텍스트 분류기가 아니면 빈 표)
# -----------------------------------------------------------------------------
def text_terms(pipe, n=25):
    m = getattr(pipe.regime, "model", None)
    if m is None or getattr(pipe.regime, "input", None) != "text":
        return pd.DataFrame(columns=["term", "coef", "direction"])
    vec, clf = m.steps[0][1], m.steps[-1][1]
    if not hasattr(vec, "get_feature_names_out"):
        return pd.DataFrame(columns=["term", "coef", "direction"])
    names, coef = vec.get_feature_names_out(), clf.coef_[0]
    o = np.argsort(coef)
    top = [(names[i], coef[i], "급등락 확률 ↑") for i in o[::-1][:n]] + [(names[i], coef[i], "급등락 확률 ↓") for i in o[:n]]
    return pd.DataFrame(top, columns=["term", "coef", "direction"])


# -----------------------------------------------------------------------------
# 기능  : 한 예측의 분해 (waterfall 대응). 모델 계산 그대로라 정확한 설명 (intrinsic)
# output: dict  gap_pct → 배율 f → s' → 경계 → label
# -----------------------------------------------------------------------------
def explain_row(pipe, row):
    X = pd.DataFrame([row])
    s = float(pipe.signal.predict(X)[0])
    f = 1.0
    if pipe.regime.var:
        f = float(pipe.regime.apply(np.array([1.0]), pipe.regime.values(X))[0])
    return {"signal": s, "regime_factor": f, "scaled_signal": s * f,
            "bounds": pipe.decision.params().get("bounds"), "label": int(pipe.predict(X)[0])}


# -----------------------------------------------------------------------------
# 기능  : LLM 설명문 검증. 설명문에 나온 모든 숫자가 프롬프트 안에 있어야 함 (없으면 모델이 만든 숫자)
# input : prompt, text
# output: 프롬프트에 없는 숫자 목록 (빈 목록이면 통과)
# -----------------------------------------------------------------------------
def verify_numbers(prompt, text):
    num = re.compile(r"[-+−]?\d+(?:[.,]\d+)*%?")
    norm = lambda s: s.replace("−", "-").replace(",", "").rstrip("%").lstrip("+")
    have = {norm(x) for x in num.findall(prompt)}
    return [x for x in num.findall(text) if norm(x) not in have]


# -----------------------------------------------------------------------------
# 기능  : 설명 묶음 실행 (DEV 만, runner 와 같은 분할·시드 → 같은 모델을 다시 만듦)
#         faithfulness: tune 구간에서 고른 1위 묶음을 평가 구간에서 섞었을 때의 하락이 다른 묶음 평균보다 큰가
#         stability   : 분할(fold × draw) 사이 묶음 중요도 순위의 Spearman 상관 평균
# input : cfg, candidate, n_rep, block_days, pdp_features
# output: 결과 폴더
# -----------------------------------------------------------------------------
def run_explain(cfg, candidate, n_rep=30, block_days=5, pdp_features=None):
    cand = next((c for c in cfg["candidates"] if c["name"] == candidate), None)
    if cand is None:
        raise SystemExit(f"후보 '{candidate}' 없음")
    sp, seed = cfg["split"], int(cfg["seed"])
    extra = {k: RESEARCH_GROUPS[k] for k in cfg["data"]["research"]}
    out = C.RUNS / f"{cfg['name']}_{cfg['hash']}_explain_{re.sub(r'[^A-Za-z0-9_+-]', '_', candidate)}"
    out.mkdir(parents=True, exist_ok=True)
    draws = universe_draws(sorted(load_tables()["daily"]["symbol"].unique()), sp, seed)
    imps, faith, pdps, terms, bases = [], [], [], [], []
    for dr in draws:
        tr_p = pd.concat([universe_panel(u, cfg["data"]["groups"], cfg["data"]["research"]) for u in dr["train_universes"]])
        te_p = universe_panel(dr["test_universe"], cfg["data"]["groups"], cfg["data"]["research"])
        for fold in sp["folds"]:
            trm, tem = fold_masks(tr_p, te_p, fold, sp["embargo_days"])
            tr, te = tr_p[trm], te_p[tem].reset_index(drop=True)
            spec = {**cand, "seed": C.derive_seed(seed, "model", cand["name"], dr["draw"], fold["name"]) % 2**31}
            pipe = Pipeline(spec, extra).fit(tr)
            groups = group_of_features([f for f in pipe.features if f not in TEXT_COLS] +
                                       [f for f in pipe.features if f in TEXT_COLS])
            rng = np.random.default_rng(C.derive_seed(seed, "explain", candidate, dr["draw"], fold["name"]))
            imp, base = permutation_importance(pipe, te, groups, n_rep, block_days, rng)
            imp[["fold", "draw"]] = fold["name"], dr["draw"]
            imps.append(imp)
            bases.append({"fold": fold["name"], "draw": dr["draw"], "score": base})
            # faithfulness: 설명(tune 구간 중요도) → 검증(평가 구간)
            tune = tr[tr["target"] > tr["target"].quantile(0.8)]
            if len(groups) > 1:
                ti, _ = permutation_importance(pipe, tune.reset_index(drop=True), groups, max(5, n_rep // 6), block_days, rng)
                top = ti.groupby("group")["drop"].mean().idxmax()
                m = imp.groupby("group")["drop"].mean()
                faith.append({"fold": fold["name"], "draw": dr["draw"], "top_group": top,
                              "top_drop": m[top], "others_mean_drop": m.drop(top).mean(), "faithful": m[top] > m.drop(top).mean()})
            for f in (pdp_features or [v for v in pipe.regime.inputs if v not in TEXT_COLS][:1]):
                if f in te.columns:
                    pdps.append(pdp_ice(pipe, te, f).assign(fold=fold["name"], draw=dr["draw"]))
            t = text_terms(pipe)
            if len(t):
                terms.append(t.assign(fold=fold["name"], draw=dr["draw"]))
            print(f"[explain] draw {dr['draw']} fold {fold['name']}: base {base:.4f}", flush=True)

    I = pd.concat(imps, ignore_index=True)
    I.to_csv(out / "importance_reps.csv", index=False)
    summ = I.groupby("group")["drop"].agg(["mean", "std", lambda s: np.percentile(s, 5), lambda s: np.percentile(s, 95)])
    summ.columns = ["mean_drop", "sd", "p5", "p95"]
    summ.sort_values("mean_drop", ascending=False).to_csv(out / "importance_summary.csv")
    unit = I.groupby(["fold", "draw", "group"])["drop"].mean().unstack("group")
    rk = unit.rank(axis=1)
    cors = [rk.iloc[i].corr(rk.iloc[j], method="spearman") for i in range(len(rk)) for j in range(i + 1, len(rk))]
    stab = {"mean_spearman": float(np.nanmean(cors)) if cors else None, "n_units": len(rk)}
    (out / "stability.json").write_text(json.dumps(stab, indent=2), encoding="utf-8")
    pd.DataFrame(bases).to_csv(out / "base_scores.csv", index=False)
    if faith:
        pd.DataFrame(faith).to_csv(out / "faithfulness.csv", index=False)
    if pdps:
        pd.concat(pdps).to_csv(out / "pdp_ice.csv", index=False)
    if terms:
        T = pd.concat(terms)
        T.groupby(["term", "direction"])["coef"].agg(["mean", "size"]).sort_values("mean").to_csv(out / "text_terms.csv")
    _plot(I, out)
    print(summ.sort_values("mean_drop", ascending=False).round(4).to_string())
    print(f"[explain] stability {stab}  faithful {np.mean([f['faithful'] for f in faith]) if faith else 'n/a'}  → {out}")
    return out


def _plot(I, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    order = I.groupby("group")["drop"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(order) + 1.5))
    ax.boxplot([I.loc[I["group"] == g, "drop"] for g in order], vert=False, tick_labels=list(order), showfliers=False)
    ax.axvline(0, color="k", lw=1)
    ax.set(title="묶음별 block permutation importance (score 하락)", xlabel="score 하락 (클수록 모델이 의존)")
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    fig.tight_layout()
    fig.savefig(out / "importance_boxplot.png", dpi=120)
    plt.close(fig)
