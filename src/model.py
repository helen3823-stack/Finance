"""제출 파일이자 실험 공용 코어.

기능  : 대상일 09:30 전 정보로 feature 를 만들고, 신호(signal) → 결정 규칙(decision)
        → regime 보정(regime) 순서로 label 0~4 를 냄
구성  : 1. 상수·registry   2. feature (build_features / day_tables / day_features)
        3. 점수            4. 신호 head   5. 결정 규칙   6. regime 보정
        7. Pipeline        8. Model / load_model
역할  : 채점(predict)과 실험(exp/)이 이 파일의 같은 함수를 씀 → 학습·검증·제출의 feature 가 구조적으로 같음
        채점자는 이 파일 외의 src/ 를 원본으로 덮어쓰므로, 제출에 필요한 코드는 전부 여기 둠

실험할 때 바꾸는 곳은 registry 이름과 spec(dict) 뿐임.
    FEATURE_GROUPS   feature 묶음 이름 → 열 이름
    SIGNALS          신호 이름 → 클래스
    DECISIONS        결정 규칙 이름 → 클래스
    REGIMES          regime 보정 이름 → 클래스
spec 예:
    {"name": "B2", "signal": {"type": "gap"}, "decision": {"type": "k"}}
"""

import pickle
import sys
import warnings

import numpy as np
import pandas as pd

from .data import CUTOFF, FLAT, WEIGHT
from .paths import ARTIFACTS

# =============================================================================
# 1. 상수 · registry
# =============================================================================

NS = "datetime64[ns]"
HIST_DAYS = 100          # predict 때 읽는 일봉 길이. 가장 긴 rolling(60) + 여유
NEWS_HIST_DAYS = 30      # predict 때 읽는 뉴스 길이. NEWS_WINDOW + 여유
NEWS_WINDOW = 20         # 뉴스 평소 수준 = 직전 20개 대상일 창의 평균
BIG_CUT = 2.5            # 급등락 경계(%)
VOL_FLOOR = 0.5          # zgap 분모의 하한(%). 변동성 0 근처에서 폭발 방지
ZGAP_CLIP = 10.0
BETA_SHRINK = 0.5        # beta = 0.5 × rolling beta + 0.5 (1 쪽으로 수축)
SURP_CLIP = 50.0         # surprise_pct 절단(%)
TONE_TAIL = 5.0          # tone 꼬리 기준
MIN_LOO = 3              # 시장 갭을 계산할 최소 다른 종목 수
BIG_GAP = 1.0            # regime: 큰 갭 종목 기준(%)
RET_CLIP = 15.0          # 회귀 목표 절단(%)
DEFAULT_K = 5.25         # artifact 가 없을 때만 쓰는 갭 규칙 배율. EDA 값이라 lockbox 가 노출된 값임 (실험·제출은 학습으로 k 를 고름)
CLASS_CENTERS = np.array([-4.0, -1.5, 0.0, 1.5, 4.0])   # 5-class 확률 → 기대 수익률(%) 신호

ARTIFACT = ARTIFACTS / "pipeline.pkl"
ARTIFACT_FORMAT = 1

# feature 묶음. 실험 config 에서는 이 이름으로 부름
FEATURE_GROUPS = {
    "core":    ["gap_pct", "gap_rank"],
    "scale":   ["vol5", "vol20", "atr14", "big_rate60", "zgap"],
    "cross":   ["mkt_gap", "resid_gap", "beta60"],
    "event":   ["earn", "surp_sign", "surp_abs", "surp_pct_w"],
    "analyst": ["an_n", "an_up", "an_down", "an_tgt_chg", "an_gap_agree"],
    "news":    ["news_n", "news_abn", "tone_mean", "tone_min", "tone_max", "tone_neg_tail", "news_missing"],
    "regime":  ["reg_med_vol20", "reg_med_absgap", "reg_gap_disp", "reg_mkt_gap", "reg_big_ratio"],
}
# 묶음마다 필요한 원천 표. day_tables() 가 필요한 표만 읽게 함
GROUP_TABLES = {
    "core": {"daily", "price"}, "scale": {"daily", "price"}, "cross": {"daily", "price"},
    "regime": {"daily", "price"}, "event": {"earnings"}, "analyst": {"analyst"}, "news": {"news"},
}
DEPLOYABLE = {c for cols in FEATURE_GROUPS.values() for c in cols}
KEY_COLS = ["symbol", "date", "target"]


# -----------------------------------------------------------------------------
# 기능  : feature 목록(묶음 이름 또는 열 이름 섞어서)을 열 이름 목록으로 풂
# input : items  ["core", "vol20", ...]
#         extra  FEATURE_GROUPS 에 없는 연구용 묶음 {이름: [열]} (exp 에서 넘김)
# output: 중복 없는 열 이름 목록 (입력 순서 유지)
# -----------------------------------------------------------------------------
def resolve_features(items, extra=None):
    groups = {**FEATURE_GROUPS, **(extra or {})}
    out = []
    for it in items or []:
        out += groups.get(it, [it])
    return list(dict.fromkeys(out))


# -----------------------------------------------------------------------------
# 기능  : 열 이름 목록을 계산하는 데 필요한 feature 묶음 이름 집합
# input : cols  열 이름 목록
# output: {"core", ...}  (core 는 신호 대체값 계산에 늘 필요해서 항상 포함)
# -----------------------------------------------------------------------------
def groups_of(cols):
    need = {"core"}
    for g, members in FEATURE_GROUPS.items():
        if set(cols) & set(members):
            need.add(g)
    return need


# =============================================================================
# 2. feature
# =============================================================================

# -----------------------------------------------------------------------------
# 기능  : 날짜별로 자기 자신을 뺀 나머지 종목의 중앙값 (leave-one-out median)
# 수식  : mkt_i,t = median_{j≠i} x_j,t   (다른 종목 유효값이 MIN_LOO 개 미만이면 NaN)
# input : values  값 Series,  groups  같은 index 의 날짜 Series
# output: values 와 같은 index 의 Series
# -----------------------------------------------------------------------------
def loo_median(values, groups):
    def _loo(a):
        m = np.tile(a, (len(a), 1))
        np.fill_diagonal(m, np.nan)
        cnt = (~np.isnan(m)).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(m, axis=1) if len(a) else m.sum(axis=1)
        med[cnt < MIN_LOO] = np.nan
        return med

    return values.groupby(groups).transform(lambda s: _loo(s.to_numpy(float)))


# -----------------------------------------------------------------------------
# 기능  : 이벤트 시각을 그 정보를 처음 쓸 수 있는 대상일로 붙임
# 수식  : 대상일 T 의 창 = [기준일 09:30, T 09:30)   ← known_at < cutoff 규칙
# input : ts       이벤트 시각 Series (known_at)
#         targets  DataFrame[date, target]  (기준일, 대상일)
# output: 대상일 Series (창 밖이면 NaT)
# -----------------------------------------------------------------------------
def map_to_target(ts, targets):
    t = targets.sort_values("target")
    lo = (t["date"] + CUTOFF).to_numpy(NS)
    hi = (t["target"] + CUTOFF).to_numpy(NS)
    tgt = t["target"].to_numpy(NS)
    v = pd.Series(ts).to_numpy(NS)
    idx = np.searchsorted(hi, v, side="right")
    ok = idx < len(hi)
    ok[ok] &= v[ok] >= lo[idx[ok]]
    out = np.full(len(v), np.datetime64("NaT"), dtype=NS)
    out[ok] = tgt[idx[ok]]
    return pd.Series(out, index=pd.Series(ts).index)


# -----------------------------------------------------------------------------
# 기능  : 일봉에서 기준일 16:00 까지 알 수 있는 종목별 특성 (모든 rolling 은 과거만)
# 수식  : ret_pct = ret × 100
#         vol_n   = std(ret_pct, 최근 n일)
#         atr14   = mean(TR, 14일),  TR = max(H−L, |H−C₋₁|, |L−C₋₁|) / C × 100
#         big_rate60 = mean(1[ret_pct < −2.5 or ≥ 2.5], 60일)
#         beta60  = 0.5 × cov(r, m)/var(m) (60일) + 0.5,  m = 자기 제외 중앙 수익률
# input : daily  일봉 (symbol, date_et, high, low, close, prev_close, ret)
# output: DataFrame[symbol, date, close_d0, vol5, vol20, atr14, big_rate60, beta60]
# -----------------------------------------------------------------------------
def _daily_block(daily):
    d = daily.assign(date=daily["date_et"].astype(NS).dt.normalize())
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    sym = d["symbol"]

    def roll(s, n, how):
        return s.groupby(sym).transform(lambda x: getattr(x.rolling(n, min_periods=max(2, n // 2)), how)())

    r = d["ret"] * 100
    big = ((r < -BIG_CUT) | (r >= BIG_CUT)).astype(float).where(r.notna())
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["prev_close"]).abs(),
                    (d["low"] - d["prev_close"]).abs()], axis=1).max(axis=1) / d["close"] * 100
    m = loo_median(r, d["date"])
    ex, em, exm, emm = roll(r, 60, "mean"), roll(m, 60, "mean"), roll(r * m, 60, "mean"), roll(m * m, 60, "mean")
    beta = (exm - ex * em) / (emm - em ** 2)
    return pd.DataFrame({
        "symbol": sym, "date": d["date"], "close_d0": d["close"],
        "vol5": roll(r, 5, "std"), "vol20": roll(r, 20, "std"), "atr14": roll(tr, 14, "mean"),
        "big_rate60": roll(big, 60, "mean"),
        "beta60": (BETA_SHRINK * beta + (1 - BETA_SHRINK)).fillna(1.0),
    })


# -----------------------------------------------------------------------------
# 기능  : 시간봉에서 하루 단위 시간외 가격
#         pre_last  그날 pre 봉 중 known_at < 그날 09:30 인 마지막 종가 (= 08:00 봉, 09:00 봉 제외)
#         post_last 그날 post 봉의 마지막 종가
# input : price  시간봉 (symbol, datetime, close, session, known_at)
# output: (pre_last Series, post_last Series)  index = (symbol, day)
# -----------------------------------------------------------------------------
def _extended_hours(price):
    p = price[price["session"] != "regular"].sort_values("datetime")
    day = p["datetime"].astype(NS).dt.normalize()
    pre = (p["session"] == "pre") & (p["known_at"].astype(NS) < day + CUTOFF)
    post = p["session"] == "post"
    last = lambda m: p[m].groupby([p.loc[m, "symbol"], day[m]])["close"].last()
    return last(pre), last(post)


# -----------------------------------------------------------------------------
# 기능  : 실적 · 애널리스트 이벤트를 (종목, 대상일)로 집계
# 수식  : earn = 1[창 안에 실적 발표],  surp_sign = sign(실제 − 예상),  surp_abs = |실제 − 예상|
#         surp_pct_w = clip(surprise_pct, ±50)
#         an_tgt_chg = mean(clip(목표가/이전 목표가 − 1, ±0.5))
# input : e / a  실적·애널리스트 표,  targets  DataFrame[date, target]
# output: DataFrame[symbol, target, <feature>]
# -----------------------------------------------------------------------------
def _earnings_block(e, targets):
    e = e.assign(target=map_to_target(e["known_at"], targets)).dropna(subset=["target"]).sort_values("known_at")
    diff = e["eps_reported"] - e["eps_estimate"]
    e = e.assign(earn=1.0, surp_sign=np.sign(diff), surp_abs=diff.abs(),
                 surp_pct_w=e["surprise_pct"].clip(-SURP_CLIP, SURP_CLIP))
    return e.groupby(["symbol", "target"])[FEATURE_GROUPS["event"]].last().reset_index()


def _analyst_block(a, targets):
    a = a.assign(target=map_to_target(a["known_at"], targets)).dropna(subset=["target"])
    a = a.assign(an_n=1.0, an_up=(a["action"] == "up").astype(float),
                 an_down=(a["action"] == "down").astype(float),
                 an_tgt_chg=(a["target_current"] / a["target_prior"] - 1).clip(-0.5, 0.5))
    return (a.groupby(["symbol", "target"])
            .agg(an_n=("an_n", "sum"), an_up=("an_up", "sum"), an_down=("an_down", "sum"),
                 an_tgt_chg=("an_tgt_chg", "mean")).reset_index())


# -----------------------------------------------------------------------------
# 기능  : 뉴스를 (종목, 대상일)로 집계. 정규화한 제목 해시로 같은 창 안의 중복을 제거
# 수식  : news_n   = 중복 제거 후 기사 수 (없으면 0)
#         news_abn = log(1 + news_n) − log(1 + 직전 20개 창의 news_n 평균)
#         news_missing = 1[최근 21개 창의 기사 합 = 0]  (커버리지 없음 표시)
# input : n  뉴스 (known_at, symbols, title, tone),  targets,  universe
# output: DataFrame[symbol, target, news_*, tone_*]
# -----------------------------------------------------------------------------
def _news_block(n, targets, universe):
    n = n.assign(symbol=n["symbols"].fillna("").str.split(",")).explode("symbol")
    n = n[n["symbol"].isin(set(universe))]
    n = n.assign(target=map_to_target(n["known_at"], targets)).dropna(subset=["target"])
    norm = n["title"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    n = n.assign(h=pd.util.hash_pandas_object(norm, index=False).to_numpy())
    u = n.drop_duplicates(["symbol", "target", "h"])
    key = [u["symbol"], u["target"]]
    agg = u.groupby(key)["tone"].agg(news_n="size", tone_mean="mean", tone_min="min", tone_max="max")
    agg["tone_neg_tail"] = (u["tone"] < -TONE_TAIL).groupby(key).sum()

    grid = pd.MultiIndex.from_product([sorted(universe), np.sort(targets["target"].unique())],
                                      names=["symbol", "target"])
    out = agg.reindex(grid)
    out["news_n"] = out["news_n"].fillna(0.0)
    out["tone_neg_tail"] = out["tone_neg_tail"].fillna(0.0)
    g = out["news_n"].groupby(level="symbol")
    base = g.transform(lambda s: s.shift().rolling(NEWS_WINDOW, min_periods=NEWS_WINDOW).mean())
    out["news_abn"] = np.log1p(out["news_n"]) - np.log1p(base)
    out["news_missing"] = (g.transform(lambda s: s.rolling(NEWS_WINDOW + 1, min_periods=NEWS_WINDOW + 1).sum()) == 0).astype(float)
    return out.reset_index()


# -----------------------------------------------------------------------------
# 기능  : 원천 표들로 (종목 × 대상일) feature 표를 만듦. 학습(전체 표)과 predict(하루치 표)가 이 함수를 같이 씀
# input : tables    {"daily", "price", "earnings"?, "analyst"?, "news"?}  (known_at < cutoff 로 이미 잘린 표)
#         targets   DataFrame[date, target]  기준일·대상일 쌍 (연속 거래일)
#         universe  그날 예측 대상 종목 목록. 종목 교차 feature 는 이 안에서만 계산
# output: DataFrame[symbol, date, target, close_d0, gap_fallback, gap_missing, <FEATURE_GROUPS 열>]
# 수식  : gap_pct   = (대상일 08:00 pre 봉 종가 / 기준일 종가 − 1) × 100
#                     pre 봉이 없으면 기준일 post 마지막 종가로 대체(gap_fallback=1), 그것도 없으면 NaN(gap_missing=1)
#         gap_rank  = 그날 universe 안 gap_pct 순위 (0~1)
#         zgap      = clip(gap_pct / max(vol20, 0.5), ±10)
#         mkt_gap   = 자기 제외 universe gap_pct 중앙값,  resid_gap = gap_pct − beta60 × mkt_gap
#         reg_*     = 그날 universe 의 median vol20, median |gap|, std gap, median gap, mean 1[|gap| ≥ 1%]
#         an_gap_agree = sign(an_up − an_down) × sign(gap_pct)
# -----------------------------------------------------------------------------
def build_features(tables, targets, universe):
    uni = set(universe)
    targets = targets.assign(date=targets["date"].astype(NS), target=targets["target"].astype(NS))
    daily = tables["daily"]
    blk = _daily_block(daily[daily["symbol"].isin(uni)])
    x = targets[["date", "target"]].merge(blk, on="date")

    pre, post = _extended_hours(tables["price"][tables["price"]["symbol"].isin(uni)])
    pre_v = pd.Series(pre.reindex(pd.MultiIndex.from_arrays([x["symbol"], x["target"]])).to_numpy(), index=x.index)
    post_v = pd.Series(post.reindex(pd.MultiIndex.from_arrays([x["symbol"], x["date"]])).to_numpy(), index=x.index)
    px = pre_v.fillna(post_v)
    x["gap_fallback"] = (pre_v.isna() & post_v.notna()).astype(float)
    x["gap_missing"] = px.isna().astype(float)
    x["gap_pct"] = (px / x["close_d0"] - 1) * 100

    x["gap_rank"] = x.groupby("target")["gap_pct"].rank(pct=True)
    x["zgap"] = (x["gap_pct"] / x["vol20"].clip(lower=VOL_FLOOR)).clip(-ZGAP_CLIP, ZGAP_CLIP)
    x["mkt_gap"] = loo_median(x["gap_pct"], x["target"])
    x["resid_gap"] = x["gap_pct"] - x["beta60"] * x["mkt_gap"]

    a = x.assign(_abs=x["gap_pct"].abs(), _big=(x["gap_pct"].abs() >= BIG_GAP).astype(float).where(x["gap_pct"].notna()))
    reg = a.groupby("target").agg(reg_med_vol20=("vol20", "median"), reg_med_absgap=("_abs", "median"),
                                  reg_gap_disp=("gap_pct", "std"), reg_mkt_gap=("gap_pct", "median"),
                                  reg_big_ratio=("_big", "mean"))
    x = x.merge(reg.reset_index(), on="target", how="left")

    key = ["symbol", "target"]
    ev, an = tables.get("earnings"), tables.get("analyst")
    if ev is not None and len(ev):
        x = x.merge(_earnings_block(ev[ev["symbol"].isin(uni)], targets), on=key, how="left")
    if an is not None and len(an):
        x = x.merge(_analyst_block(an[an["symbol"].isin(uni)], targets), on=key, how="left")
    if tables.get("news") is not None:
        x = x.merge(_news_block(tables["news"], targets, universe), on=key, how="left")
    for c in DEPLOYABLE - set(x.columns):
        x[c] = np.nan
    for c in ("earn", "an_n", "an_up", "an_down"):
        x[c] = x[c].fillna(0.0)
    x["an_gap_agree"] = (np.sign(x["an_up"] - x["an_down"]) * np.sign(x["gap_pct"])).fillna(0.0)

    cols = KEY_COLS + ["close_d0", "gap_fallback", "gap_missing"] + [c for g in FEATURE_GROUPS.values() for c in g]
    return x[cols].sort_values(["target", "symbol"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 기능  : Day 하나에서 build_features 에 넣을 원천 표를 day API 로 꺼냄 (cutoff 는 API 가 보장)
# input : day     src.data.Day
#         groups  필요한 feature 묶음 이름 집합 (필요한 표만 읽어 추론 시간을 줄임)
# output: (tables dict, targets DataFrame)
# -----------------------------------------------------------------------------
def day_tables(day, groups):
    need = set().union(*(GROUP_TABLES[g] for g in groups)) | {"daily", "price"}
    t = {"daily": day.daily(days=HIST_DAYS, columns=["symbol", "date_et", "high", "low", "close", "prev_close", "ret"]),
         "price": day.price(since=day.date, columns=["symbol", "datetime", "close", "session"])}
    since = day.date + CUTOFF
    if "earnings" in need:
        t["earnings"] = day.earnings(since=since)
    if "analyst" in need:
        t["analyst"] = day.analyst(since=since)
    if "news" in need:
        t["news"] = day.news(days=NEWS_HIST_DAYS, columns=["symbols", "title", "tone"])
    dates = sorted(pd.to_datetime(t["daily"]["date_et"]).dt.normalize().unique())
    dates = [d for d in dates if d <= day.date]
    targets = pd.DataFrame({"date": dates, "target": dates[1:] + [day.target]})
    return t, targets


# -----------------------------------------------------------------------------
# 기능  : Day 하나의 feature 표 (그날 day.symbols 를 universe 로 씀)
# input : day, groups
# output: build_features 결과 중 대상일 = day.target 인 행
# -----------------------------------------------------------------------------
def day_features(day, groups):
    tables, targets = day_tables(day, groups)
    x = build_features(tables, targets, day.symbols)
    return x[x["target"] == pd.Timestamp(day.target)].reset_index(drop=True)


# =============================================================================
# 3. 점수
# =============================================================================

# -----------------------------------------------------------------------------
# 기능  : 과제 score 의 빠른 버전 (src.data.score 와 같은 값, 결정 규칙 탐색용)
# 수식  : score = 1 − Σ w·O / Σ w·E,  E_ij = n_i · m_j / N,  w_ij = (i−j)² (i−2)²
# input : y, p  정답·예측 label 배열 (0~4 정수)
# output: float (분모가 0 이면 NaN)
# -----------------------------------------------------------------------------
def fast_score(y, p):
    O = np.bincount(np.asarray(y, int) * 5 + np.asarray(p, int), minlength=25).reshape(5, 5).astype(float)
    n = O.sum()
    if n == 0:
        return float("nan")
    den = (WEIGHT * np.outer(O.sum(1), O.sum(0)) / n).sum()
    return float(1 - (WEIGHT * O).sum() / den) if den > 0 else float("nan")


# -----------------------------------------------------------------------------
# 기능  : 결정 규칙·regime 을 고를 때의 목적 함수 (Domain-robust 선택 원칙)
# 수식  : pooled   score(tune 전체)
#         mean     mean_d score_d          (d = tune 구간을 시간순으로 n_chunks 개로 나눈 조각)
#         mean_sd  mean_d score_d − λ · sd_d score_d
#         min      min_d score_d
# input : spec  {"criterion", "lam", "n_chunks"},  dates  tune 행의 대상일
# output: 호출 가능 객체  obj(y, p) → float (클수록 좋음)
# -----------------------------------------------------------------------------
class Objective:
    def __init__(self, spec, dates):
        spec = spec or {}
        self.criterion = spec.get("criterion", "pooled")
        self.lam = float(spec.get("lam", 1.0))
        n_chunks = int(spec.get("n_chunks", 3))
        u = np.sort(pd.unique(pd.Series(dates).to_numpy(NS)))
        edges = u[np.linspace(0, len(u), n_chunks + 1).astype(int)[1:-1]] if len(u) else []
        self.chunk = np.searchsorted(np.asarray(edges, dtype=NS), pd.Series(dates).to_numpy(NS), side="right")

    def __call__(self, y, p):
        if self.criterion == "pooled":
            return fast_score(y, p)
        s = np.array([fast_score(y[self.chunk == c], p[self.chunk == c]) for c in np.unique(self.chunk)])
        if self.criterion == "mean":
            return float(np.nanmean(s))
        if self.criterion == "mean_sd":
            return float(np.nanmean(s) - self.lam * np.nanstd(s))
        if self.criterion == "min":
            return float(np.nanmin(s))
        raise ValueError(f"criterion 은 pooled / mean / mean_sd / min 중 하나: {self.criterion}")


# =============================================================================
# 4. 신호 head
#    공통 인터페이스  fit(X, y_ret, y_label, seed) → self,  predict(X) → 신호
#    learnable=False 인 신호는 학습이 없어 tune 구간 전체로 결정 규칙만 고름
# =============================================================================

def _hgb_params(params, seed):
    p = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
             l2_regularization=1.0, early_stopping=False, random_state=seed)
    p.update(params or {})
    return p


# -----------------------------------------------------------------------------
# 기능  : 학습 없는 신호. 열 하나를 그대로 신호로 씀 (기본 gap_pct)
# output: s = X[col]  (%, NaN 이면 결정 규칙에서 보합)
# -----------------------------------------------------------------------------
class ColumnSignal:
    learnable = False
    kind = "scalar"

    def __init__(self, features=None, params=None):
        self.col = (params or {}).get("col", "gap_pct")
        self.features = [self.col]

    def fit(self, X, y_ret, y_label, seed=0):
        return self

    def predict(self, X):
        return X[self.col].to_numpy(float)

    def state(self):
        return {}

    def load(self, st):
        return self


# -----------------------------------------------------------------------------
# 기능  : 당일 수익률 회귀 (B3)
# input : X[features],  목표 y = clip(ret_pct, ±15)
# loss  : absolute_error (중앙값 회귀, 급등락 이상치에 강건)
# output: s = ŷ (%)
# -----------------------------------------------------------------------------
class RegressSignal:
    learnable = True
    kind = "scalar"

    def __init__(self, features, params=None):
        self.features, self.params, self.model = list(features), dict(params or {}), None

    def fit(self, X, y_ret, y_label, seed=0):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(loss="absolute_error", **_hgb_params(self.params, seed))
        self.model.fit(X[self.features], np.clip(y_ret, -RET_CLIP, RET_CLIP))
        return self

    def predict(self, X):
        return self.model.predict(X[self.features])

    def state(self):
        return {"model": self.model}

    def load(self, st):
        self.model = st["model"]
        return self


# -----------------------------------------------------------------------------
# 기능  : 5-class 직접 분류 (B4 / D0)
# input : X[features],  목표 label 0~4
# loss  : multinomial log loss
# output: s = Σ_c P(c) · center_c   (center = −4, −1.5, 0, 1.5, 4 %) → 결정 규칙이 score 기준으로 경계를 고름
# -----------------------------------------------------------------------------
class Classify5Signal:
    learnable = True
    kind = "scalar"

    def __init__(self, features, params=None):
        self.features, self.params, self.model = list(features), dict(params or {}), None

    def fit(self, X, y_ret, y_label, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(**_hgb_params(self.params, seed))
        self.model.fit(X[self.features], y_label)
        return self

    def proba(self, X):
        P = np.zeros((len(X), 5))
        P[:, self.model.classes_.astype(int)] = self.model.predict_proba(X[self.features])
        return P

    def predict(self, X):
        return self.proba(X) @ CLASS_CENTERS

    def state(self):
        return {"model": self.model}

    def load(self, st):
        self.model = st["model"]
        return self


# -----------------------------------------------------------------------------
# 기능  : Direction + Magnitude 분해 (D1 / D2)
# 정의  : Move    = 1[|r| ≥ 0.6%]                 (전체 표본으로 학습)
#         Extreme = 1[|r| ≥ 2.5%]                 (Move 표본만으로 학습)
#         Up      = 1[r > 0]                      (Move 표본만으로 학습, 보합에는 방향을 강제하지 않음)
#         magnitude="ordinal"    Move head + Extreme head (이진 2개)  → D2
#         magnitude="multiclass" flat/normal/extreme 3-class 하나     → D1
# loss  : 각 head 의 log loss
# output: (n, 3) 배열 [P(move), P(extreme | move), P(up | move)]
# -----------------------------------------------------------------------------
class DecompSignal:
    learnable = True
    kind = "decomp"

    def __init__(self, features, params=None):
        params = dict(params or {})
        self.magnitude = params.pop("magnitude", "ordinal")
        if self.magnitude not in ("ordinal", "multiclass"):
            raise ValueError(f"magnitude 는 ordinal / multiclass 중 하나: {self.magnitude}")
        self.features, self.params, self.heads = list(features), params, {}

    def fit(self, X, y_ret, y_label, seed=0):
        from sklearn.ensemble import HistGradientBoostingClassifier
        mk = lambda: HistGradientBoostingClassifier(**_hgb_params(self.params, seed))
        y = np.asarray(y_label, int)
        move = y != FLAT
        ext = np.isin(y, (0, 4))
        Xf = X[self.features]
        if self.magnitude == "ordinal":
            self.heads["move"] = mk().fit(Xf, move.astype(int))
            self.heads["ext"] = mk().fit(Xf[move], ext[move].astype(int))
        else:
            self.heads["mag3"] = mk().fit(Xf, np.where(ext, 2, np.where(move, 1, 0)))
        self.heads["up"] = mk().fit(Xf[move], (np.asarray(y_ret)[move] > 0).astype(int))
        return self

    @staticmethod
    def _p(model, X, cls):
        P = model.predict_proba(X)
        idx = list(model.classes_).index(cls) if cls in model.classes_ else None
        return P[:, idx] if idx is not None else np.zeros(len(X))

    def predict(self, X):
        Xf = X[self.features]
        if self.magnitude == "ordinal":
            p_move, p_ext = self._p(self.heads["move"], Xf, 1), self._p(self.heads["ext"], Xf, 1)
        else:
            p0, p1, p2 = (self._p(self.heads["mag3"], Xf, c) for c in (0, 1, 2))
            p_move = 1 - p0
            p_ext = np.divide(p2, p1 + p2, out=np.zeros(len(Xf)), where=(p1 + p2) > 0)
        return np.column_stack([p_move, p_ext, self._p(self.heads["up"], Xf, 1)])

    def state(self):
        return {"heads": self.heads, "magnitude": self.magnitude}

    def load(self, st):
        self.heads, self.magnitude = st["heads"], st["magnitude"]
        return self


SIGNALS = {"gap": ColumnSignal, "column": ColumnSignal, "regress": RegressSignal,
           "classify5": Classify5Signal, "decomp": DecompSignal}


# =============================================================================
# 5. 결정 규칙
#    스칼라 신호용 규칙은 모두 경계 4개 b0 < b1 < b2 < b3 로 바뀜
#        label = #{b_k ≤ s}   (s = b 이면 위 구간. label_of 의 왼쪽 포함과 같음),  s = NaN 이면 보합
#    공통 인터페이스  fit(s, y, obj) → self,  predict(s) → label,  params() → dict
# =============================================================================

def _bounds_predict(s, b):
    s = np.asarray(s, float)
    out = np.searchsorted(np.asarray(b, float), np.nan_to_num(s, nan=0.0), side="right")
    out[np.isnan(s)] = FLAT
    return out.astype(int)


class _BoundsRule:
    """스칼라 신호 규칙의 공통 부분. 하위 클래스는 _search 만 구현함."""

    def __init__(self, params=None):
        self.cfg, self.b = dict(params or {}), None

    def fit(self, s, y, obj):
        self.b = np.asarray(self._search(np.asarray(s, float), np.asarray(y, int), obj), float)
        return self

    def predict(self, s):
        return _bounds_predict(s, self.b)

    def params(self):
        return {"bounds": [float(v) for v in self.b]}

    def state(self):
        return {"b": self.b, "cfg": self.cfg}

    def load(self, st):
        self.b, self.cfg = np.asarray(st["b"], float), st["cfg"]
        return self


def _k_grid(cfg):
    lo, hi, step = cfg.get("k_grid", (0.5, 8.0, 0.25))
    return np.round(np.arange(lo, hi + 1e-9, step), 4)


def _k_bounds(k_up, k_dn):
    return [-2.5 / k_dn, -0.6 / k_dn, 0.6 / k_up, 2.5 / k_up]


# -----------------------------------------------------------------------------
# 기능  : 전부 보합 (B0)
# output: label = 2
# -----------------------------------------------------------------------------
class FlatRule(_BoundsRule):
    def _search(self, s, y, obj):
        return [-np.inf, -np.inf, np.inf, np.inf]


# -----------------------------------------------------------------------------
# 기능  : 과제 구간 그대로 (B1)
# 수식  : label = label_of(s)  → 경계 (−2.5, −0.6, 0.6, 2.5)
# -----------------------------------------------------------------------------
class IdentityRule(_BoundsRule):
    def _search(self, s, y, obj):
        return [-2.5, -0.6, 0.6, 2.5]


# -----------------------------------------------------------------------------
# 기능  : 배율 1개 (B2)
# 수식  : label = label_of(k × s)  → 경계 (−2.5/k, −0.6/k, 0.6/k, 2.5/k)
# param : k_grid = (시작, 끝, 간격)  기본 (0.5, 8.0, 0.25)
# -----------------------------------------------------------------------------
class KRule(_BoundsRule):
    def _search(self, s, y, obj):
        best = max(_k_grid(self.cfg), key=lambda k: obj(y, _bounds_predict(s, _k_bounds(k, k))))
        self.k = float(best)
        return _k_bounds(best, best)

    def params(self):
        return {**super().params(), "k": self.k}

    def state(self):
        return {**super().state(), "k": self.k}

    def load(self, st):
        self.k = st["k"]
        return super().load(st)


# -----------------------------------------------------------------------------
# 기능  : 대칭 경계 2개 (B2-Sym)
# 수식  : |s| < a → 2,  a ≤ |s| < b → 1/3,  |s| ≥ b → 0/4    → 경계 (−b, −a, a, b)
# param : q_grid  a, b 후보로 쓸 |s| 의 분위수 목록 (scale 에 무관하게 탐색)
# -----------------------------------------------------------------------------
class Sym2Rule(_BoundsRule):
    def _search(self, s, y, obj):
        q = self.cfg.get("q_grid", list(np.round(np.arange(0.05, 0.99, 0.03), 3)))
        cand = np.unique(np.nanquantile(np.abs(s), q))
        best, arg = -np.inf, (0.6, 2.5)
        for i, a in enumerate(cand):
            for b in cand[i + 1:]:
                v = obj(y, _bounds_predict(s, [-b, -a, a, b]))
                if v > best:
                    best, arg = v, (a, b)
        return [-arg[1], -arg[0], arg[0], arg[1]]


# -----------------------------------------------------------------------------
# 기능  : 상승·하락 배율 분리 (B2-Asym)
# 수식  : s ≥ 0 이면 label_of(k_up × s),  s < 0 이면 label_of(k_dn × s)
# -----------------------------------------------------------------------------
class AsymKRule(_BoundsRule):
    def _search(self, s, y, obj):
        g = _k_grid(self.cfg)
        best, arg = -np.inf, (1.0, 1.0)
        for ku in g:
            for kd in g:
                v = obj(y, _bounds_predict(s, _k_bounds(ku, kd)))
                if v > best:
                    best, arg = v, (ku, kd)
        self.k_up, self.k_dn = map(float, arg)
        return _k_bounds(*arg)

    def params(self):
        return {**super().params(), "k_up": self.k_up, "k_dn": self.k_dn}

    def state(self):
        return {**super().state(), "k_up": self.k_up, "k_dn": self.k_dn}

    def load(self, st):
        self.k_up, self.k_dn = st["k_up"], st["k_dn"]
        return super().load(st)


# -----------------------------------------------------------------------------
# 기능  : 경계 4개 직접 최적화 (B2-4T, D0)
# 수식  : label = #{b_k ≤ s},  b0 < b1 < b2 < b3
# 방법  : AsymK 해에서 출발해, 경계 하나씩 s 의 분위수 후보 위에서 바꾸는 좌표 하강을 n_pass 번
# param : n_q (후보 분위수 개수, 기본 80),  n_pass (기본 3)
# -----------------------------------------------------------------------------
class FourTRule(_BoundsRule):
    def _search(self, s, y, obj):
        b = list(AsymKRule(self.cfg)._search(s, y, obj))
        cand = np.unique(np.nanquantile(s, np.linspace(0.005, 0.995, int(self.cfg.get("n_q", 80)))))
        for _ in range(int(self.cfg.get("n_pass", 3))):
            for k in range(4):
                lo = b[k - 1] if k else -np.inf
                hi = b[k + 1] if k < 3 else np.inf
                opts = [c for c in cand if lo < c < hi] + [b[k]]
                b[k] = max(opts, key=lambda c: obj(y, _bounds_predict(s, b[:k] + [c] + b[k + 1:])))
        return b


# -----------------------------------------------------------------------------
# 기능  : 분해 구조의 결정 규칙 (D1 / D2 / D2-C)
# 수식  : P(move) < t_move                      → 2
#         P(ext|move) ≥ t_ext                   → 0 / 4 (방향 = 1[P(up) ≥ 0.5])
#         그 외                                  → 1 / 3
#         conf_gate=True 이면 급등락이라도 |P(up) − 0.5| < c 일 때 1 / 3 으로 낮춤 (D2-C)
# param : q_grid (t_move, t_ext 후보 분위수),  c_grid (conf 후보),  conf_gate
# 입력  : s = DecompSignal.predict() 의 (n, 3) 배열
# -----------------------------------------------------------------------------
class DecompRule:
    def __init__(self, params=None):
        self.cfg = dict(params or {})
        self.t_move, self.t_ext, self.c = 0.5, 0.5, 0.0

    @staticmethod
    def _apply(s, t_move, t_ext, c):
        p_move, p_ext, p_up = s[:, 0], s[:, 1], s[:, 2]
        up = p_up >= 0.5
        ext = (p_ext >= t_ext) & (np.abs(p_up - 0.5) >= c)
        lab = np.where(up, np.where(ext, 4, 3), np.where(ext, 0, 1))
        return np.where(p_move < t_move, FLAT, lab).astype(int)

    def fit(self, s, y, obj):
        q = self.cfg.get("q_grid", list(np.round(np.arange(0.05, 0.96, 0.05), 2)))
        tm = np.unique(np.quantile(s[:, 0], q))
        te = np.unique(np.quantile(s[:, 1], q))
        cs = self.cfg.get("c_grid", [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]) if self.cfg.get("conf_gate") else [0.0]
        best = -np.inf
        for a in tm:
            for b in te:
                for c in cs:
                    v = obj(y, self._apply(s, a, b, c))
                    if v > best:
                        best, (self.t_move, self.t_ext, self.c) = v, (float(a), float(b), float(c))
        return self

    def predict(self, s):
        return self._apply(np.asarray(s, float), self.t_move, self.t_ext, self.c)

    def params(self):
        return {"t_move": self.t_move, "t_ext": self.t_ext, "c": self.c}

    def state(self):
        return {"cfg": self.cfg, **self.params()}

    def load(self, st):
        self.cfg, self.t_move, self.t_ext, self.c = st["cfg"], st["t_move"], st["t_ext"], st["c"]
        return self


DECISIONS = {"flat": FlatRule, "identity": IdentityRule, "k": KRule, "sym2": Sym2Rule,
             "asym": AsymKRule, "4t": FourTRule, "decomp": DecompRule}


# =============================================================================
# 6. regime 보정 (Test-Time Regime Calibration)
#    스칼라 신호의 크기만 바꿈:  s' = s × f(r)   → 부호(방향)는 그대로, 크기 판단만 regime 에 따라 바뀜
#    r 은 그날 universe 로 계산한 reg_* 열. 정답은 쓰지 않음 (학습 구간에서 f 만 정함)
#    공통 인터페이스  fit(s, r, y, decision, obj) → self,  apply(s, r) → s'
# =============================================================================

class NoRegime:
    """T0. f = 1"""

    def __init__(self, params=None):
        self.var = None

    def fit(self, s, r, y, decision, obj):
        return self

    def apply(self, s, r):
        return s

    def params(self):
        return {}

    def state(self):
        return {}

    def load(self, st):
        return self


# -----------------------------------------------------------------------------
# 기능  : T1. regime 변수 삼분위별 배율
# 수식  : f = f_low (r < e1), 1 (e1 ≤ r < e2), f_high (r ≥ e2),  e1, e2 = tune 구간 날짜별 r 의 1/3, 2/3 분위
# param : var (regime 열 이름, 기본 reg_med_vol20),  f_grid (기본 0.6 ~ 1.6, 0.1 간격)
# -----------------------------------------------------------------------------
class TercileRegime:
    def __init__(self, params=None):
        p = dict(params or {})
        self.var = p.get("var", "reg_med_vol20")
        self.grid = np.round(np.arange(*p.get("f_grid", (0.6, 1.61, 0.1))), 3)
        self.edges, self.f = None, (1.0, 1.0, 1.0)

    def _factor(self, r):
        t = np.digitize(np.nan_to_num(r, nan=np.nanmedian(r) if np.isfinite(r).any() else 0.0), self.edges)
        return np.asarray(self.f)[t]

    def fit(self, s, r, y, decision, obj):
        self.edges = np.nanquantile(np.unique(r[np.isfinite(r)]), [1 / 3, 2 / 3])
        best, keep = -np.inf, self.f
        for lo in self.grid:
            for hi in self.grid:
                self.f = (lo, 1.0, hi)
                v = obj(y, decision.predict(self.apply(s, r)))
                if v > best:
                    best, keep = v, self.f
        self.f = tuple(float(v) for v in keep)
        return self

    def apply(self, s, r):
        return s * self._factor(r)

    def params(self):
        return {"var": self.var, "edges": [float(e) for e in self.edges], "f": list(self.f)}

    def state(self):
        return {"var": self.var, "edges": self.edges, "f": self.f, "grid": self.grid}

    def load(self, st):
        self.var, self.edges, self.f, self.grid = st["var"], st["edges"], st["f"], st["grid"]
        return self


# -----------------------------------------------------------------------------
# 기능  : T2. regime 값에 따른 연속 배율
# 수식  : f = clip(exp(α · (r − med) / iqr), 0.5, 2),   med·iqr 는 tune 구간 날짜별 r 로 계산
# param : var,  a_grid (기본 −1.0 ~ 1.0, 0.1 간격)
# -----------------------------------------------------------------------------
class LinearRegime:
    def __init__(self, params=None):
        p = dict(params or {})
        self.var = p.get("var", "reg_med_vol20")
        self.grid = np.round(np.arange(*p.get("a_grid", (-1.0, 1.01, 0.1))), 3)
        self.med, self.iqr, self.alpha = 0.0, 1.0, 0.0

    def apply(self, s, r):
        z = np.nan_to_num((r - self.med) / self.iqr, nan=0.0)
        return s * np.clip(np.exp(self.alpha * z), 0.5, 2.0)

    def fit(self, s, r, y, decision, obj):
        u = np.unique(r[np.isfinite(r)])
        self.med = float(np.median(u))
        q1, q3 = np.quantile(u, [0.25, 0.75])
        self.iqr = float(q3 - q1) or 1.0
        self.alpha = float(max(self.grid, key=lambda a: obj(y, decision.predict(self._with(a).apply(s, r)))))
        return self

    def _with(self, a):
        c = LinearRegime({"var": self.var})
        c.med, c.iqr, c.alpha = self.med, self.iqr, a
        return c

    def params(self):
        return {"var": self.var, "med": self.med, "iqr": self.iqr, "alpha": self.alpha}

    def state(self):
        return self.params()

    def load(self, st):
        self.var, self.med, self.iqr, self.alpha = st["var"], st["med"], st["iqr"], st["alpha"]
        return self


REGIMES = {"none": NoRegime, "tercile": TercileRegime, "linear": LinearRegime}


# =============================================================================
# 7. Pipeline  (spec 하나 = 실험 후보 하나 = 제출 모델 하나)
# =============================================================================

# -----------------------------------------------------------------------------
# 기능  : spec 검사. 이름이 registry 에 없거나 조합이 맞지 않으면 바로 멈춤
# input : spec dict
# output: 기본값을 채운 spec 사본
# -----------------------------------------------------------------------------
def validate_spec(spec):
    s = {"signal": {"type": "gap"}, "decision": {"type": "k"}, "regime": {"type": "none"},
         "selection": {"criterion": "pooled"}, "tune_frac": 0.2, "tune_on": "auto", "seed": 0, **spec}
    for part, reg in (("signal", SIGNALS), ("decision", DECISIONS), ("regime", REGIMES)):
        if s[part].get("type") not in reg:
            raise ValueError(f"{s.get('name')}: {part}.type '{s[part].get('type')}' 없음. 가능: {sorted(reg)}")
    decomp_sig = SIGNALS[s["signal"]["type"]].kind == "decomp"
    if decomp_sig != (s["decision"]["type"] == "decomp"):
        raise ValueError(f"{s.get('name')}: signal 'decomp' 와 decision 'decomp' 는 함께 써야 함")
    if decomp_sig and s["regime"]["type"] != "none":
        raise ValueError(f"{s.get('name')}: regime 보정은 스칼라 신호에만 씀")
    return s


class Pipeline:
    """신호 → 결정 규칙 → regime 보정을 묶은 모델 하나.

    fit:  학습 구간을 시간순으로 [fit | tune] 으로 나눔 (tune = 마지막 tune_frac 비율의 대상일)
          learnable 신호는 fit 부분으로 학습, 결정 규칙과 regime 은 tune 부분으로 고름
          학습 없는 신호(gap)는 tune_on="auto" 일 때 학습 구간 전체로 결정 규칙을 고름
    """

    # -------------------------------------------------------------------------
    # input : spec  {"name", "signal": {"type", "features", "params"}, "decision": {"type", "params"},
    #                "regime": {"type", "params"}, "selection": {...}, "tune_frac", "tune_on", "seed"}
    #         extra_groups  연구용 feature 묶음 {이름: [열]}
    # -------------------------------------------------------------------------
    def __init__(self, spec, extra_groups=None):
        self.spec = validate_spec(spec)
        sig = self.spec["signal"]
        feats = resolve_features(sig.get("features", ["core"]), extra_groups)
        self.signal = SIGNALS[sig["type"]](feats, sig.get("params"))
        self.decision = DECISIONS[self.spec["decision"]["type"]](self.spec["decision"].get("params"))
        self.regime = REGIMES[self.spec["regime"]["type"]](self.spec["regime"].get("params"))

    @property
    def features(self):
        return list(self.signal.features) + ([self.regime.var] if self.regime.var else [])

    @property
    def groups(self):
        return groups_of(self.features)

    @property
    def deployable(self):
        return set(self.features) <= DEPLOYABLE

    def _split(self, X):
        t = X["target"].to_numpy(NS)
        if not self.signal.learnable and self.spec["tune_on"] == "auto":
            return np.zeros(len(X), bool), np.ones(len(X), bool)
        edge = np.quantile(np.unique(t).astype("int64"), 1 - float(self.spec["tune_frac"]))
        tune = t.astype("int64") > edge
        return ~tune, tune

    # -------------------------------------------------------------------------
    # 기능  : 학습. X 는 feature + 정답(label, ret_pct) 을 가진 패널
    # input : X  DataFrame[target, label, ret_pct, <features>]
    # output: self
    # -------------------------------------------------------------------------
    def fit(self, X):
        X = X.sort_values(["target", "symbol"]).reset_index(drop=True)
        fit_m, tune_m = self._split(X)
        seed = int(self.spec["seed"])
        if self.signal.learnable:
            F = X[fit_m]
            self.signal.fit(F, F["ret_pct"].to_numpy(float), F["label"].to_numpy(int), seed)
        T = X[tune_m]
        y = T["label"].to_numpy(int)
        obj = Objective(self.spec["selection"], T["target"])
        s = self.signal.predict(T)
        self.decision.fit(s, y, obj)
        if self.regime.var:
            self.regime.fit(s, T[self.regime.var].to_numpy(float), y, self.decision, obj)
        self.fit_info = {"n_fit": int(fit_m.sum()), "n_tune": int(tune_m.sum()),
                         "tune_from": str(T["target"].min().date()) if len(T) else None}
        return self

    # -------------------------------------------------------------------------
    # input : X  feature 표
    # output: label 배열 (0~4 정수)
    # -------------------------------------------------------------------------
    def predict(self, X):
        s = self.signal.predict(X)
        if self.regime.var:
            s = self.regime.apply(s, X[self.regime.var].to_numpy(float))
        return self.decision.predict(s)

    def params(self):
        return {"decision": self.decision.params(), "regime": self.regime.params(),
                **getattr(self, "fit_info", {})}

    # -------------------------------------------------------------------------
    # 기능  : 저장·복원. 클래스가 아니라 spec 과 학습 결과(sklearn 객체·숫자)만 저장함
    # -------------------------------------------------------------------------
    def state(self):
        return {"format": ARTIFACT_FORMAT, "spec": self.spec, "signal": self.signal.state(),
                "decision": self.decision.state(), "regime": self.regime.state(),
                "params": self.params()}

    @classmethod
    def from_state(cls, st):
        if st.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {st.get('format')} != {ARTIFACT_FORMAT}")
        p = cls(st["spec"])
        p.signal.load(st["signal"])
        p.decision.load(st["decision"])
        p.regime.load(st["regime"])
        p.fit_info = {k: v for k, v in st.get("params", {}).items() if k.startswith(("n_", "tune"))}
        return p


# -----------------------------------------------------------------------------
# 기능  : artifact 가 없을 때 쓰는 기본 모델. 갭 규칙 label_of(DEFAULT_K × gap_pct)
# -----------------------------------------------------------------------------
def default_pipeline():
    p = Pipeline({"name": "default_gap_k", "signal": {"type": "gap"}, "decision": {"type": "k"}})
    p.decision.k, p.decision.b = DEFAULT_K, np.asarray(_k_bounds(DEFAULT_K, DEFAULT_K))
    return p


# =============================================================================
# 8. 제출 인터페이스
# =============================================================================

class Model:
    """Pipeline 을 감싸 day 하나를 예측함. feature 를 못 만든 종목은 보합으로 둠."""

    def __init__(self, pipeline=None):
        self.pipeline = pipeline or default_pipeline()

    # -------------------------------------------------------------------------
    # input : day  src.data.Day
    # output: DataFrame[symbol, label]  day.symbols 를 빠짐없이, 그것만
    # -------------------------------------------------------------------------
    def predict(self, day):
        out = pd.DataFrame({"symbol": day.symbols, "label": FLAT})
        try:
            x = day_features(day, self.pipeline.groups)
            if len(x):
                lab = pd.Series(self.pipeline.predict(x), index=x["symbol"].to_numpy())
                out["label"] = out["symbol"].map(lab).fillna(FLAT).astype(int)
        except Exception as e:      # 데이터 결손으로 예측 전체가 멈추지 않게 함
            print(f"[model] {day!r} 예측 실패, 전부 보합으로 대체: {type(e).__name__}: {e}", file=sys.stderr)
        return out


def load_model():
    """artifacts/pipeline.pkl 을 읽어 Model 을 돌려줌. 없으면 기본 갭 규칙."""
    if ARTIFACT.exists():
        with open(ARTIFACT, "rb") as f:
            return Model(Pipeline.from_state(pickle.load(f)))
    return Model()
