"""EDA 공용 함수. eda/ 노트북에서 import 해서 씀.

속도를 위해 day API를 하루씩 부르지 않고 parquet을 한 번에 읽은 뒤 벡터 연산으로
(종목 × 대상일) 패널을 만듦. 자르는 규칙은 src/data.py 와 같음.

    대상일 T 의 정보 = known_at < T 09:30 인 행   (Reddit은 created_et 기준)

check_against_day_api() 로 day API 결과와 같은지 표본을 뽑아 검증함.
무거운 결과(패널, 뉴스, Reddit 집계)는 eda/cache/ 에 parquet으로 저장해 두고 다시 씀.
패널은 lockbox(대상일 ≥ 2026-06-01) 행을 빼고 만듦 (exp/lockbox.py). 2026-10-05 이전 노트북 출력에는
lockbox 가 섞여 있었고, 그 이력은 exp/lockbox_ledger.json 에 기록되어 있음.
"""

import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import CUTOFF, LABELS, START, WEIGHT, Dataset, label_of, score  # noqa: E402
from exp.lockbox import LOCKBOX_START, dev_only  # noqa: E402  lockbox 행은 EDA 에서도 빼고 봄

DATA = ROOT / "dataset"
CACHE = ROOT / "eda" / "cache"
BIG = (0, 4)                       # 급하락, 급상승
NS = "datetime64[ns]"

# 일반 영어 단어와 겹쳐서 `$` 표기가 있을 때만 티커로 인정하는 것들
COMMON_WORDS = {"NOW", "COST", "WELL", "GILD", "COP", "BA", "DE", "GE", "KO",
                "MS", "PM", "CB", "V"}


# =============================================================================
# 캐시
# =============================================================================

# -----------------------------------------------------------------------------
# 역할  : 오래 걸리는 계산 결과를 eda/cache/<name>.parquet 에 저장하고 다시 씀
# Input : name    캐시 파일 이름
#         build   인자 없이 DataFrame 을 돌려주는 함수 (캐시가 없을 때만 실행)
#         rebuild True 면 캐시를 무시하고 다시 계산
# Output: DataFrame
# -----------------------------------------------------------------------------
def cached(name, build, rebuild=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.parquet"
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    t0 = time.time()
    df = build()
    df.to_parquet(path, index=False)
    print(f"[cache] {name}: {len(df):,}행, {time.time() - t0:.1f}초")
    return df


# =============================================================================
# 달력과 시점
# =============================================================================

# -----------------------------------------------------------------------------
# 역할  : 거래일 목록으로 (기준일, 대상일, cutoff) 표를 만듦
# Input : daily  일봉 (date_et 열)
# Output: DataFrame[date, target, cutoff]  기준일 / 다음 거래일 / 다음 거래일 09:30
# -----------------------------------------------------------------------------
def trading_calendar(daily):
    dates = np.sort(daily["date_et"].astype(NS).unique())
    cal = pd.DataFrame({"date": dates[:-1], "target": dates[1:]})
    cal["cutoff"] = cal["target"] + CUTOFF
    return cal


# -----------------------------------------------------------------------------
# 역할  : 시각(known_at 등)을 그 정보를 처음 쓸 수 있는 대상일로 바꿈
#         규칙은 known_at < 대상일 09:30. cutoff 와 정확히 같으면 다음 대상일로 감
# Input : ts   시각 Series
#         cal  trading_calendar() 결과
# Output: 대상일 Series (ts 와 같은 index). 달력 범위 밖이면 NaT
# -----------------------------------------------------------------------------
def map_to_target(ts, cal):
    cut = cal["cutoff"].to_numpy(NS)
    idx = np.searchsorted(cut, pd.Series(ts).to_numpy(NS), side="right")
    tgt = np.append(cal["target"].to_numpy(NS), np.datetime64("NaT"))
    out = tgt[np.minimum(idx, len(cut))]
    out[idx == 0] = np.datetime64("NaT")       # 첫 cutoff 보다 이른 과거 이벤트
    return pd.Series(out, index=pd.Series(ts).index)


# =============================================================================
# 가격 패널
# =============================================================================

# -----------------------------------------------------------------------------
# 역할  : 날짜별로 자기 자신을 뺀 나머지 종목의 중앙값(leave-one-out median)
#         변동성이 큰 종목 하나가 시장 신호를 끌고 가지 않게 함
# Input : values  값 Series
#         groups  같은 index 의 날짜 Series
# Output: values 와 같은 index 의 Series
# -----------------------------------------------------------------------------
def loo_median(values, groups):
    def _loo(a):
        n = len(a)
        if n < 3:
            return np.full(n, np.nan)
        m = np.tile(a, (n, 1))
        np.fill_diagonal(m, np.nan)
        return np.nanmedian(m, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return values.groupby(groups).transform(lambda s: _loo(s.to_numpy(float)))


# -----------------------------------------------------------------------------
# 역할  : 일봉에서 기준일 장 마감(16:00)까지 알 수 있는 종목별 특성을 계산함
#         모든 rolling 은 그날까지의 과거만 씀
# Input : daily  일봉 원본
# Output: DataFrame[symbol, date, close, ret, label, big, vol5, vol20, atr14,
#                   big_rate60, ret5, mkt_ret, beta60]
# -----------------------------------------------------------------------------
def daily_features(daily):
    d = (daily.assign(date=daily["date_et"].astype(NS))
         .sort_values(["symbol", "date"]).reset_index(drop=True))
    d["label"] = label_of(d["ret"] * 100)
    d["big"] = d["label"].isin(BIG).astype(float).where(d["ret"].notna())
    g = d.groupby("symbol", sort=False)

    def roll(col, n, how):
        return g[col].transform(lambda s: getattr(s.rolling(n, min_periods=n // 2), how)())

    d["vol5"] = roll("ret", 5, "std")
    d["vol20"] = roll("ret", 20, "std")
    d["big_rate60"] = roll("big", 60, "mean")
    d["ret5"] = g["close"].pct_change(5)

    tr = pd.concat([d["high"] - d["low"],
                    (d["high"] - d["prev_close"]).abs(),
                    (d["low"] - d["prev_close"]).abs()], axis=1).max(axis=1) / d["close"]
    d["atr14"] = tr.groupby(d["symbol"]).transform(lambda s: s.ewm(alpha=1 / 14, adjust=False).mean())

    # 시장 수익률(자기 제외 중앙값)에 대한 60일 rolling 베타
    d["mkt_ret"] = loo_median(d["ret"], d["date"])
    d["_xm"], d["_mm"] = d["ret"] * d["mkt_ret"], d["mkt_ret"] ** 2
    g = d.groupby("symbol", sort=False)
    ex, em, exm, emm = (roll(c, 60, "mean") for c in ("ret", "mkt_ret", "_xm", "_mm"))
    d["beta60"] = (exm - ex * em) / (emm - em ** 2)

    keep = ["symbol", "date", "close", "ret", "label", "big", "vol5", "vol20",
            "atr14", "big_rate60", "ret5", "mkt_ret", "beta60"]
    return d[keep]


# -----------------------------------------------------------------------------
# 역할  : 시간봉에서 하루 단위 시간외 가격을 뽑음
#         pre_last      그날 pre 봉 중 known_at < 그날 09:30 인 마지막 종가 (08:00 봉)
#         pre_last_leak known_at <= 09:30 까지 넣은 값 (09:00 봉 포함, 누수 비교용)
#         post_last     그날 post 봉의 마지막 종가
# Input : 없음 (dataset/price.parquet 을 읽음)
# Output: DataFrame[symbol, day, pre_last, pre_last_leak, post_last]
# -----------------------------------------------------------------------------
def extended_hours():
    p = pd.read_parquet(DATA / "price.parquet",
                        columns=["symbol", "datetime", "close", "session", "known_at"])
    p = p[p["session"] != "regular"].sort_values("datetime")
    p["day"] = p["datetime"].astype(NS).dt.normalize()
    cut = p["day"] + CUTOFF
    known = p["known_at"].astype(NS)
    pre = p["session"] == "pre"
    last = lambda m, name: p[m].groupby(["symbol", "day"])["close"].last().rename(name)
    return pd.concat([last(pre & (known < cut), "pre_last"),
                      last(pre & (known <= cut), "pre_last_leak"),
                      last(p["session"] == "post", "post_last")], axis=1).reset_index()


# -----------------------------------------------------------------------------
# 역할  : 분석의 기본 표인 (종목 × 대상일) 패널을 만듦
#         행 하나 = 기준일 장 마감 후 ~ 대상일 09:30 사이에 서서 대상일 label 을 맞히는 상황
# Input : 없음 (dataset/ 을 읽음)
# Output: DataFrame. 대상일 < LOCKBOX_START (2026-06-01) 인 DEV 행만. 주요 열
#         정답   ret_pct, label, M(급등락 여부), D(방향 부호), rest(갭 이후 나머지 로그수익률)
#         기준일 ret_d0, big_d0, vol5, vol20, atr14, big_rate60, ret5, beta60
#         시간외 gap, gap_leak, post_ret, pre_ret, abs_gap, zgap, gap_rank,
#                mkt_gap, resid_gap
# -----------------------------------------------------------------------------
def build_panel():
    daily = pd.read_parquet(DATA / "daily.parquet")
    f = daily_features(daily)
    cal = trading_calendar(daily)
    cal = cal[cal["date"] >= START]

    d0 = f.rename(columns={"ret": "ret_d0", "label": "label_d0", "big": "big_d0",
                           "close": "close_d0"}).drop(columns="mkt_ret")
    t = f[["symbol", "date", "ret", "label"]].rename(columns={"date": "target", "ret": "ret_t"})
    panel = (cal.merge(d0, on="date")
             .merge(t, on=["symbol", "target"])
             .dropna(subset=["ret_t"]))

    ext = extended_hours()
    panel = (panel.merge(ext[["symbol", "day", "pre_last", "pre_last_leak"]],
                         left_on=["symbol", "target"], right_on=["symbol", "day"], how="left")
             .drop(columns="day")
             .merge(ext[["symbol", "day", "post_last"]],
                    left_on=["symbol", "date"], right_on=["symbol", "day"], how="left")
             .drop(columns="day"))

    panel["ret_pct"] = panel["ret_t"] * 100
    panel["label"] = panel["label"].astype(int)
    panel["M"] = panel["label"].isin(BIG).astype(int)
    panel["D"] = np.sign(panel["ret_t"]).astype(int)

    panel["gap"] = panel["pre_last"] / panel["close_d0"] - 1
    panel["gap_leak"] = panel["pre_last_leak"] / panel["close_d0"] - 1
    panel["post_ret"] = panel["post_last"] / panel["close_d0"] - 1
    panel["pre_ret"] = panel["pre_last"] / panel["post_last"] - 1
    panel["gap_pct"] = panel["gap"] * 100
    panel["abs_gap"] = panel["gap"].abs()
    panel["zgap"] = panel["gap"] / panel["vol20"]
    panel["gap_rank"] = panel.groupby("target")["gap"].rank(pct=True)
    panel["mkt_gap"] = loo_median(panel["gap"], panel["target"])
    panel["resid_gap"] = panel["gap"] - panel["beta60"].fillna(1) * panel["mkt_gap"]
    panel["rest"] = np.log1p(panel["ret_t"]) - np.log1p(panel["gap"])
    return dev_only(panel).sort_values(["target", "symbol"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 역할  : 패널 값이 day API(src/data.py)로 꺼낸 값과 같은지 표본 검증함
#         벡터 연산으로 만든 패널에 미래 정보가 섞이지 않았는지 확인하는 장치
# Input : panel   build_panel() 결과
#         n_days  검사할 기준일 수
#         seed    난수 시드
# Output: DataFrame[date, n, label_mismatch, gap_max_abs_diff]  (0 이면 일치)
# -----------------------------------------------------------------------------
def check_against_day_api(panel, n_days=5, seed=0):
    ds = Dataset()
    dates = pd.Series(panel["date"].unique()).sample(n_days, random_state=seed).sort_values()
    rows = []
    for d in dates:
        day = ds.day(d)
        y = day.y.set_index("symbol")
        bars = day.price(since=day.target)                 # known_at in [대상일 00:00, cutoff)
        bars = bars[bars["session"] == "pre"].sort_values("datetime")
        pre_last = bars.groupby("symbol")["close"].last()
        close_d0 = day.daily(days=1).set_index("symbol")["close"]
        gap_api = (pre_last / close_d0 - 1).rename("gap_api")
        sub = panel[panel["date"] == d].set_index("symbol").join(gap_api).join(y["label"].rename("label_api"))
        rows.append({"date": d.date(), "n": len(sub),
                     "label_mismatch": int((sub["label"] != sub["label_api"].astype(int)).sum()),
                     "gap_max_abs_diff": float((sub["gap"] - sub["gap_api"]).abs().max())})
    return pd.DataFrame(rows)


# =============================================================================
# 이벤트: 실적, 애널리스트
# =============================================================================

# -----------------------------------------------------------------------------
# 역할  : 실적 발표를 반영 대상일로 붙이고 서프라이즈를 세 가지 형태로 만듦
#         surprise_pct 는 예상 EPS 가 0 근처면 폭발하므로 부호/절대차/절단값을 같이 씀
# Input : cal  trading_calendar() 결과
#         clip surprise_pct 절단 범위(%)
# Output: DataFrame[symbol, target, earn, surp_sign, surp_abs, surp_pct_w]
# -----------------------------------------------------------------------------
def earnings_features(cal, clip=50):
    e = pd.read_parquet(DATA / "earnings.parquet")
    e["target"] = map_to_target(e["known_at"], cal)
    e = e.dropna(subset=["target"]).sort_values("known_at")
    diff = e["eps_reported"] - e["eps_estimate"]
    e["earn"] = 1
    e["surp_sign"] = np.sign(diff)
    e["surp_abs"] = diff.abs()
    e["surp_pct_w"] = e["surprise_pct"].clip(-clip, clip)
    return (e.groupby(["symbol", "target"])[["earn", "surp_sign", "surp_abs", "surp_pct_w"]]
            .last().reset_index())


# -----------------------------------------------------------------------------
# 역할  : 애널리스트 등급 변경과 목표가 변경을 반영 대상일 단위로 집계함
# Input : cal  trading_calendar() 결과
# Output: DataFrame[symbol, target, an_n, an_up, an_down, an_init, an_tgt_chg]
# -----------------------------------------------------------------------------
def analyst_features(cal):
    a = pd.read_parquet(DATA / "analyst.parquet")
    a["target"] = map_to_target(a["known_at"], cal)
    a = a.dropna(subset=["target"])
    a = a.assign(an_n=1,
                 an_up=(a["action"] == "up").astype(int),
                 an_down=(a["action"] == "down").astype(int),
                 an_init=(a["action"] == "init").astype(int),
                 an_tgt_chg=(a["target_current"] / a["target_prior"] - 1).clip(-0.5, 0.5))
    return (a.groupby(["symbol", "target"])
            .agg(an_n=("an_n", "sum"), an_up=("an_up", "sum"), an_down=("an_down", "sum"),
                 an_init=("an_init", "sum"), an_tgt_chg=("an_tgt_chg", "mean"))
            .reset_index())


# =============================================================================
# 텍스트: 뉴스, Reddit
# =============================================================================

# -----------------------------------------------------------------------------
# 역할  : 뉴스 320만 건을 row group 단위로 읽어 (종목 × 대상일)로 집계함
#         제목을 정규화해 해시로 바꾼 뒤 같은 대상일 안의 중복(재전송, 신디케이션)을 제거함
#         문자열은 바로 버리고 숫자만 남겨 메모리를 아낌
# Input : cal      trading_calendar() 결과
#         symbols  집계할 종목 목록
# Output: DataFrame[symbol, target, news_raw, news_n, tone_mean, tone_min, tone_max,
#                   tone_std, tone_neg_tail, tone_pos_tail]
#         news_raw 는 중복 포함 건수, news_n 은 중복 제거 후 건수
# -----------------------------------------------------------------------------
def build_news(cal, symbols, tail=5.0):
    pf = pq.ParquetFile(DATA / "news.parquet")
    univ, parts = set(symbols), []
    first = cal["cutoff"].iloc[0] - pd.Timedelta(days=7)
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["known_at", "symbols", "title", "tone"]).to_pandas()
        t = t[t["known_at"].astype(NS) >= first]
        if t.empty:
            continue
        norm = t["title"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
        t["h"] = pd.util.hash_pandas_object(norm, index=False).to_numpy()
        t["symbol"] = t["symbols"].fillna("").str.split(",")
        t = t.explode("symbol")
        parts.append(t.loc[t["symbol"].isin(univ), ["symbol", "known_at", "h", "tone"]])

    n = pd.concat(parts, ignore_index=True)
    n["target"] = map_to_target(n["known_at"], cal)
    n = n.dropna(subset=["target"])
    raw = n.groupby(["symbol", "target"]).size().rename("news_raw")
    u = n.drop_duplicates(["symbol", "target", "h"])
    key = [u["symbol"], u["target"]]
    out = u.groupby(key)["tone"].agg(news_n="size", tone_mean="mean", tone_min="min",
                                     tone_max="max", tone_std="std")
    out["tone_neg_tail"] = (u["tone"] < -tail).groupby(key).sum()
    out["tone_pos_tail"] = (u["tone"] > tail).groupby(key).sum()
    return out.join(raw).reset_index()


# -----------------------------------------------------------------------------
# 역할  : 건수를 그 종목 평소 수준 대비 얼마나 많은지(비정상 관심도)로 바꿈
#         log1p(오늘 건수) - log1p(직전까지의 EMA). 건수 0인 날도 채워서 계산함
#         EMA 는 하루 밀어서(shift) 오늘 값이 기준에 섞이지 않게 함
# Input : df       DataFrame[symbol, target, <col>]
#         col      건수 열 이름
#         targets  전체 대상일 목록 (0 건인 날을 채우는 기준)
#         span     EMA 길이(거래일)
# Output: DataFrame[symbol, target, <col>, <col>_abn]
# -----------------------------------------------------------------------------
def abnormal(df, col, targets, span=20):
    grid = pd.MultiIndex.from_product([df["symbol"].unique(), np.sort(targets)],
                                      names=["symbol", "target"])
    s = df.set_index(["symbol", "target"])[col].reindex(grid, fill_value=0).astype(float)
    base = s.groupby(level="symbol").transform(lambda x: x.ewm(span=span, adjust=False).mean().shift())
    return pd.DataFrame({col: s, f"{col}_abn": np.log1p(s) - np.log1p(base)}).reset_index()


# -----------------------------------------------------------------------------
# 역할  : 티커 언급을 찾는 정규식 두 개를 만듦
#         `$TICKER` 는 모든 종목, 맨 대문자 단어는 일반 단어와 겹치지 않는 종목만 인정
# Input : symbols  종목 목록
# Output: (arrow 사전 필터용 패턴 문자열, 파이썬 re 패턴)
# -----------------------------------------------------------------------------
def ticker_patterns(symbols):
    bare = sorted((s for s in symbols if s not in COMMON_WORDS and len(s) > 2 and "-" not in s),
                  key=len, reverse=True)
    alt = "|".join(re.escape(s) for s in bare)
    pre = r"\$[A-Za-z]" + (rf"|\b(?:{alt})\b" if alt else "")
    rx = re.compile(r"\$([A-Za-z]{1,5}(?:[.\-/][A-Za-z])?)\b" + (rf"|\b({alt})\b" if alt else ""))
    return pre, rx


# -----------------------------------------------------------------------------
# 역할  : Reddit 글·댓글에서 대상일별 티커 언급량과 subreddit 활동량을 집계함
#         score, n_comments 는 36시간 뒤 값이라 읽지 않음 (created_et 와 본문만 읽음)
#         arrow(C++) 정규식으로 후보 행만 먼저 거른 뒤 파이썬 정규식을 돌려 빠르게 처리함
# Input : cal      trading_calendar() 결과
#         symbols  종목 목록
#         subs     subreddit 목록 (None 이면 전부)
# Output: (mentions DataFrame[symbol, target, sub, mentions],
#          activity DataFrame[sub, target, items])  items 는 글+댓글 수
#         한 글·댓글 안에서 같은 티커는 한 번만 셈
# -----------------------------------------------------------------------------
def build_reddit(cal, symbols, subs=None):
    folder = DATA / "reddit"
    subs = subs or sorted({p.name.split(".")[0] for p in folder.glob("*.parquet")})
    univ = set(symbols)
    pre, rx = ticker_patterns(symbols)
    first = np.datetime64(cal["cutoff"].iloc[0] - pd.Timedelta(days=7), "ns")
    men, act = [], []
    for sub in subs:
        for kind in ("posts", "comments"):
            pf = pq.ParquetFile(folder / f"{sub}.{kind}.parquet")
            cols = ["created_et", "title", "body"] if kind == "posts" else ["created_et", "body"]
            for i in range(pf.num_row_groups):
                t = pf.read_row_group(i, columns=cols)
                ts = pd.Series(t.column("created_et").to_numpy().astype(NS))
                keep = ts.to_numpy() >= first
                if not keep.any():
                    continue
                tgt = map_to_target(ts, cal)
                act.append(pd.DataFrame({"target": tgt[keep]}).assign(sub=sub))
                text = pc.fill_null(t.column("body"), "")
                if kind == "posts":
                    text = pc.binary_join_element_wise(pc.fill_null(t.column("title"), ""), text,
                                                    pa.scalar(" ", text.type))
                hit = pc.fill_null(pc.match_substring_regex(text, pre), False).to_numpy() & keep
                if not hit.any():
                    continue
                found = pd.Series(pc.filter(text, pa.array(hit)).to_pylist()).map(
                    lambda s: {(a.upper() or b).replace(".", "-").replace("/", "-")
                               for a, b in rx.findall(s)} & univ)
                m = pd.DataFrame({"target": tgt[hit].to_numpy(), "symbol": found.to_numpy()}).explode("symbol")
                men.append(m.dropna().assign(sub=sub))
    mentions = (pd.concat(men).groupby(["symbol", "target", "sub"]).size()
                .rename("mentions").reset_index())
    activity = (pd.concat(act).dropna().groupby(["sub", "target"]).size()
                .rename("items").reset_index())
    return mentions, activity


# -----------------------------------------------------------------------------
# 역할  : Reddit 집계를 캐시에서 읽고, 없으면 build_reddit() 으로 만들어 저장함
#         (전체 8개 subreddit 기준 첫 실행 약 2분)
# Input : cal, symbols  build_reddit() 과 같음
#         rebuild       True 면 다시 계산
# Output: (mentions DataFrame, activity DataFrame)
# -----------------------------------------------------------------------------
def load_reddit(cal, symbols, rebuild=False):
    pm, pa_ = CACHE / "reddit_mentions.parquet", CACHE / "reddit_activity.parquet"
    if pm.exists() and pa_.exists() and not rebuild:
        return pd.read_parquet(pm), pd.read_parquet(pa_)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m, a = build_reddit(cal, symbols)
    m.to_parquet(pm, index=False)
    a.to_parquet(pa_, index=False)
    print(f"[cache] reddit: 언급 {len(m):,}행, 활동 {len(a):,}행, {time.time() - t0:.1f}초")
    return m, a


# -----------------------------------------------------------------------------
# 역할  : 가격 패널에 실적·애널리스트·뉴스(·Reddit) feature 를 모두 붙인 분석용 표를 만듦
#         건수형 feature(earn, an_*)는 이벤트가 없는 날을 0 으로 채움
# Input : reddit   Reddit feature 포함 여부
#         rebuild  캐시를 무시하고 다시 계산
# Output: (panel DataFrame, cal DataFrame)
#         추가 열  earn, surp_*, an_*, news_n, news_n_abn, tone_*, news_raw,
#                 mentions, mentions_abn, act_<subreddit>_abn
# -----------------------------------------------------------------------------
def load_all(reddit=True, rebuild=False):
    panel = dev_only(cached("panel_dev", build_panel, rebuild))
    daily = pd.read_parquet(DATA / "daily.parquet", columns=["symbol", "date_et"])
    cal = trading_calendar(daily)
    syms = sorted(panel["symbol"].unique())
    targets = cal["target"].to_numpy()
    key = ["symbol", "target"]

    news = cached("news", lambda: build_news(cal, syms), rebuild)
    p = (panel.merge(earnings_features(cal), on=key, how="left")
         .merge(analyst_features(cal), on=key, how="left")
         .merge(news.drop(columns="news_n"), on=key, how="left")
         .merge(abnormal(news, "news_n", targets), on=key, how="left"))
    for c in ("earn", "an_n", "an_up", "an_down", "an_init", "news_raw"):
        p[c] = p[c].fillna(0)

    if reddit:
        m, a = load_reddit(cal, syms, rebuild)
        tot = m.groupby(key)["mentions"].sum().reset_index()
        p = p.merge(abnormal(tot, "mentions", targets), on=key, how="left")
        act = abnormal(a.rename(columns={"sub": "symbol"}), "items", targets)
        act = act.pivot(index="target", columns="symbol", values="items_abn")
        act.columns = [f"act_{c}_abn" for c in act.columns]
        p = p.merge(act.reset_index(), on="target", how="left")
    return p, cal


# =============================================================================
# 점수 분석
# =============================================================================

# -----------------------------------------------------------------------------
# 역할  : 과제 score 에 두 지표를 더해 점수를 어디서 잃었는지 보여 줌
#         flip_rate  P(예측 4 | 정답 0) + P(예측 0 | 정답 4)  (가중 64인 최악의 실수)
#         loss_c     정답 클래스 i 마다 Σ_j w_ij·O_ij / Σ w·E   (합 = 1 - score)
# Input : y_true, y_pred  label 배열 (0~4)
# Output: dict  score, accuracy, direction, big_recall, big_prec, n, flip_rate, loss_c0~c4
# -----------------------------------------------------------------------------
def score_ext(y_true, y_pred):
    t, p = np.asarray(y_true, int), np.asarray(y_pred, int)
    out = score(t, p)
    O = np.zeros((5, 5))
    np.add.at(O, (t, p), 1)
    E = np.outer(O.sum(1), O.sum(0)) / len(t)
    denom = (WEIGHT * E).sum()
    rate = lambda i, j: (p[t == i] == j).mean() if (t == i).any() else 0.0
    out["flip_rate"] = float(rate(0, 4) + rate(4, 0))
    for i, v in enumerate((WEIGHT * O).sum(1) / denom if denom else np.full(5, np.nan)):
        out[f"loss_c{i}"] = float(v)
    return out


# -----------------------------------------------------------------------------
# 역할  : 점수의 상·하한을 보는 기준 전략과 oracle 전략의 예측을 만듦
#         oracle 은 정답을 일부 아는 상한 실험이라 모델에는 쓰지 않음
# Input : y     정답 label 배열
#         seed  무작위 전략용 시드
# Output: dict  전략 이름 -> 예측 배열
# -----------------------------------------------------------------------------
def baseline_preds(y, seed=0):
    y = np.asarray(y, int)
    rng = np.random.default_rng(seed)
    big = np.isin(y, BIG)
    side = np.where(y < 2, 1, np.where(y > 2, 3, 2))
    return {
        "전부 보합": np.full_like(y, 2),
        "전부 급상승": np.full_like(y, 4),
        "무작위 (정답 분포대로)": rng.choice(y, len(y)),
        "방향만 oracle": side,
        "급등락만 oracle, 나머지 보합": np.where(big, y, 2),
        "급등락을 반대로, 나머지 보합": np.where(big, 4 - y, 2),
        "전부 맞힘": y.copy(),
    }


# -----------------------------------------------------------------------------
# 역할  : 정답이 보합인 행의 예측만 바꿨을 때 점수가 변하는지 확인함
#         분자(O)는 그대로지만 예측 분포가 바뀌어 분모(E)가 변하는 효과를 보여 줌
# Input : y, p   정답과 기준 예측
#         fracs  보합 행 중 급상승(4)으로 바꿀 비율 목록
#         seed   시드
# Output: DataFrame[frac, score, num, denom]
# -----------------------------------------------------------------------------
def flat_row_effect(y, p, fracs=(0, 0.05, 0.1, 0.2, 0.4), seed=0):
    y, p = np.asarray(y, int), np.asarray(p, int)
    flat = np.flatnonzero(y == 2)
    rng = np.random.default_rng(seed)
    order = rng.permutation(flat)
    rows = []
    for f in fracs:
        q = p.copy()
        q[order[:int(len(flat) * f)]] = 4
        O = np.zeros((5, 5))
        np.add.at(O, (y, q), 1)
        E = np.outer(O.sum(1), O.sum(0)) / len(y)
        num, den = (WEIGHT * O).sum(), (WEIGHT * E).sum()
        rows.append({"frac": f, "score": 1 - num / den, "num": num, "denom": den})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 역할  : 갭 하나로 label 을 정하는 규칙. label_of(k × 갭%)
#         k 가 1이면 갭을 그대로 당일 수익률로 보는 것, 크면 갭 이후 추세가 이어진다고 보는 것
# Input : gap_pct  갭(%) 배열,  k  배율
# Output: 예측 label 배열 (갭이 없으면 보합)
# -----------------------------------------------------------------------------
def gap_rule(gap_pct, k=1.0):
    x = pd.Series(np.asarray(gap_pct, float) * k)
    return label_of(x).fillna(2).astype(int).to_numpy()


# -----------------------------------------------------------------------------
# 역할  : 학습 구간에서 score 가 가장 높은 k 를 격자 탐색으로 찾음
# Input : df    패널 (gap_pct, label)
#         grid  후보 k 목록
# Output: (best_k, DataFrame[k, score])
# -----------------------------------------------------------------------------
def fit_gap_rule(df, grid=np.round(np.arange(0.5, 8.01, 0.25), 2)):
    res = pd.DataFrame({"k": grid,
                        "score": [score(df["label"], gap_rule(df["gap_pct"], k))["score"] for k in grid]})
    return float(res.loc[res["score"].idxmax(), "k"]), res


# =============================================================================
# 증분 가치(Δscore)와 검증
# =============================================================================

# -----------------------------------------------------------------------------
# 역할  : 클래스 확률에서 기대 벌점이 가장 작은 label 을 고름
#         예측 j 의 기대 벌점 = Σ_i P(정답 i)·w_ij
# Input : proba    (n, k) 확률,  classes  proba 열에 해당하는 label
# Output: 예측 label 배열
# -----------------------------------------------------------------------------
def cost_min_predict(proba, classes):
    P = np.zeros((len(proba), 5))
    P[:, np.asarray(classes, int)] = proba
    return (P @ WEIGHT).argmin(axis=1)


# -----------------------------------------------------------------------------
# 역할  : 학습 구간으로 모델을 학습하고 평가 구간의 score 를 잼
#         HistGradientBoosting 은 결측값을 그대로 받고 빠름
#         decision="regress" (기본)
#             당일 수익률(%)을 회귀로 예측한 뒤 label_of(k × 예측값)으로 바꿈
#             k 는 학습 구간의 마지막 20%(시간순)에서 score 가 가장 높은 값으로 고름
#             갭 규칙과 같은 결정 방식이라 갭 규칙과 공정하게 비교됨
#         decision="cost"
#             분류 확률에서 기대 벌점이 가장 작은 label. 분모(E) 효과를 무시해서
#             급등락을 너무 아끼는 경향이 있음 (00 노트북 5절)
# Input : train, test  패널 부분집합
#         feats        사용할 열 목록
#         decision     "regress" 또는 "cost"
#         seed         시드
# Output: score_ext() dict (+ "k": 고른 배율, regress 일 때)
# -----------------------------------------------------------------------------
def fit_eval(train, test, feats, decision="regress", seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    params = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=False, random_state=seed)
    if decision == "cost":
        clf = HistGradientBoostingClassifier(**params).fit(train[feats], train["label"])
        return score_ext(test["label"], cost_min_predict(clf.predict_proba(test[feats]), clf.classes_))

    edge = train["target"].quantile(0.8)
    fit, tune = train[train["target"] <= edge], train[train["target"] > edge]
    reg = HistGradientBoostingRegressor(loss="absolute_error", **params)
    reg.fit(fit[feats], fit["ret_pct"].clip(-15, 15))
    k, _ = fit_gap_rule(pd.DataFrame({"gap_pct": reg.predict(tune[feats]), "label": tune["label"]}))
    out = score_ext(test["label"], gap_rule(reg.predict(test[feats]), k))
    out["k"] = k
    return out


# -----------------------------------------------------------------------------
# 역할  : 시간순 walk-forward 분할. 학습과 평가 사이에 embargo 를 둠
# Input : panel, starts(평가 시작일들), horizon_days(평가 길이), embargo_days
# Output: [(이름, 학습 mask, 평가 mask), ...]
# -----------------------------------------------------------------------------
def walk_forward(panel, starts=("2025-07-01", "2025-11-01", "2026-03-01"),
                 horizon_days=120, embargo_days=14):
    t, out = panel["target"], []
    for s in map(pd.Timestamp, starts):
        out.append((f"wf {s:%Y-%m}",
                    (t < s - pd.Timedelta(days=embargo_days)).to_numpy(),
                    ((t >= s) & (t < s + pd.Timedelta(days=horizon_days))).to_numpy()))
    return out


# -----------------------------------------------------------------------------
# 역할  : 종목 holdout. 같은 기간에서 k개 종목을 학습에서 빼고 그 종목으로 평가
# Input : panel, n_sets(세트 수), k(빼는 종목 수), seed
# Output: [(이름, 학습 mask, 평가 mask), ...]
# -----------------------------------------------------------------------------
def symbol_holdout(panel, n_sets=5, k=10, seed=0):
    rng = np.random.default_rng(seed)
    syms = np.sort(panel["symbol"].unique())
    out = []
    for i in range(n_sets):
        held = panel["symbol"].isin(rng.choice(syms, k, replace=False)).to_numpy()
        out.append((f"sym {i}", ~held, held))
    return out


# -----------------------------------------------------------------------------
# 역할  : 시간 + 종목 holdout. 실제 채점(미래 구간 × 처음 보는 종목)을 흉내 냄
#         학습: start 이전 & 남은 종목,  평가: start 이후 & 뺀 종목
# Input : panel, start, n_sets, k, embargo_days, seed
# Output: [(이름, 학습 mask, 평가 mask), ...]
# -----------------------------------------------------------------------------
def time_symbol_holdout(panel, start="2026-01-01", n_sets=5, k=10, embargo_days=14, seed=1):
    rng = np.random.default_rng(seed)
    syms = np.sort(panel["symbol"].unique())
    s, t = pd.Timestamp(start), panel["target"]
    out = []
    for i in range(n_sets):
        held = panel["symbol"].isin(rng.choice(syms, k, replace=False)).to_numpy()
        out.append((f"time+sym {i}",
                    ((t < s - pd.Timedelta(days=embargo_days)).to_numpy()) & ~held,
                    (t >= s).to_numpy() & held))
    return out


# -----------------------------------------------------------------------------
# 역할  : 후보 feature 묶음마다 단독 score 와 기준선 대비 Δscore 를 모든 분할에서 잼
#         기준선 모델은 분할마다 한 번만 학습함
# Input : panel   패널
#         base    기준선 feature 목록 (예: 갭 계열)
#         groups  {이름: feature 목록}
#         splits  [(이름, 학습 mask, 평가 mask)]
# Output: DataFrame[split, group, solo, with_base, base, d_score, flip_rate, big_recall, big_prec]
# -----------------------------------------------------------------------------
def delta_table(panel, base, groups, splits):
    rows = []
    for name, tr, te in splits:
        train, test = panel[tr], panel[te]
        b = fit_eval(train, test, base)
        rows.append({"split": name, "group": "기준선", "solo": b["score"], "with_base": b["score"],
                     "base": b["score"], "d_score": 0.0, "flip_rate": b["flip_rate"],
                     "big_recall": b["big_recall"], "big_prec": b["big_prec"]})
        for g, feats in groups.items():
            solo = fit_eval(train, test, feats)["score"]
            w = fit_eval(train, test, list(dict.fromkeys(base + feats)))
            rows.append({"split": name, "group": g, "solo": solo, "with_base": w["score"],
                         "base": b["score"], "d_score": w["score"] - b["score"],
                         "flip_rate": w["flip_rate"], "big_recall": w["big_recall"],
                         "big_prec": w["big_prec"]})
    return pd.DataFrame(rows)


# =============================================================================
# 표와 그림
# =============================================================================

# -----------------------------------------------------------------------------
# 역할  : 한글이 깨지지 않게 matplotlib 기본 설정을 맞춤
# Input : 없음
# Output: 없음
# -----------------------------------------------------------------------------
def setup_plot():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Malgun Gothic", "DejaVu Sans"], "axes.unicode_minus": False,
                         "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


# -----------------------------------------------------------------------------
# 역할  : x 를 분위 구간으로 나누고 구간별로 y 의 평균과 건수를 냄
# Input : df  DataFrame,  x  구간을 나눌 열,  y  평균낼 열 (문자열 또는 목록),  q  구간 수
# Output: DataFrame[bin, x_mid, <y 평균들>, n]
# -----------------------------------------------------------------------------
def binned(df, x, y, q=10):
    ys = [y] if isinstance(y, str) else list(y)
    d = df[[x, *ys]].dropna(subset=[x])
    b = pd.qcut(d[x], q, duplicates="drop")
    out = d.groupby(b, observed=True).agg(x_mid=(x, "median"), n=(x, "size"),
                                          **{c: (c, "mean") for c in ys})
    return out.reset_index(names="bin")


# -----------------------------------------------------------------------------
# 역할  : 혼동행렬처럼 생긴 표를 칸마다 숫자를 적은 heatmap 으로 그림
# Input : ax  matplotlib 축,  mat  2차원 배열 또는 DataFrame,  title  제목
#         fmt  칸 숫자 형식,  cmap  색 지도,  xlabel / ylabel  축 이름
# Output: 없음 (ax 에 그림)
# -----------------------------------------------------------------------------
def plot_heatmap(ax, mat, title="", fmt="{:.0f}", cmap="Blues", xlabel="예측", ylabel="정답"):
    m = pd.DataFrame(mat)
    ax.imshow(m.to_numpy(float), cmap=cmap)
    ax.set_xticks(range(m.shape[1]), m.columns, rotation=0)
    ax.set_yticks(range(m.shape[0]), m.index)
    v = m.to_numpy(float)
    hi = np.nanmax(v) if np.isfinite(v).any() else 1
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center",
                    color="white" if v[i, j] > hi * 0.6 else "black", fontsize=9)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(False)
