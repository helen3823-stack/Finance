"""E4 갭 조건부 분석.

기능  : 같은 갭 크기 구간 안에서, 비가격 이벤트가 있을 때와 없을 때의 방향·크기·갭 이후 움직임을 비교
구성  : EVENTS (이벤트 정의 registry) · gap_conditional() · run_analysis()
역할  : "이 데이터가 갭에 없는 정보를 주는가"를 모델 없이 표로 확인 (ΔScore 와 함께 제외·채택 근거)

수식
    gap_bin   = |gap_pct| 구간 (기본 [0, 0.5), [0.5, 1), [1, 2.5), [2.5, ∞))
    up_rate   = mean 1[ret_pct > 0]
    M_rate    = mean 1[label ∈ {0, 4}]
    rest_mean = mean rest,  rest = log(1 + ret) − log(1 + gap)
    agree     = sign(이벤트 방향) × sign(gap)   (+1 같은 방향, −1 반대)
"""

import json

import numpy as np
import pandas as pd

from . import config as C
from .panel import RESEARCH_GROUPS, load_tables, universe_panel
from .splits import partition

# 이벤트 정의. 이름 → (패널 → 범주 Series) 함수. 새 이벤트는 여기 한 줄 추가
EVENTS = {
    "analyst": lambda x: pd.Series(np.select([x["an_up"] > x["an_down"], x["an_down"] > x["an_up"]],
                                             ["upgrade", "downgrade"], "none"), index=x.index),
    "earnings": lambda x: pd.Series(np.where(x["earn"] > 0, np.where(x["surp_sign"] > 0, "beat", "miss_or_meet"), "none"),
                                    index=x.index),
    "news_tone_vs_gap": lambda x: _agree(np.sign(x["tone_mean"]), x),
    "news_volume": lambda x: pd.qcut(x["news_abn"], 4, labels=["q1 적음", "q2", "q3", "q4 많음"], duplicates="drop").astype(str),
    "llm_anon_vs_gap": lambda x: _agree(x["llm_anon_sign"], x),
    "llm_raw_vs_gap": lambda x: _agree(x["llm_raw_sign"], x),
    "reddit_mentions": lambda x: pd.qcut(x["mentions_abn"], 4, labels=["q1", "q2", "q3", "q4 급증"], duplicates="drop").astype(str),
}


def _agree(sign, x):
    s = np.sign(sign) * np.sign(x["gap_pct"])
    return pd.Series(np.select([sign.isna() | (sign == 0), s > 0, s < 0], ["no_event", "same_dir", "opposite"], "gap_0"),
                     index=x.index)


# -----------------------------------------------------------------------------
# 기능  : 갭 조건부 표
# input : x  패널,  event  EVENTS 이름,  bins  |gap| 구간 경계(%)
# output: DataFrame[gap_bin, category, n, up_rate, M_rate, rest_mean, ret_mean]
# -----------------------------------------------------------------------------
def gap_conditional(x, event, bins=(0, 0.5, 1.0, 2.5, np.inf)):
    cat = EVENTS[event](x)
    g = x.assign(gap_bin=pd.cut(x["gap_pct"].abs(), list(bins), right=False), category=cat,
                 up=(x["ret_pct"] > 0).astype(float), M=x["label"].isin((0, 4)).astype(float))
    return (g.groupby(["gap_bin", "category"], observed=True)
            .agg(n=("up", "size"), up_rate=("up", "mean"), M_rate=("M", "mean"),
                 rest_mean=("rest", "mean"), ret_mean=("ret_pct", "mean")).reset_index())


# -----------------------------------------------------------------------------
# 기능  : config [analysis] 의 이벤트 표를 모두 만들어 runs/<name>_<hash>/analysis_<event>.csv 로 저장
#         DEV 구간(lockbox 이전)만 씀. 교차 feature 는 전체 종목을 20종목 묶음으로 나눠 계산
# input : cfg  ([analysis] events, bins)
# -----------------------------------------------------------------------------
def run_analysis(cfg):
    a = cfg.get("analysis", {})
    events = a.get("events", ["analyst", "earnings", "news_tone_vs_gap", "news_volume"])
    seed = int(cfg["seed"])
    syms = sorted(load_tables()["daily"]["symbol"].unique())
    unis = partition(syms, cfg["split"]["universe_size"], np.random.default_rng(C.derive_seed(seed, "analysis")))
    x = pd.concat([universe_panel(u, cfg["data"]["groups"], cfg["data"]["research"]) for u in unis], ignore_index=True)
    x = x[x["target"] < pd.Timestamp(cfg["split"]["lockbox_start"])]
    out = C.RUNS / f"{cfg['name']}_{cfg['hash']}_analysis"
    out.mkdir(parents=True, exist_ok=True)
    bins = tuple(a.get("bins", (0, 0.5, 1.0, 2.5))) + (np.inf,)
    for ev in events:
        need = {"news_tone_vs_gap": "news", "news_volume": "news", "reddit_mentions": "reddit",
                "llm_anon_vs_gap": "news_llm_anon", "llm_raw_vs_gap": "news_llm_raw"}.get(ev)
        if need in RESEARCH_GROUPS and need not in cfg["data"]["research"]:
            print(f"[analysis] {ev}: data.research 에 '{need}' 가 없어 건너뜀")
            continue
        t = gap_conditional(x, ev, bins)
        t.to_csv(out / f"analysis_{ev}.csv", index=False)
        print(f"\n== {ev} ==\n{t.round(4).to_string(index=False)}")
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[analysis] → {out}")
    return out
