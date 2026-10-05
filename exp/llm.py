"""E4-L LLM 스크리닝 (연구용, 제출 모델에서는 쓰지 않음).

기능  : 뉴스 제목 / Reddit 글을 LLM 으로 구조화하고, 그 결과가 갭 이후 움직임을 설명하는지 볼 feature 를 만듦
구성  : SCHEMAS·SYSTEM (출력 형식·지시문) · anonymize() · sample_news() · sample_reddit()
        · call() (캐시) · run_llm()
역할  : LLM 은 "주가에 좋은가"를 판단하지 않고 사실만 추출하는 parser 로만 씀. 방향 부호는 코드가 규칙으로 계산
        누수 통제: 회사명·티커·날짜를 지운 익명 입력(anon)과 원문 입력(raw)을 둘 다 돌려 비교
        재현성: 모든 호출을 (모델, 프롬프트 버전, 스키마, 입력) 해시로 캐시 → 다시 돌리면 API 를 부르지 않음
안전장치: dry_run=true (기본) 이면 API 를 부르지 않고 보낼 입력만 저장. max_calls 로 새 호출 수 상한

feature (대상일 단위)
    news   llm_<v>_sign       = sign(Σ 이벤트 부호),  beat/raise/upgrade = +1, miss/cut/downgrade = −1
           llm_<v>_severity   = 1~3,  llm_<v>_unexpected = 1~3,  llm_<v>_n = 처리한 기사 수
    reddit llm_reddit_bull    = mean(+1 bullish, −1 bearish, 0 그 외),  llm_reddit_hype = mean(1~3),  llm_reddit_n
"""

import hashlib
import json
import re
import time

import numpy as np
import pandas as pd

from src.model import map_to_target

from . import config as C
from .panel import calendar, load_tables, universe_panel
from .splits import partition

PROMPT_VERSION = "v1"
LLM_CACHE = C.CACHE / "llm"
ALIASES = C.ROOT / "configs" / "company_aliases.json"

SCHEMAS = {
    "news_event": {
        "type": "object", "additionalProperties": False,
        "required": ["event_type", "earnings_result", "guidance_action", "analyst_action", "legal_event",
                     "mna_event", "product_event", "management_event", "severity", "unexpectedness", "mixed_event"],
        "properties": {
            "event_type": {"type": "string", "enum": ["earnings", "guidance", "analyst", "legal_regulatory", "mna",
                                                      "product", "management", "macro", "other", "none"]},
            "earnings_result": {"type": "string", "enum": ["beat", "meet", "miss", "not_mentioned"]},
            "guidance_action": {"type": "string", "enum": ["raise", "maintain", "cut", "not_mentioned"]},
            "analyst_action": {"type": "string", "enum": ["upgrade", "maintain", "downgrade", "not_mentioned"]},
            "legal_event": {"type": "boolean"}, "mna_event": {"type": "boolean"},
            "product_event": {"type": "boolean"}, "management_event": {"type": "boolean"},
            "severity": {"type": "integer", "enum": [1, 2, 3]},
            "unexpectedness": {"type": "integer", "enum": [1, 2, 3]},
            "mixed_event": {"type": "boolean"},
        },
    },
    "reddit_sentiment": {
        "type": "object", "additionalProperties": False,
        "required": ["stance", "hype_intensity", "speculation", "short_squeeze", "event_reaction", "disagreement"],
        "properties": {
            "stance": {"type": "string", "enum": ["bullish", "bearish", "neutral", "unclear"]},
            "hype_intensity": {"type": "integer", "enum": [1, 2, 3]},
            "speculation": {"type": "boolean"}, "short_squeeze": {"type": "boolean"},
            "event_reaction": {"type": "boolean"}, "disagreement": {"type": "boolean"},
        },
    },
}
SYSTEM = {
    "news_event": ("You extract facts from one financial news headline. Report only what the headline states. "
                   "Do not judge whether the news is good or bad for any stock price, and do not use knowledge of "
                   "what happened after the headline. Use 'not_mentioned' or false when the headline does not say. "
                   "severity: 1 routine, 2 notable, 3 major. unexpectedness: 1 expected or scheduled, 2 somewhat, "
                   "3 described as surprising."),
    "reddit_sentiment": ("You label one Reddit post or comment about a stock. Report the author's stated stance "
                         "and tone only, not your own view of the stock. hype_intensity: 1 calm, 2 excited, 3 extreme."),
}


# -----------------------------------------------------------------------------
# 기능  : 익명화. 회사 이름·티커($XXX 포함)·날짜 표현을 지움
# input : text,  aliases  {티커: [이름]}
# output: 익명화한 문자열  (회사 → [COMPANY], 날짜 → [DATE])
# -----------------------------------------------------------------------------
_MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_DATE_RX = re.compile(rf"\b{_MONTHS}\.?\s*\d{{0,2}}(?:,?\s*\d{{4}})?\b|\b(?:19|20)\d{{2}}\b|\b\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?\b|"
                      r"\b(?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\b", re.I)


def anonymize(text, aliases):
    names = sorted({n for v in aliases.values() for n in v}, key=len, reverse=True)
    out = re.sub(r"\$[A-Za-z]{1,5}(?:[.\-][A-Za-z])?\b", "[COMPANY]", text)
    for n in names:
        out = re.sub(rf"(?<!\w){re.escape(n)}(?:'s)?(?!\w)", "[COMPANY]", out, flags=re.I)
    for t in aliases:
        out = re.sub(rf"\b{re.escape(t)}\b", "[COMPANY]", out)          # 티커는 대문자 그대로일 때만
    return re.sub(r"\s+", " ", _DATE_RX.sub("[DATE]", out)).strip()


def _aliases():
    return {k: v for k, v in json.loads(ALIASES.read_text(encoding="utf-8")).items() if not k.startswith("_")}


# -----------------------------------------------------------------------------
# 기능  : DEV 구간의 (종목, 대상일) 패널 (gap_pct, rest, label). 교차 feature 는 20종목 묶음으로 계산
# -----------------------------------------------------------------------------
def _dev_panel(cfg):
    syms = sorted(load_tables()["daily"]["symbol"].unique())
    unis = partition(syms, cfg["split"]["universe_size"], np.random.default_rng(C.derive_seed(cfg["seed"], "llm")))
    x = pd.concat([universe_panel(u, ["core"]) for u in unis], ignore_index=True)
    return x[x["target"] < pd.Timestamp(cfg["split"]["lockbox_start"])][["symbol", "target", "gap_pct", "rest", "label"]]


# -----------------------------------------------------------------------------
# 기능  : 갭 층화 뉴스 표본
# 방법  : 대상일 창 안의 기사를 정규화 제목으로 중복 제거하고, (종목, 대상일)마다 가장 많이 반복된 제목을 대표로 고름
#         |gap| 층 (small < 0.5, medium < 1.5, large ≥ 1.5 %) × 갭 부호 6칸에서 n_per_stratum 개씩 무작위 추출
# input : cfg [llm] n_per_stratum, seed
# output: DataFrame[symbol, target, title, gap_pct, stratum]
# -----------------------------------------------------------------------------
def sample_news(cfg):
    L = cfg["llm"]
    n = load_tables(with_news=True)["news"]
    x = _dev_panel(cfg)
    n = n.assign(symbol=n["symbols"].fillna("").str.split(",")).explode("symbol")
    n = n[n["symbol"].isin(set(x["symbol"]))]
    n = n.assign(target=map_to_target(n["known_at"], calendar())).dropna(subset=["target"])
    n["norm"] = n["title"].fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    n = n[n["norm"].str.len() > 0]
    cnt = n.groupby(["symbol", "target", "norm"]).agg(dups=("title", "size"), title=("title", "first"),
                                                       first=("known_at", "min")).reset_index()
    rep = cnt.sort_values(["dups", "first"], ascending=[False, True]).drop_duplicates(["symbol", "target"])
    rep = rep.merge(x, on=["symbol", "target"]).dropna(subset=["gap_pct"])
    size = pd.cut(rep["gap_pct"].abs(), [0, 0.5, 1.5, np.inf], right=False, labels=["small", "medium", "large"])
    rep["stratum"] = size.astype(str) + np.where(rep["gap_pct"] >= 0, "_pos", "_neg")
    rng = np.random.default_rng(C.derive_seed(cfg["seed"], "llm_news"))
    k = int(L.get("n_per_stratum", 50))
    pick = [g.iloc[np.sort(rng.choice(len(g), min(k, len(g)), replace=False))] for _, g in rep.groupby("stratum", sort=True)]
    return pd.concat(pick, ignore_index=True)[["symbol", "target", "title", "gap_pct", "stratum"]]


# -----------------------------------------------------------------------------
# 기능  : Reddit 표본. 시드로 섞은 row group 순서대로 읽으며 티커가 언급된 댓글을 모음
# input : cfg [llm] reddit_sub, reddit_n
# output: DataFrame[symbol, target, text]  (한 댓글에 티커가 여럿이면 종목마다 한 행)
# -----------------------------------------------------------------------------
def sample_reddit(cfg):
    import sys
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sys.path.insert(0, str(C.ROOT / "eda"))
    import eda_utils as E

    L = cfg["llm"]
    syms = sorted(load_tables()["daily"]["symbol"].unique())
    pre, rx = E.ticker_patterns(syms)
    pf = pq.ParquetFile(C.ROOT / "dataset" / "reddit" / f"{L.get('reddit_sub', 'stocks')}.comments.parquet")
    rng = np.random.default_rng(C.derive_seed(cfg["seed"], "llm_reddit"))
    lb, cal, want, rows = pd.Timestamp(cfg["split"]["lockbox_start"]), calendar(), int(L.get("reddit_n", 300)), []
    for i in rng.permutation(pf.num_row_groups):
        t = pf.read_row_group(int(i), columns=["created_et", "body"])
        hit = pc.fill_null(pc.match_substring_regex(pc.fill_null(t.column("body"), ""), pre), False).to_numpy()
        if not hit.any():
            continue
        d = t.filter(hit).to_pandas()
        d["target"] = map_to_target(d["created_et"], cal)
        d = d[d["target"].notna() & (d["target"] < lb)]
        for _, r in d.sample(frac=1.0, random_state=int(rng.integers(2**31))).iterrows():
            found = {(a.upper() or b).replace(".", "-") for a, b in rx.findall(r["body"])} & set(syms)
            rows += [{"symbol": s, "target": r["target"], "text": r["body"][:1500]} for s in sorted(found)]
            if len(rows) >= want:
                return pd.DataFrame(rows[:want])
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 기능  : LLM 호출 1건 (캐시 우선)
# input : schema  SCHEMAS 이름,  text,  L  [llm] 설정 (model, effort),  client  anthropic 클라이언트 또는 None
# output: (결과 dict 또는 None, 캐시에서 읽었는지)
# 캐시  : exp/cache/llm/<sha256>.json = {request(model, prompt_version, schema, system_sha, text), response, meta}
#         temperature 는 이 모델에서 지정할 수 없어 기록하지 않음 (모델 기본값)
# -----------------------------------------------------------------------------
def _key(schema, text, L):
    req = {"model": L["model"], "effort": L["effort"], "prompt_version": PROMPT_VERSION, "schema": schema,
           "system_sha": hashlib.sha256(SYSTEM[schema].encode()).hexdigest()[:12], "text": text}
    return hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest(), req


def call(schema, text, L, client):
    k, req = _key(schema, text, L)
    path = LLM_CACHE / f"{k}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"], True
    if client is None:
        return None, False
    resp = client.beta.messages.create(
        model=L["model"], max_tokens=1024,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        system=SYSTEM[schema], messages=[{"role": "user", "content": text}],
        output_config={"effort": L["effort"], "format": {"type": "json_schema", "schema": SCHEMAS[schema]}})
    out = None
    if resp.stop_reason != "refusal":
        out = json.loads(next(b.text for b in resp.content if b.type == "text"))
    meta = {"served_model": resp.model, "stop_reason": resp.stop_reason, "request_id": resp._request_id,
            "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    LLM_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"request": req, "response": out, "meta": meta}, ensure_ascii=False, indent=1), encoding="utf-8")
    return out, False


# -----------------------------------------------------------------------------
# 기능  : 구조화 결과 → 숫자 feature (방향 부호는 규칙으로 계산)
# -----------------------------------------------------------------------------
def _news_row(r):
    s = {"beat": 1, "miss": -1}.get(r["earnings_result"], 0) + {"raise": 1, "cut": -1}.get(r["guidance_action"], 0) \
        + {"upgrade": 1, "downgrade": -1}.get(r["analyst_action"], 0)
    return {"sign": float(np.sign(s)), "severity": float(r["severity"]), "unexpected": float(r["unexpectedness"])}


def _reddit_row(r):
    return {"bull": {"bullish": 1.0, "bearish": -1.0}.get(r["stance"], 0.0), "hype": float(r["hype_intensity"])}


# -----------------------------------------------------------------------------
# 기능  : E4-L 실행
#         1) 표본 추출 → 2) 변형(anon / raw)별 입력 → 3) 캐시 또는 API → 4) feature parquet 저장
#         5) 요약: anon·raw 부호 일치율, 부호 × 갭 방향별 rest 평균 (누수 의심 지표)
# input : cfg [llm] model, effort, dry_run, max_calls, kinds(["news", "reddit"]), variants(["anon", "raw"]) ...
# output: 결과 폴더 runs/<name>_<hash>_llm/
# -----------------------------------------------------------------------------
def run_llm(cfg):
    L = {"model": "claude-opus-5-5", "effort": "low", "dry_run": True, "max_calls": 200,
         "kinds": ["news"], "variants": ["anon", "raw"], **cfg.get("llm", {})}
    out = C.RUNS / f"{cfg['name']}_{cfg['hash']}_llm"
    out.mkdir(parents=True, exist_ok=True)
    client = None
    if not L["dry_run"]:
        import anthropic
        client = anthropic.Anthropic()
    aliases, budget, pending, summary = _aliases(), int(L["max_calls"]), [], {}

    jobs = []
    if "news" in L["kinds"]:
        s = sample_news(cfg)
        s.to_csv(out / "sample_news.csv", index=False)
        for v in L["variants"]:
            jobs.append(("news_event", v, s, [anonymize(t, aliases) if v == "anon" else t for t in s["title"]]))
    if "reddit" in L["kinds"]:
        s = sample_reddit(cfg)
        s.to_csv(out / "sample_reddit.csv", index=False)
        jobs.append(("reddit_sentiment", "anon", s, [anonymize(t, aliases) for t in s["text"]]))

    for schema, v, s, texts in jobs:
        res = []
        for text in texts:
            r, hit = call(schema, text, L, client if budget > 0 else None)
            if not hit and client is not None and budget > 0:
                budget -= 1
            if r is None and not hit:
                pending.append({"schema": schema, "variant": v, "text": text})
            res.append(r)
        conv = _news_row if schema == "news_event" else _reddit_row
        f = pd.DataFrame([conv(r) if r else {} for r in res], index=s.index)
        d = pd.concat([s[["symbol", "target"]], f], axis=1).dropna(subset=f.columns.tolist() or ["symbol"])
        if not len(f.columns):
            continue
        if schema == "news_event":
            g = d.groupby(["symbol", "target"]).agg(sign=("sign", "sum"), severity=("severity", "mean"),
                                                    unexpected=("unexpected", "mean"), n=("sign", "size"))
            g["sign"] = np.sign(g["sign"])
            g.columns = [f"llm_{v}_{c}" for c in g.columns]
            g.reset_index().to_parquet(C.CACHE / f"llm_features_{v}.parquet", index=False)
        else:
            g = d.groupby(["symbol", "target"]).agg(llm_reddit_bull=("bull", "mean"), llm_reddit_hype=("hype", "mean"),
                                                    llm_reddit_n=("bull", "size"))
            g.reset_index().to_parquet(C.CACHE / "llm_reddit_features.parquet", index=False)
        summary[f"{schema}/{v}"] = {"items": len(s), "parsed": int(sum(r is not None for r in res))}

    if {"news_event/anon", "news_event/raw"} <= set(summary):
        a = pd.read_parquet(C.CACHE / "llm_features_anon.parquet")
        b = pd.read_parquet(C.CACHE / "llm_features_raw.parquet")
        m = a.merge(b, on=["symbol", "target"]).merge(_dev_panel(cfg), on=["symbol", "target"])
        summary["sign_agreement_anon_raw"] = float((m["llm_anon_sign"] == m["llm_raw_sign"]).mean())
        for v in ("anon", "raw"):
            agree = np.sign(m[f"llm_{v}_sign"]) * np.sign(m["gap_pct"])
            summary[f"rest_by_agree_{v}"] = m.groupby(agree)["rest"].agg(["size", "mean"]).round(5).to_dict()
    with open(out / "pending.jsonl", "w", encoding="utf-8") as fh:
        for p in pending:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    summary.update({"dry_run": L["dry_run"], "pending_calls": len(pending), "model": L["model"], "effort": L["effort"],
                    "prompt_version": PROMPT_VERSION})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"[llm] → {out}  (dry_run 이면 pending.jsonl 에 보낼 입력 {len(pending)}건)")
    return out
