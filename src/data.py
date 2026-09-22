#!/usr/bin/env python
"""데이터를 읽고 기간을 나눔.

    ds = Dataset()
    train, test = ds.split("2026-06-01")

    for day in test:
        day.daily(days=5)          # 최근 5거래일 일봉
        day.price(days=1)          # 전날 장 마감 후부터 개장 전까지
        day.news(hours=24)
        day.reddit("stocks", hours=24)
        day.y                      # 그날 정답

day에서 꺼내는 모든 데이터는 예측 시점에서 잘림. 그 이후 데이터는 나오지 않음.
정답은 day.y 로만 확인할 수 있음.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .paths import DATASET as DATA
MARKET_TZ = "America/New_York"
CUTOFF = pd.Timedelta(hours=9, minutes=30)      # 장 시작

# 기준일의 시작. 표마다 시작이 다른데(일봉·시간봉 2023-10, Reddit 2024-01,
# 뉴스 2024-08-13) 모든 표가 갖춰진 날부터만 기준일로 씀. 그 앞의 일봉·시간봉은
# 지표를 계산할 때 과거로 거슬러 보는 용도로만 쓰임.
START = pd.Timestamp("2024-08-13")

# 전날 종가 대비 등락률로 다섯 구간을 나눔.
# 기준일 구간(2024-08-13 이후)의 실제 분포는 7.4 / 24.1 / 32.9 / 27.4 / 8.3 % 임.
RET_CUTS = (-2.5, -0.6, 0.6, 2.5)
LABELS = ("급하락", "하락", "보합", "상승", "급상승")
CRASH, DOWN, FLAT, UP, SURGE = 0, 1, 2, 3, 4   # schema.py 와 같은 값

# 평가에 쓰는 가중치. 행이 정답, 열이 예측임.
#
#   w[i][j] = (i - j)**2 * (|i - 2|)**2
#
#   (i-j)**2   거리가 멀수록 급격히 커짐. 급하락을 급상승으로 찍는 게 가장 큼
#   |i-2|**2   정답이 급등락이면 4배, 하락·상승이면 1배, 보합이면 0배
#
# 급등락을 놓치는 것이 하락·상승을 놓치는 것보다 네 배 무겁고,
# 정답이 보합인 날은 점수에 영향을 주지 않음.
_LEVEL = np.arange(5)
WEIGHT = ((_LEVEL[:, None] - _LEVEL[None, :]) ** 2) * (np.abs(_LEVEL - 2)[:, None] ** 2)

TABLES = ("price", "daily", "news", "earnings", "analyst")
_NEWS_LIGHT = ["known_at", "symbols", "source", "url", "title",
               "tone", "positive", "negative"]


def label_of(ret_pct) -> pd.Series:
    """등락률(%)을 0 급하락 / 1 하락 / 2 보합 / 3 상승 / 4 급상승 으로 바꿈."""
    return pd.cut(ret_pct, [-np.inf, *RET_CUTS, np.inf],
                  right=False, labels=list(range(5))).astype("Int8")


def score(y_true, y_pred, ret_pct=None) -> dict:
    """예측에 점수를 매김.

    주 지표는 `score` 임. 다섯 등급에 순서가 있으므로, 얼마나 멀리 틀렸는지를
    가중치로 반영함.

        score = 1 - Σ w·O / Σ w·E

    `O` 는 실제 혼동행렬, `E` 는 같은 분포에서 무작위로 찍었을 때의 기대치임.
    우연으로 맞은 몫을 빼기 때문에 **0 이 기준선**이 됨.

        +1    전부 맞힘
         0    정보가 없는 것과 같음
        음수  우연보다 못함

    한 클래스만 계속 찍으면 정확히 0 이 나옴. 전부 보합이든 전부 급상승이든
    마찬가지라, 안전하게 한쪽으로 몰아 찍어서는 점수가 오르지 않음.

    가중치(`WEIGHT`)에 세 가지가 들어 있음.

        급등락을 맞히는 게 가치가 큼    정답이 급등락인 행은 가중이 4배
        보합은 가치가 적음             정답이 보합인 행은 전부 0
        반대로 찍으면 크게 깎임         거리 제곱이라 멀수록 급격히 커짐

    기준일 구간 전체에서 실측하면 이럼. 보합인 날은 보합으로 두고 나머지만 전략대로 찍은 값임.

        급등락만 잡고 나머지 보합    +0.862   정확도 0.485
        방향만 맞고 세기는 틀림      +0.834   정확도 0.844
        급등락을 정반대로           -1.818   정확도 0.329

    정확도가 높아도 급등락을 놓치면 점수가 낮음. 가중이 비대칭이라 아주 나쁜
    예측은 -1 아래로 내려갈 수 있음.

    가중치가 (i-j)² 만인 보통의 quadratic weighted kappa 와는 다름. sklearn 의
    cohen_kappa_score(weights="quadratic") 와 값이 일치하지 않음.

    Args:
        y_true: 정답 label.
        y_pred: 예측 label.
        ret_pct: 안 써도 됨. 옛 호출부와 맞추려고 남겨 둔 인자임.

    Returns:
        score      주 지표. 1 이 완벽, 0 이 우연 수준
        accuracy   맞힌 비율
        direction  방향 적중률. 보합을 뺀 나머지에서 오르내림만 봄
        big_recall 실제 급등락 중 급등락으로 맞힌 비율
        big_prec   급등락으로 예측한 것 중 실제로 급등락이던 비율
    """
    t = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    n = len(t)

    observed = np.zeros((5, 5))
    np.add.at(observed, (t, p), 1)
    expected = np.outer(observed.sum(1), observed.sum(0)) / n
    denom = (WEIGHT * expected).sum()

    moved = t != FLAT
    side_t, side_p = np.sign(t - FLAT), np.sign(p - FLAT)
    big_t, big_p = np.isin(t, (CRASH, SURGE)), np.isin(p, (CRASH, SURGE))

    return {
        "score": float(1 - (WEIGHT * observed).sum() / denom) if denom else float("nan"),
        "accuracy": float((t == p).mean()),
        "direction": (float((side_t[moved] == side_p[moved]).mean())
                      if moved.any() else float("nan")),
        "big_recall": float((p[big_t] == t[big_t]).mean()) if big_t.any() else float("nan"),
        "big_prec": float((t[big_p] == p[big_p]).mean()) if big_p.any() else float("nan"),
        "n": int(n),
    }


def _ts(t):
    if t is None:
        return None
    x = pd.Timestamp(t)
    return x.tz_convert(MARKET_TZ).tz_localize(None) if x.tz is not None else x


class Day:
    """어떤 거래일 기준으로 다음 거래일 개장 직전까지 쓸 수 있는 데이터.

        day = ds.day("2026-02-20")      # 금요일
        day.target                       # 2026-02-24 (다음 거래일)
        day.cutoff                       # 2026-02-24 09:30
        day.y                            # 2026-02-24 정답

    기준일 장 마감 시점에서 다음 거래일을 예측하는 구조임. 주말 뉴스나 다음날
    개장 전 시간외 거래까지 모두 쓸 수 있음. 금요일로 부르면 월요일 09:30
    직전까지가 들어옴.

    구간은 days로 지정하고 거래일 기준임. 기준일을 포함해서 셈.

        day.daily(days=5)      # 2-13 ~ 2-20 (5거래일) + 그 뒤 cutoff 까지
        day.news(days=1)       # 2-20 00:00 ~ 2-24 09:30 (주말 포함)

    종목을 좁히려면 symbols를 넘긴다. Reddit은 시황 이야기라 종목 구분이 없음.
    """

    def __init__(self, ds, date, target=None):
        self.ds = ds
        self.date = pd.Timestamp(date).normalize()
        self.target = (pd.Timestamp(target).normalize() if target is not None
                       else ds.next_trading_day(self.date))
        if self.target is None:
            last = ds.dates()[-1]
            raise ValueError(
                f"{self.date:%Y-%m-%d} 다음 거래일이 데이터에 없음. "
                f"마지막 거래일은 {last:%Y-%m-%d} 라 그 전날까지만 기준일로 쓸 수 있음")
        self.cutoff = self.target + CUTOFF

    # ---------------------------------------------------------------- 구간

    def start_of(self, days=None, hours=None, since=None):
        """구간의 시작 지점을 정함. days는 거래일, hours는 시각 단위다."""
        if since is not None:
            return _ts(since)
        if hours is not None:
            return self.cutoff - pd.Timedelta(hours=hours)
        if days is not None:
            past = self.ds.dates(until=self.date + pd.Timedelta(days=1))
            past = past[-int(days):] if len(past) >= days else past
            return pd.Timestamp(past[0]) if len(past) else None
        return None

    def table(self, name, symbols=None, since=None, days=None, hours=None, **kw):
        return self.ds.table(name, self.start_of(days, hours, since), self.cutoff,
                             symbols=symbols, **kw)

    # ---------------------------------------------------------------- 표별

    def price(self, symbols=None, **kw):
        return self.table("price", symbols, **kw)

    def daily(self, symbols=None, **kw):
        return self.table("daily", symbols, **kw)

    def news(self, symbols=None, *, with_text=False, **kw):
        cols = kw.pop("columns", None) or (None if with_text else _NEWS_LIGHT)
        return self.table("news", symbols, columns=cols, **kw)

    def earnings(self, symbols=None, **kw):
        return self.table("earnings", symbols, **kw)

    def analyst(self, symbols=None, **kw):
        return self.table("analyst", symbols, **kw)

    def reddit(self, subreddit="stocks", kind="comments",
               since=None, days=None, hours=None):
        """그 구간에 올라온 글과 댓글. 종목 구분은 없음.

        kind 는 posts, comments, both 중 하나임.

        Reddit 에는 known_at 이 없고 올라온 시각(created_et)으로 자름. 본문은
        올라온 직후 수집되어 cutoff 에 알 수 있는 정보가 맞음. 단 score 와
        n_comments 는 36시간 뒤 다시 긁은 값이라 cutoff 시점에는 모르는 미래
        정보임. feature 로 쓰지 말 것.
        """
        return self.ds.reddit(subreddit, self.start_of(days, hours, since),
                              self.cutoff, kind)

    # ---------------------------------------------------------------- 묶어서

    def load(self, days=5, symbols=None, *, price=True, daily=True,
             news=True, earnings=False, analyst=False,
             reddit=None, with_text=False, reddit_kind="comments"):
        """필요한 것만 골라 한 번에 가져옴.

            x = day.load(days=5, news=False, reddit=["stocks", "investing"])
            x["daily"], x["price"], x["reddit"]["stocks"]

        Args:
            days: 거래일 기준 구간 길이. 기준일을 포함함.
            symbols: 종목을 좁힌다. price, daily, news, earnings, analyst에 적용됨.
            price / daily / news / earnings / analyst: 각각 가져올지 여부.
            reddit: 가져올 subreddit 목록. None이면 가져오지 않음.
            with_text: 기사 본문까지 가져올지 여부.
            reddit_kind: posts, comments, both 중 하나.

        Returns:
            이름별 DataFrame. Reddit은 subreddit 이름으로 한 번 더 나뉨.
        """
        out = {}
        if daily:
            out["daily"] = self.daily(symbols, days=days)
        if price:
            out["price"] = self.price(symbols, days=days)
        if news:
            out["news"] = self.news(symbols, days=days, with_text=with_text)
        if earnings:
            out["earnings"] = self.earnings(symbols, days=days)
        if analyst:
            out["analyst"] = self.analyst(symbols, days=days)
        if reddit:
            subs = [reddit] if isinstance(reddit, str) else list(reddit)
            out["reddit"] = {s: self.reddit(s, reddit_kind, days=days) for s in subs}
        return out

    # ---------------------------------------------------------------- 정답

    @property
    def symbols(self):
        """예측 대상 종목 목록."""
        return sorted(self.y.symbol)

    @property
    def y(self):
        """예측 대상일의 정답. symbol, ret_pct, label 이 들어 있음."""
        return self.ds.labels(self.target, self.target + pd.Timedelta(days=1))

    def __repr__(self):
        return (f"<Day {self.date:%Y-%m-%d} -> {self.target:%Y-%m-%d} "
                f"cutoff {self.cutoff:%m-%d %H:%M}>")


class Dataset:
    """날짜를 주면 그 시점에 쓸 수 있는 데이터와 정답을 꺼냄.

    Args:
        folder: 데이터 폴더. 생략하면 project/dataset을 씀.
        symbols: 다룰 종목을 직접 지정함. 채점할 때 대상을 좁히는 데 씀.
    """

    def __init__(self, folder=None, symbols=None):
        self.dir = Path(folder) if folder else DATA
        if not self.dir.exists():
            raise FileNotFoundError(f"{self.dir} 가 없음")
        self._files = {}
        self._labels = None
        self._only = set(symbols) if symbols is not None else None

    # ---------------------------------------------------------------- 읽기

    def _open(self, name: str) -> pq.ParquetFile:
        if name not in self._files:
            self._files[name] = pq.ParquetFile(self.dir / f"{name}.parquet")
        return self._files[name]

    def table(self, name: str, since=None, until=None, *,
              symbols=None, columns=None) -> pd.DataFrame:
        """표 하나를 [since, until) 구간으로 잘라서 읽음.

        기준은 datetime이 아니라 known_at임. 그 값을 실제로 알 수 있게 된
        시각이라는 뜻임. 1시간봉의 datetime은 봉이 시작한 시각이라 09:30 봉은
        10:30이 되어야 채워짐. datetime으로 자르면 09:30에 그 봉을 이미 아는
        셈이 되어 미래를 보게 됨.
        """
        pf = self._open(name)
        names = pf.schema_arrow.names
        lo, hi = _ts(since), _ts(until)
        # 대상 종목을 정해 뒀으면 그것만 보이게 함. 개발할 때와 채점할 때의
        # 환경을 같게 맞추려는 것임.
        if symbols is None:
            symbols = self._only

        need = columns
        if need is not None:
            need = list(need) + ["known_at"]
            if symbols is not None:
                need.append("symbols" if "symbols" in names else "symbol")
            need = list(dict.fromkeys(n for n in need if n in names))

        col = names.index("known_at")
        md = pf.metadata
        want = [i for i in range(md.num_row_groups)
                if (st := md.row_group(i).column(col).statistics) is None
                or ((hi is None or st.min < hi) and (lo is None or st.max >= lo))]
        if not want:
            return pd.DataFrame(columns=need or names)

        df = pf.read_row_groups(want, columns=need).to_pandas()
        if lo is not None:
            df = df[df.known_at >= lo]
        if hi is not None:
            df = df[df.known_at < hi]

        if symbols is not None:
            want_syms = {symbols} if isinstance(symbols, str) else set(symbols)
            if "symbol" in df.columns:
                df = df[df.symbol.isin(want_syms)]
            elif "symbols" in df.columns:        # 기사 하나에 종목이 여러 개 붙어 있음
                tagged = df.symbols.fillna("").str.split(",")
                df = df[tagged.apply(lambda xs: not want_syms.isdisjoint(xs))]
        return df.reset_index(drop=True)

    def reddit(self, subreddit="stocks", since=None, until=None,
               kind="comments"):
        """Reddit 글과 댓글. 시황 이야기라 종목 구분이 없음.

        kind 는 posts, comments, both 중 하나임. both면 둘을 합치고 kind
        column으로 구분함.

        [since, until) 은 created_et(올라온 시각) 기준임. score 와 n_comments 는
        36시간 뒤 2차 수집값이라 그 시점에는 알 수 없는 값임.
        """
        if kind == "both":
            parts = []
            for k in ("posts", "comments"):
                d = self.reddit(subreddit, since, until, k)
                if len(d):
                    parts.append(d.assign(kind=k))
            if not parts:
                return pd.DataFrame()
            return (pd.concat(parts, ignore_index=True)
                    .sort_values("created_et").reset_index(drop=True))

        path = self.dir / "reddit" / f"{subreddit}.{kind}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 가 없음. `ds.subreddits()` 로 이름을 확인하라")
        df = pq.read_table(path).to_pandas()
        lo, hi = _ts(since), _ts(until)
        if lo is not None:
            df = df[df.created_et >= lo]
        if hi is not None:
            df = df[df.created_et < hi]
        return df.reset_index(drop=True)

    def subreddits(self) -> list[str]:
        folder = self.dir / "reddit"
        return sorted({p.name.split(".")[0] for p in folder.glob("*.parquet")})

    # ---------------------------------------------------------------- label

    def labels(self, since=None, until=None) -> pd.DataFrame:
        """날짜별 정답. symbol, date, ret_pct, label 을 담는다.

        known_at이 16:00이라 장중에 꺼내는 데이터에는 섞이지 않음.
        """
        if self._labels is None:
            d = self.table("daily").dropna(subset=["ret"]).copy()
            d["ret_pct"] = d["ret"] * 100
            d["label"] = label_of(d["ret_pct"])
            if self._only is not None:
                d = d[d.symbol.isin(self._only)]
            self._labels = (d[["symbol", "date_et", "prev_close", "close",
                               "ret_pct", "label"]]
                            .rename(columns={"date_et": "date"})
                            .sort_values(["date", "symbol"]).reset_index(drop=True))
        out = self._labels
        if since is not None:
            out = out[out.date >= _ts(since)]
        if until is not None:
            out = out[out.date < _ts(until)]
        return out.reset_index(drop=True)

    def dates(self, since=None, until=None):
        """거래일 목록."""
        return sorted(self.labels(since, until).date.unique())

    def next_trading_day(self, date):
        """다음 거래일. 없으면 None.

        금요일을 주면 월요일이 나옴. 주말과 휴장일은 건너뛴다.
        """
        d = pd.Timestamp(date).normalize()
        after = [x for x in self.dates() if x > d]
        return pd.Timestamp(after[0]) if after else None

    def day(self, date):
        """기준일 하나. 다음 거래일 개장 직전까지를 다룬다."""
        return Day(self, date)

    def days(self, since=None, until=None):
        """기준일 목록. START(2024-08-13) 앞과 다음 거래일이 없는 마지막 날은 제외함."""
        every = self.dates()
        last = every[-1] if every else None
        lo = START if since is None else max(_ts(since), START)
        return [Day(self, d) for d in self.dates(lo, until)
                if last is None or d < last]

    def split(self, on, *, since=None, until=None):
        """기준일을 경계로 앞뒤를 나눔. 기준일 당일은 평가 쪽에 들어감.

        나누는 기준은 Day.target, 즉 예측 대상일임. 그래야 학습에 쓴 날의
        정답이 평가 구간으로 새지 않음.

        교차검증도 이 함수로 만듦. 기준일을 앞에서부터 옮겨 가며 여러 번
        부르면 walk-forward가 됨.

            for d in ["2026-03-01", "2026-05-01", "2026-07-01"]:
                train, test = ds.split(d)

        Returns:
            `(학습용 Day 리스트, 평가용 Day 리스트)`
        """
        on = pd.Timestamp(on).normalize()
        days = self.days(since, until)
        return ([d for d in days if d.target < on],
                [d for d in days if d.target >= on])

    @property
    def symbols(self) -> list[str]:
        """다룰 종목. symbols 인자가 있으면 그것을, 없으면 symbols.json을,
        그것도 없으면 데이터에 있는 종목 전부를 씀."""
        if self._only is not None:
            return sorted(self._only)
        path = self.dir / "symbols.json"
        if path.exists():
            v = json.loads(path.read_text())
            return sorted(v if isinstance(v, list) else sum(v.values(), []))
        return sorted(self.labels().symbol.unique())

    def balance(self, since=None, until=None) -> pd.DataFrame:
        """label 비율. 기본은 기준일 구간(START 이후)임."""
        v = self.labels(START if since is None else since, until).label.value_counts(normalize=True).sort_index()
        return pd.DataFrame({"이름": LABELS, "비율%": (v * 100).round(1).values})

    def __repr__(self) -> str:
        lab = self.labels()
        days = self.days()
        return (f"<Dataset {lab.symbol.nunique()}종목 "
                f"기준일 {days[0].date:%Y-%m-%d}~{days[-1].date:%Y-%m-%d} {len(days)}일 "
                f"(일봉 {lab.date.min():%Y-%m-%d}~{lab.date.max():%Y-%m-%d})>")
