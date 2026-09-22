"""예측 결과의 형식. 하루치를 예측할 때마다 여기를 통과함."""

import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator

CRASH, DOWN, FLAT, UP, SURGE = 0, 1, 2, 3, 4


class Prediction(BaseModel):
    """예측 한 줄. predict가 돌려준 행은 모두 여기를 거친다."""

    symbol: str
    label: int

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("종목 이름이 비어 있음")
        return v

    @field_validator("label")
    @classmethod
    def _label(cls, v):
        if v not in (CRASH, DOWN, FLAT, UP, SURGE):
            raise ValueError(
                "label 은 0(급하락)/1(하락)/2(보합)/3(상승)/4(급상승) 중 하나여야 함. "
                f"받은 값: {v}")
        return v


def check_output(pred, symbols) -> pd.DataFrame:
    """predict 결과를 검사해서 정리된 DataFrame으로 돌려줌.

    DataFrame과 dict 리스트를 모두 받음. 아래를 어기면 그 자리에서 멈춤.

        - symbol, label column 이 있어야 함
        - label 은 0에서 4 사이의 정수만 됨
        - 같은 종목이 두 번 나오면 안 됨
        - day.symbols 를 빠짐없이, 그리고 그것만 채워야 함

    Args:
        pred: Model.predict 가 돌려준 값.
        symbols: 채워야 하는 종목 목록.

    Returns:
        `symbol` / `label` column. label 은 int.
    """
    if isinstance(pred, pd.DataFrame):
        records = pred.to_dict("records")
    elif isinstance(pred, (list, tuple)):
        records = list(pred)
    else:
        raise TypeError(f"predict 는 DataFrame 이나 dict 리스트를 돌려줘야 함. "
                        f"받은 것: {type(pred).__name__}")

    missing_cols = {"symbol", "label"} - set(records[0]) if records else {"symbol", "label"}
    if missing_cols:
        raise ValueError(f"predict 결과에 {sorted(missing_cols)} column 이 없음")

    rows, bad = [], []
    for i, r in enumerate(records):
        try:
            rows.append(Prediction(symbol=r["symbol"], label=r["label"]))
        except ValidationError as e:
            if len(bad) < 5:
                why = "; ".join(f"{d['loc'][0]}: {d['msg']}" for d in e.errors())
                bad.append(f"  {i}행 {r!r} -> {why}")
        except (ValueError, TypeError, KeyError) as e:
            if len(bad) < 5:
                bad.append(f"  {i}행 {r!r} -> {e}")
    if bad:
        raise ValueError("predict 결과가 형식에 맞지 않는다\n" + "\n".join(bad))

    out = pd.DataFrame([r.model_dump() for r in rows])

    dup = out.symbol[out.symbol.duplicated()].unique()
    if len(dup):
        raise ValueError(f"같은 종목이 두 번 나왔다: {sorted(dup)[:5]}")

    want, got = set(symbols), set(out.symbol)
    if want - got:
        raise ValueError(f"{len(want - got)}종목 예측 누락: {sorted(want - got)[:5]} ...")
    if got - want:
        raise ValueError(f"예측 대상이 아닌 종목이 들어왔다: {sorted(got - want)[:5]} ...")
    return out
