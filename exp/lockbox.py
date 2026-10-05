"""Lockbox 보호.

기능  : lockbox 구간(대상일 ≥ LOCKBOX_START)의 정답·점수가 개발 과정에 새지 않게 막고, 노출 이력을 기록함
구성  : LOCKBOX_START (유일한 기준값) · dev_only() · ledger (노출 장부) · freeze() · open_run() · close_run()
        · allow_training() · caveat()
역할  : 1) 기본은 DEV 전용 — 패널·분석·LLM·점검·main.py eval 은 lockbox 행을 아예 받지 않음
        2) 사전 등록 — lockbox 평가는 freeze 로 후보 spec 을 먼저 고정해야 하고, 고정한 spec 은 DEV 실행 기록이 있어야 함
        3) 1회 사용 — 고정한 계획당 한 번만 실행. 다시 쓰면 멈추고, 강제하면 그 사실이 장부와 보고서에 남음
        4) 학습 순서 — lockbox 평가 전에는 lockbox 기간을 학습에 쓰는 export 를 막음
        5) 노출 장부 — 이미 일어난 노출(EDA, main.py eval 등)을 기록하고 lockbox 보고서에 함께 실음

장부 파일: exp/lockbox_ledger.json  (git 에 올려 팀 전체가 같은 이력을 봄)
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

LOCKBOX_START = pd.Timestamp("2026-06-01")      # 바꾸려면 이 줄과 장부에 이유를 함께 남길 것
LEDGER = Path(__file__).resolve().parent / "lockbox_ledger.json"


# -----------------------------------------------------------------------------
# 기능  : DEV 행만 남김 (대상일 < LOCKBOX_START)
# input : df  target 열이 있는 표,  allow  True 면 그대로 돌려줌 (lockbox 평가에서만 씀)
# output: 걸러진 표
# -----------------------------------------------------------------------------
def dev_only(df, allow=False, col="target"):
    return df if allow else df[pd.to_datetime(df[col]) < LOCKBOX_START]


# -----------------------------------------------------------------------------
# 기능  : 장부 읽기·쓰기
# 구조  : {"lockbox_start", "exposures": [...], "plans": [...], "runs": [...]}
# -----------------------------------------------------------------------------
def load_ledger():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"lockbox_start": str(LOCKBOX_START.date()), "exposures": [], "plans": [], "runs": []}


def _save(ledger):
    LEDGER.write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")


def _now():
    return datetime.now().isoformat(timespec="seconds")


# -----------------------------------------------------------------------------
# 기능  : 노출 기록 추가 (lockbox 기간의 정답·점수를 본 모든 일)
# input : source (무엇이), detail (어떻게), models (노출된 모델 계열 목록)
# -----------------------------------------------------------------------------
def record_exposure(source, detail, models=()):
    led = load_ledger()
    led["exposures"].append({"time": _now(), "source": source, "detail": detail, "models": list(models)})
    _save(led)


# -----------------------------------------------------------------------------
# 기능  : 후보 spec 의 지문. 이름이 아니라 내용(신호·feature·결정 규칙·regime·선택 기준)으로 만듦
# -----------------------------------------------------------------------------
def spec_hash(spec):
    body = {k: v for k, v in spec.items() if k not in ("name", "seed")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:12]


# -----------------------------------------------------------------------------
# 기능  : DEV 실행 기록에 있는 spec 지문 집합 (runs/*/config.json 중 완료된 DEV 실행)
# -----------------------------------------------------------------------------
def _dev_specs(runs_dir):
    out = {}
    for cfg_path in Path(runs_dir).glob("*/config.json"):
        if not (cfg_path.parent / "DONE").exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg["split"].get("use_lockbox"):
            continue
        for c in cfg.get("candidates", []):
            out.setdefault(spec_hash(c), []).append(cfg_path.parent.name)
    return out


# -----------------------------------------------------------------------------
# 기능  : 사전 등록. lockbox 에서 평가할 후보를 고정함
# 검사  : 모든 후보 spec 이 DEV 실행 기록에 있어야 함 (DEV 에서 검증하지 않은 모델은 lockbox 에 못 올림)
#         이미 실행된 계획이 있으면 새 계획을 받지 않음
# input : cfg (use_lockbox = true 인 config),  runs_dir
# -----------------------------------------------------------------------------
def freeze(cfg, runs_dir):
    led = load_ledger()
    if led["runs"]:
        raise SystemExit(f"lockbox 는 이미 평가됨: {led['runs'][0]['plan']}. 새 계획을 받지 않음")
    seen = _dev_specs(runs_dir)
    plan = {"config": cfg["path"], "config_hash": cfg["hash"], "time": _now(), "candidates": {}}
    for c in cfg["candidates"]:
        h = spec_hash(c)
        if h not in seen:
            raise SystemExit(f"후보 '{c['name']}' (spec {h}) 의 DEV 실행 기록이 없음. 먼저 DEV config 로 돌릴 것")
        plan["candidates"][c["name"]] = {"spec_hash": h, "dev_runs": sorted(seen[h])}
    led["plans"] = [p for p in led["plans"] if p["config_hash"] != cfg["hash"]] + [plan]
    _save(led)
    print(f"[lockbox] 계획 고정: {cfg['hash']} 후보 {list(plan['candidates'])}")


# -----------------------------------------------------------------------------
# 기능  : lockbox 실행 허가. 고정된 계획과 config 가 같고 아직 실행 전이어야 함
# input : cfg, force (재실행 강제. 장부에 forced 로 남고 보고서에 오염 표시)
# output: 실행 기록 dict (close_run 에 넘김)
# -----------------------------------------------------------------------------
def open_run(cfg, force=False):
    led = load_ledger()
    plan = next((p for p in led["plans"] if p["config_hash"] == cfg["hash"]), None)
    if plan is None:
        raise SystemExit("고정된 계획이 없음. 먼저 python -m exp freeze <config>")
    for c in cfg["candidates"]:
        if plan["candidates"].get(c["name"], {}).get("spec_hash") != spec_hash(c):
            raise SystemExit(f"후보 '{c['name']}' 가 고정한 계획과 다름 (freeze 이후 수정됨)")
    if led["runs"] and not force:
        raise SystemExit(f"lockbox 는 이미 평가됨 ({led['runs'][0]['time']}). --force-lockbox 는 장부와 보고서에 오염으로 남음")
    return {"plan": cfg["hash"], "time": _now(), "forced": bool(led["runs"])}


def close_run(entry, out_dir):
    led = load_ledger()
    led["runs"].append({**entry, "result": str(out_dir)})
    if entry["forced"]:
        led["exposures"].append({"time": _now(), "source": "exp run --force-lockbox",
                                 "detail": f"lockbox 재평가 {entry['plan']}", "models": ["*"]})
    _save(led)


def evaluated():
    return bool(load_ledger()["runs"])


# -----------------------------------------------------------------------------
# 기능  : lockbox 기간을 학습에 써도 되는지 (export 등)
# 규칙  : lockbox 평가가 끝난 뒤에만 허용. 그 전에는 allow=True 일 때만 허용하고 노출로 기록
# input : until (학습 끝 날짜), allow, source
# -----------------------------------------------------------------------------
def allow_training(until, allow=False, source="export"):
    if pd.Timestamp(until) <= LOCKBOX_START or evaluated():
        return
    if not allow:
        raise SystemExit(f"{source}: lockbox 평가 전에 lockbox 기간({LOCKBOX_START.date()}~)을 학습에 쓸 수 없음. "
                         f"train_until 을 {LOCKBOX_START.date()} 이전으로 두거나 --allow-lockbox (노출로 기록)")
    record_exposure(source, f"lockbox 평가 전 학습에 lockbox 기간 포함 (train_until={until})", ["export"])


# -----------------------------------------------------------------------------
# 기능  : lockbox 보고서에 넣을 노출 이력·해석 주의 문구
# input : names  보고서의 후보 이름 목록
# output: markdown 문자열
# -----------------------------------------------------------------------------
def caveat(names):
    led = load_ledger()
    lines = ["## Lockbox 노출 이력", "",
             "아래 노출이 있었던 모델 계열은 lockbox 절대 점수가 '처음 보는 구간' 추정이 아님. "
             "후보 간 paired Δ 는 후보 설계가 lockbox 를 보고 정해지지 않았다면 해석 가능함.", ""]
    for e in led["exposures"]:
        lines.append(f"- {e['time']} · {e['source']} · {e['detail']} · 계열 {', '.join(e['models']) or '-'}")
    if any(r["forced"] for r in led["runs"]):
        lines += ["", "**경고: lockbox 를 강제로 재평가함. 이 결과는 오염됨.**"]
    return "\n".join(lines) + "\n"
