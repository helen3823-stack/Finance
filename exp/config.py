"""실험 설정.

기능  : configs/*.toml 을 읽어 기본값을 채우고, 검사하고, 내용 지문(hash)을 냄
구성  : DEFAULTS · load() · fingerprint() · derive_seed() · environment()
역할  : 재현성의 기준점. 같은 config + 같은 코드 + 같은 데이터 → 같은 결과
        결과 폴더 이름에 config hash 가 들어가므로 무엇으로 돌린 결과인지 항상 추적됨
"""

import copy
import hashlib
import json
import platform
import subprocess
import sys
import tomllib
from importlib import metadata
from pathlib import Path

from .lockbox import LOCKBOX_START

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
CACHE = ROOT / "exp" / "cache"

# config 에 없는 값은 여기서 채움. 키 이름을 바꿀 때는 여기와 configs/ 만 바꾸면 됨
DEFAULTS = {
    "name": "unnamed",
    "stage": "E?",
    "seed": 20261005,
    "data": {"groups": ["core", "scale", "cross", "event", "regime"], "research": []},
    "split": {
        "folds": [
            {"name": "A", "start": "2025-07-01", "end": "2025-11-01"},
            {"name": "B", "start": "2025-11-01", "end": "2026-03-01"},
            {"name": "C", "start": "2026-03-01", "end": "2026-06-01"},
        ],
        "embargo_days": 14,
        "n_draws": 3,
        "n_seen": 10,
        "n_unseen": 10,
        "universe_size": 20,
        "lockbox_start": str(LOCKBOX_START.date()),   # exp/lockbox.py 가 유일한 기준. config 로 바꿀 수 없음
        "use_lockbox": False,
    },
    "eval": {"baseline": None, "bootstrap": 1000, "block_days": 5},
    "adopt": {
        "min_positive_folds": 3,     # 채택: Δ>0 인 fold 수
        "reject_negative_folds": 2,  # 제외: Δ < −reject_fold_tol 인 fold 수
        "reject_fold_tol": 0.005,    # 노이즈 수준의 음수는 악화로 세지 않음 (paired SE ≈ 0.003~0.005)
        "worst_fold_tol": 0.02,      # 채택: 최악 fold Δ ≥ −tol
        "flip_tol": 0.02,            # 채택: flip 지수 증가 ≤ tol
        "extreme_band": 0.25,        # 채택: 예측 급등락 비율이 기준선의 ±25% 안
        "min_gain": 0.0,             # 보류: 평균 Δ 가 이 값 미만이면 비용 대비 효과 부족
    },
    "candidates": [],
}


# -----------------------------------------------------------------------------
# 기능  : 중첩 dict 병합 (b 가 a 를 덮어씀, 목록은 통째로 교체)
# -----------------------------------------------------------------------------
def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


# -----------------------------------------------------------------------------
# 기능  : config 읽기 + 기본값 + 검사
# input : path  TOML 경로
# output: dict  (cfg["hash"] = 내용 지문 12자리, cfg["path"] = 경로)
# 검사  : 후보 이름 중복, baseline 존재, fold 가 lockbox 와 겹치지 않음, lockbox 시작일 고정, 미등록 키
# -----------------------------------------------------------------------------
def load(path):
    path = Path(path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = set(raw) - set(DEFAULTS) - {"llm", "analysis", "export"}
    if unknown:
        raise ValueError(f"{path.name}: 알 수 없는 최상위 키 {sorted(unknown)}")
    cfg = _merge(DEFAULTS, raw)
    names = [c["name"] for c in cfg["candidates"]]
    if len(names) != len(set(names)):
        raise ValueError(f"{path.name}: 후보 이름 중복 {names}")
    if cfg["eval"]["baseline"] is not None and cfg["eval"]["baseline"] not in names:
        raise ValueError(f"{path.name}: eval.baseline '{cfg['eval']['baseline']}' 가 후보에 없음 {names}")
    lb = cfg["split"]["lockbox_start"]
    if lb != str(LOCKBOX_START.date()):
        raise ValueError(f"{path.name}: split.lockbox_start 는 config 로 바꿀 수 없음 (exp/lockbox.py: {LOCKBOX_START.date()})")
    for f in cfg["split"]["folds"]:
        if f["end"] > lb and not cfg["split"]["use_lockbox"]:
            raise ValueError(f"{path.name}: fold {f['name']} 끝 {f['end']} 이 lockbox 시작 {lb} 을 넘음")
    cfg["hash"] = hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:12]
    cfg["path"] = str(path)
    return cfg


# -----------------------------------------------------------------------------
# 기능  : 목적별로 독립된 시드를 만듦. 한 곳의 난수 사용이 다른 곳 결과를 바꾸지 않게 함
# input : base  기본 시드,  *keys  목적 이름·번호 (예: "draw", 2)
# output: 0 ~ 2³²−1 정수
# -----------------------------------------------------------------------------
def derive_seed(base, *keys):
    h = hashlib.sha256(json.dumps([base, *keys], default=str).encode()).digest()
    return int.from_bytes(h[:4], "little")


# -----------------------------------------------------------------------------
# 기능  : 코드·데이터 지문
#         code  src/model.py, src/data.py, exp/*.py 내용의 sha256
#         data  dataset/ 파일들의 (이름, 크기, 수정 시각)
# output: {"code": 12자리, "data": 12자리}
# -----------------------------------------------------------------------------
def fingerprint():
    files = [ROOT / "src" / "model.py", ROOT / "src" / "data.py", *sorted((ROOT / "exp").glob("*.py"))]
    code = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()[:12]
    ds = sorted((ROOT / "dataset").rglob("*.parquet"))
    sig = json.dumps([(p.relative_to(ROOT).as_posix(), p.stat().st_size, int(p.stat().st_mtime)) for p in ds])
    return {"code": code, "data": hashlib.sha256(sig.encode()).hexdigest()[:12]}


# -----------------------------------------------------------------------------
# 기능  : 실행 환경 기록 (python, 패키지 버전, git commit)
# -----------------------------------------------------------------------------
def environment():
    pkgs = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "anthropic"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = None, None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs,
            "git_commit": commit, "git_dirty": dirty, **fingerprint()}
