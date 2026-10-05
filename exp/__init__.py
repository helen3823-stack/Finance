"""실험 하네스 (G0 ~ E7).

기능  : config(TOML) 하나로 후보 모델들을 같은 분할·같은 universe 에서 학습·평가하고,
        paired bootstrap 으로 기준선 대비 ΔScore 를 판정함
구성  : config   설정 읽기·검사·지문(hash)·시드
        panel    연구용 (종목 × 대상일) 패널. src.model.build_features 를 그대로 씀
        splits   DEV fold · universe draw (seen 10 + unseen 10) · lockbox 보호
        metrics  score 와 보조 지표
        stats    paired moving-block bootstrap, 채택/보류/제외 판정
        runner   실험 실행·저장 (runs/<이름>_<hash>/)
        checks   E0 재현성, G0 실행 시간
        analysis E4 갭 조건부 표
        llm      E4-L LLM 이벤트 파서 (익명화·캐시)
역할  : 실험마다 바뀌는 것은 configs/*.toml 의 이름과 값뿐이고, 코드 구조는 그대로 둠

사용
    python -m exp run     configs/e1_baseline.toml
    python -m exp check   e0 | g0
    python -m exp analyze configs/e4_nonprice.toml
    python -m exp llm     configs/e4_llm.toml
    python -m exp export  configs/final.toml --candidate NAME
"""
