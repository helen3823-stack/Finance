# 실행 가이드 (처음 보는 팀원용)

> 이 문서만 보고 **설치 → 동작 확인 → 실험 → 결과 확인 → 제출**까지 할 수 있게 정리했음.
> 모르는 용어는 맨 아래 [용어](#12-용어) 참고.

---

## 0. 한눈에 보기

```text
1. 준비 (한 번만)          uv 설치 → 코드 받기 → 데이터 넣기 → 패키지 설치
2. 동작 확인 (5분)         python -m exp check e0
3. 실험                   python -m exp run configs/e1_baseline.toml   ← config 파일만 바꿔 가며 반복
4. 결과 확인              runs/<실험이름>_.../report.md 열기
5. 제출                   python -m exp export ... → python main.py check
```

**꼭 지킬 것 세 가지**

1. `python main.py eval --split 2026-06-01` 처럼 **2026-06-01 이후를 채점하지 않기** (lockbox, [7절](#7-lockbox-규칙-반드시-지킬-것))
2. 코드를 고치지 말고 **`configs/*.toml` 만 고쳐서** 실험하기
3. 실험 결과를 말할 때는 **`runs/` 폴더 이름(해시 포함)을 같이 공유**하기

---

## 1. 준비 (한 번만)

### 1-1. uv 설치

uv 는 파이썬과 패키지를 한 번에 맞춰 주는 도구임. 팀원 모두 같은 버전을 쓰게 됨.

Windows (PowerShell):

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치 뒤 터미널을 **새로 열고** 확인:

```bash
uv --version
```

### 1-2. 코드 받기

```bash
git clone https://github.com/jangs03/Finance.git
```

```bash
cd Finance
```

> 이미 받아 둔 사람은 `git pull` 로 최신 코드를 받음.

### 1-3. 데이터 넣기

1. 과제 안내 자료(w4-1 PDF 5쪽)의 Dropbox 링크에서 `dataset.tar.gz` 를 받음
   (로그인 창이 뜨면 아래쪽 "또는 다운로드만 하고 계속 진행하세요" 클릭)
2. 압축을 풀어 **`Finance` 폴더 안에 `dataset` 폴더**가 오게 둠

```text
Finance/
├── dataset/              ← 여기
│   ├── daily.parquet
│   ├── price.parquet
│   ├── news.parquet
│   ├── earnings.parquet
│   ├── analyst.parquet
│   ├── symbols.json
│   └── reddit/  (parquet 16개)
├── src/
├── exp/
└── ...
```

> `dataset/` 은 3.7GB 라 git 에 올리지 않음 (`.gitignore` 에 있음). 각자 넣어야 함.

### 1-4. 패키지 설치

```bash
uv sync
```

처음에는 torch 때문에 몇 분 걸림. 끝나면 `.venv/` 폴더가 생김.

이후 모든 명령은 앞에 `uv run` 을 붙여 실행함 (예: `uv run python -m exp ...`).
아래 예시는 짧게 쓰려고 `uv run` 을 생략했으니 **직접 칠 때는 붙일 것.**

> uv 를 쓰지 않고 이미 파이썬 3.11 이상이 있다면 `pip install pandas pyarrow numpy scikit-learn pydantic matplotlib` 로도 돌아감.

---

## 2. 폴더 지도

| 위치 | 내용 | 고치나? |
|---|---|---|
| `src/model.py` | **제출 파일.** feature 계산 + 모델 + 예측. 실험도 이 파일을 그대로 씀 | 새 규칙을 추가할 때만 |
| `src/data.py` 등 나머지 `src/` | 과제 제공 코드 | ❌ (채점 때 원본으로 덮어씀) |
| `configs/*.toml` | 실험 설정. 단계별로 하나씩 | ✅ **여기만 고침** |
| `exp/` | 실험 실행 코드 (`python -m exp ...`) | ❌ |
| `exp/lockbox_ledger.json` | lockbox 노출 기록 장부 | ❌ (프로그램이 씀) |
| `runs/` | 실험 결과 (자동 생성, git 에 안 올라감) | ❌ |
| `eda/` | EDA 노트북과 함수 | 분석할 때만 |
| `docs/` | 설계 문서(`experiment_design.md`), 이 가이드 | — |
| `artifacts/` | 제출용 학습 결과 (`pipeline.pkl`) | ❌ (export 가 씀) |

---

## 3. 동작 확인 (처음 한 번, 5분)

### 3-1. 재현성 점검 (E0)

```bash
python -m exp check e0
```

- 약 1분. 마지막 줄이 **`[E0] 통과`** 면 정상
- 실험용 feature 와 제출용 feature 가 같은 값을 내는지 날짜 8개로 대조함
- `실패` 가 나오면 실험하지 말고 바로 공유할 것

### 3-2. 제출 형식 점검

```bash
python main.py check
```

- 마지막 줄 **`형식 확인 완료`** 면 정상 (점수는 계산하지 않음)

---

## 4. 실험 돌리기

### 4-1. 기본 명령

```bash
python -m exp run configs/e1_baseline.toml
```

- 화면에 `[run] draw 0 fold A: ...` 가 9줄 나오고, `[run] 완료 → runs/...` 로 끝남
- E1 기준 약 3분. 처음 실행할 때는 데이터 캐시를 만드느라 조금 더 걸림
- **같은 config 와 같은 코드로 다시 돌리면** 계산하지 않고 이전 결과 폴더를 알려 줌
  (일부러 다시 계산하려면 `--rerun`)

### 4-2. 단계별 순서

설계 문서(`docs/experiment_design.md`)의 순서 그대로임. **앞 단계 결과를 보고 다음 단계 config 의 빈자리를 채운 뒤** 실행함.

| 단계 | 명령 | 무엇을 보나 | 다음 단계에서 바꿀 곳 |
|---|---|---|---|
| E0 | `python -m exp check e0` | feature 재현성 | — |
| G0 | `python -m exp check g0` | 하루 예측 시간, LLM 사용 조건 | — |
| E1 | `python -m exp run configs/e1_baseline.toml` | 갭 규칙 · 결정 규칙 비교 | E2 의 후보 `B` |
| E2 | `python -m exp run configs/e2_structure.toml` | 5-class vs 분해 구조 | E3 의 `signal`·`decision` type |
| E3 | `python -m exp run configs/e3_ablation.toml` | 가격·이벤트 feature 묶음 | E4 의 후보 `price` features |
| E4 | `python -m exp run configs/e4_nonprice.toml` | 애널리스트·뉴스·Reddit 증분 | E5 의 신호·features |
| E4 표 | `python -m exp analyze configs/e4_nonprice.toml` | 갭 크기별 조건부 표 | — |
| E4-L | `python -m exp llm configs/e4_llm.toml` | LLM 이벤트 파서 (선택, [8절](#8-llm-선택)) | — |
| E5 | `python -m exp run configs/e5_regime.toml` | 당일 regime 에 따른 보정 | E6 의 후보 `final` |
| E6 | [7절](#7-lockbox-규칙-반드시-지킬-것) 순서대로 | 최종 1회 평가 | `final.toml` |
| E7 | [9절](#9-제출) | 제출 | — |

---

## 5. 결과 읽는 법

결과는 `runs/<config이름>_<config해시>_<코드해시>/` 에 쌓임.

| 파일 | 내용 |
|---|---|
| **`report.md`** | **먼저 이것만 보면 됨.** 후보별 지표 표 + 기준선 대비 비교 + 판정 |
| `compare.csv` | 기준선 대비 ΔScore, 95% 신뢰구간, fold 별 Δ, 판정 |
| `metrics.csv` | 후보 · fold · draw · 범위(all / seen / unseen)별 지표 |
| `predictions.parquet` | 모든 예측 (다시 분석할 때) |
| `params.jsonl` | 학습 구간에서 고른 k · 경계값 |
| `config.json`, `env.json` | 무엇으로, 어떤 환경에서 돌렸는지 (재현용) |

### report.md 의 주요 열

| 열 | 뜻 | 좋은 방향 |
|---|---|---|
| `score` | 과제 점수 (0 = 찍기 수준, 1 = 완벽) | ↑ |
| `delta` | 후보 score − 기준선 score (같은 날·같은 종목에서 비교) | ↑ |
| `ci_low` ~ `ci_high` | delta 의 95% 신뢰구간 | `ci_low > 0` 이면 확실히 좋음 |
| `fold_A/B/C` | 기간별 delta | 셋 다 양수가 좋음 |
| `unseen_delta` | 처음 보는 종목에서의 delta | ↑ |
| `flip_index` / `flip_diff` | 급락↔급상승을 반대로 찍은 정도 (0~2) / 기준선 대비 증가량 | ↓ |
| `pred_extreme` / `extreme_ratio` | 급등락으로 찍은 비율 / 기준선 대비 배수 | 1 근처 |

### 판정

| 판정 | 뜻 | 할 일 |
|---|---|---|
| `adopt` | 모든 채택 조건 통과 | 다음 단계에 반영 |
| `hold` | 일부 조건 실패 (`✗` 로 표시된 항목) | 보통 반영하지 않음. 이유를 기록 |
| `reject` | 신뢰구간 상한 < 0, 또는 0.005 넘게 나빠진 기간이 2개 이상 | 제외. 이것도 결과로 기록 |

> 채택 기준 숫자는 `exp/config.py` 의 `DEFAULTS["adopt"]` 에 있고, config 에 `[adopt]` 를 적어 바꿀 수 있음.

---

## 6. 실험 바꾸는 법 (config 수정)

config 의 후보 하나는 이렇게 생김.

```toml
[[candidates]]
name = "core+scale"                                   # 결과표에 나올 이름 (자유)
signal = { type = "regress", features = ["core", "scale"] }
decision = { type = "k" }
regime = { type = "none" }                            # 생략 가능
```

### 쓸 수 있는 이름

| 칸 | 이름 | 뜻 |
|---|---|---|
| `signal.type` | `gap` | 프리마켓 갭을 그대로 신호로 씀 (학습 없음) |
| | `regress` | 당일 수익률 회귀 |
| | `classify5` | 5-class 직접 분류 |
| | `decomp` | 방향 + 크기 분해 (`params = { magnitude = "ordinal" 또는 "multiclass" }`) |
| `signal.features` | `core` | gap_pct, gap_rank |
| | `scale` | vol5, vol20, atr14, big_rate60, zgap |
| | `cross` | mkt_gap, resid_gap, beta60 |
| | `event` | earn, surp_sign, surp_abs, surp_pct_w |
| | `analyst` | an_n, an_up, an_down, an_tgt_chg, an_gap_agree |
| | `news` | news_n, news_abn, tone_*, news_missing |
| | `reddit`, `news_llm_anon`, `news_llm_raw`, `reddit_llm` | **연구용** (제출 불가). config 의 `[data] research` 에도 적어야 함 |
| `decision.type` | `flat` · `identity` · `k` · `sym2` · `asym` · `4t` | 스칼라 신호용 결정 규칙 (단순 → 복잡) |
| | `decomp` | `signal.type = "decomp"` 전용. `params = { conf_gate = true }` 면 D2-C |
| `regime.type` | `none` · `tercile` · `linear` | 당일 regime 보정. `params = { var = "reg_med_vol20" }` 등 |

- 묶음 이름 대신 열 이름을 직접 써도 됨 (예: `features = ["gap_pct", "vol20"]`)
- 이름을 잘못 쓰면 실행 즉시 **가능한 이름 목록**을 보여 주고 멈춤
- 비교 기준은 `[eval] baseline = "후보이름"`

### 하지 말 것

- `seed` 를 바꾸지 않음 (바꾸면 universe 구성이 달라져 이전 결과와 비교가 안 됨)
- `[split]` 의 fold 를 2026-06-01 이후로 늘리지 않음 (자동으로 막힘)

---

## 7. Lockbox 규칙 (반드시 지킬 것)

**lockbox = 대상일 2026-06-01 ~ 2026-09-14.** 최종 모델을 고른 뒤 **딱 한 번** 평가하는 구간임.
여기 점수를 미리 보면 그 점수에 맞춰 모델을 고르게 되어 최종 성능을 믿을 수 없게 됨.

| ✅ 해도 됨 | ❌ 하지 말 것 |
|---|---|
| `python main.py check` (점수 없음) | `python main.py eval --split 2026-06-01` (자동으로 막힘) |
| `python main.py eval --split 2026-03-01` (2026-03 ~ 05 만 채점) | `--allow-lockbox`, `--force-lockbox` (쓰면 장부에 "노출"로 남음) |
| `python main.py predict --date 2026-07-01` (정답 없이 예측만) | 노트북에서 `dataset/` 을 직접 읽어 2026-06 이후 정답·점수 계산 |

### E6: lockbox 평가 순서 (팀이 함께 정한 뒤 한 사람만)

1. `configs/e6_lockbox.toml` 의 `baseline`, `final` 후보를 DEV 에서 고른 spec 으로 채움
   (두 spec 모두 E1~E5 중 하나로 **이미 DEV 실행된 것**이어야 함)
2. 사전 등록:

```bash
python -m exp freeze configs/e6_lockbox.toml
```

3. 1회 평가:

```bash
python -m exp run configs/e6_lockbox.toml
```

4. 장부를 git 에 올려 팀 전체가 "평가 완료"를 알게 함:

```bash
git add exp/lockbox_ledger.json
```

- freeze 뒤에 후보를 고치면 실행이 멈춤 → 고치지 말 것
- 두 번째 실행은 막힘
- 장부 확인: `python -m exp lockbox`

> 이미 일어난 노출(EDA, 점수 1회 확인)은 장부에 기록되어 있음. 그래서 보고서에서는 lockbox 의 **절대 점수보다 최종 후보 − 기준선의 차이(delta)** 를 주 결과로 씀.

---

## 8. LLM (선택)

뉴스 제목·Reddit 글을 LLM 으로 구조화해, 갭에 없는 정보가 있는지 보는 **연구용** 단계임. 제출 모델에는 쓰지 않음.

1. 기본은 **dry run** (API 호출 없음). 보낼 입력만 저장됨:

```bash
python -m exp llm configs/e4_llm.toml
```

→ `runs/e4_llm_..._llm/pending.jsonl` 에 보낼 입력과 건수가 나옴

2. 실제로 호출하려면 (팀 예산 확인 후):
   - 패키지: `uv add anthropic`
   - 인증: API 키를 환경 변수 `ANTHROPIC_API_KEY` 로 설정 (키를 코드나 config 에 적지 말 것)
   - `configs/e4_llm.toml` 에서 `dry_run = false`, `max_calls` 를 원하는 상한으로
3. 같은 입력은 `exp/cache/llm/` 에 저장돼서 다시 돌려도 **API 를 다시 부르지 않음**
4. 결과를 실험에 붙이려면 `configs/e4_nonprice.toml` 의 `research` 에 `"news_llm_anon"` 등을 추가

---

## 8-1. 텍스트 파이프라인 (T 단계)

설계는 `docs/text_pipeline.md`. 뉴스 제목으로 **"크기(급등락 여부)"** 를 예측하고, 방향은 갭이 맡는 구조임.

### 준비 (한 번만)

1. **LM 금융 사전**: [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) CSV 를 받아 `resources/` 폴더에 둠 (파일 이름에 `MasterDictionary` 가 들어가면 됨)
2. 텍스트 자원 만들기 (LM 사전 + 회사 이름 → `artifacts/text_resources.json`, 제출에도 쓰임):

```bash
python -m exp text resources
```

3. 확인: E0 가 텍스트 feature 까지 대조함

```bash
python -m exp check e0
```

### FinBERT (Colab, 연구용)

FinBERT 결과는 채점 기간(10월)의 새 제목에는 없어서 **제출 모델에는 못 씀**. 비교 실험(T3, T5)용임.

1. 채점할 제목 내보내기 → `exp/cache/text/titles_for_finbert.parquet` (약 34만 개)

```bash
python -m exp text export-titles
```

2. Colab 에 `colab/finbert_colab.py` 와 위 파일을 올리고 GPU 런타임에서:
   `!pip install -q transformers pyarrow` → `!python finbert_colab.py titles_for_finbert.parquet finbert_scores.parquet`
3. 받은 `finbert_scores.parquet` 를 가져오기:

```bash
python -m exp text import-finbert finbert_scores.parquet
```

4. `configs/t5_vectorizer.toml` 의 `finbert_emb` 후보 주석을 풀고 다시 실행

### 실험 순서

| 단계 | 명령 | 질문 |
|---|---|---|
| T1 | `python -m exp run configs/t1_relevance.toml` | 태그만 / 제목 언급 / 그 회사만 언급 중 어디에 크기 신호가 있나 |
| T2 | `python -m exp run configs/t2_count.toml` | 어떤 "건수" 가 가장 쓸 만한가 (count 기준선) |
| T3 | `python -m exp run configs/t3_tone.toml` | 톤(GDELT · LM · FinBERT)이 count 위에 정보를 더하나 |
| T4 | `python -m exp run configs/t4_event.toml` | 신규성·사건 유형이 count 위에 정보를 더하나 |
| T5 | `python -m exp run configs/t5_vectorizer.toml` | vectorizer 별 텍스트 분류기 ablation |
| T6 | `python -m exp llm configs/t6_llm.toml` | LLM silver label · placebo · 사건 파서 (기본 dry run) |
| 표 | `python -m exp analyze configs/t1_relevance.toml` | 같은 갭 구간 안에서 텍스트 범주별 급등락 비율 |
| 설명 | `python -m exp explain configs/t5_vectorizer.toml --candidate tfidf_uni` | 묶음별 block permutation, 단어 계수, 안정성 |

- T 단계는 lockbox 를 쓴 뒤의 실험이라 **Holm 보정**(`[adopt] multiplicity = "holm"`)이 채택 조건에 들어감
- 텍스트 분류기는 `regime = { type = "textclf", params = { input = "text", vectorizer = "tfidf_uni" } }` 처럼 config 이름만 바꿔 씀 (vectorizer 후보: `count_uni` `tfidf_uni` `tfidf_bi` `tfidf_char` `tfidf_stop` `tfidf_df` `lm_vocab` `hashing`)

---

## 9. 제출

1. lockbox 평가(E6)가 끝난 뒤, `configs/final.toml` 의 후보를 E6 의 `final` 과 같게 맞춤
2. 학습해서 저장:

```bash
python -m exp export configs/final.toml --candidate final
```

→ `artifacts/pipeline.pkl` 과 사람이 읽는 `artifacts/pipeline.json` 이 생김

3. 형식 확인:

```bash
python main.py check
```

4. 제출물은 **`src/model.py` 와 `artifacts/`** 뿐임. 둘 다 git 에 올림

> lockbox 평가 전에 export 하면 lockbox 기간을 학습에 넣을 수 없다며 멈춤 (정상 동작).
> 연구용 feature(Reddit, LLM)를 쓴 후보는 저장을 거부함 (채점 환경에서 만들 수 없기 때문).

---

## 10. 자주 나는 오류

| 메시지 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: No module named 'pandas'` 등 | 패키지 미설치, 또는 `uv run` 을 빠뜨림 | `uv sync` 후 `uv run python ...` 으로 실행 |
| `dataset 가 없음` / `FileNotFoundError ... dataset` | 데이터 위치가 다름 | [1-3](#1-3-데이터-넣기) 처럼 `Finance/dataset/` 에 둘 것 |
| `split=2026-06-01 는 lockbox ... 안임` | lockbox 채점 시도 | 정상 차단. `--split 2026-03-01` 처럼 DEV 기간만 |
| `고정된 계획이 없음` | E6 를 freeze 없이 실행 | `python -m exp freeze configs/e6_lockbox.toml` 먼저 |
| `DEV 실행 기록이 없음` | E6 후보를 DEV 에서 한 번도 안 돌림 | 그 spec 을 DEV config 에 넣어 먼저 `run` |
| `lockbox 는 이미 평가됨` | 누군가 이미 E6 를 실행함 | 다시 하지 말고 `python -m exp lockbox` 로 결과 위치 확인 |
| `signal.type 'xxx' 없음. 가능: [...]` | config 이름 오타 | 메시지에 나온 이름 중에서 고름 |
| `같은 config·코드의 결과가 있음` | 이미 돌린 실험 | 그 폴더를 보면 됨. 다시 계산은 `--rerun` |
| 한글이 깨짐 (`main.py`) | Windows 콘솔 인코딩 | PowerShell 에서 `$env:PYTHONUTF8 = 1` 후 다시 실행 |
| 디스크가 부족함 | 캐시 누적 | `exp/cache/`, `eda/cache/` 폴더를 지워도 됨 (다시 만들어짐) |

---

## 11. 명령 요약

| 하고 싶은 것 | 명령 |
|---|---|
| 재현성 점검 | `python -m exp check e0` |
| 예측 시간·LLM 조건 점검 | `python -m exp check g0` |
| 실험 실행 | `python -m exp run configs/<파일>.toml` |
| 같은 실험 강제 재계산 | `python -m exp run configs/<파일>.toml --rerun` |
| 갭 조건부 표 | `python -m exp analyze configs/e4_nonprice.toml` |
| LLM 스크리닝 | `python -m exp llm configs/e4_llm.toml` |
| lockbox 사전 등록 | `python -m exp freeze configs/e6_lockbox.toml` |
| lockbox 장부 보기 | `python -m exp lockbox` |
| 제출용 학습 | `python -m exp export configs/final.toml --candidate final` |
| 제출 형식 확인 | `python main.py check` |
| DEV 기간 채점 | `python main.py eval --split 2026-03-01` |
| 하루 예측 | `python main.py predict --date 2026-05-01` |

---

## 12. 용어

| 용어 | 뜻 |
|---|---|
| 기준일 / 대상일 | 기준일 장 마감 뒤에 서서, 다음 거래일(대상일)의 종가 등락을 맞힘 |
| cutoff | 대상일 09:30. 이 시각 이전에 알 수 있던 정보만 씀 |
| 프리마켓 갭 (`gap_pct`) | 대상일 08:00 시간외 가격 ÷ 기준일 종가 − 1 (%) |
| DEV | 실험·선택에 쓰는 기간 (대상일 ~ 2026-05-31) |
| lockbox | 최종 1회 평가용 기간 (2026-06-01 ~). [7절](#7-lockbox-규칙-반드시-지킬-것) |
| fold | DEV 안의 평가 기간 A(2025-07~10) · B(2025-11~2026-02) · C(2026-03~05) |
| universe / draw | 채점처럼 20종목(아는 10 + 처음 보는 10)을 고른 구성 / 그 구성을 바꾼 횟수 (3번) |
| seen / unseen | 학습에 쓴 종목 / 학습에서 뺀 종목 |
| baseline | 비교 기준 후보 |
| paired Δ | 같은 날·같은 종목에서 잰 두 후보의 점수 차이 |
| 95% 신뢰구간 (CI) | 날짜를 다시 뽑아 1,000번 계산한 Δ 의 범위. 하한이 0보다 크면 개선이 우연이 아님 |
