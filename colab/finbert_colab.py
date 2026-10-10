"""FinBERT 채점 (Google Colab 에서 실행, GPU 권장).

기능  : 대표 뉴스 제목마다 FinBERT 감성 확률(p_pos, p_neg, p_neu)과 문장 임베딩(mean pooling)을 계산
구성  : main() 하나. 입력 parquet → 배치 추론 → 출력 parquet
역할  : 로컬에서 만든 titles_for_finbert.parquet 를 채점해 결과 파일만 돌려줌 (연구용 feature)
        채점 구간(10월)의 새 제목은 채점할 수 없으므로 제출 모델에는 쓰지 않음

사용 (Colab)
    1) 로컬에서  python -m exp text export-titles   → exp/cache/text/titles_for_finbert.parquet
    2) Colab 에 이 파일과 titles_for_finbert.parquet 를 올림
    3) !pip install -q transformers pyarrow
       !python finbert_colab.py titles_for_finbert.parquet finbert_scores.parquet
    4) finbert_scores.parquet 를 내려받아 로컬에서
       python -m exp text import-finbert finbert_scores.parquet

입력 : parquet [title_id, title]
출력 : parquet [title_id, p_pos, p_neg, p_neu, emb_0 … emb_767]  (+ 같은 이름의 .json 에 모델·리비전 기록)
수식 : p = softmax(logits),  레이블 순서는 model.config.id2label 로 맞춤 (모델마다 다름)
       emb = Σ_t mask_t · h_t / Σ_t mask_t   (마지막 층 토큰 벡터의 padding 제외 평균, w4-2 p33)
재현성: eval 모드 · no_grad · 같은 모델 리비전이면 결과가 같음. 리비전 해시를 .json 에 남김
"""

import json
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MODEL = "ProsusAI/finbert"
BATCH = 256
MAX_LEN = 64            # 뉴스 제목은 짧아서 64 토큰이면 충분 (512 한계에 닿지 않음, w4-2 p37)


def main(src, dst, model_name=MODEL):
    d = pd.read_parquet(src)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    clf = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()
    enc = AutoModel.from_pretrained(model_name).to(dev).eval()
    labels = {v.lower(): int(k) for k, v in clf.config.id2label.items()}
    order = np.argsort(d["title"].str.len().to_numpy())          # 길이순 배치로 padding 을 줄임 (결과는 원래 순서로 복원)
    probs = np.zeros((len(d), 3), np.float32)
    embs = np.zeros((len(d), enc.config.hidden_size), np.float16)
    with torch.no_grad():
        for i in range(0, len(d), BATCH):
            idx = order[i:i + BATCH]
            x = tok(d["title"].iloc[idx].tolist(), padding=True, truncation=True, max_length=MAX_LEN,
                    return_tensors="pt").to(dev)
            p = torch.softmax(clf(**x).logits, dim=-1).float().cpu().numpy()
            probs[idx] = p[:, [labels["positive"], labels["negative"], labels["neutral"]]]
            h = enc(**x).last_hidden_state
            m = x["attention_mask"].unsqueeze(-1).to(h.dtype)
            embs[idx] = ((h * m).sum(1) / m.sum(1)).float().cpu().numpy().astype(np.float16)
            if (i // BATCH) % 50 == 0:
                print(f"{i:,}/{len(d):,}", flush=True)
    out = pd.DataFrame({"title_id": d["title_id"].astype(str), "p_pos": probs[:, 0], "p_neg": probs[:, 1],
                        "p_neu": probs[:, 2]})
    out = pd.concat([out, pd.DataFrame(embs, columns=[f"emb_{j}" for j in range(embs.shape[1])])], axis=1)
    out.to_parquet(dst, index=False)
    meta = {"model": model_name, "revision": getattr(clf.config, "_commit_hash", None), "max_len": MAX_LEN,
            "n": len(out), "torch": torch.__version__}
    with open(dst.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print("done", meta)


if __name__ == "__main__":
    main(*sys.argv[1:3], *(sys.argv[3:4] or [MODEL]))
