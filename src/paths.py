"""프로젝트 안에서 쓰는 경로.

경로를 여기 한곳에 모아 둠. 폴더를 옮기거나 이름을 바꿀 때
이 파일만 고치면 됨.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "dataset"          # 데이터
ARTIFACTS = ROOT / "artifacts"      # 학습 결과를 저장하는 곳
