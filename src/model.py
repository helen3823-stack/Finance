"""제출 파일.

필요한 건 두 가지임.

    load_model()      학습을 마친 모델을 돌려줌
    Model.predict()   Day 하나를 받아 symbol / label 을 돌려줌 (label 은 0~4)

학습 방법이나 위치에는 제약이 없음. 노트북에서 돌려도 되고 스크립트를 따로
만들어도 됨. 결과를 저장해 두고 load_model()에서 읽어오기만 하면 됨.

day에서 꺼내는 데이터는 예측 시점에서 잘려 있고, 정답인 day.y는 학습에만 씀.
predict 안에서 day.y를 쓰면 미래를 보는 셈임.

완성된 예시는 example/sample_model.py 에 있음.
"""

import pandas as pd

from .data import FLAT


class Model:
    """전부 보합으로 예측함. 여기서부터 고쳐 나가면 됨."""

    def predict(self, day):
        """symbol / label 을 돌려줌.

        label 은 0 급하락 / 1 하락 / 2 보합 / 3 상승 / 4 급상승.
        day.symbols 에 있는 종목을 빠짐없이, 그것만 채워야 함.
        """
        return pd.DataFrame({"symbol": day.symbols, "label": FLAT})


def load_model():
    """학습을 마친 모델을 돌려줌.

    채점할 때 이 함수로 모델을 가져와 predict만 돌린다. 학습은 다시 하지 않으므로
    배운 내용을 파일로 저장해 두고 여기서 읽어야 함.

    클래스를 통째로 pickle하면 안 됨. 노트북에서 정의한 클래스는 다른 곳에서
    읽을 때 깨짐. 학습 결과만 저장하고 클래스는 이 파일에 둠.

        from .paths import ARTIFACTS

        def load_model():
            with open(ARTIFACTS / "model.pkl", "rb") as f:
                clf = pickle.load(f)
            return Model(clf)

    torch를 쓴다면 state_dict를 저장하고 여기서 불러옴.
    저장할 것이 없다면 그냥 Model()을 돌려줘도 됨.
    """
    return Model()
