# 基本ライブラリのインポート
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def iris_logistic_model_training():
    # データセット読み込み
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # 目標値
    df["target"] = iris.target

    # 目標値を数字から花の名前に変更
    df.loc[df["target"]==0, "target"] = "setosa"
    df.loc[df["target"]==1, "target"] = "versicolor"
    df.loc[df["target"]==2, "target"] = "virginica"

    # 予測モデル構築
    x = iris.data[:, [0, 2]]
    y = iris.target

    # ロジスティック回帰
    clf = LogisticRegression(random_state=0)
    result = clf.fit(x, y)
    
    return result