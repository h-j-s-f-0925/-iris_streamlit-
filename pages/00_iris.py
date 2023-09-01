# 基本ライブラリのインポート
import os
import pandas as pd
from PIL import Image

from model_training import iris_model_training
# # Streamlit のインポート
import streamlit as st

st.set_page_config(layout="centered")

result = iris_model_training.iris_logistic_model_training()

# サイドバー(入力画面)
sepalValue = st.sidebar.slider("sepal length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petalValue = st.sidebar.slider("petal length (cm)", min_value=0.0, max_value=10.0, step=0.1)


# メインパネル(出力画面)
st.title("Iris Classifier")
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# image_path = os.path.join(CURRENT_DIR, "images", "iris.jpg")
# image = Image.open(image_path)
# st.write(f"Current directory: {CURRENT_DIR}")
# if os.path.exists(image_path):
#     st.image(image)
# else:
#     st.error(f"Image not found at: {image_path}")
image = Image.open("./images/iris.jpg")
st.image(image)

# ローカルで表示されるが、streamlit share 上でエラー
# st.image("pages/images/iris.jpg")
st.write('## Input Value')

# インプットデータ(1行のデータフレーム)
value_df = pd.DataFrame([], columns=["data", "sepal length(cm)", "petal length(cm)"])
record = pd.Series(["data", sepalValue, petalValue], index=value_df.columns)
# value_df = value_df.append(record, ignore_index=True)
value_df = pd.concat([value_df, record.to_frame().T], ignore_index=True)
value_df.set_index("data", inplace=True)

# 入力値の値
st.write(value_df)

# 予測値のデータフレーム
pred_probs = result.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs, columns=['setosa', 'versicolor', 'virginica'], index=['probability'])

st.write('## Prediction')
st.write(pred_df)

# 予測値の出力
name = pred_df.idxmax(axis=1).tolist()
st.write('## Result')
st.write('このアイリスはきっと', str(name[0]), 'です！')