import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")

cancer = load_breast_cancer()
x = cancer.data
t = cancer.target

df = pd.DataFrame(data=x, columns=cancer.feature_names)
df["Target"] = t

df = df[["worst radius", "mean radius", "Target"]]


from sklearn.model_selection import train_test_split
x_train, x_test, t_train, t_test = train_test_split(df.iloc[:, :-1], df["Target"], random_state=0, shuffle=True)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C=1, penalty="l2")
log_reg.fit(x_train, t_train)

# オッズ比
odds_ratio = np.exp(log_reg.coef_)

# plt.barh(y=cancer.feature_names, width=odds_ratio[0])
# plt.show()

# 推論
y_pred = log_reg.predict(x_test)

from sklearn import metrics

labels = list(set(t))

cm = metrics.confusion_matrix(t_test, y_pred, labels=[1, 0])

ax = sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=["正例(1:良性: benign)", "負例(0: 悪性: malignant)"], yticklabels=["正例(1:良性: benign)", "負例(0: 悪性: malignant)"])
# x軸のラベルを上に表示
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xlabel("Pridiction")
plt.ylabel("Target")

    
st.title("Cancer Classifier")


col1, col2, = st.columns([3, 5])

with col1:
    # サイドバー インプットデータ
    worstRadiusValue = st.sidebar.slider("worst radius", min_value=1, max_value=40)
    meanRadiusValue = st.sidebar.slider("mean radius", min_value=1, max_value=40)


    st.write("## Input Value")

    # インプットデータ(1行のデータフレーム)
    value_df = pd.DataFrame([], columns=["data", "worst radius", "mean radius"])
    record = pd.Series(["data", worstRadiusValue, meanRadiusValue], index=value_df.columns)
    # value_df = value_df.append(record, ignore_index=True)
    value_df = pd.concat([value_df, record.to_frame().T], ignore_index=True)
    value_df.set_index("data", inplace=True)
    st.write(value_df)

    st.write("### Odds Ratio")
    df_odds = pd.DataFrame(data=odds_ratio, columns=["worst radius","mean radius"], index=["odds"])
    df_odds
    st.write("worst radius が1増えると、カテゴリ１に当てはまる確率が約", round(odds_ratio[0][0], 2) ,"倍になる")
    st.write("mean radius が1増えると、カテゴリ１に属する確率が約", round(odds_ratio[0][1], 2) ,"倍になる")

    # 予測値のデータフレーム
    pred_probs = log_reg.predict_proba(value_df)
    pred_df = pd.DataFrame(pred_probs, columns=["malignan", "benign"], index=['probability'])

    st.write('## Prediction')
    st.write(pred_df)

    # 予測値の出力
    name = pred_df.idxmax(axis=1).tolist()
    st.write('## Result')
    st.write('この患者の腫瘍はきっと', '良性' if name[0] == 'benign' else '悪性', 'です！')
    
with col2:
    st.write("## Confusion Matrix")
    st.pyplot(plt.gcf()) # # plt.gcf() を使用して現在アクティブな図を取得します