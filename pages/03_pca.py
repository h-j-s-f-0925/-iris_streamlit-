# 基本ライブラリのインポート
import pandas as pd
import  matplotlib.pyplot as plt
from model_training import hotel_review_pca
# # Streamlit のインポート
import streamlit as st

st.set_page_config(layout="centered")

# # サイドバー(入力画面)
st.sidebar.markdown("## 主成分数")
input_n_components = st.sidebar.slider("n_components", min_value=1, max_value=9, step=1, value=3)
# petalValue = st.sidebar.slider("petal length (cm)", min_value=0.0, max_value=10.0, step=0.1)

st.sidebar.markdown("## 散布図：主成分の組み合わせ")
# i の選択
i = st.sidebar.selectbox('Select value for i', range(1, input_n_components + 1))

# i が選択された場合、j の選択肢から i を除外
options_for_j = list(range(1, input_n_components + 1))
options_for_j.remove(i)

# j の選択
j = st.sidebar.selectbox('Select value for j', options_for_j)


# HotelReviewsPCAをインスタンス化
pca_analyzer = hotel_review_pca.HotelReviewsPCA(n_components=input_n_components)

# データの読み込み
pca_analyzer.load_data()

# 前処理
pca_analyzer.preprocess_data()

# 主成分分析の実行
pca_analyzer.perform_pca()

# 結果の取得
df_pca_score, df_pca_eigenvalue_contribution_rate, df_eighenvector = pca_analyzer.get_results()


# メインパネル(出力画面)
st.title("主成分分析")
# st.image("src/images/iris.jpg")

st.write('## 主成分得点')
st.write(df_pca_score)

st.write('## 固有値・寄与率・累積寄与率')
st.write(df_pca_eigenvalue_contribution_rate)

st.write('## 固有値ベクトル')
st.write(df_eighenvector)

st.write('## 累積寄与率グラフ')
fig, ax = plt.subplots() 
pca_analyzer.cumsum_variance_ratio_plot(ax)
st.pyplot(fig)

# 閾値を超える最初の主成分を取得
n_components_for_threshold = pca_analyzer.get_threshold_component()
if n_components_for_threshold :
    st.write("第", n_components_for_threshold ,"主成分までで元データの70%以上を表現できています。")

st.write("## 固有値ベクトル")
fig = pca_analyzer.eigenvalue_vector_heatmap()
st.pyplot(fig)

st.write("## 固有値ベクトルの散布図")
fig = pca_analyzer.eigenvalue_vector_scatter_plot(i=i, j=j)
st.pyplot(fig)

st.write("## 主成分負荷量")
fig = pca_analyzer.factor_loading_heatmap()
st.pyplot(fig)

st.write("## 主成分得点の散布図")
fig = pca_analyzer.principal_component_score_plot(x=i, y=j)
st.pyplot(fig)

