import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_breast_cancer

st.set_page_config(layout="wide")

st.title("Cancer Dashboard")

cancer = load_breast_cancer()
x = cancer.data
t = cancer.target

df = pd.DataFrame(data=x, columns=cancer.feature_names)
df["Target"] = t

from sklearn.model_selection import train_test_split
x_train, x_test, t_train, t_test = train_test_split(df.iloc[:, :-1], df["Target"], random_state=0, shuffle=True)

from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C=1, penalty="l2")
log_reg.fit(x_train, t_train)

# オッズ比
odds_ratio = np.exp(log_reg.coef_)


# vars_cat = [var for var in df.columns if var.startswith('cat')]
# vars_cont = [var for var in df.columns if var.startswith('cont')]
vars_cont = [var for var in df.columns]


# st.set_page_config(layout="wide")


# Graph (Pie Chart in Sidebar)
df_target = df["Target"].value_counts(normalize=True)
fig_target = go.Figure(data=[go.Pie(labels=df_target.index,
                                    values=df_target,
                                    hole=.3)])
fig_target.update_layout(showlegend=False,
                         height=200,
                         margin={'l': 20, 'r': 60, 't': 0, 'b': 0})
fig_target.update_traces(textposition='inside', textinfo='label+percent')

# Layout (Sidebar)
st.sidebar.markdown("## Settings")
# cat_selected = st.sidebar.selectbox('Categorical Variables', vars_cat)
cont_selected = st.sidebar.selectbox('Continuous Variables', vars_cont)
cont_multi_selected = st.sidebar.multiselect('Correlation Matrix', vars_cont,
                                     default=vars_cont)


# Categorical Variable Bar Chart in Content
# df_cat = df.groupby([cat_selected, 'target']).count()[['id']].reset_index()

# cat0 = df_cat[df_cat['target'] == 0]
# cat1 = df_cat[df_cat['target'] == 1]

# fig_cat = go.Figure(data=[
#     go.Bar(name='target=0', x=cat0[cat_selected], y=cat0['id']),
#     go.Bar(name='target=1', x=cat1[cat_selected], y=cat1['id'])
# ])

# fig_cat.update_layout(height=300,
#                       width=500,
#                       margin={'l': 20, 'r': 20, 't': 0, 'b': 0},
#                       legend=dict(
#                           yanchor="top",
#                           y=0.99,
#                           xanchor="right",
#                           x=0.99),
#                       barmode='stack')
# fig_cat.update_xaxes(title_text=None)
# fig_cat.update_yaxes(title_text='# of samples')

# Continuous Variable Distribution in Content
li_cont0 = df[df['Target'] == 0][cont_selected].values.tolist()
li_cont1 = df[df['Target'] == 1][cont_selected].values.tolist()

cont_data = [li_cont0, li_cont1]
group_labels = ['Target=0', 'Target=1']

fig_cont = ff.create_distplot(cont_data, group_labels,
                              show_hist=False,
                              show_rug=False)
fig_cont.update_layout(height=300,
                       width=500,
                       margin={'l': 20, 'r': 20, 't': 0, 'b': 0},
                       legend=dict(
                           yanchor="top",
                           y=0.99,
                           xanchor="right",
                           x=0.99)
                       )

# Correlation Matrix in Content
df_corr = df[cont_multi_selected].corr()
fig_corr = go.Figure([go.Heatmap(z=df_corr.values,
                                 x=df_corr.index.values,
                                 y=df_corr.columns.values)])
fig_corr.update_layout(height=800,
                       width=1300,
                       margin={'l': 20, 'r': 20, 't': 0, 'b': 0})

# Layout (Content)
left_column, right_column = st.columns(2)
# left_column.subheader('Categorical Variable Distribution: ' + cat_selected)
left_column.markdown("## Target Variables")
left_column.plotly_chart(fig_target, use_container_width=True)

right_column.subheader('Continuous Variable Distribution: ' + cont_selected)
# left_column.plotly_chart(fig_cat)
right_column.plotly_chart(fig_cont)

st.subheader('Correlation Matrix')
st.plotly_chart(fig_corr)

st.subheader('Odds Ratio')
# PlotlyのBarオブジェクトを作成
fig = go.Figure(data=[go.Bar(
    y=cancer.feature_names,
    x=log_reg.coef_[0],
    orientation='h'  # horizontal bar chartを作成
)])
# layout属性を設定して、グラフのサイズを変更
fig.update_layout(
    autosize=False,  # If True, the figure size will be automatically adjusted based on the dimensions of your plot elements.
    width=800,       # Set figure width
    height=600,      # Set figure height
    margin={'l': 20, 'r': 20, 't': 0, 'b': 0}
)
# Streamlitで表示
st.plotly_chart(fig)
