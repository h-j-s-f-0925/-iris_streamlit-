import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

plt.style.use("bmh")

plt.rcParams["figure.figsize"] = 10, 10


class HotelReviewsPCA:
    def __init__(self, n_components=3):
        self.n_components = n_components

    def load_data(self):
        # 現在のディレクトリのパスを取得
        current_directory = os.getcwd()
        file_path = os.path.join(current_directory, "model_training/data", "hotel_reviews.csv")
        self.df = pd.read_csv(file_path)#このクラスの他のメソッドからもself.dfにアクセスしてデータフレームを利用することができます
        # このクラスのオブジェクトを作成した後に、オブジェクトからも直接アクセスすることができます。pca_analyzer.df.head()

    def preprocess_data(self):
        # 数値型のみ取得
        df_numeric = self.df.select_dtypes(include="number")
        # 不要カラムの削除
        self.data = df_numeric.drop(["年齢", "満足度"], axis=1)
        # 標準化
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data)
        self.df_scaled = pd.DataFrame(data_scaled, columns=self.data.columns)

    def perform_pca(self):
        # 主成分分析を実行し(fit())し、主成分の生成(transform())
        self.pca = PCA(n_components=self.n_components, random_state=0)
        feature = self.pca.fit_transform(self.df_scaled)
        
        # 主成分得点
        self.df_pca_score = pd.DataFrame(feature, columns=[f"PC{x+1}" for x in range(feature.shape[1])])
        
        # 固有ベクトル:各主成分の分散の大きさ
        eigenvector = self.pca.components_
        self.df_eighenvector = pd.DataFrame(eigenvector,index=[f"PC{x+1}" for x in range(len(eigenvector))],columns=self.df_scaled.columns)
        
        # 固有値、寄与率、累積寄与率
        combined_data = np.vstack((self.pca.explained_variance_, self.pca.explained_variance_ratio_, np.cumsum(self.pca.explained_variance_ratio_)))
        self.df_pca_eigenvalue_contribution_rate = pd.DataFrame(combined_data, index=["固有値", "寄与率", "累積寄与率"], columns=[f"PC{x+1}" for x in range(combined_data.shape[1])])

    def get_results(self):
        return self.df_pca_score, self.df_pca_eigenvalue_contribution_rate, self.df_eighenvector 
    
    def cumsum_variance_ratio_plot(self, ax):
        variance_ratio = self.pca.explained_variance_ratio_
        cumsum_variance_ratio = np.cumsum(variance_ratio)
        # 主成分の数に応じてx軸のティックを設定
        x_ticks = np.arange(1, len(cumsum_variance_ratio) + 1)
        ax.plot(x_ticks, cumsum_variance_ratio, "-o")
        ax.set_xticks(x_ticks)
        ax.set_xlabel('主成分')
        ax.set_ylabel('累積寄与率')
    
    def get_threshold_component(self, lower_threshold=0.7):
        """
        累積寄与率が指定された閾値を超える最初の主成分を取得します。
        
        Parameters:
        - lower_threshold: 下限の閾値
        
        Returns:
        - int: 閾値を超える最初の主成分の数
        """
        cumsum_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)
        component_index =  np.argmax(cumsum_variance_ratio >= lower_threshold)
        
        # 70%以上の累積寄与率を持つ主成分が存在しない場合はNoneを返す
        if cumsum_variance_ratio[component_index] < lower_threshold:
            return None
        
        return component_index + 1
    
    def eigenvalue_vector_heatmap(self):
        fig, ax = plt.subplots() 
        components_shape = self.pca.components_.T.shape
        
        sns.heatmap(pd.DataFrame(self.pca.components_,
                        columns=self.df_scaled.columns, 
                        index=[f"PC{x+1}" for x in range(self.pca.components_.shape[0])]),
            vmax=1.0, center=0.0, vmin=-1.0, square=True,annot=True, fmt='.2f', cbar=False)

        # カラーバーを横軸に設定
        cbar = fig.colorbar(ax.collections[0], ax=ax, orientation="horizontal", pad=0.20)
        plt.title("固有値ベクトル")
        ax.set_title("固有値ベクトル")
        plt.tight_layout()
        return fig
    
    def eigenvalue_vector_scatter_plot(self, i=1, j=2):
        fig, ax = plt.subplots()
        
        # 指定された主成分における観測変数の寄与度をプロットする
        for x, y, name in zip(self.pca.components_[i-1], self.pca.components_[j-1], self.df.columns):
            plt.text(x, y, name)

        plt.scatter(self.pca.components_[i-1], self.pca.components_[j-1], alpha=0.8)

        plt.xlabel(f'PC{i}')
        plt.ylabel(f'PC{j}')
        return fig
        
    def factor_loading_heatmap(self):
        fig, ax = plt.subplots()
        # 主成分負荷量
        factor_loading = self.pca.components_ * np.c_[np.sqrt(self.pca.explained_variance_)]

        df_factor_loading = pd.DataFrame(factor_loading,
                    index=[f'PC{x+1}' for x in range(len(factor_loading))],
                    columns=self.df_scaled.columns)
        
        sns.heatmap(df_factor_loading, vmax=1.0, center=0.0, vmin=-1.0, square=True, annot=True, fmt='.2f', cbar=False)
        cbar = ax.figure.colorbar(ax.collections[0], ax=ax, orientation="horizontal", pad=0.20)
        plt.title("主成分負荷量")
        plt.tight_layout()
        return fig
    
    def principal_component_score_plot(self, x=1, y=2):
        fig, ax = plt.subplots(figsize=(16,12))
        plt.title(f"第{x}, 第{y}主成分得点のプロット")
        
        # クラスの属性を使用してプロットを作成
        scatter = plt.scatter(self.df_pca_score.iloc[:, x-1], self.df_pca_score.iloc[:, y-1],
                    alpha=.8, c=list(self.df.iloc[:, -1])) #満足度の値ごとに色を設定
        cbar = ax.figure.colorbar(scatter, ax=ax, orientation="vertical", pad=0.20)
        # colorbarにラベルを追加
        cbar.set_label('満足度', rotation=0, labelpad=15, fontsize=10)
        plt.xlabel(f'PC{x}')
        plt.ylabel(f'PC{y}')
        return fig
    

pca_analyzer = HotelReviewsPCA(n_components=1)
pca_analyzer.load_data()
pca_analyzer.preprocess_data()
pca_analyzer.perform_pca()
df_pca_score, df_pca_eigenvalue_contribution_rate , df_eighenvector= pca_analyzer.get_results()
# pca_analyzer.eigenvalue_vector_scatter_plot()
