import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class UserBehaviorClustering:
    def __init__(self, df_path, n_clusters=3, numeric_cols=None, categorical_cols=None):
        self.df_path = df_path
        self.df = pd.read_csv(self.df_path)
        self.n_clusters = n_clusters
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.pipeline = None
        self.score = None

    def load_and_prepare_data(self):
        # Event sequence string olarak hazırlanır
        self.df["event_sequence_str"] = self.df["event_sequence"].apply(lambda x: " ".join(eval(x)))

    def build_pipeline_and_cluster(self):
        # Kullanılmayacak kolonlar
        exclude_cols = {"user_id", "event_sequence", "event_sequence_str"}

        # Otomatik kolon belirleme
        if self.numeric_cols is None:
            self.numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns.difference(exclude_cols).tolist()

        if self.categorical_cols is None:
            self.categorical_cols = [
                col for col in self.df.select_dtypes(include=["object", "category", "bool"]).columns
                if col not in exclude_cols
            ]

        # Ön işleyici
        preprocessor = ColumnTransformer([
            ("tfidf", TfidfVectorizer(), "event_sequence_str"),
            ("num", StandardScaler(), self.numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)
        ])

        # Pipeline oluştur
        self.pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("clustering", KMeans(n_clusters=self.n_clusters, random_state=42))
        ])

        # Fit + Predict
        self.df["behavior_cluster"] = self.pipeline.fit_predict(self.df)

        # Silhouette skoru
        X_all = self.pipeline.named_steps["preprocessing"].transform(self.df)
        self.score = silhouette_score(X_all, self.df["behavior_cluster"])
        print("Silhouette Skoru:", round(self.score, 3))

    def visualize_clusters(self):
        print("PCA ile 2D görselleştiriliyor...")
        X_all = self.pipeline.named_steps["preprocessing"].transform(self.df)
        X_2d = PCA(n_components=2).fit_transform(X_all.toarray())

        plt.figure(figsize=(8, 6))
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=self.df["behavior_cluster"], cmap="viridis")
        plt.title("Kullanıcı Kümeleri (PCA)")
        plt.xlabel("PCA1")
        plt.ylabel("PCA2")
        plt.show()

    def analyze_clusters(self):
        for i in range(self.n_clusters):
            print(f"\nCluster {i} için en yaygın olaylar:")
            cluster_users = self.df[self.df['behavior_cluster'] == i]
            all_events = " ".join(cluster_users['event_sequence_str'].tolist()).split()
            most_common = pd.Series(all_events).value_counts().head(5)
            print(most_common)

    def assign_behavior_segments(self, cluster_labels):
        self.df["behavior_segment"] = self.df["behavior_cluster"].map(cluster_labels)

    def cross_tabulate_segments(self):
        crosstab = pd.crosstab(self.df["behavior_segment"], self.df["segment_rf"], normalize='index') * 100
        print("\nDavranış Segmenti vs RF Segment Karşılaştırması (%):")
        print(crosstab)
        return crosstab

    def save_pipeline(self, path="saved_model/pipeline.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"Pipeline '{path}' dosyasına kaydedildi.")

    def load_pipeline(self, path="saved_model/pipeline.pkl"):
        self.pipeline = joblib.load(path)
        print(f"Pipeline '{path}' dosyasından yüklendi.")

    def run(self, cluster_labels=None, visualize=True):
        self.load_and_prepare_data()
        self.build_pipeline_and_cluster()

        if cluster_labels:
            self.assign_behavior_segments(cluster_labels)
            self.cross_tabulate_segments()

        self.analyze_clusters()

        if visualize:
            self.visualize_clusters()

        return self.df

if __name__ == "__main__":
    clustering = UserBehaviorClustering(
        df_path="/home/eplinux/ikas_case_study/final_datasets/final_data.csv",
        n_clusters=3
    )

    cluster_label_map = {
        0: "Growth Potential",
        1: "Churn Risk",
        2: "High Value Users"
    }

    df_with_clusters = clustering.run(cluster_labels=cluster_label_map)

    # Kullanıcı ve tahmin edilen davranış segmentini içeren dataframe
    user_segment_df = df_with_clusters[["user_id", "behavior_segment"]]

    # Kayıt işlemi
    output_path = "/home/eplinux/ikas_case_study/result_show/user_behavior_segments.csv"
    user_segment_df.to_csv(output_path, index=False)
    print(f"\nKullanıcı segmentleri '{output_path}' dosyasına kaydedildi.")

