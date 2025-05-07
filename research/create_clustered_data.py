import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Segment eşleştirme fonksiyonu
def map_segment_rf_to_group(segment):
    if segment in ['champions_rf', 'loyal_customers_rf', 'cant_loose_rf']:
        return 'High Value Users'
    elif segment in ['hibernating_rf', 'at_risk_rf', 'about_to_sleep_rf', 'need_attention_rf']:
        return 'Churn Risk'
    elif segment in ['new_customers_rf', 'potential_loyalists_rf', 'promising_rf']:
        return 'Growth Potential Users'
    else:
        return 'Other'

def run_segmentation(df, features, model_type="kmeans", model_params=None):
    if model_params is None:
        model_params = {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    if model_type == "kmeans":
        model = KMeans(**model_params)
    elif model_type == "agglomerative":
        model = AgglomerativeClustering(**model_params)
    else:
        raise ValueError("Geçersiz model tipi. Seçenekler: 'kmeans', 'dbscan', 'agglomerative'")

    clusters = model.fit_predict(X_scaled)

    df_copy = df.copy()
    df_copy["cluster"] = clusters
    df_copy["mapped_segment"] = df_copy["segment_rf"].apply(map_segment_rf_to_group)

    return df_copy

def evaluate_clusters_by_segment(df):
    cluster_summary = df.groupby('cluster')['mapped_segment'].value_counts().unstack().fillna(0)
    print("Segment Dağılımı (Adet):")
    print(cluster_summary)

    print("\nSegment Dağılımı (Yüzde):")
    cluster_pct = cluster_summary.div(cluster_summary.sum(axis=1), axis=0) * 100
    print(cluster_pct.round(2))

    print("\nBaskın Segment (Her Küme İçin):")
    dominant_segments = cluster_summary.idxmax(axis=1)
    for cluster_id, segment in dominant_segments.items():
        print(f"Cluster {cluster_id}: {segment}")

    cluster_summary.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Cluster Dağılımı ve Segmentler (Mapped)")
    plt.xlabel("Cluster")
    plt.ylabel("Frekans")
    plt.show()

# Veriyi yükle
df = pd.read_csv("/home/eplinux/ikas_case_study/final_datasets/pca_cluster_ready_data.csv")

print("\n---------------------------------------------------------------------")
print("EXPECTED RESULTS:")
print("High Value Users: Champions, Loyal Customers, Can't Loose")
print("Churn Risk: Hibernating, At Risk, About To Sleep, Need Attention")
print("Growth Potential Users: Potential Loyalists, Promising")
print("---------------------------------------------------------------------\n")

# Kümeleme için kullanılacak özellikler
features = [
    "0","1","2","3","4","5","6","7","8","9","10","11"
]

# KMeans ile kümeleme
kmeans_df = run_segmentation(df, features, model_type="kmeans", model_params={"n_clusters": 3, "random_state": 42})
print("KMeans Kümeleme Sonuçları:")
evaluate_clusters_by_segment(kmeans_df)

# Agglomerative Clustering ile kümeleme
agg_df = run_segmentation(df, features, model_type="agglomerative", model_params={"n_clusters": 3})
print("Agglomerative Kümeleme Sonuçları:")
evaluate_clusters_by_segment(agg_df)

# Kümeleme sonuçlarını görselleştirme
sns.pairplot(kmeans_df, hue="cluster", vars=features)
plt.suptitle("Küme Görselleştirme - KMeans", y=1.02)
plt.show()
