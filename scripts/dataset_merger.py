import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


class ProcessedDataGenerator:
    def __init__(self, base_path, demog_data, output_path):
        self.base_path = base_path
        self.demog_data = demog_data
        self.output_path = output_path
        self.threshold = 0.75
        self.merged_df = None
        self.highly_correlated_features = set()
        self.categorical_cols = []
        self.pca_df = None

    def load_data(self):
        self.payment_df = pd.read_csv(os.path.join(self.base_path, "payment_extended.csv"))
        self.purchase_df = pd.read_csv(os.path.join(self.base_path, "purchase_intended.csv"))
        self.segmented_df = pd.read_csv(os.path.join(self.base_path, "rf_segmented_users.csv"))
        self.categorical_df = pd.read_csv(self.demog_data)

    def preprocess_and_merge(self):
        merged = pd.merge(self.payment_df, self.purchase_df, on='user_id', how='outer')
        merged = pd.merge(merged, self.segmented_df, on='user_id', how='outer')
        merged = pd.merge(merged, self.categorical_df, on='user_id', how='outer')

        merged = merged.loc[:, ~merged.columns.str.startswith("Unnamed:")]
        merged = merged.drop(["first_event_type", "last_event_before_first_payment"], axis=1)
        merged.rename(columns={"segment": "segment_rf"}, inplace=True)
        merged["segment_rf"] = merged["segment_rf"] + "_rf"
        merged['steps_before_first_payment'].fillna(merged['steps_before_first_payment'].max(), inplace=True)

        self.merged_df = merged

        # Kategorik sütunları kaydet
        self.categorical_cols = self.merged_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    def compute_correlations(self):
        drop_cols = self.categorical_cols
        corr_matrix = self.merged_df.drop(columns=drop_cols, errors="ignore").corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    self.highly_correlated_features.add(corr_matrix.columns[i])

    def apply_pca(self):
        drop_cols = set(self.categorical_cols + list(self.highly_correlated_features) + ["user_id", "segment_rf", "event_sequence"])
        numerical_data = self.merged_df.drop(columns=drop_cols, errors="ignore")

        scaler = StandardScaler()
        scaled = scaler.fit_transform(numerical_data)

        pca = PCA(n_components=0.95)
        components = pca.fit_transform(scaled)

        self.pca_df = pd.DataFrame(data=components)

    def save_results(self):
        # Kategorik değişkenler ve user_id'yi içeren orijinal alt küme
        categorical_part = self.merged_df[["user_id"] + list(self.categorical_cols) + ["segment_rf"]]

        # PCA sonuçlarını user_id ile birleştir
        final_df = pd.concat([categorical_part, self.pca_df], axis=1)

        # Kaydet
        final_df.to_csv(self.output_path, index=False)


    def run(self):
        self.load_data()
        self.preprocess_and_merge()
        self.compute_correlations()
        self.apply_pca()
        self.save_results()

        print("Çıkarılan yüksek korelasyona sahip özellikler:", self.highly_correlated_features)


"""def main():
    features_path = "/home/eplinux/ikas_case_study/feature_data"
    output_path = "/home/eplinux/ikas_case_study/final_datasets/final_data.csv"
    demog_data = "/home/eplinux/ikas_case_study/data/advanced_user_profiles_with_uuid.csv"

    processor = ProcessedDataGenerator(features_path, demog_data, output_path)
    processor.run()


if __name__ == "__main__":
    main()"""
