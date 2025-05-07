import pandas as pd
import ast
from collections import Counter
from nltk import ngrams
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineer:
    def __init__(self, input_file, feature_data_path):
        self.df = pd.read_csv(input_file)
        self.feature_data_path = feature_data_path
        self.df1 = self.df.copy()
        self.df2 = self.df.copy()
        self.df3 = self.df.copy()

    def generate_purchase_intended_features(self):
        # PURCHASE INTENDED FEATURES
        self.df1['timestamp'] = pd.to_datetime(self.df1['timestamp'])
        self.df1['day'] = self.df1['timestamp'].dt.date

        # Her kullanıcı için toplam kaç farklı gün verisi var?
        user_total_days = self.df1.groupby('user_id')['day'].nunique().reset_index()
        user_total_days.columns = ['user_id', 'total_days']

        # Her kullanıcı için "payment_success" veya "checkout_start" içeren günleri al
        filtered_df = self.df1[self.df1['event_type'].isin(['payment_success', 'checkout_start'])]
        user_target_days = filtered_df.groupby('user_id')['day'].nunique().reset_index()
        user_target_days.columns = ['user_id', 'purchase_intented_event_days']

        # Merge ve oran hesapla
        df1_target = pd.merge(user_total_days, user_target_days, on='user_id', how='left')
        df1_target['purchase_intented_event_days'] = df1_target['purchase_intented_event_days'].fillna(0)
        df1_target['purchase_intented_event_days_ratio'] = df1_target['purchase_intented_event_days'] / df1_target['total_days']

        df1_target.to_csv(f"{self.feature_data_path}/purchase_intended.csv")

    def generate_rf_segments(self):
        self.df2['timestamp'] = pd.to_datetime(self.df2['timestamp'])
        today = self.df2['timestamp'].max() + pd.Timedelta(days=1)

        df_filtered = self.df2[self.df2["event_type"].isin(["payment_success", "checkout_start"])]
        df_recency = df_filtered.groupby('user_id')['timestamp'].max().reset_index()
        df_recency['recency'] = (today - df_recency['timestamp']).dt.days

        df_payment_success = self.df2[self.df2['event_type'] == 'payment_success']
        df_frequency = df_payment_success.groupby('user_id').size().reset_index(name='frequency')

        rf = df_recency[['user_id', 'recency']].merge(df_frequency, on='user_id', how='left')
        rf['frequency'] = rf['frequency'].fillna(0)

        rf['recency_score'] = pd.qcut(rf['recency'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
        rf['frequency_score'] = pd.qcut(rf['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        rf['RF_score'] = rf['recency_score'].astype(str) + rf['frequency_score'].astype(str)

        def segment_rf_only(row):
            r, f = row['recency_score'], row['frequency_score']
            if r == 5 and f in [4, 5]: return 'champions'
            elif r == 5 and f == 1: return 'new_customers'
            elif r in [4, 5] and f in [2, 3]: return 'potential_loyalists'
            elif r == 4 and f == 1: return 'promising'
            elif r in [3, 4] and f in [4, 5]: return 'loyal_customers'
            elif r == 3 and f == 3: return 'need_attention'
            elif r == 3 and f in [1, 2]: return 'about_to_sleep'
            elif r in [1, 2] and f == 5: return 'cant_loose'
            elif r in [1, 2] and f in [3, 4]: return 'at_risk'
            elif r in [1, 2] and f in [1, 2]: return 'hibernating'
            else: return 'other'

        rf['segment'] = rf.apply(segment_rf_only, axis=1)
        rf_cleaned = rf[['user_id', 'segment']]
        rf_cleaned.to_csv(f"{self.feature_data_path}/rf_segmented_users.csv", index=False)
        self.rf_data = rf

    def visualize_rf_segments(self):
        if not hasattr(self, 'rf_data'):
            print("RF segment verisi bulunamadı. Önce generate_rf_segments() çağırılmalı.")
            return

        rf = self.rf_data
        segment_order = rf['segment'].value_counts().index
        colors = ['red' if seg in ['cant_loose', 'at_risk'] else 'gray' for seg in segment_order]

        plt.figure(figsize=(10, 6))
        sns.countplot(data=rf, x='segment', order=segment_order, palette=colors)
        plt.title('Distribution of Customer Segments')
        plt.xlabel('Segment')
        plt.ylabel('Number of Users')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def parse_event_sequence(self, x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except Exception as e:
                print(f"Hatalı satır: {x}\nHata: {e}")
                return []
        return x

    def generate_payment_pattern_features(self):
        # PAYMENT PATTERN RELATED FEATURES

        # timestamp ve sıralama
        self.df3["timestamp"] = pd.to_datetime(self.df3["timestamp"])
        self.df3 = self.df3.sort_values(by=["user_id", "timestamp"])

        # Her kullanıcı için event_sequence oluştur
        event_sequences = self.df3.groupby("user_id")["event_type"].apply(list).reset_index()
        event_sequences.columns = ["user_id", "event_sequence"]

        # Unique kullanıcı listesiyle birleştir
        df_users = self.df3[["user_id"]].drop_duplicates()
        self.df3 = pd.merge(df_users, event_sequences, on="user_id", how="left")

        # Stringse parse et (muhtemelen gerek kalmayacak ama güvenli olsun)
        self.df3["event_sequence"] = self.df3["event_sequence"].apply(self.parse_event_sequence)

        # Yeni sütunlar
        first_event_type = []
        last_event_before_first_payment = []
        steps_before_first_payment = []
        total_payment_success = []
        pre_payment_mod_repeat = []

        for events in self.df3["event_sequence"]:
            # İlk olay
            first_event = events[0] if events else None

            # İlk ödeme öncesi olay ve adım sayısı
            try:
                idx = events.index("payment_success")
                event_before = events[idx - 1] if idx > 0 else None
                steps = idx
            except ValueError:
                event_before = None
                steps = None

            # Toplam ödeme sayısı
            total_payments = events.count("payment_success")

            # 2-gram tekrarı
            try:
                idx = events.index("payment_success")
                pre_payment_seq = events[:idx]
                if len(pre_payment_seq) >= 2:
                    pre_payment_2grams = list(ngrams(pre_payment_seq, 2))
                    repeat_count = Counter(pre_payment_2grams)
                    most_common_repeat_count = max(repeat_count.values()) if repeat_count else 0
                else:
                    most_common_repeat_count = 0
            except ValueError:
                most_common_repeat_count = 0

            # Listeye ekle
            first_event_type.append(first_event)
            last_event_before_first_payment.append(event_before)
            steps_before_first_payment.append(steps)
            total_payment_success.append(total_payments)
            pre_payment_mod_repeat.append(most_common_repeat_count)

        # Sütunlara ata
        self.df3["first_event_type"] = first_event_type
        self.df3["last_event_before_first_payment"] = last_event_before_first_payment
        self.df3["steps_before_first_payment"] = steps_before_first_payment
        self.df3["total_payment_success"] = total_payment_success
        self.df3["pre_payment_mod_repeat"] = pre_payment_mod_repeat

        # Kaydet
        self.df3.to_csv(f"{self.feature_data_path}/payment_extended.csv", index=False)

    def run(self):
        self.generate_purchase_intended_features()
        self.generate_payment_pattern_features()
        self.generate_rf_segments()


"""# çalışma
base_data_path = "/home/eplinux/ikas_case_study/advanced_user_events_with_uuid.csv"
feature_data_path = "/home/eplinux/ikas_case_study/feature_data"
feature_engineer = FeatureEngineer(base_data_path, feature_data_path)
feature_engineer.run()"""

