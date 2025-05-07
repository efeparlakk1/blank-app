import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# CSV dosyalarını okuma
payment_extended_df = pd.read_csv('/home/eplinux/ikas_case_study/new_feature_datas/payment_extended.csv')
purchase_intended_df = pd.read_csv('/home/eplinux/ikas_case_study/new_feature_datas/purchase_intented.csv')
rf_segmented_users_df = pd.read_csv('/home/eplinux/ikas_case_study/new_feature_datas/PAY_rf_segmented_users.csv')
categorical_data_df = pd.read_csv('/home/eplinux/ikas_case_study/advanced_user_profiles_with_uuid.csv')

# 'user_id' analiz için değil, eşleşme için kullanılır
categorical_df_encoded = pd.get_dummies(
    categorical_data_df.drop(columns=["user_id"]), 
    prefix_sep='__', 
    drop_first=True  # istersen True yapıp bir dummy sütunu düşebilirsin
)

# Gerekirse user_id'yi sakla
categorical_df_encoded["user_id"] = categorical_data_df["user_id"]

# "user_id" sütununa göre veri setlerini birleştirme
merged_df = pd.merge(payment_extended_df, purchase_intended_df, on='user_id', how='outer')
merged_df = pd.merge(merged_df, rf_segmented_users_df, on='user_id', how='outer')
merged_df = pd.merge(merged_df, categorical_df_encoded, on='user_id', how='outer')

merged_df = merged_df.loc[:, ~merged_df.columns.str.startswith("Unnamed:")]
merged_df = merged_df.drop(["start_day", "end_day", "first_event_type", "last_event_before_first_payment"], axis=1)
merged_df.rename(columns={"segment": "segment_rf"}, inplace=True)
merged_df["segment_rf"] = merged_df["segment_rf"] + "_rf"

# null değerleri kontrol etme ve doldurma
merged_df['steps_before_first_payment'].fillna(merged_df['steps_before_first_payment'].max(), inplace=True)

# Korelasyon matrisini hesaplama (özellikler hariç)
correlation_matrix = merged_df.drop(["user_id", "segment_rf", "event_sequence"], axis=1).corr()

# Korelasyon matrisini görselleştirme
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.show()

# Korelasyonu yüksek olan özellikleri belirleme
threshold = 0.75  # Korelasyon eşik değeri
highly_correlated_features = set()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            highly_correlated_features.add(colname)

# Yüksek korelasyona sahip değişkenlerden yalnızca birini çıkarma
numerical_features = merged_df.drop(["user_id", "segment_rf", "event_sequence"], axis=1)
numerical_features = numerical_features.drop(columns=highly_correlated_features)

# Veriyi normalleştirme (PCA'dan önce gereklidir)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_features)

# PCA uygulama
pca = PCA(n_components=0.95)  # Verinin %95'ini açıklayacak kadar bileşen seçelim
principal_components = pca.fit_transform(scaled_data)

# PCA ile elde edilen bileşenlerden DataFrame oluşturma
pca_df = pd.DataFrame(data=principal_components)

# PCA sonrası verileri mevcut DataFrame'e ekleme
merged_df_pca = pd.concat([merged_df[['user_id', 'segment_rf']], pca_df], axis=1)

# PCA sonrası sonuçları görselleştirme
#plt.figure(figsize=(8, 6))
#sns.scatterplot(x=pca_df[0], y=pca_df[1], hue=merged_df['segment_rf'], palette='coolwarm')
#plt.title('PCA ile Boyut İndirgeme Sonuçları')
#plt.xlabel('Bileşen 1')
#plt.ylabel('Bileşen 2')
#plt.show()

# Çıkarılan yüksek korelasyonlu özellikleri yazdırma
print("Çıkarılan yüksek korelasyona sahip özellikler:", highly_correlated_features)

# Sonuçları kaydetme
merged_df = merged_df.drop(columns=highly_correlated_features)
merged_df.to_csv("/home/eplinux/ikas_case_study/final_datasets/pay_pca_cluster_ready_data.csv", index=False)
