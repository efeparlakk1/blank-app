import os
import pandas as pd
import streamlit as st
from feature_generator import FeatureEngineer
from dataset_merger import ProcessedDataGenerator
from cluster_model import UserBehaviorClustering  # Gerekli modüller

# Streamlit başlığı
st.title("Model Eğitim ve Sonuç Görüntüleme")

# Streamlit session_state kontrolü
if 'feature_engineered' not in st.session_state:
    st.session_state.feature_engineered = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# 1. Feature dataset creation
st.subheader("1. Feature Dataset Creation")
base_data_path = "/home/eplinux/ikas_case_study/data/advanced_user_events_with_uuid.csv"
feature_data_path = "/home/eplinux/ikas_case_study/feature_data"

st.text(f"Base Data Path: {base_data_path}")
st.text(f"Feature Data Path: {feature_data_path}")

# Feature Engineer işlemi
if st.button("Feature Engineering Başlat") and not st.session_state.feature_engineered:
    st.text("Feature Engineering Başlatılıyor...")
    feature_engineer = FeatureEngineer(base_data_path, feature_data_path)
    feature_engineer.run()
    st.session_state.feature_engineered = True
    st.text("Feature Engineering Tamamlandı!")

# 2. Dataset processor and merger
if st.session_state.feature_engineered:
    st.subheader("2. Dataset Processor and Merger")
    features_path = "/home/eplinux/ikas_case_study/feature_data" 
    demog_data_path = "/home/eplinux/ikas_case_study/data/advanced_user_profiles_with_uuid.csv"
    output_path = "/home/eplinux/ikas_case_study/final_datasets/final_data.csv" 

    st.text(f"Features Path: {features_path}")
    st.text(f"Categorical Data Path: {demog_data_path}")
    st.text(f"Output Path: {output_path}")

    # Veri işleme işlemi
    if st.button("Veri İşleme ve Birleştirme Başlat") and not st.session_state.data_processed:
        st.text("Veri işleniyor...")
        processor = ProcessedDataGenerator(features_path, demog_data_path, output_path)
        processor.run()
        st.session_state.data_processed = True
        st.text("Veri işleme tamamlandı!")

# 3. Cluster model training ve kaydetme
if st.session_state.data_processed:
    st.subheader("3. Cluster Model Training ve Kaydetme")
    if st.button("Model Eğitimi Başlat") and not st.session_state.model_trained:
        st.text("Model Eğitimi Başlatılıyor...")
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
        st.session_state.model_trained = True
        st.text("Model Eğitimi Tamamlandı!")

        # Kullanıcı ve tahmin edilen davranış segmentini içeren dataframe
        user_segment_df = df_with_clusters[["user_id", "behavior_segment"]]

        # Kayıt işlemi
        output_path = "/home/eplinux/ikas_case_study/result_show/user_behavior_segments.csv"
        user_segment_df.to_csv(output_path, index=False)

        st.text(f"Kullanıcı segmentleri '{output_path}' dosyasına kaydedildi.")

        # Resim ve .txt dosyalarını görüntüleme seçeneği
        st.subheader("Sonuçları Görüntüle")
        result_show_dir = "/home/eplinux/ikas_case_study/result_show"

        # Klasördeki dosyaları listele
        result_files = os.listdir(result_show_dir)
        images = [file for file in result_files if file.endswith(('.png', '.jpg', '.jpeg'))]
        txt_files = [file for file in result_files if file.endswith('.txt')]

        # Resim dosyalarını göster
        st.write("Resimler:")
        for image in images:
            img_path = os.path.join(result_show_dir, image)
            st.image(img_path, caption=image)

        # .txt dosyalarını göster
        st.write("Metin Dosyaları:")
        for txt_file in txt_files:
            txt_path = os.path.join(result_show_dir, txt_file)
            with open(txt_path, 'r') as file:
                st.text(f"--- {txt_file} ---")
                st.text(file.read())

# 4. Model Results Görüntüleme
if st.session_state.model_trained:
    if st.button("Model Results Göster"):
        st.subheader("Model Sonuçları")

        # CSV dosyasını yükle
        results_path = "/home/eplinux/ikas_case_study/result_show/user_behavior_segments.csv"

        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)

            # Scrollable table olarak göster
            st.dataframe(results_df, height=300)  # Burada 300px yükseklik verildi, ihtiyaca göre değiştirebilirsiniz
        else:
            st.text("Sonuç dosyası bulunamadı.")
