import os
import pandas as pd
import streamlit as st
from scripts.feature_generator import FeatureEngineer
from scripts.dataset_merger import ProcessedDataGenerator
from scripts.cluster_model import UserBehaviorClustering  # Gerekli modüller

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
base_data_path = "data/advanced_user_events_with_uuid.csv"
feature_data_path = "./feature_data"

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
    features_path = "./feature_data" 
    demog_data_path = "data/advanced_user_profiles_with_uuid.csv"
    output_path = "final_datasets/final_data.csv" 

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
            df_path="final_datasets/final_data.csv",
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
        output_path = "result_show/user_behavior_segments.csv"
        user_segment_df.to_csv(output_path, index=False)

        st.text(f"Kullanıcı segmentleri '{output_path}' dosyasına kaydedildi.")

        # Sonuçları doğrudan göster
        st.subheader("Model Sonuçları")
        st.dataframe(user_segment_df, height=300)

        # Görsel ve .txt dosyalarını da göster
        st.subheader("Sonuçları Görüntüle")
        result_show_dir = "result_show"
        result_files = os.listdir(result_show_dir)
        images = [file for file in result_files if file.endswith(('.png', '.jpg', '.jpeg'))]
        txt_files = [file for file in result_files if file.endswith('.txt')]

        st.write("Resimler:")
        for image in images:
            img_path = os.path.join(result_show_dir, image)
            st.image(img_path, caption=image)

st.subheader("4. Uygulamayı Yeniden Başlatma")
if st.button("🔁 Yeniden Başlat"):
    st.session_state.clear()
    st.experimental_rerun()


