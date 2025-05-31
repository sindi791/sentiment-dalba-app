
import streamlit as st
import pandas as pd
import pickle
from utils import preprocess_text

st.set_page_config(page_title="Sentiment Analysis - D'Alba", layout="wide")
st.title("Analisis Sentimen Komentar Instagram Produk D'Alba")

uploaded_file = st.file_uploader("Upload file komentar (.xlsx / .csv)", type=["xlsx", "csv"])
model_choice = st.selectbox("Pilih Model Klasifikasi:", ["SVM", "Naive Bayes"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)

    # Deteksi otomatis kolom komentar
    kolom_komentar = None
    for col in data.columns:
        if col.lower() in ['komentar', 'comment']:
            kolom_komentar = col
            break

    if not kolom_komentar:
        st.error("File harus memiliki kolom bernama 'komentar' atau 'comment'")
    else:
        if st.button("Klasifikasikan"):
            data['cleaned'] = data[kolom_komentar].astype(str).apply(preprocess_text)

            # Load model dan vectorizer
            try:
                vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
                model_svm = pickle.load(open("model_svm.pkl", "rb"))
                model_nb = pickle.load(open("model_nb.pkl", "rb"))
            except Exception as e:
                st.error(f"Gagal memuat model/vectorizer: {e}")
                st.stop()

            # Transformasi TF-IDF
            X = vectorizer.transform(data['cleaned'])

            # Cek apakah ada fitur
            if X.shape[1] == 0:
                st.error("Komentar tidak menghasilkan fitur setelah preprocessing. Coba upload data yang lebih bervariasi.")
                st.stop()

            # Prediksi
            if model_choice == "SVM":
                prediction = model_svm.predict(X)
            else:
                prediction = model_nb.predict(X)

            data['sentimen'] = prediction

            # Tampilkan hasil
            st.subheader("Hasil Klasifikasi")
            st.dataframe(data[[kolom_komentar, 'sentimen']])
            st.subheader("Visualisasi Sentimen")
            st.bar_chart(data['sentimen'].value_counts())
