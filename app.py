# Buat ulang file app.py versi final tanpa error indentasi atau key
final_app_code = """
import streamlit as st
import pandas as pd
import pickle
from utils import preprocess_text

st.title("Analisis Sentimen Komentar Instagram Produk D'Alba")

uploaded_file = st.file_uploader("Upload file komentar (.xlsx / .csv)", type=["xlsx", "csv"])
model_choice = st.selectbox("Pilih Model Klasifikasi:", ["SVM", "Naive Bayes"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)

    # Cek apakah kolom komentar tersedia
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

            vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
            X = vectorizer.transform(data['cleaned'])

            if model_choice == "SVM":
                model = pickle.load(open("model_svm.pkl", "rb"))
            else:
                model = pickle.load(open("model_nb.pkl", "rb"))

            data['sentimen'] = model.predict(X)
            st.write(data[[kolom_komentar, 'sentimen']])
            st.bar_chart(data['sentimen'].value_counts())
"""

# Simpan ke file app.py
fixed_app_file = "/mnt/data/app.py"
with open(fixed_app_file, "w") as f:
    f.write(final_app_code)

fixed_app_file
