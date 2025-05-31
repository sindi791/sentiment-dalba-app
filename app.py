import streamlit as st
import pandas as pd
import pickle
from utils import preprocess_text

# Load model dan vectorizer
model_svm = pickle.load(open('model_svm.pkl', 'rb'))
model_nb = pickle.load(open('model_nb.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Judul aplikasi
st.title("Analisis Sentimen Komentar Instagram Produk D'Alba")

# Pilih input
option = st.radio("Pilih metode input komentar:", ['Manual', 'Upload CSV'])

# Input manual
if option == 'Manual':
    text_input = st.text_area("Masukkan komentar:")
    if text_input:
        data = pd.DataFrame([text_input], columns=['komentar'])
    else:
        st.stop()

# Input dari file CSV
else:
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        st.stop()

# Pilih model
model_choice = st.selectbox("Pilih Model Klasifikasi:", ['SVM', 'Naive Bayes'])

# Tombol klasifikasi
if st.button("Klasifikasikan"):
    # Preprocessing
    data['cleaned'] = data['komentar'].astype(str).apply(preprocess_text)
    X = vectorizer.transform(data['cleaned'])

    # Prediksi
    if model_choice == 'SVM':
        prediction = model_svm.predict(X)
    else:
        prediction = model_nb.predict(X)

    # Tampilkan hasil
    data['sentimen'] = prediction
    st.write("Hasil Klasifikasi:")
    st.dataframe(data[['komentar', 'sentimen']])

    # Visualisasi
    st.subheader("Visualisasi Sentimen")
    st.bar_chart(data['sentimen'].value_counts())
