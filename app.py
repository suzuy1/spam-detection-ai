import streamlit as st
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Inisialisasi
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model dan vectorizer
@st.cache_resource
def load_resources():
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_resources()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def predict_spam(text):
    text_clean = preprocess_text(text)
    text_vec = vectorizer.transform([text_clean]).toarray()
    prob = model.predict_proba(text_vec)[0][1]
    label = 'Spam' if prob > 0.5 else 'Ham'
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence, prob

# UI STREAMLIT
st.set_page_config(page_title="Spam Detection AI", page_icon="🛡️", layout="centered")

st.markdown("""
    <div style="text-align: center;">
        <h1>🛡️ Spam Detection AI</h1>
        <p style="font-size: 18px; color: #666;">
            Deteksi Spam SMS menggunakan Jaringan Saraf Tiruan (JST)
        </p>
        <hr>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📋 Informasi Project")
    st.markdown("""
    **Mata Kuliah:** Kecerdasan Buatan  
    **Metode:** Jaringan Saraf Tiruan (JST) - MLPClassifier  
    **Dataset:** SMS Spam Collection (Kaggle)  
    **Jumlah Data:** 5.574 SMS
    """)
    st.markdown("---")
    st.markdown("### 🏗️ Arsitektur JST")
    st.markdown("""
    - Input Layer (3000 neuron)
    - Hidden Layer 1 (128 neuron, ReLU)
    - Hidden Layer 2 (64 neuron, ReLU)
    - Hidden Layer 3 (32 neuron, ReLU)
    - Output Layer (1 neuron, Sigmoid)
    """)
    st.markdown("---")
    st.markdown("### 👥 Kelompok")
    st.text_input("Nama Anggota 1")
    st.text_input("Nama Anggota 2")
    st.text_input("Nama Anggota 3")
    st.text_input("Nama Anggota 4")

st.markdown("### ✍️ Masukkan Teks SMS")
user_input = st.text_area("Ketik atau paste pesan SMS di sini:", height=120, 
                          placeholder="Contoh: Congratulations! You've won a $1000 cash prize...")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Deteksi Spam", use_container_width=True)

if predict_btn and user_input:
    with st.spinner('Sedang menganalisis...'):
        label, confidence, raw_prob = predict_spam(user_input)
    
    st.markdown("---")
    st.markdown("### 📊 Hasil Prediksi")
    
    if label == 'Spam':
        st.error(f"🚨 **SPAM TERDETEKSI!**")
        st.markdown(f"""
            <div style="background-color: #ffebee; padding: 20px; border-radius: 10px; border-left: 5px solid #e74c3c;">
                <h4 style="color: #c0392b; margin: 0;">Confidence: {confidence*100:.2f}%</h4>
                <p style="color: #666; margin-top: 10px;">Pesan ini terdeteksi sebagai spam. Hati-hati!</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ **HAM (Bukan Spam)**")
        st.markdown(f"""
            <div style="background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #2ecc71;">
                <h4 style="color: #27ae60; margin: 0;">Confidence: {confidence*100:.2f}%</h4>
                <p style="color: #666; margin-top: 10px;">Pesan ini aman, bukan spam.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Tingkat Keyakinan")
    st.progress(float(confidence))
    st.caption(f"Probabilitas: {raw_prob*100:.2f}%")
    
    with st.expander("🔧 Lihat Hasil Preprocessing"):
        cleaned = preprocess_text(user_input)
        st.markdown("**Teks setelah preprocessing:**")
        st.code(cleaned)

elif predict_btn and not user_input:
    st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")

st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 12px;">
        <p>Project Akhir - Kecerdasan Buatan | Dataset: Kaggle SMS Spam Collection</p>
    </div>
""", unsafe_allow_html=True)
