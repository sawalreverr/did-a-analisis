import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
import torch
import services.google_drive_service as gd 
import services.dropbox_service as db
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from st_aggrid import AgGrid, GridOptionsBuilder

# Streamlit Config
st.set_page_config(page_icon="🐳", page_title="did-a-analisis" ,layout="centered", )

# Load resources
kamus_normalisasi = pd.read_excel('data/kamus_normalisasi.xlsx')
normalization_dict = dict(zip(kamus_normalisasi['Original'], kamus_normalisasi['Replacement']))

# Cached download of stopwords to avoid re-downloading every time
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('indonesian'))

stop_words = load_stopwords()

# Load IndoBERT
@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model = AutoModel.from_pretrained('indobenchmark/indobert-base-p1')
    return tokenizer, model

tokenizer, bert_model = load_bert()

factory = StemmerFactory()
stemmer = factory.create_stemmer()

xgboost_model = joblib.load('model/xgboost_optimized_model.pkl')

# Cached stemmer
def cached_stem(word, cache={}):
    if word not in cache:
        cache[word] = stemmer.stem(word)
    return cache[word]

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'(@\w+)|(#\w+)|(http\S+|www\S+|https\S+)|', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = re.sub(r"[\U0001F600-\U0001F64F]", " ", text)
    text = re.sub(r"[\U0001F300-\U0001F5FF]", " ", text)
    text = re.sub(r"[\U0001F680-\U0001F6FF]", " ", text)
    text = re.sub(r"[\U0001F1E0-\U0001F1FF]", " ", text)
    text = re.sub(r"[\U00002700-\U000027BF]", " ", text)
    text = re.sub(r"[\U0001F900-\U0001F9FF]", " ", text)
    text = re.sub(r"[\U0001FAD0-\U0001FAFF]", " ", text)
    text = re.sub(r"[^\x00-\x7F]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    tokens = [normalization_dict.get(word, word) for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [cached_stem(word) for word in tokens]

    return ' '.join(tokens)

# Feature extraction using IndoBERT
def extract_features(texts, tokenizer, model):
    features = []
    for text in tqdm(texts, desc="Extracting features"):
        inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        features.append(outputs.last_hidden_state[:, 0, :].numpy().flatten())
    return np.array(features)

# Display Helper
def display_info(message):
    if 'info_shown' not in st.session_state:
        st.session_state['info_shown'] = st.empty()

    st.session_state.info_shown.info(message)

# Display Table
def custom_table(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationPageSize=10)
    gb.configure_default_column(editable=True, groupable=True)
    grid_options = gb.build()
    
    AgGrid(df, gridOptions=grid_options, height=500, fit_columns_on_grid_load=True)

# Analyze text
def analyze_text(text):
    display_info("Memproses...")

    preprocessed_text = preprocess_text(text)
    features = extract_features([preprocessed_text], tokenizer, bert_model)
    prediction = xgboost_model.predict(features)

    st.session_state.info_shown.empty()
    
    sentiment = "Positif 😊" if prediction[0] == 1 else "Negatif ☹️"
    st.success(f"Hasil Sentimen: **{sentiment}**")

# Analyze file
def analyze_file(data):
    progress = st.progress(0)

    # Preprocessing --
    with st.spinner(text="Preprocessing..."):
        data['cleaned_text'] = data['tweet'].apply(preprocess_text)
    progress.progress(40)

    # Feature extraction --
    with st.spinner(text="Feature Extraction (IndoBERT)..."):
        features = extract_features(data['cleaned_text'], tokenizer, bert_model)
    progress.progress(70)
    
    # Model testing --
    with st.spinner(text="Model Testing (XGBoost)..."):
        predictions = xgboost_model.predict(features)
    progress.progress(100)

    col1, col2, col3 = st.columns([2, 3, 2])
    col1.markdown("Preprocessing ✅")
    col2.markdown("Feature Extraction (IndoBERT) ✅")
    col3.markdown("Model Testing (XGBoost) ✅")

    # Display the DataFrame
    data['final_prediction'] = predictions
    data.rename(columns={'target': 'actual_label'}, inplace=True)
    
    result_df = data[['tweet', 'actual_label', 'final_prediction']]
    
    st.subheader("Predicted Results")
    custom_table(result_df)
    
    # Confusion Matrix
    cm = confusion_matrix(data['actual_label'], predictions)
    cm_df = pd.DataFrame(cm, columns=["Predicted Negative", "Predicted Positive"], index=["Actual Negative", "Actual Positive"])

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=1, linecolor='black')
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    # Performance Evaluation
    report = classification_report(data['actual_label'], predictions, output_dict=True)
    eval_df = pd.DataFrame(report).transpose()
    st.subheader("Performance Evaluation")
    st.table(eval_df)

    # Pie chart
    sentiment_counts = pd.Series(predictions).value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = sns.color_palette("Set2", 2)
    ax.pie(sentiment_counts, labels=["Negatif", "Positif"], autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title("Distribusi Sentimen")
    st.pyplot(fig)

""" # 🐳 did-a-analisis 
**did-a-analisis** adalah aplikasi sederhana untuk menganalisis sentimen publik menggunakan IndoBERT dan XGBoost. Aplikasi ini memproses data mengenai topik Ibu Kota Nusantara (IKN), untuk membantu memahami opini masyarakat dengan mudah dan cepat.
"""
st.divider()
input_option = st.radio("Pilih metode input:", ["Input Text", "Upload File (.csv/.xlsx)"])

if input_option == "Input Text":
    text_input = st.text_area("Masukkan text untuk di analisis:")
    if st.button("Analisis"):
        if text_input.strip():
            analyze_text(text_input)
else:
    upload_option = st.selectbox("Pilih metode upload:", ["Unggah File Lokal", "Google Drive", "Dropbox"])

    if upload_option == "Unggah File Lokal":
        uploaded_file = st.file_uploader("Upload file (.csv atau .xlsx):", type=["csv", "xlsx"])
        if uploaded_file:
            file_type = uploaded_file.name.split('.')[-1]
            data = pd.read_csv(uploaded_file) if file_type == 'csv' else pd.read_excel(uploaded_file)
            
            if "tweet" in data.columns and "target" in data.columns:
                display_info("Memproses file...")
                
                st.subheader("Data yang Diupload")
                custom_table(data)

                st.session_state.info_shown.empty()

                if st.button("Analisis"):
                    analyze_file(data)
            else:
                st.error("File harus memiliki kolom 'tweet' dan 'target'.")

    elif upload_option == "Google Drive":
        st.write("Upload via Google Drive")

    elif upload_option == "Dropbox":
        st.write("Upload Via Dropbox")
