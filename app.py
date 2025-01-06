import io
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import torch
import xgboost as xgb
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit.runtime.scriptrunner import get_script_run_ctx


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

# Load resources (fine-tuned BERT and XGBoost model)
@st.cache_resource
def load_fine_tuned_model():
    fine_tuned_model_path = './model/fine_tuned_model'
    xgb_model_path = './model/xgboost_model.json'

    tokenizer = BertTokenizer.from_pretrained(fine_tuned_model_path)
    fine_tuned_model = BertForSequenceClassification.from_pretrained(fine_tuned_model_path)
    
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_model_path)

    label_encoder = LabelEncoder()
    label_encoder.fit([0, 1])
    
    return tokenizer, fine_tuned_model, xgb_model, label_encoder

tokenizer, fine_tuned_model, xgb_model, label_encoder = load_fine_tuned_model()

factory = StemmerFactory()
stemmer = factory.create_stemmer()

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

# Feature extraction using fine-tuned BERT
def extract_features(texts, model, tokenizer, max_length=128):
    model.eval()
    features = []
    for text in tqdm(texts, desc="Extracting features"):
        with torch.no_grad():
            encodings = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            outputs = model.bert(**encodings)
            last_hidden_state = outputs.last_hidden_state
            features.append(last_hidden_state.mean(dim=1).numpy().flatten())
    return np.array(features)

# Display Helper
def display_info(message):
    if 'info_shown' not in st.session_state:
        st.session_state['info_shown'] = st.empty()

    st.session_state.info_shown.info(message)

# Save file
def download_results(dataframe, file_format):
    output = io.BytesIO()

    if file_format == "CSV":
        dataframe.to_csv(output, index=False)
        output.seek(0)
        return output.getvalue()
    
    elif file_format == "XLSX":
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            dataframe.to_excel(writer, index=False)
        output.seek(0)
        return output.getvalue()
    
    else:
        return None

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
    features = extract_features([preprocessed_text], fine_tuned_model, tokenizer)
    prediction = xgb_model.predict(features)

    st.session_state.info_shown.empty()

    if prediction[0] == 1: 
        st.success("Sentimen Positif 😊")
    else:
        st.error("Sentimen Negatif ☹️")


@st.cache_data
def perform_analysis(_data):
    """Perform the actual analysis and return all results"""
    # Preprocessing
    data = _data.copy()
    data['cleaned_text'] = data['tweet'].apply(preprocess_text)
    
    # Feature extraction
    features = extract_features(data['cleaned_text'], fine_tuned_model, tokenizer)
    
    # Model testing
    predictions = xgb_model.predict(features)
    
    # Process results
    predicted_labels = ["negatif" if label == 0 else "positif" for label in predictions]
    data['final_prediction'] = predicted_labels
    data.rename(columns={'label': 'actual_label'}, inplace=True)
    
    result_df = data[['tweet', 'actual_label', 'final_prediction']]
    
    # Calculate metrics
    cm = confusion_matrix(result_df['actual_label'], result_df['final_prediction'])
    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted Negative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"]
    )
    
    report = classification_report(
        result_df['actual_label'],
        result_df['final_prediction'],
        output_dict=True
    )
    eval_df = pd.DataFrame(report).transpose()
    
    sentiment_counts = result_df['final_prediction'].value_counts()
    
    return {
        'result_df': result_df,
        'cm_df': cm_df,
        'eval_df': eval_df,
        'sentiment_counts': sentiment_counts
    }

if 'analysis_state' not in st.session_state:
    st.session_state['analysis_state'] = None
if 'download_clicked' not in st.session_state:
    st.session_state['download_clicked'] = False

def analyze_file(data):
    # Perform analysis if not already done or if explicitly requested
    if st.session_state['analysis_state'] is None:
        progress = st.progress(0)
        
        with st.spinner(text="Preprocessing..."):
            progress.progress(30)
            
        with st.spinner(text="Feature Extraction (IndoBERT)..."):
            progress.progress(60)
            
        with st.spinner(text="Model Testing (XGBoost)..."):
            results = perform_analysis(data)
            progress.progress(100)
            
        st.session_state['analysis_state'] = results

def display_results():
    if st.session_state['analysis_state'] is not None:
        results = st.session_state['analysis_state']
        
        # Display completion indicators
        col1, col2, col3 = st.columns([2, 3, 2])
        col1.markdown("Preprocessing ✅")
        col2.markdown("Feature Extraction (IndoBERT) ✅")
        col3.markdown("Model Testing (XGBoost) ✅")

        # Display predicted results
        st.subheader("Predicted Results")
        custom_table(results['result_df'])
        
        # Display confusion matrix
        st.subheader("Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(results['cm_df'], annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm)
        st.pyplot(fig_cm)

        # Display performance evaluation
        st.subheader("Performance Evaluation")
        st.table(results['eval_df'])

        # Display pie chart
        st.subheader("Sentiment Distribution")
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        ax_pie.pie(
            results['sentiment_counts'],
            labels=results['sentiment_counts'].index,
            autopct='%1.1f%%',
            startangle=90,
            colors=['#ff9999','#66b3ff']
        )
        ax_pie.axis('equal')
        st.pyplot(fig_pie)

        # Download buttons in container
        with st.container():
            st.subheader("Download Hasil Analisis")
            col1, col2 = st.columns([1, 1])
            
            def on_download_click():
                st.session_state['download_clicked'] = True
            
            with col1:
                csv = download_results(results['result_df'], "CSV")
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="hasil_analisis.csv",
                    mime="text/csv",
                    use_container_width=True,
                    on_click=on_download_click
                )
            
            with col2:
                xlsx = download_results(results['result_df'], "XLSX")
                st.download_button(
                    label="Download as XLSX",
                    data=xlsx,
                    file_name="hasil_analisis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    on_click=on_download_click
                )
    # Streamlit UI
""" # 🐳 did-a-analisis 
**did-a-analisis** adalah aplikasi sederhana untuk menganalisis sentimen publik menggunakan IndoBERT dan XGBoost. Aplikasi ini memproses data mengenai topik Ibu Kota Nusantara (IKN), untuk membantu memahami opini masyarakat dengan mudah dan cepat.
"""
st.divider()
input_option = st.selectbox("Pilih metode input:", ["Input text", "Upload file"])

if input_option == "Input text":
    text_input = st.text_area("Masukkan text untuk di analisis:")
    if st.button("Analisis"):
        if text_input.strip():
            analyze_text(text_input)
else:
    uploaded_file = st.file_uploader("Upload file (.csv atau .xlsx):", type=["csv", "xlsx"])

    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1]
        data = pd.read_csv(uploaded_file) if file_type == 'csv' else pd.read_excel(uploaded_file)

        if "tweet" in data.columns and "label" in data.columns:
            st.subheader("Data yang Diupload")
            custom_table(data)

            # Add a clear results button
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Analisis", key="analyze_button"):
                    st.session_state['analysis_state'] = None
                    analyze_file(data)
            with col2:
                if st.button("Clear Results", key="clear_button"):
                    st.session_state['analysis_state'] = None
            
            # Display results if they exist
            display_results()
        else:
            st.error("File harus memiliki kolom 'tweet' dan 'label'.")
