# utils.py

import os
import re
import string
import pickle
import pandas as pd
import nltk
import logging

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
import fitz  # PyMuPDF

# --- Setup ---
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

logging.basicConfig(level=logging.INFO)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ========== File Handling ==========

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ========== File Reading for Streamlit ==========

def read_uploaded_file(file):
    try:
        if file.type == "text/plain":
            return [file.read().decode("utf-8")]

        elif file.type == "application/pdf":
            text = ""
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
            return [text]

        elif file.type == "text/csv":
            df = pd.read_csv(file)
        elif file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
            df = pd.read_excel(file)
        elif file.type == "application/json":
            df = pd.read_json(file)
        else:
            return []

        text_col = df.select_dtypes(include=["object"]).columns
        return df[text_col[0]].astype(str).tolist() if len(text_col) else []
    except Exception as e:
        print(f"❌ File read error: {e}")
        return []

# ========== Text Cleaning ==========

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

def clean_and_tokenize(text):
    text = clean_text(text)
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]

def preprocess_bulk_texts(texts):
    return [" ".join(clean_and_tokenize(text)) for text in texts]

# ========== LDA Prediction ==========

def predict_topic_gensim(texts, lda_model, dictionary):
    results = []
    for text in texts:
        tokens = clean_and_tokenize(text)
        bow = dictionary.doc2bow(tokens)
        topic_probs = lda_model.get_document_topics(bow)

        if topic_probs:
            topic_num, prob = max(topic_probs, key=lambda x: x[1])
            results.append({
                "text": text[:150] + ("..." if len(text) > 150 else ""),
                "predicted_topic": topic_num,
                "confidence": round(prob * 100, 2)
            })
        else:
            results.append({
                "text": text[:150] + ("..." if len(text) > 150 else ""),
                "predicted_topic": "N/A",
                "confidence": 0.0
            })
    return pd.DataFrame(results)

# ========== KMeans Prediction ==========

def predict_cluster_kmeans(texts, pipeline, kmeans_model):
    try:
        cleaned = pipeline.named_steps["preprocessor"].transform(texts)
        vectorized = pipeline.transform(texts)
        return kmeans_model.predict(vectorized).tolist()
    except Exception as e:
        print(f"❌ KMeans error: {e}")
        return [-1] * len(texts)
