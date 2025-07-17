# generategensim.py

import os
import pickle
import pandas as pd
import re
import nltk
import logging
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------- NLTK Downloads --------------------
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -------------------- Preprocessing --------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_and_tokenize(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    tokens = nltk.word_tokenize(text.lower())
    return [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]

# -------------------- Paths --------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(script_dir, "processed_data")
models_dir = os.path.join(script_dir, "models")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

# -------------------- Dataset Configs --------------------
datasets = {
    "20": "df_20_meta_with_models.csv",
    "mini": "df_mini_meta_with_models.csv",
    "merged": "df_merged_meta_with_models.csv"
}

for name, filename in datasets.items():
    logging.info(f"\nProcessing corpus for: {name.upper()}")

    df_path = os.path.join(processed_data_dir, filename)
    if not os.path.exists(df_path):
        logging.warning(f" File not found: {df_path}")
        continue

    df = pd.read_csv(df_path)

    # Tokenize
    if "tokenized" in df.columns:
        tokenized_docs = df["tokenized"].apply(eval).tolist()
        logging.info(f"[{name}] Using existing tokenized column.")
    else:
        logging.info(f"[{name}] Tokenizing from 'preprocessed_text' column...")
        tokenized_docs = [clean_and_tokenize(text) for text in df["preprocessed_text"]]

    # Save tokenized for evaluation.py
    token_out_path = os.path.join(processed_data_dir, f"tokenized_{name}.pkl")
    with open(token_out_path, "wb") as f:
        pickle.dump(tokenized_docs, f)

    # Build Dictionary and Corpus
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    dictionary.compactify()

    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Save
    dictionary.save(os.path.join(models_dir, f"dictionary_{name}.dict"))
    with open(os.path.join(models_dir, f"corpus_{name}.pkl"), "wb") as f:
        pickle.dump(corpus, f)

    logging.info(f"Saved: dictionary_{name}.dict, corpus_{name}.pkl, tokenized_{name}.pkl")

logging.info("\n All Gensim resources generated successfully!")

