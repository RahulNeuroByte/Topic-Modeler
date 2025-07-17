
# pipeline_export.py

import os
import pickle
import pandas as pd
import logging
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import TextPreprocessor

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -------------------- Paths --------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(script_dir, "processed_data")
models_dir = os.path.join(script_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# -------------------- Dataset Configs --------------------
datasets = {
    "20": "df_20_meta_with_models.csv",
    "mini": "df_mini_meta_with_models.csv",
    "merged": "df_merged_meta_with_models.csv"
}

logging.info(" Starting pipeline export process...")

for name, filename in datasets.items():
    start = time()
    logging.info(f"Processing: {name.upper()}")

    try:
        filepath = os.path.join(processed_data_dir, filename)
        df = pd.read_csv(filepath)

        if "preprocessed_text" not in df.columns:
            raise ValueError(f"Missing 'preprocessed_text' column in {filename}")

        texts = df["preprocessed_text"].astype(str).tolist()

        # Create TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        X_tfidf = vectorizer.fit_transform(texts)

        # Save vectorizer
        vectorizer_path = os.path.join(models_dir, f"vectorizer_{name}.pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        # Build and Save Full Pipeline
        pipeline = Pipeline([
            ("preprocessor", TextPreprocessor()),
            ("tfidf", vectorizer)
        ])
        pipeline_path = os.path.join(models_dir, f"pipeline_{name}.pkl")
        with open(pipeline_path, "wb") as f:
            pickle.dump(pipeline, f)

        logging.info(f"[{name}]  Pipeline saved to: {pipeline_path} (Time: {time() - start:.2f}s)")

    except Exception as e:
        logging.error(f"[{name}]  Failed: {e}")

logging.info(" All pipelines exported successfully.")
