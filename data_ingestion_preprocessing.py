# data_ingestion_and_preprocessing.py
# Preprocess all datasets (20, mini, merged), tokenize, vectorize, save .pkl files

import os
import pickle
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Import from your reusable preprocessing.py
from preprocessing import preprocess_text_pipeline

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Preprocessing + Vectorization --------------------
def preprocess_and_vectorize(df, dataset_name):
    logger.info(f"Preprocessing dataset: {dataset_name}")
    
    # Clean and tokenize
    preprocessed_texts = preprocess_text_pipeline(df["preprocessed_text_raw"].tolist())
    tokenized_texts = [text.split() for text in preprocessed_texts]
    
    logger.info(f"Vectorizing dataset: {dataset_name}")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", min_df=5, max_df=0.8)
    X_vectorized = vectorizer.fit_transform(preprocessed_texts)
    
    return preprocessed_texts, tokenized_texts, X_vectorized, vectorizer

# -------------------- Save All Outputs --------------------
def save_outputs(preprocessed_texts, tokenized_texts, X_vectorized, vectorizer, y, target_names, filenames, dataset_name):
    output_dir = os.path.join("processed_data")
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving outputs for: {dataset_name}")
    
    with open(os.path.join(output_dir, f"X_{dataset_name}_vectorized.pkl"), "wb") as f:
        pickle.dump(X_vectorized, f)
    with open(os.path.join(output_dir, f"vectorizer_{dataset_name}.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(output_dir, f"y_{dataset_name}.pkl"), "wb") as f:
        pickle.dump(y, f)
    with open(os.path.join(output_dir, f"target_names_{dataset_name}.pkl"), "wb") as f:
        pickle.dump(target_names, f)
    with open(os.path.join(output_dir, f"tokenized_{dataset_name}.pkl"), "wb") as f:
        pickle.dump(tokenized_texts, f)

    # Save metadata
    df_meta = pd.DataFrame({
        "filename": filenames,
        "target": y,
        "target_name": [target_names[i] for i in y],
        "preprocessed_text": preprocessed_texts
    })
    df_meta.to_csv(os.path.join(output_dir, f"df_{dataset_name}_meta.csv"), index=False)
    logger.info(f"Saved df_{dataset_name}_meta.csv")


# -------------------- Main Script --------------------
if __name__ == "__main__":
    logger.info("🚀 Starting Data Ingestion & Preprocessing Pipeline...")

    processed_data_dir = "processed_data"

    datasets = {
        "20": os.path.join(processed_data_dir, "df_20_meta.csv"),
        "mini": os.path.join(processed_data_dir, "df_mini_meta.csv"),
        "merged": os.path.join(processed_data_dir, "df_merged_meta.csv")
    }

    for dataset_name, path in datasets.items():
        if not os.path.exists(path):
            logger.warning(f"[SKIPPED] {dataset_name} not found at {path}")
            continue

        logger.info(f"\n📂 Loading {dataset_name.upper()} dataset...")
        df = pd.read_csv(path)

        # Make sure there's a column to process (assume preprocessed_text exists or fallback to raw)
        if "preprocessed_text" in df.columns:
            df["preprocessed_text_raw"] = df["preprocessed_text"]
        elif "body" in df.columns:
            df["preprocessed_text_raw"] = df["body"]
        else:
            logger.error(f"No text column found in {path}")
            continue

        # Assign fake labels and filenames if needed (merged set might not have them)
        y = df["target"].values if "target" in df.columns else [0] * len(df)
        target_names = df["target_name"].unique().tolist() if "target_name" in df.columns else ["general"]
        filenames = df["filename"] if "filename" in df.columns else [f"{dataset_name}_{i}" for i in range(len(df))]

        # Preprocess, Vectorize, Save
        pre_texts, tokens, X_vec, vectorizer = preprocess_and_vectorize(df, dataset_name)
        save_outputs(pre_texts, tokens, X_vec, vectorizer, y, target_names, filenames, dataset_name)

    logger.info("✅ All datasets processed and saved successfully.")
