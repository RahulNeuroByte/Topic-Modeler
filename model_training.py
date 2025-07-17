
# model_training.py

import os
import pickle
import warnings
import pandas as pd
from sklearn.cluster import KMeans
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

warnings.filterwarnings("ignore")

# -------------------- Train Gensim LDA --------------------
def train_gensim_lda(tokenized_docs, num_topics=20, passes=30):
    print(f"[INFO] Training Gensim LDA with {num_topics} topics and {passes} passes...")
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        per_word_topics=True
    )

    coherence_model = CoherenceModel(model=lda_model, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    print(f"[INFO] Coherence Score: {coherence_score:.4f}")

    return lda_model, dictionary, corpus

# -------------------- Train Sklearn KMeans --------------------
def train_kmeans(X_vectorized, n_clusters=20):
    print(f"[INFO] Training KMeans with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_vectorized)
    print("[INFO] KMeans training complete.")
    return kmeans

# -------------------- Main --------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(script_dir, "processed_data")
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # ==================== 20 Newsgroups ====================
    print("\n--- 20 Newsgroups Training ---")
    try:
        with open(os.path.join(processed_data_dir, 'tokenized_20.pkl'), 'rb') as f:
            tokens_20 = pickle.load(f)
        with open(os.path.join(processed_data_dir, 'X_20_vectorized.pkl'), 'rb') as f:
            X_20 = pickle.load(f)
        df_20_meta = pd.read_csv(os.path.join(processed_data_dir, 'df_20_meta.csv'))
    except Exception as e:
        print(f"[ERROR] Loading 20_newsgroups failed: {e}")
        tokens_20, X_20, df_20_meta = None, None, None

    if tokens_20:
        lda_20, dictionary_20, corpus_20 = train_gensim_lda(tokens_20, num_topics=20)
        lda_20.save(os.path.join(models_dir, "gensim_lda_model_20.model"))
        dictionary_20.save(os.path.join(models_dir, "dictionary_20.dict"))
        with open(os.path.join(models_dir, "corpus_20.pkl"), "wb") as f:
            pickle.dump(corpus_20, f)

    if X_20 is not None:
        kmeans_20 = train_kmeans(X_20, n_clusters=20)
        with open(os.path.join(models_dir, "kmeans_model_20.pkl"), "wb") as f:
            pickle.dump(kmeans_20, f)
        df_20_meta["kmeans_cluster"] = kmeans_20.labels_
        df_20_meta.to_csv(os.path.join(processed_data_dir, "df_20_meta_with_models.csv"), index=False)

    # ==================== Mini Newsgroups ====================
    print("\n--- Mini Newsgroups Training ---")
    try:
        with open(os.path.join(processed_data_dir, 'tokenized_mini.pkl'), 'rb') as f:
            tokens_mini = pickle.load(f)
        with open(os.path.join(processed_data_dir, 'X_mini_vectorized.pkl'), 'rb') as f:
            X_mini = pickle.load(f)
        df_mini_meta = pd.read_csv(os.path.join(processed_data_dir, 'df_mini_meta.csv'))
    except Exception as e:
        print(f"[ERROR] Loading mini_newsgroups failed: {e}")
        tokens_mini, X_mini, df_mini_meta = None, None, None

    if tokens_mini:
        lda_mini, dictionary_mini, corpus_mini = train_gensim_lda(tokens_mini, num_topics=10)
        lda_mini.save(os.path.join(models_dir, "gensim_lda_model_mini.model"))
        dictionary_mini.save(os.path.join(models_dir, "dictionary_mini.dict"))
        with open(os.path.join(models_dir, "corpus_mini.pkl"), "wb") as f:
            pickle.dump(corpus_mini, f)

    if X_mini is not None:
        kmeans_mini = train_kmeans(X_mini, n_clusters=10)
        with open(os.path.join(models_dir, "kmeans_model_mini.pkl"), "wb") as f:
            pickle.dump(kmeans_mini, f)
        df_mini_meta["kmeans_cluster"] = kmeans_mini.labels_
        df_mini_meta.to_csv(os.path.join(processed_data_dir, "df_mini_meta_with_models.csv"), index=False)

    # ==================== Merged Dataset ====================
    print("\n--- Merged Dataset Training ---")
    try:
        with open(os.path.join(processed_data_dir, 'tokenized_merged.pkl'), 'rb') as f:
            tokens_merged = pickle.load(f)
        with open(os.path.join(processed_data_dir, 'X_merged_vectorized.pkl'), 'rb') as f:
            X_merged = pickle.load(f)
        df_merged_meta = pd.read_csv(os.path.join(processed_data_dir, 'df_merged_meta.csv'))
    except Exception as e:
        print(f"[ERROR] Loading merged dataset failed: {e}")
        tokens_merged, X_merged, df_merged_meta = None, None, None

    if tokens_merged:
        lda_merged, dictionary_merged, corpus_merged = train_gensim_lda(tokens_merged, num_topics=25, passes=30)
        lda_merged.save(os.path.join(models_dir, "gensim_lda_model_merged.model"))
        dictionary_merged.save(os.path.join(models_dir, "dictionary_merged.dict"))
        with open(os.path.join(models_dir, "corpus_merged.pkl"), "wb") as f:
            pickle.dump(corpus_merged, f)

    if X_merged is not None:
        kmeans_merged = train_kmeans(X_merged, n_clusters=25)
        with open(os.path.join(models_dir, "kmeans_model_merged.pkl"), "wb") as f:
            pickle.dump(kmeans_merged, f)
        df_merged_meta["kmeans_cluster"] = kmeans_merged.labels_
        df_merged_meta.to_csv(os.path.join(processed_data_dir, "df_merged_meta_with_models.csv"), index=False)

    print("\n[DONE] All LDA + KMeans models trained and saved successfully.")
