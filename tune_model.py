

# tune_model.py (Enhanced with better hyperparameters and accuracy)

import os
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim import corpora

start = time.time()

# ----------------- Setup Paths -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(script_dir, "processed_data")
models_dir = os.path.join(script_dir, "models")
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# ----------------- Load Functions -----------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_dict(path):
    return corpora.Dictionary.load(path)

# ----------------- Enhanced LDA Tuning -----------------
def tune_lda(dictionary, corpus, texts, name):
    scores_cv, scores_umass = [], []
    best_model = None
    best_score = -1

    for num_topics in range(5, 31, 5):
        print(f"[{name}] Training LDA with {num_topics} topics...")
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=30,
            iterations=400,
            alpha='auto',
            eta='auto',
            random_state=42
        )

        cm_cv = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        cm_umass = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')

        score_cv = cm_cv.get_coherence()
        score_umass = cm_umass.get_coherence()
        scores_cv.append((num_topics, score_cv))
        scores_umass.append((num_topics, score_umass))

        print(f"[{name}] Topics={num_topics}, c_v={score_cv:.4f}, u_mass={score_umass:.4f}")

        if score_cv > best_score:
            best_score = score_cv
            best_model = model

    # Save best model
    best_model.save(os.path.join(models_dir, f"tuned_lda_model_{name}.model"))

    # Plotting
    x_cv, y_cv = zip(*scores_cv)
    x_umass, y_umass = zip(*scores_umass)

    plt.figure(figsize=(10, 5))
    plt.plot(x_cv, y_cv, label='Coherence (c_v)', marker='o')
    plt.plot(x_umass, y_umass, label='Coherence (u_mass)', marker='s')
    plt.xlabel("Number of Topics")
    plt.ylabel("Score")
    plt.title(f"LDA Coherence Scores ({name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"lda_coherence_{name}.png"))
    plt.close()
    print(f"[{name}]  Best LDA model saved with {best_score:.4f} coherence.")

# ----------------- Enhanced KMeans Tuning -----------------
def tune_kmeans(X, name):
    scores = []
    best_model = None
    best_score = -1

    for k in range(5, 31, 5):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append((k, score))
        print(f"[{name}] K={k}, Silhouette Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_model = kmeans

    # Save best model
    with open(os.path.join(models_dir, f"tuned_kmeans_model_{name}.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    # Plotting
    x, y = zip(*scores)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', color='green')
    plt.xlabel("K (Clusters)")
    plt.ylabel("Silhouette Score")
    plt.title(f"KMeans Silhouette Scores ({name})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"kmeans_silhouette_{name}.png"))
    plt.close()
    print(f"[{name}]  Best KMeans model saved with silhouette={best_score:.4f}")

# ----------------- Main -----------------
if __name__ == "__main__":
    print("\n🔧 Starting enhanced model tuning...")

    datasets = {
        "20_newsgroups": {
            "vec": "X_20_vectorized.pkl",
            "dict": "dictionary_20.dict",
            "corpus": "corpus_20.pkl",
            "meta": "df_20_meta.csv"
        },
        "mini_newsgroups": {
            "vec": "X_mini_vectorized.pkl",
            "dict": "dictionary_mini.dict",
            "corpus": "corpus_mini.pkl",
            "meta": "df_mini_meta.csv"
        },
        "merged": {
            "vec": "X_merged_vectorized.pkl",
            "dict": "dictionary_merged.dict",
            "corpus": "corpus_merged.pkl",
            "meta": "df_merged_meta.csv"
        }
    }

    for name, paths in datasets.items():
        try:
            print(f"\n Tuning: {name.upper()}")
            X = load_pickle(os.path.join(processed_data_dir, paths["vec"]))
            corpus = load_pickle(os.path.join(models_dir, paths["corpus"]))
            dictionary = load_dict(os.path.join(models_dir, paths["dict"]))
            df_meta = pd.read_csv(os.path.join(processed_data_dir, paths["meta"]))
            texts = [text.split() for text in df_meta["preprocessed_text"]]

            tune_lda(dictionary, corpus, texts, name)
            tune_kmeans(X, name)

        except Exception as e:
            print(f"[ ERROR] Failed for {name}: {e}")

    end = time.time()
    print(f"\n All tuning completed. Time taken: {end - start:.2f} seconds")
