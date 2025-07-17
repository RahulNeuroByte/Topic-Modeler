



# evaluation_tuning.py
# evaluation_tuning.py

import os
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

# ----------- Start Time -------------------
start = time.time()

# ---------------------- Helper ----------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------- Coherence Calculation ----------------------


def compute_coherence(dictionary, corpus, texts, start=2, limit=15, step=1, passes=10):
    scores = []
    for num_topics in range(start, limit, step):
        print(f"[INFO] Training LDA with {num_topics} topics and {passes} passes...")
        model = LdaModel(corpus=corpus,
                         num_topics=num_topics,
                         id2word=dictionary,
                         passes=passes,
                         random_state=42)
        cm = CoherenceModel(model=model, texts=texts, corpus=corpus,
                            dictionary=dictionary, coherence='c_v')
        score = cm.get_coherence()
        scores.append(score)
        print(f"[INFO] Coherence for {num_topics} topics: {score:.4f}")
    return scores

# ---------------------- Main Script ----------------------
if __name__ == "__main__":
    print("\n--- Paths Setup ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_dir = os.path.join(script_dir, "processed_data")
    models_dir = os.path.join(script_dir, "models")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("[INFO] Loading tokenized docs, corpus, and dictionaries...")

    # Tokenized text
    tokens_20 = load_pickle(os.path.join(processed_data_dir, "tokenized_20.pkl"))
    tokens_mini = load_pickle(os.path.join(processed_data_dir, "tokenized_mini.pkl"))
    tokens_merged = load_pickle(os.path.join(processed_data_dir, "tokenized_merged.pkl"))

    # Gensim corpus
    corpus_20 = load_pickle(os.path.join(models_dir, "corpus_20.pkl"))
    corpus_mini = load_pickle(os.path.join(models_dir, "corpus_mini.pkl"))
    corpus_merged = load_pickle(os.path.join(models_dir, "corpus_merged.pkl"))

    # Gensim dictionaries
    dictionary_20 = corpora.Dictionary.load(os.path.join(models_dir, "dictionary_20.dict"))
    dictionary_mini = corpora.Dictionary.load(os.path.join(models_dir, "dictionary_mini.dict"))
    dictionary_merged = corpora.Dictionary.load(os.path.join(models_dir, "dictionary_merged.dict"))

    # ---------------------- Coherence Scores ----------------------
    print("\n[INFO] Calculating coherence for 20 Newsgroups...")
    coherence_20 = compute_coherence(dictionary_20, corpus_20, tokens_20, passes=30)

    print("\n[INFO] Calculating coherence for Mini Newsgroups...")
    coherence_mini = compute_coherence(dictionary_mini, corpus_mini, tokens_mini, passes=20)

    print("\n[INFO] Calculating coherence for Merged Dataset...")
    coherence_merged = compute_coherence(dictionary_merged, corpus_merged, tokens_merged, passes=30)

    # ---------------------- Plot Results ----------------------
    x = list(range(2, 15))

    best_20 = max(coherence_20)
    best_topic_20 = x[coherence_20.index(best_20)]

    best_mini = max(coherence_mini)
    best_topic_mini = x[coherence_mini.index(best_mini)]

    best_merged = max(coherence_merged)
    best_topic_merged = x[coherence_merged.index(best_merged)]

    plt.figure(figsize=(12, 6))
    plt.plot(x, coherence_20, label="20 Newsgroups", marker='o')
    plt.plot(x, coherence_mini, label="Mini Newsgroups", marker='s')
    plt.plot(x, coherence_merged, label="Merged Dataset", marker='^')

    # Annotate best points
    plt.annotate(f"Best: {best_topic_20}", (best_topic_20, best_20), xytext=(best_topic_20+0.3, best_20),
                 arrowprops=dict(facecolor='green', arrowstyle='->'))
    plt.annotate(f"Best: {best_topic_mini}", (best_topic_mini, best_mini), xytext=(best_topic_mini+0.3, best_mini),
                 arrowprops=dict(facecolor='blue', arrowstyle='->'))
    plt.annotate(f"Best: {best_topic_merged}", (best_topic_merged, best_merged), xytext=(best_topic_merged+0.3, best_merged),
                 arrowprops=dict(facecolor='orange', arrowstyle='->'))

    plt.title("LDA Coherence Score (c_v) vs. Number of Topics")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "coherence_scores.png"))
    plt.show()

    # ---------------------- Save Coherence Scores ----------------------
    df_scores = pd.DataFrame({
        "num_topics": x,
        "coherence_20": coherence_20,
        "coherence_mini": coherence_mini,
        "coherence_merged": coherence_merged
    })
    df_scores.to_csv(os.path.join(results_dir, "coherence_scores.csv"), index=False)

    print("\n Coherence Evaluation Completed. Plot and CSV Saved.")

end = time.time()
print(f" Evaluatoin Time : {end - start:.2f} seconds")




'''



# evaluation_tuned.py (Modified to use tuned models)

import os
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

start = time.time()
print("\n Evaluating Tuned LDA Models...")

# ---------- Paths ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(script_dir, "processed_data")
models_dir = os.path.join(script_dir, "models")
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# ---------- Loaders ----------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------- Config ----------
datasets = {
    "20_newsgroups": {
        "dict": "dictionary_20.dict",
        "corpus": "corpus_20.pkl",
        "tokens": "tokenized_20.pkl",
        "model": "tuned_lda_model_20_newsgroups.model"
    },
    "mini_newsgroups": {
        "dict": "dictionary_mini.dict",
        "corpus": "corpus_mini.pkl",
        "tokens": "tokenized_mini.pkl",
        "model": "tuned_lda_model_mini_newsgroups.model"
    },
    "merged": {
        "dict": "dictionary_merged.dict",
        "corpus": "corpus_merged.pkl",
        "tokens": "tokenized_merged.pkl",
        "model": "tuned_lda_model_merged.model"
    }
}

# ---------- Results ----------
results = []

for name, files in datasets.items():
    try:
        print(f"\nEvaluating: {name.upper()}")

        dictionary = corpora.Dictionary.load(os.path.join(models_dir, files["dict"]))
        corpus = load_pickle(os.path.join(models_dir, files["corpus"]))
        tokens = load_pickle(os.path.join(processed_data_dir, files["tokens"]))
        lda_model = LdaModel.load(os.path.join(models_dir, files["model"]))

        cm_cv = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')
        cm_umass = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')

        score_cv = cm_cv.get_coherence()
        score_umass = cm_umass.get_coherence()

        print(f"[{name}] Coherence c_v: {score_cv:.4f} | u_mass: {score_umass:.4f}")
        results.append((name, score_cv, score_umass))

    except Exception as e:
        print(f" Error evaluating {name}: {e}")

# ---------- Plot ----------
if results:
    df = pd.DataFrame(results, columns=["Dataset", "Coherence_c_v", "Coherence_u_mass"])
    df.set_index("Dataset", inplace=True)

    ax = df.plot(kind='bar', figsize=(10, 5), rot=0, colormap='Set2', title="Tuned LDA Coherence Scores")
    plt.ylabel("Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "tuned_lda_coherence_scores.png"))
    plt.show()

    df.to_csv(os.path.join(results_dir, "tuned_lda_coherence_scores.csv"))
    print("\n Saved: tuned_lda_coherence_scores.csv and .png")

end = time.time()
print(f"\n⏱ Evaluation completed in {end - start:.2f} seconds.")


'''
