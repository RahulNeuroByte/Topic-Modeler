# pyLDAvis_generator.py


import os
import pickle
import numpy as np
import pyLDAvis
import pyLDAvis._prepare as prep
from gensim.models.ldamodel import LdaModel

# ------------------ Model Names and File Mapping ------------------
model_map = {
    "20_newsgroups": "20",
    "mini_newsgroups": "mini",
    "merged": "merged"
}

# ------------------ Paths ------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")
results_dir = os.path.join(script_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# ------------------ Function ------------------
def generate_vis(display_name, file_prefix):
    try:
        print(f"\n Processing model: {display_name}")

        # File paths
        lda_model_path = os.path.join(models_dir, f"tuned_lda_model_{display_name}.model")
        corpus_path = os.path.join(models_dir, f"corpus_{file_prefix}.pkl")

        # Load LDA model
        print(f"[INFO] Loading LDA model: {lda_model_path}")
        lda_model = LdaModel.load(lda_model_path)

        # Load corpus
        print(f"[INFO] Loading corpus: {corpus_path}")
        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)

        dictionary = lda_model.id2word
        vocab = [dictionary[i] for i in range(len(dictionary))]

        # Topic-Term Distributions
        print("[INFO] Preparing topic-term distributions...")
        topic_term_dists = lda_model.get_topics()
        assert topic_term_dists.shape[1] == len(vocab), (
            f"Mismatch: topic_term_dists has {topic_term_dists.shape[1]} terms, but vocab has {len(vocab)}"
        )

        # Document-Topic Distributions
        doc_topic_dists = np.array([
            [prob for _, prob in lda_model.get_document_topics(doc, minimum_probability=0.0)]
            for doc in corpus
        ])

        # Document Lengths
        doc_lengths = [sum(cnt for _, cnt in doc) for doc in corpus]

        # Term Frequency
        term_frequency = np.zeros(len(vocab))
        for doc in corpus:
            for idx, count in doc:
                term_frequency[idx] += count
        term_frequency = term_frequency.astype(int)

        # Generate Visualization
        print(f"[INFO] Generating pyLDAvis HTML for {display_name}...")
        vis_data = pyLDAvis.prepare(
            topic_term_dists=topic_term_dists,
            doc_topic_dists=doc_topic_dists,
            doc_lengths=doc_lengths,
            vocab=vocab,
            term_frequency=term_frequency
        )

        output_path = os.path.join(results_dir, f"lda_topics_visual_{display_name}.html")
        pyLDAvis.save_html(vis_data, output_path)
        print(f" Saved: {output_path}")

    except Exception as e:
        print(f" Error for model '{display_name}': {e}")

# ------------------ Loop ------------------
if __name__ == "__main__":
    print(" Starting pyLDAvis generation for all tuned LDA models...")
    for display_name, file_prefix in model_map.items():
        generate_vis(display_name, file_prefix)
    print("\n All visualizations generated (or attempted).")
