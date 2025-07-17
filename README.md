

# 🧐 Document Topic Modeling & Clustering (LDA + KMeans)

This project implements an end-to-end document topic modeling and clustering system using Latent Dirichlet Allocation (LDA) and KMeans on the 20 Newsgroups, Mini Newsgroups, and their merged dataset. It supports automated preprocessing, model tuning, evaluation, and a responsive Streamlit-based web interface for real-time document analysis and visualization.

---

## 📁 Project Structure

```
D:\Project_DC\
│
├── app.py                         # Streamlit app for prediction and visualization
├── utils.py                       # Preprocessing, model utils, file handling
├── data_ingestion_preprocessing.py  # Loads, cleans, tokenizes raw text
├── generate_gensim_data.py        # Generates corpus + dictionary for Gensim LDA
├── pipeline_export.py             # Saves TF-IDF + preprocessing pipelines
├── model_training.py              # Trains LDA + KMeans (base version)
├── tune_model.py                  # Tunes LDA and KMeans with coherence/silhouette scores
├── evaluation_tuning.py           # Evaluates tuned models and plots coherence curves
├── pyLDAvis_generator.py          # Generates interactive HTML visualizations for LDA
├── models/                        # All trained models (LDA, KMeans, pipeline, vectorizer)
├── processed_data/                # Preprocessed CSVs, tokenized files, vectorized data
├── results/                       # Plots, pyLDAvis HTML outputs, CSVs
└── README.md                      # You’re here!
```

---

## 📌 Features

* ✅ Text cleaning, lemmatization, tokenization
* ✅ LDA topic modeling (Gensim)
* ✅ KMeans clustering (Scikit-learn)
* ✅ Auto-tuning (coherence & silhouette score)
* ✅ Interactive Streamlit app:

  * 📄 Upload file (TXT, PDF, CSV, JSON)
  * ✍️ Paste custom text
  * 🧠 Predict topic and cluster
  * 📊 Visualize topic keywords (pie chart, word cloud, distribution)
  * 📅 Download graphs
* ✅ pyLDAvis visualization of full corpus
* ✅ Compatible with both mobile and desktop
* ✅ Modular, production-ready code

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/document-topic-modeling.git
cd document-topic-modeling
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Make sure to include:
>
> * `streamlit`
> * `gensim`
> * `scikit-learn`
> * `pandas`, `matplotlib`, `wordcloud`, `pyldavis`
> * `PyMuPDF` (for PDFs)
> * `nltk`

### 3. Download NLTK Assets

```python
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
```

---

## ⚙️ Pipeline

### 🔹 Step 1: Preprocessing

```bash
python data_ingestion_preprocessing.py
```

* Loads raw data
* Cleans, lemmatizes, tokenizes
* Saves preprocessed CSVs and tokenized lists

### 🔹 Step 2: Generate Gensim Dictionary & Corpus

```bash
python generate_gensim_data.py
```

* Creates `dictionary_*.dict` and `corpus_*.pkl`

### 🔹 Step 3: Export TF-IDF Pipelines

```bash
python pipeline_export.py
```

* Saves `pipeline_*.pkl` and `vectorizer_*.pkl`

### 🔹 Step 4: Tune Models

```bash
python tune_model.py
```

* Trains LDA with `passes=30`, `iterations=400`, `alpha/eta='auto'`
* Saves best `tuned_lda_model_*.model` based on coherence
* Tunes KMeans and saves best `tuned_kmeans_model_*.pkl`

### 🔹 Step 5: Evaluate (Optional)

```bash
python evaluation_tuning.py
```

* Evaluates models across topic ranges
* Saves plots + CSVs of coherence scores

### 🔹 Step 6: Generate pyLDAvis (Optional)

```bash
python pyLDAvis_generator.py
```

* Creates `lda_topics_visual_*.html` for each dataset

---

## 🌐 Run the Streamlit App

```bash
streamlit run app.py
```

* Choose dataset (`20`, `mini`, `merged`)
* Upload a file or paste text
* Click **Analyze**
* View predictions and download visualizations

---

## 📊 Visualizations

* Pie chart of top topic keywords
* Word cloud for the topic
* Topic score distribution for the document
* pyLDAvis (full corpus, embedded)

---

## 📌 Supported File Types

* `.txt` (plain text)
* `.pdf` (via PyMuPDF)
* `.csv` (extracts first object-type column)
* `.json` (extracts first string column)

---

## 📊 Model Quality

* LDA models tuned using:

  * `c_v` coherence score
  * `u_mass` (secondary)
* KMeans models tuned using silhouette score
* Coherence plots saved in `/results/`

---

## ✅ Final Deliverables

* ✅ All tuned models in `/models/`
* ✅ Evaluation plots in `/results/`
* ✅ Streamlit app for real-time prediction
* ✅ Interactive pyLDAvis for each dataset

---

## 📌 To-Do / Extensions

* [ ] Topic labeling with human-readable labels
* [ ] Deep learning topic models (BERTopic, Top2Vec)
* [ ] Online prediction API (Flask or FastAPI)
* [ ] Add feedback loop for label correction

---

## 🙌 Credits

* Scikit-learn, Gensim, PyLDAvis, Streamlit, NLTK
* 20 Newsgroups Dataset (sklearn.datasets)
* Project structure and optimization by **\[Your Name]**

---
## 💻 Input Formats Supported
- PDF
- CSV
- Plain Text
- JSON

## 📊 Visualizations
- pyLDAvis
- Word clouds
- Cluster distances