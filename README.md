

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
git clone https://github.com/RahulNeuroByte/Topic-Modeler.git

### 2. Install Dependencies from requirements.txt
> Make sure to include:
>
absl-py==2.1.0
aiohappyeyeballs==2.6.1
aiohttp==3.12.14
aiosignal==1.4.0
asgiref==3.8.1
asttokens==3.0.0
astunparse==1.6.3
attrs==25.3.0
certifi==2025.1.31
cffi==1.17.1
charset-normalizer==3.4.2
click==8.2.1
colorama==0.4.6
comm==0.2.2
contourpy==1.3.2
cycler==0.12.1
debugpy==1.8.14
decorator==5.2.1
Deprecated==1.2.18
distlib==0.3.9
Django==5.0.6
executing==2.2.0
filelock==3.18.0
flatbuffers==24.3.7
fonttools==4.58.4
frozenlist==1.7.0
gast==0.5.4
geopandas==1.0.1
git-filter-repo==2.47.0
google-pasta==0.2.0
grpcio==1.62.1
gTTS==2.5.4
h5py==3.10.0
idna==3.10
imblearn==0.0
ipykernel==6.29.5
ipython==9.3.0
ipython_pygments_lexers==1.1.1
jedi==0.19.2
joblib==1.5.1
jupyter_client==8.6.3
jupyter_core==5.8.1
keras==3.1.0
kiwisolver==1.4.8
libclang==18.1.1
matplotlib==3.10.3
matplotlib-inline==0.1.7
ml-dtypes==0.3.2
mlxtend==0.23.4
multidict==6.6.3
namex==0.0.7
narwhals==1.43.1
nest-asyncio==1.6.0
nltk==3.9.1
numpy==2.3.1
opencv-python==4.10.0.84
opencv-python-headless==4.10.0.84
opt-einsum==3.3.0
optree==0.10.0
packaging==24.2
pandas==2.3.0
parso==0.8.4
pillow==11.2.1
pipenv==2024.4.1
platformdirs==4.3.7
playsound==1.3.0
plotly==6.1.2
prompt_toolkit==3.0.51
propcache==0.3.2
psutil==7.0.0
pure_eval==0.2.3
pycparser==2.22
PyGithub==2.6.1
Pygments==2.19.1
PyJWT==2.10.1
PyNaCl==1.5.0
pyogrio==0.10.0
pyparsing==3.2.3
pyproj==3.7.0
python-dateutil==2.9.0.post0
python-dotenv==1.1.1
pytz==2025.2
pywin32==310
pyzmq==27.0.0
regex==2024.11.6
requests==2.32.4
scikit-learn==1.6.1
scipy==1.15.3
seaborn==0.13.2
shapely==2.0.6
six==1.17.0
sqlparse==0.5.0
stack-data==0.6.3
tensorboard==2.16.2
tensorboard-data-server==0.7.2
tensorflow==2.16.1
tensorflow-intel==2.16.1
tensorflow-io-gcs-filesystem==0.31.0
termcolor==2.4.0
textblob==0.19.0
threadpoolctl==3.6.0
tornado==6.5.1
tqdm==4.67.1
traitlets==5.14.3
typing_extensions==4.14.0
tzdata==2025.2
urllib3==2.5.0
virtualenv==20.30.0
wcwidth==0.2.13
wordcloud==1.9.4
wrapt==1.17.2
yarl==1.20.1

---

## ⚙️ Pipeline

### 🔹 Step 1: Preprocessing

 data_ingestion_preprocessing.py

* Loads raw data
* Cleans, lemmatizes, tokenizes
* Saves preprocessed CSVs and tokenized lists

### 🔹 Step 2: Generate Gensim Dictionary & Corpus

* Creates dictionary_*.dict and corpus_*.pkl`

### 🔹 Step 3: Export TF-IDF Pipelines

* Saves pipeline_*.pkl and vectorizer_*.pkl

### 🔹 Step 4: Tune Models

* Trains LDA with `passes=30`, `iterations=400`, `alpha/eta='auto'`
* Saves best `tuned_lda_model_*.model` based on coherence
* Tunes KMeans and saves best `tuned_kmeans_model_*.pkl`

### 🔹 Step 5: Evaluate (Optional)

* Evaluates models across topic ranges
* Saves plots + CSVs of coherence scores

### 🔹 Step 6: Generate pyLDAvis (Optional)

* Creates `lda_topics_visual_*.html` for each dataset

## 🌐 Run the Streamlit App

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
* Project structure and optimization by **Rahul Kumar Dubey**

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