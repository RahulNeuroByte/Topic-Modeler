# preprocessing.py

import re
import nltk
import logging
from sklearn.preprocessing import FunctionTransformer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Download required NLTK resources once ---
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# --- Global resources ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------- Text Cleaning --------------------

def clean_text(text: str) -> str:
    """Remove email quotes, special characters, and normalize whitespace."""
    _QUOTE_RE = re.compile(r"(writes in|writes:|wrote:|says:|said:)[^\n]*\n[>\\|]+.*", re.MULTILINE)
    text = re.sub(_QUOTE_RE, '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------------------- Batch Preprocessing --------------------

def preprocess_text_pipeline(texts: list[str]) -> list[str]:
    """
    Clean, tokenize, lemmatize, and remove stopwords from a list of texts.
    Returns list of cleaned strings.
    """
    logger.info(f"Preprocessing {len(texts)} texts...")
    processed_texts = []
    for text in texts:
        cleaned = clean_text(text)
        tokens = cleaned.lower().split()
        processed = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
        processed_texts.append(" ".join(processed))
    return processed_texts

# -------------------- For Sklearn Pipeline --------------------

class TextPreprocessor(FunctionTransformer):
    """
    Sklearn-compatible transformer for preprocessing pipeline steps.
    Usage: pipeline = Pipeline([('pre', TextPreprocessor()), ...])
    """
    def __init__(self):
        super().__init__(func=preprocess_text_pipeline)

# -------------------- For Single User Input --------------------

def preprocess_single_text(text: str) -> str:
    """
    Clean and preprocess a single input string (e.g., from user or Streamlit app).
    Returns cleaned string.
    """
    return preprocess_text_pipeline([text])[0]
