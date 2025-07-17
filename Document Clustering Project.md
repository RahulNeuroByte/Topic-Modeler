# Document Clustering Project

This project provides a comprehensive solution for document clustering using LDA and K-Means, with a focus on a robust, modular, and deployable architecture. All code is designed to run on VS Code on Windows.

## Project Structure

```
. \
├── data/                 # Contains the raw newsgroup datasets
│   ├── 20_newsgroups/ 
│   └── mini_newsgroups/
├── src/                  # Source code for different modules
│   ├── data_ingestion_preprocessing.py
│   ├── model_training.py
│   ├── evaluation_tuning.py
│   ├── pipeline_export.py
│   └── app.py            # Streamlit application
├── requirements.txt      # Python dependencies
├── RUN_GUIDE.md          # Step-by-step guide to run the project
└── README.md             # This file
```

## Setup Instructions

### 1. Prepare your Environment

*   **Python:** Ensure Python 3.8+ is installed. During installation, make sure to check 'Add Python to PATH'.
*   **VS Code:** Download and install Visual Studio Code from [https://code.visualstudio.com/](https://code.visualstudio.com/).
*   **VS Code Extensions:** Install the 'Python' and 'Jupyter' extensions from the VS Code Extensions marketplace.

### 2. Download and Extract Dataset

1.  Download the `twenty+newsgroups.zip` file (the one you provided).
2.  Extract its contents. You should find `20_newsgroups.tar.gz` and `mini_newsgroups.tar.gz` inside.
3.  Create a `data` folder in your project root directory (e.g., `C:\YourProject\data`).
4.  Extract `20_newsgroups.tar.gz` into `data` folder. This will create `data\20_newsgroups`.
5.  Extract `mini_newsgroups.tar.gz` into `data` folder. This will create `data\mini_newsgroups`.

### 3. Install Dependencies

Open your terminal/command prompt in the project root directory and run:

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data

Open a Python interpreter in your terminal and run:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
exit()
```

## How to Run the Project

Refer to `RUN_GUIDE.md` for detailed instructions on how to execute each part of the project sequentially. Each script is designed to be run independently and will save necessary outputs for the next step.


