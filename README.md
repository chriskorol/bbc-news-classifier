# BBC News Klassifikator

A complete Machine Learning pipeline that automatically classifies BBC news articles into 5 categories — with **99.1% accuracy**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-4.6+-47A248?logo=mongodb&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)

## Overview

| Step | Description |
|------|-------------|
| **1. Data Import** | 2,225 BBC articles → MongoDB with metadata |
| **2. Feature Extraction** | TF-IDF with 10,000 features (unigrams + bigrams) |
| **3. Model Training** | Naive Bayes (99.1%) vs. SVM (98.9%) |
| **4. Web App** | Streamlit app for live text classification |

## Categories

| Category | Articles | Precision | Recall |
|----------|----------|-----------|--------|
| Sport | 511 | 1.00 | 0.99 |
| Business | 510 | 0.99 | 0.99 |
| Politics | 417 | 0.98 | 1.00 |
| Tech | 401 | 0.99 | 0.99 |
| Entertainment | 386 | 1.00 | 0.98 |

## Quick Start

### Prerequisites

- Python 3.10+
- MongoDB running locally (`mongod`)

### Installation

```bash
git clone https://github.com/chriskorol/bbc-news-classifier.git
cd bbc-news-classifier
pip install -r requirements.txt
```

### Run the Pipeline

Open `script.ipynb` in Jupyter and run all cells. This will:

1. Import all articles into MongoDB
2. Extract TF-IDF features
3. Train & evaluate both models
4. Save the best model as `.pkl` files

### Run the Web App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) — paste any English news text and get an instant classification.

## Project Structure

```
bbc-news-classifier/
├── script.ipynb            # Full ML pipeline (Jupyter Notebook)
├── app.py                  # Streamlit web app (Anthropic-inspired UI)
├── best_model.pkl          # Trained Naive Bayes model
├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── categories.pkl          # Category labels
├── requirements.txt        # Python dependencies
├── PRESENTATION_SCRIPT.md  # Presentation script (German)
├── News Articles/          # 2,225 BBC articles (5 categories)
│   ├── business/
│   ├── entertainment/
│   ├── politics/
│   ├── sport/
│   └── tech/
└── Summaries/              # Article summaries
```

## How It Works

### TF-IDF Feature Extraction

Each article is transformed into a vector of 10,000 numbers:

- **TF (Term Frequency):** How often a word appears in a specific article
- **IDF (Inverse Document Frequency):** Penalizes common words ("the", "is"), rewards rare informative words ("goal", "profit")
- **TF × IDF:** Captures how important a word is for a specific article relative to all others

### Model Comparison

| Model | Accuracy | Training Time |
|-------|----------|--------------|
| **Multinomial Naive Bayes** | **99.10%** | < 1 sec |
| Linear SVM | 98.88% | ~ 2 sec |

Naive Bayes was selected as the final model — slightly better accuracy, faster training, and native probability outputs for confidence scores.

## Tech Stack

- **Database:** MongoDB + pymongo
- **ML:** scikit-learn (TF-IDF, MultinomialNB, LinearSVC)
- **Visualization:** matplotlib + seaborn
- **Web App:** Streamlit
- **Serialization:** joblib

## Dataset

[BBC News Summary](https://www.kaggle.com/datasets/pariza/bbc-news-summary) — 2,225 articles from BBC News (2004–2005), 5 categories.

---

*THWS · Datenbanken-Projekt · 2026*
