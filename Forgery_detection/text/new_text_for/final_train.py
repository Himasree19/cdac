# ==========================================
# TEXT FORGERY DETECTION (ADVANCED)
# TF-IDF + NGRAM + STYLOMETRIC + EMBEDDINGS
# Human = 0 | AI = 1
# ==========================================

import os
import pandas as pd
import numpy as np
import re
import string
import joblib

import nltk
from nltk import word_tokenize, pos_tag
from textstat import flesch_reading_ease

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sentence_transformers import SentenceTransformer

# download once
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')

# ----------------------------------
# PATH
# ----------------------------------

TRAIN_FOLDER = r"C:\Users\LENOVO\Desktop\new_text_for\dataset_2\Training"

fake_path = os.path.join(TRAIN_FOLDER, "Fake.csv")
real_path = os.path.join(TRAIN_FOLDER, "Real.csv")

# ----------------------------------
# LOAD DATA
# ----------------------------------

fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

if "sentence" in fake_df.columns:
    fake_df = fake_df.rename(columns={"sentence": "text"})

if "sentence" in real_df.columns:
    real_df = real_df.rename(columns={"sentence": "text"})

fake_df["label"] = 1
real_df["label"] = 0

df = pd.concat(
    [fake_df[["text","label"]], real_df[["text","label"]]],
    ignore_index=True
)

df = df.dropna().drop_duplicates(subset=["text"])

print("Total Samples:", len(df))

texts = df["text"].astype(str)
y = df["label"]

# ==========================================
# 1. STYLOMETRIC FEATURES
# ==========================================

def extract_stylometric(text):
    words = word_tokenize(text)

    # Sentence length variance
    sentences = re.split(r'[.!?]', text)
    sent_lengths = [len(s.split()) for s in sentences if s.strip() != ""]
    sent_var = np.var(sent_lengths) if len(sent_lengths) > 0 else 0

    # Word frequency (total words)
    word_freq = len(words)

    # Punctuation usage
    punct_count = sum(1 for c in text if c in string.punctuation)

    # POS tag distribution
    pos_tags = pos_tag(words)
    noun_count = sum(1 for w, t in pos_tags if "NN" in t)
    verb_count = sum(1 for w, t in pos_tags if "VB" in t)

    # Readability score
    readability = flesch_reading_ease(text)

    return [sent_var, word_freq, punct_count, noun_count, verb_count, readability]


stylometric_features = np.array([extract_stylometric(t) for t in texts])

# ==========================================
# 2. LINGUISTIC FEATURES
# ==========================================

# TF-IDF (IMPORTANT)
tfidf_vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1,2)
)
tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()

# N-grams (separate explicit feature)
ngram_vectorizer = CountVectorizer(
    ngram_range=(2,2),
    max_features=1000
)
ngram_features = ngram_vectorizer.fit_transform(texts).toarray()

# Syntax patterns → already captured via POS counts above

# ==========================================
# 3. SEMANTIC FEATURES
# ==========================================

# Using Sentence-BERT (practical)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_embeddings = sbert_model.encode(texts.tolist())

# NOTE:
# You can also use:
# :contentReference[oaicite:0]{index=0}
# :contentReference[oaicite:1]{index=1}
# :contentReference[oaicite:2]{index=2}

# ==========================================
# 4. COMBINE ALL FEATURES
# ==========================================

X = np.hstack([
    stylometric_features,     # Stylometric
    tfidf_features,           # TF-IDF
    ngram_features,           # N-grams
    sbert_embeddings          # Semantic
])

# ==========================================
# 5. MODEL
# ==========================================

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

# ==========================================
# 6. EVALUATION
# ==========================================

pred = rf.predict(X)

print("\nAccuracy:", accuracy_score(y, pred))

print("\nClassification Report:\n")
print(classification_report(y, pred, target_names=["Human","AI"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y, pred))

# ==========================================
# CONFUSION MATRIX (VISUAL + VALUES)
# ==========================================

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y, pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Human", "AI"]
)

disp.plot(cmap="Blues")
plt.title("Text Forgery Detection - Confusion Matrix")
plt.show()

# ==========================================
# 7. SAVE EVERYTHING
# ==========================================

joblib.dump(rf, "final_text_model.pkl")
joblib.dump(tfidf_vectorizer, "tf_idf.pkl")
joblib.dump(ngram_vectorizer, "n_gram.pkl")

print("\n✅ Model & vectorizers saved")