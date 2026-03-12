# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 14:34:59 2025

@author: simic
"""

# first part of the project
# - Load dataset
# - Preprocess (cleaning, token filtering)
# - EDA (class distribution, length histograms, top words per class)
# - Train classical models (LogReg, MultinomialNB, LinearSVC)
# - Light hyperparameter search (GridSearchCV) with safe defaults
# - Evaluate and save models & artifacts

# %% Libraries

import os
import re
import string
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics

# %% Configuration
DATA_PATH = "IMDB Dataset.csv"
OUTPUT_DIR = "./sentiment_student1_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# %% Utility functions

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Expected columns 'review' and 'sentiment' in CSV.")
    return df

def clean_text(text):
    """Lightweight cleaning: remove HTML, lowercase, remove urls, punctuation, digits, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", " ", text)                      # remove html tags
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)    # remove urls
    # replace punctuation with spaces
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"\d+", " ", text)                        # remove digits
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords_and_short(tokens):
    stopwords = ENGLISH_STOP_WORDS
    return [t for t in tokens if t not in stopwords and len(t) > 1]

def preprocess_series(series):
    cleaned = series.fillna("").map(clean_text)
    tokenized = cleaned.map(lambda s: s.split())
    filtered = tokenized.map(remove_stopwords_and_short)
    return filtered.map(lambda tokens: " ".join(tokens))

# %% Load and preprocess

print("Loading dataset...")
df = load_data(DATA_PATH)
print("Dataset shape:", df.shape)

print("Preprocessing text (cleaning, stopwords removal)...")
df['clean_review'] = preprocess_series(df['review'])
df['review_len_raw'] = df['review'].fillna("").map(lambda s: len(s.split()))
df['review_len_clean'] = df['clean_review'].map(lambda s: len(s.split()))

# Save small sample for inspection
df.sample(200, random_state=RANDOM_STATE).to_csv(os.path.join(OUTPUT_DIR, "cleaned_sample.csv"), index=False)

# %% EDA 

# Class distribution
class_counts = df['sentiment'].value_counts()
print("\nClass counts:\n", class_counts)

plt.figure(figsize=(6,4))
plt.bar(class_counts.index, class_counts.values)
plt.title("Class distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"))
plt.show()

# Length histograms
plt.figure(figsize=(8,4))
plt.hist(df['review_len_raw'], bins=50)
plt.title("Raw review lengths (tokens)")
plt.xlabel("Tokens (raw)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hist_len_raw.png"))
plt.show()

plt.figure(figsize=(8,4))
plt.hist(df['review_len_clean'], bins=50)
plt.title("Cleaned review lengths (tokens)")
plt.xlabel("Tokens (cleaned)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hist_len_clean.png"))
plt.show()

# Top words per class (CountVectorizer with limited feature size for speed)
def top_n_words_for_class(df, label, n=25, max_features=5000):
    vect = CountVectorizer(max_features=max_features)
    texts = df[df['sentiment'] == label]['clean_review']
    X = vect.fit_transform(texts)
    sums = np.array(X.sum(axis=0)).ravel()
    idx = np.argsort(sums)[::-1][:n]
    features = np.array(vect.get_feature_names_out())[idx]
    counts = sums[idx]
    return list(zip(features, counts))

top_pos = top_n_words_for_class(df, 'positive', n=20)
top_neg = top_n_words_for_class(df, 'negative', n=20)
print("\nTop positive words:", top_pos)
print("\nTop negative words:", top_neg)

# Save top words to file
pd.DataFrame(top_pos, columns=["word", "count"]).to_csv(os.path.join(OUTPUT_DIR, "top_positive_words.csv"), index=False)
pd.DataFrame(top_neg, columns=["word", "count"]).to_csv(os.path.join(OUTPUT_DIR, "top_negative_words.csv"), index=False)

# %% Modeling

# Labels: positive -> 1, negative -> 0
y = df['sentiment'].map({'positive': 1, 'negative': 0}).astype(int)
X = df['clean_review']

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    stratify=y, random_state=RANDOM_STATE)
print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])

# TF-IDF vectorizer (safe defaults)
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=5)

# Helper evaluation function
def evaluate(pipeline, X_test, y_test, name):
    preds = pipeline.predict(X_test)
    acc = metrics.accuracy_score(y_test, preds)
    report = metrics.classification_report(y_test, preds, target_names=['negative','positive'])
    cm = metrics.confusion_matrix(y_test, preds)
    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print(report)
    print("Confusion matrix:\n", cm)

    # ROC AUC if available
    y_score = None
    final_step = list(pipeline.named_steps.keys())[-1]
    clf = pipeline.named_steps[final_step]
    try:
        if hasattr(clf, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:,1]
        elif hasattr(clf, "decision_function"):
            y_score = pipeline.decision_function(X_test)
    except Exception:
        y_score = None

    if y_score is not None:
        auc = metrics.roc_auc_score(y_test, y_score)
        print("ROC AUC:", auc)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr)
        plt.title(f"ROC curve - {name} (AUC={auc:.4f})")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"roc_{name}.png"))
        plt.show()

# Baseline Logistic Regression
pipe_lr = Pipeline([
    ('tfidf', tfidf),
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear', random_state=RANDOM_STATE))
])
print("\nTraining LogisticRegression (baseline)...")
pipe_lr.fit(X_train, y_train)
evaluate(pipe_lr, X_test, y_test, "LogisticRegression_baseline")

# Multinomial Naive Bayes
pipe_nb = Pipeline([('tfidf', tfidf), ('clf', MultinomialNB())])
print("\nTraining MultinomialNB...")
pipe_nb.fit(X_train, y_train)
evaluate(pipe_nb, X_test, y_test, "MultinomialNB")

# Linear SVM
pipe_svc = Pipeline([('tfidf', tfidf), ('clf', LinearSVC(max_iter=2000, random_state=RANDOM_STATE))])
print("\nTraining LinearSVC...")
pipe_svc.fit(X_train, y_train)
evaluate(pipe_svc, X_test, y_test, "LinearSVC")

# Cross-validation on train set (quick)
print("\nCross-val accuracy (5-fold) on training data:")
for name, model in [("LogReg", pipe_lr), ("NaiveBayes", pipe_nb), ("LinearSVC", pipe_svc)]:
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

# ---------- Light Grid Search (fast) ----------
# Use smaller TF-IDF for speed during grid search
tfidf_grid = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=5)

pipe_lr_gs = Pipeline([('tfidf', tfidf_grid), ('clf', LogisticRegression(max_iter=1000, solver='liblinear', random_state=RANDOM_STATE))])
pipe_svc_gs = Pipeline([('tfidf', tfidf_grid), ('clf', LinearSVC(max_iter=2000, random_state=RANDOM_STATE))])

param_grid_lr = {'clf__C': [0.1, 1.0, 5.0]}
param_grid_svc = {'clf__C': [0.1, 1.0, 5.0]}

print("\nRunning small GridSearchCV for LogisticRegression (cv=3)...")
gs_lr = GridSearchCV(pipe_lr_gs, param_grid_lr, cv=3, scoring='f1', n_jobs=-1, verbose=1)
gs_lr.fit(X_train, y_train)
print("Best LR params:", gs_lr.best_params_, "Best CV f1:", gs_lr.best_score_)

print("\nRunning small GridSearchCV for LinearSVC (cv=3)...")
gs_svc = GridSearchCV(pipe_svc_gs, param_grid_svc, cv=3, scoring='f1', n_jobs=-1, verbose=1)
gs_svc.fit(X_train, y_train)
print("Best SVC params:", gs_svc.best_params_, "Best CV f1:", gs_svc.best_score_)

# Evaluate best estimators
best_lr = gs_lr.best_estimator_
best_svc = gs_svc.best_estimator_
evaluate(best_lr, X_test, y_test, "LogReg_gridbest")
evaluate(best_svc, X_test, y_test, "LinearSVC_gridbest")

# %% Save models & artifacts
models = {
    "pipe_lr_baseline": pipe_lr,
    "pipe_nb": pipe_nb,
    "pipe_svc": pipe_svc,
    "best_lr": best_lr,
    "best_svc": best_svc
}
for name, model in models.items():
    p = os.path.join(OUTPUT_DIR, f"{name}.pkl")
    with open(p, "wb") as f:
        pickle.dump(model, f)
    print("Saved", name, "->", p)

# Save summarized results to CSV
results = []
for name, est in [("LogReg_baseline", pipe_lr), ("NB", pipe_nb), ("SVC", pipe_svc),
                  ("LogReg_best", best_lr), ("SVC_best", best_svc)]:
    preds = est.predict(X_test)
    acc = metrics.accuracy_score(y_test, preds)
    try:
        y_score = est.predict_proba(X_test)[:,1]
        auc = metrics.roc_auc_score(y_test, y_score)
    except Exception:
        try:
            y_score = est.decision_function(X_test)
            auc = metrics.roc_auc_score(y_test, y_score)
        except Exception:
            auc = np.nan
    results.append({"model": name, "accuracy": acc, "roc_auc": auc})
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_summary.csv"), index=False)

# Save some misclassified examples for error analysis
def save_misclassified(estimator, X_test, y_test, name, n=200):
    preds = estimator.predict(X_test)
    mask = preds != y_test
    mis = pd.DataFrame({"text": X_test[mask], "true": y_test[mask], "pred": preds[mask]})
    mis = mis.sample(min(len(mis), n), random_state=RANDOM_STATE)
    mis.to_csv(os.path.join(OUTPUT_DIR, f"misclassified_{name}.csv"), index=False)

save_misclassified(pipe_lr, X_test, y_test, "LogReg_baseline", n=300)
save_misclassified(pipe_nb, X_test, y_test, "NaiveBayes", n=300)
save_misclassified(pipe_svc, X_test, y_test, "LinearSVC", n=300)

print("\nStudent 1 workflow finished. Artifacts saved in:", OUTPUT_DIR)
print("Files created (examples):", os.listdir(OUTPUT_DIR)[:20])
print("Suggested next steps: deeper error analysis, examine sentences in misclassified files, pass artifacts to Student 2 for embedding-based deep models.")
