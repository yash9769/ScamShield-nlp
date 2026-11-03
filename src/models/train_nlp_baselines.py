# =============================================================================
# FILE: src/models/train_nlp_baselines.py   (PIPELINE STAGE 4: CLASSICAL ML MODELS)
# PURPOSE: Train and evaluate three CLASSICAL machine-learning classifiers on
#          TF-IDF features:  Logistic Regression, Naive Bayes, Linear SVM.
#          These are the "baselines" that the deep-learning models are compared to.
# INPUT:   dataset/processed/train.csv, val.csv, test.csv
# OUTPUT:  models/tfidf_vectorizer.joblib, logistic_regression.joblib,
#          naive_bayes.joblib, svm.joblib
#          reports/nlp_baseline_results.json, reports/nlp_baseline_report.md
# =============================================================================
import os
import json
import joblib                      # saves/loads Python objects (trained models) to disk
import pandas as pd
import numpy as np
import sys

from sklearn.feature_extraction.text import TfidfVectorizer   # text -> number vectors
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV         # adds probability output to the SVM
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PROCESSED_DIR, BASE_DIR, RANDOM_SEED

MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

def train_and_evaluate():
    """Fit TF-IDF + 3 classifiers, score them on the test set, save models + reports."""
    print("Starting NLP Baseline Models Training & Evaluation...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load dataset splits
    train_path = os.path.join(PROCESSED_DIR, 'train.csv')
    val_path = os.path.join(PROCESSED_DIR, 'val.csv')
    test_path = os.path.join(PROCESSED_DIR, 'test.csv')

    print(f"Loading data splits from {PROCESSED_DIR}...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # fillna('') -> an empty/missing text becomes an empty string so TF-IDF does not crash
    # Handle any potential NaNs in text
    X_train_raw = train_df['text'].fillna('')
    y_train = train_df['label']

    X_val_raw = val_df['text'].fillna('')
    y_val = val_df['label']

    X_test_raw = test_df['text'].fillna('')
    y_test = test_df['label']

    print(f"Dataset sizes -> Train: {len(X_train_raw):,}, Val: {len(X_val_raw):,}, Test: {len(X_test_raw):,}")

    # ---- STEP 1: FEATURE EXTRACTION with TF-IDF ----
    # TF-IDF turns each message into a vector of 10,000 numbers.
    #   TF  (term frequency)          = how often a word/phrase appears in THIS message
    #   IDF (inverse doc. frequency)  = how RARE it is across all messages
    #   weight = TF x IDF -> common words ("the") get low weight, distinctive ones ("claim") high.
    # Settings:
    #   ngram_range=(1,2)  -> use single words AND two-word phrases ("click here")
    #   max_features=10000 -> keep only the 10,000 most frequent terms
    #   min_df=2           -> ignore terms that appear in fewer than 2 messages (typos, noise)
    #   sublinear_tf=True  -> use 1+log(TF) so a word repeated 50 times is not 50x more important
    # 1. Feature Extraction (TF-IDF)
    print("\n1. Fitting TF-IDF Vectorizer (max_features=10,000, n-gram=(1,2))...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        sublinear_tf=True
    )

    # fit_transform on TRAIN: learns the vocabulary + IDF values AND converts train text.
    # transform on val/test: converts using the SAME vocabulary learned from train.
    # (Fitting on test data would leak information, so we never do that.)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_val_tfidf = vectorizer.transform(X_val_raw)     # NOTE: validation features are created but never used to tune these models
    X_test_tfidf = vectorizer.transform(X_test_raw)

    # Save the vectorizer: at prediction time new text MUST be converted with the same vocabulary
    vec_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
    joblib.dump(vectorizer, vec_path)
    print(f"Saved TF-IDF Vectorizer to: {vec_path}")
    print(f"Vocabulary Size: {len(vectorizer.vocabulary_):,} features")

    # ---- STEP 2: DEFINE THE THREE CLASSIFIERS ----
    # Fixed hyper-parameters, no tuning/grid search was done.
    #  * Logistic Regression: weighted sum of features -> sigmoid -> probability. C=1 is the
    #    regularisation strength (smaller C = stronger regularisation). max_iter=1000 lets the
    #    optimiser run long enough to converge.
    #  * Multinomial Naive Bayes: applies Bayes' rule assuming words are independent; works on
    #    word counts/weights. alpha=0.1 is Laplace smoothing so unseen words do not give probability 0.
    #  * Linear SVM (LinearSVC): finds the maximum-margin separating line. It has no probability
    #    output, so CalibratedClassifierCV wraps it (Platt scaling, internally 5-fold CV, i.e.
    #    it trains 5 LinearSVCs and averages) to produce predict_proba().
    # 2. Define Classifier Models
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED),
            "filename": "logistic_regression.joblib"
        },
        "Naive Bayes": {
            "model": MultinomialNB(alpha=0.1),
            "filename": "naive_bayes.joblib"
        },
        "SVM (Calibrated Linear)": {
            "model": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=RANDOM_SEED)),
            "filename": "svm.joblib"
        }
    }

    results = {}

    print("\n2. Training and Evaluating Classifiers on Test Set...")
    for name, info in models.items():
        print(f"\n--- Training {name} ---")
        clf = info["model"]
        # THE ACTUAL TRAINING: learn from training features + training labels
        clf.fit(X_train_tfidf, y_train)

        # Predictions & Probabilities
        # predict(): hard 0/1 decision. predict_proba(): [:, 1] = probability of class 1 (scam).
        y_pred = clf.predict(X_test_tfidf)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test_tfidf)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_prob = clf.decision_function(X_test_tfidf)
        else:
            y_prob = y_pred

        # ---- METRICS (pos_label=1 means "scam" is the positive class) ----
        #  accuracy  = correct / all
        #  precision = of messages flagged scam, how many truly were    (TP / (TP+FP))
        #  recall    = of all real scams, how many we caught            (TP / (TP+FN))
        #  F1        = harmonic mean of precision and recall
        #  ROC-AUC   = chance a random scam gets a higher score than a random legit message
        # Metrics Calculation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_prob)
        # Confusion matrix layout: [[TN, FP], [FN, TP]]; ravel() flattens it to those 4 numbers.
        # FN (missed scams) is the costliest error for a security tool.
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Save model file
        save_path = os.path.join(MODELS_DIR, info["filename"])
        joblib.dump(clf, save_path)

        results[name] = {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1": float(f1),
            "ROC-AUC": float(auc),
            "Confusion Matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            },
            "Saved File": save_path
        }

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f} (Phishing Recall)")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print(f"Confusion Matrix [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
        print(f"Saved model to: {save_path}")

    # The JSON file is read later by compare_models.py and by the API (/models/info)
    # 3. Save JSON Results
    json_path = os.path.join(REPORTS_DIR, 'nlp_baseline_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw JSON metrics to: {json_path}")

    # ---- STEP 4: WRITE A HUMAN-READABLE MARKDOWN REPORT ----
    # (Comments are kept OUTSIDE the triple-quoted strings below, since text inside them
    #  becomes part of the report.)
    # 4. Generate Markdown Comparison Report
    report_path = os.path.join(REPORTS_DIR, 'nlp_baseline_report.md')

    md_content = f"""# ScamShield — Phase 4: NLP Baseline Models Report

This document presents the experimental results for traditional NLP baseline models evaluated on the held-out **ScamShield test dataset** ({len(y_test):,} samples).

## Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for name, m in results.items():
        md_content += f"| **{name}** | {m['Accuracy']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1']:.4f} | {m['ROC-AUC']:.4f} |\n"

    md_content += """
---

## Detailed Confusion Matrices & Error Analysis

"""
    for name, m in results.items():
        cm = m['Confusion Matrix']
        md_content += f"""### {name}
* **True Negatives (Legitimate correctly identified):** {cm['TN']:,}
* **False Positives (Legitimate flagged as Scam):** {cm['FP']:,}
* **False Negatives (Scams missed by model):** {cm['FN']:,}
* **True Positives (Scams correctly detected):** {cm['TP']:,}
* **Phishing Recall:** {m['Recall']*100:.2f}% (Higher is critical to avoid false negatives)
* **Precision:** {m['Precision']*100:.2f}%

"""

    md_content += """---

## Key Performance Insights & Recommendations

1. **Phishing Recall vs Precision:**
   In scam/phishing detection, **False Negatives (FN)** represent missed scams, which pose a direct security risk to users.
   
2. **Best Model Selection:**
"""
    # max(..., key=...) picks the model name whose metric is the largest
    # Find model with highest F1 and highest Recall
    best_f1_model = max(results.keys(), key=lambda k: results[k]['F1'])
    best_recall_model = max(results.keys(), key=lambda k: results[k]['Recall'])

    md_content += f"""   * **Highest F1-Score:** `{best_f1_model}` ({results[best_f1_model]['F1']:.4f})
   * **Highest Phishing Recall:** `{best_recall_model}` ({results[best_recall_model]['Recall']:.4f})

3. **Conclusion:**
   The traditional NLP baselines establish a very strong performance benchmark. These saved model artifacts will serve as our baseline for comparison against Deep Learning models (LSTM & DistilBERT) in Phase 5.
"""

    with open(report_path, 'w') as f:
        f.write(md_content)
    print(f"Saved NLP baseline report to: {report_path}")
    print("\nPhase 4 NLP Baseline training completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
