import os
import json
import joblib
import pandas as pd
import numpy as np
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
    
    # Handle any potential NaNs in text
    X_train_raw = train_df['text'].fillna('')
    y_train = train_df['label']
    
    X_val_raw = val_df['text'].fillna('')
    y_val = val_df['label']
    
    X_test_raw = test_df['text'].fillna('')
    y_test = test_df['label']
    
    print(f"Dataset sizes -> Train: {len(X_train_raw):,}, Val: {len(X_val_raw):,}, Test: {len(X_test_raw):,}")
    
    # 1. Feature Extraction (TF-IDF)
    print("\n1. Fitting TF-IDF Vectorizer (max_features=10,000, n-gram=(1,2))...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_val_tfidf = vectorizer.transform(X_val_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)
    
    vec_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
    joblib.dump(vectorizer, vec_path)
    print(f"Saved TF-IDF Vectorizer to: {vec_path}")
    print(f"Vocabulary Size: {len(vectorizer.vocabulary_):,} features")
    
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
        clf.fit(X_train_tfidf, y_train)
        
        # Predictions & Probabilities
        y_pred = clf.predict(X_test_tfidf)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test_tfidf)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_prob = clf.decision_function(X_test_tfidf)
        else:
            y_prob = y_pred
            
        # Metrics Calculation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1)
        rec = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        auc = roc_auc_score(y_test, y_prob)
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

    # 3. Save JSON Results
    json_path = os.path.join(REPORTS_DIR, 'nlp_baseline_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw JSON metrics to: {json_path}")

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
