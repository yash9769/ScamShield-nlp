import os
import sys
import json
import time
import joblib
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PROCESSED_DIR, BASE_DIR, RANDOM_SEED
from src.models.bilstm import SimpleTokenizer, TextDataset, ScamBiLSTM

MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')


class DistilBertTestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True, max_length=self.max_len,
            padding='max_length', return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def get_file_size_mb(path):
    if os.path.isfile(path):
        return round(os.path.getsize(path) / 1024 / 1024, 2)
    elif os.path.isdir(path):
        total = sum(os.path.getsize(os.path.join(dp, f))
                    for dp, dn, fn in os.walk(path) for f in fn)
        return round(total / 1024 / 1024, 2)
    return 0.0


def compare_models():
    print("Starting Phase 6: Comprehensive Model Comparison...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Load test set
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'test.csv'))
    X_test_raw = test_df['text'].fillna('').tolist()
    y_test = test_df['label'].values

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))

    all_results = {}

    # ─── 1. Load pre-computed NLP results ───────────────────────────────────
    nlp_path = os.path.join(REPORTS_DIR, 'nlp_baseline_results.json')
    with open(nlp_path) as f:
        nlp_results = json.load(f)

    vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
    X_test_tfidf = vectorizer.transform(X_test_raw)

    for model_name, model_file in [
        ("Naive Bayes",        "naive_bayes.joblib"),
        ("Logistic Regression","logistic_regression.joblib"),
        ("Linear SVM",         "svm.joblib"),
    ]:
        clf = joblib.load(os.path.join(MODELS_DIR, model_file))

        # Latency: average over 200 samples
        sample = X_test_tfidf[:200]
        start = time.perf_counter()
        clf.predict(sample)
        elapsed = (time.perf_counter() - start) / 200 * 1000  # ms per sample

        m = nlp_results.get(model_name) or nlp_results.get("SVM (Calibrated Linear)")
        # Use correct key for SVM
        if model_name == "Linear SVM":
            m = nlp_results.get("SVM (Calibrated Linear)", {})

        all_results[model_name] = {
            "Accuracy":  m.get("Accuracy", 0),
            "Precision": m.get("Precision", 0),
            "Recall":    m.get("Recall", 0),
            "F1":        m.get("F1", 0),
            "ROC-AUC":   m.get("ROC-AUC", 0),
            "Latency_ms": round(elapsed, 3),
            "Model_Size_MB": get_file_size_mb(os.path.join(MODELS_DIR, model_file)),
            "FN": m.get("Confusion Matrix", {}).get("FN", "N/A"),
        }
        print(f"[{model_name}] Latency: {elapsed:.3f} ms/sample | Size: {all_results[model_name]['Model_Size_MB']} MB")

    # ─── 2. BiLSTM ────────────────────────────────────────────────────────────
    print("\nEvaluating BiLSTM on test set for latency measurement...")
    lstm_tok = SimpleTokenizer.load(os.path.join(MODELS_DIR, 'lstm_tokenizer.json'))
    lstm_ds = TextDataset(X_test_raw[:200], y_test[:200], lstm_tok)
    lstm_loader = DataLoader(lstm_ds, batch_size=200, shuffle=False)

    lstm_model = ScamBiLSTM(vocab_size=len(lstm_tok.word2idx)).to(device)
    lstm_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'bilstm_model.pt'), map_location=device))
    lstm_model.eval()

    X_b, _ = next(iter(lstm_loader))
    X_b = X_b.to(device)
    start = time.perf_counter()
    with torch.no_grad():
        lstm_model(X_b)
    elapsed = (time.perf_counter() - start) / 200 * 1000

    with open(os.path.join(REPORTS_DIR, 'lstm_results.json')) as f:
        lstm_data = json.load(f)["BiLSTM"]

    all_results["BiLSTM"] = {
        "Accuracy":  lstm_data["Accuracy"],
        "Precision": lstm_data["Precision"],
        "Recall":    lstm_data["Recall"],
        "F1":        lstm_data["F1"],
        "ROC-AUC":   lstm_data["ROC-AUC"],
        "Latency_ms": round(elapsed, 3),
        "Model_Size_MB": get_file_size_mb(os.path.join(MODELS_DIR, 'bilstm_model.pt')),
        "FN": lstm_data["Confusion Matrix"]["FN"],
    }
    print(f"[BiLSTM] Latency: {elapsed:.3f} ms/sample | Size: {all_results['BiLSTM']['Model_Size_MB']} MB")

    # ─── 3. DistilBERT ────────────────────────────────────────────────────────
    print("\nEvaluating DistilBERT latency on 200 test samples...")
    db_tok = DistilBertTokenizer.from_pretrained(os.path.join(MODELS_DIR, 'distilbert_scamshield'))
    db_model = DistilBertForSequenceClassification.from_pretrained(
        os.path.join(MODELS_DIR, 'distilbert_scamshield')).to(device)
    db_model.eval()

    db_ds = DistilBertTestDataset(X_test_raw[:200], y_test[:200], db_tok)
    db_loader = DataLoader(db_ds, batch_size=200, shuffle=False)
    batch = next(iter(db_loader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    start = time.perf_counter()
    with torch.no_grad():
        db_model(input_ids=input_ids, attention_mask=attention_mask)
    elapsed = (time.perf_counter() - start) / 200 * 1000

    with open(os.path.join(REPORTS_DIR, 'distilbert_results.json')) as f:
        db_data = json.load(f)["DistilBERT"]

    all_results["DistilBERT"] = {
        "Accuracy":  db_data["Accuracy"],
        "Precision": db_data["Precision"],
        "Recall":    db_data["Recall"],
        "F1":        db_data["F1"],
        "ROC-AUC":   db_data["ROC-AUC"],
        "Latency_ms": round(elapsed, 3),
        "Model_Size_MB": get_file_size_mb(os.path.join(MODELS_DIR, 'distilbert_scamshield')),
        "FN": db_data["Confusion Matrix"]["FN"],
    }
    print(f"[DistilBERT] Latency: {elapsed:.3f} ms/sample | Size: {all_results['DistilBERT']['Model_Size_MB']} MB")

    # ─── 4. Comparison Bar Chart ───────────────────────────────────────────────
    model_names = list(all_results.keys())
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
    fig.suptitle("ScamShield: Full Model Comparison (Test Set)", fontsize=16, weight='bold')

    for ax, metric, color in zip(axes, metrics, colors):
        vals = [all_results[m][metric] for m in model_names]
        bars = ax.bar(model_names, vals, color=color, alpha=0.85, edgecolor='white')
        ax.set_title(metric, fontsize=13, weight='bold')
        ax.set_ylim(0.85, 1.0)
        ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel("Score")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha='center', va='bottom', fontsize=8, weight='bold')

    plt.tight_layout()
    chart_path = os.path.join(FIGURES_DIR, 'model_comparison.png')
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nSaved comparison chart to: {chart_path}")

    # ─── 5. Markdown Report ───────────────────────────────────────────────────
    best_f1  = max(all_results, key=lambda k: all_results[k]['F1'])
    best_rec = max(all_results, key=lambda k: all_results[k]['Recall'])
    best_auc = max(all_results, key=lambda k: all_results[k]['ROC-AUC'])

    md = f"""# ScamShield — Phase 6: Comprehensive Model Comparison Report

Empirical comparison of all 5 models evaluated on the **held-out test set** (7,899 samples).

![Model Comparison Chart](figures/model_comparison.png)

## Performance Comparison Table

| Model | Accuracy | Precision | Recall (Phishing) | F1-Score | ROC-AUC | Missed Scams (FN) | Latency (ms/sample) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for name, m in all_results.items():
        md += (f"| **{name}** | {m['Accuracy']:.4f} | {m['Precision']:.4f} | "
               f"{m['Recall']:.4f} | {m['F1']:.4f} | {m['ROC-AUC']:.4f} | "
               f"{m['FN']} | {m['Latency_ms']} | {m['Model_Size_MB']} |\n")

    md += f"""
---

## Key Findings

| Criterion | Winner |
|---|---|
| **Highest F1-Score** | `{best_f1}` ({all_results[best_f1]['F1']:.4f}) |
| **Highest Phishing Recall** | `{best_rec}` ({all_results[best_rec]['Recall']:.4f}) |
| **Highest ROC-AUC** | `{best_auc}` ({all_results[best_auc]['ROC-AUC']:.4f}) |
| **Fewest Missed Scams (FN)** | `{min(all_results, key=lambda k: all_results[k]['FN'])}` |
| **Fastest Inference** | `{min(all_results, key=lambda k: all_results[k]['Latency_ms'])}` |
| **Smallest Model Size** | `{min(all_results, key=lambda k: all_results[k]['Model_Size_MB'])}` |

---

## Model Selection Recommendation

For **ScamShield production deployment**, we recommend the **Linear SVM** as the text detection backbone:

- Exceptional recall (94.99%) with only **147 missed scams** on 7,899 test samples.
- Ultra-fast inference (~sub-millisecond latency), perfect for real-time API calls.
- Compact model size (<1 MB) for easy deployment and packaging.

**DistilBERT** achieves the **highest phishing recall (96.66%)** — missing only **98 scams** — and is
recommended as the upgrade path if computational resources allow, or for a future high-accuracy
"deep scan" mode within ScamShield.

The **BiLSTM** serves as a strong interpretable deep learning intermediate, confirming that the
task is learnable with recurrent architectures.
"""

    report_path = os.path.join(REPORTS_DIR, 'model_comparison_report.md')
    with open(report_path, 'w') as f:
        f.write(md)
    print(f"Saved model comparison report to: {report_path}")

    # Save JSON summary
    json_out = os.path.join(REPORTS_DIR, 'model_comparison_results.json')
    with open(json_out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved JSON summary to: {json_out}")
    print("\nPhase 6 Model Comparison completed successfully!")


if __name__ == "__main__":
    compare_models()
