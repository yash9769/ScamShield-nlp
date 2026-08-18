# Scam & Phishing Text Detection using NLP

**A comparative study of classical ML and deep learning approaches to
scam/phishing text classification.**

> This is a standalone academic project for an NLP course. It is
> independent of the ScamShield mobile application — no app, backend
> integration, or product-specific code lives here, only the NLP
> pipeline: data engineering, exploratory analysis, model training, and
> comparative evaluation.

---

## 1. Problem Statement

Scam and phishing messages (SMS, email) are a widespread social
engineering threat. This project builds and compares multiple NLP text
classifiers that label a message as **scam/phishing (1)** or
**legitimate (0)**, evaluating each on accuracy, recall (catching
actual scams matters more than overall accuracy — a missed scam is
worse than a false alarm), inference latency, and model size.

## 2. Datasets

Messages were pooled from multiple public sources and unified into a
single labelled dataset:

| Source | Content type |
|---|---|
| MeAJOR Phishing Email Dataset | Phishing emails |
| Enron Email Dataset | Legitimate emails (negative class) |
| Enron-Spam | Spam/phishing emails |
| SMS Spam Collection | SMS spam and legitimate SMS |

Labels: `0` = legitimate, `1` = scam/phishing.

Cleaning steps applied (see `src/data/clean.py`): deduplication,
HTML/header stripping from raw emails, normalization of whitespace and
encoding artifacts, and removal of empty/near-empty records. See
`reports/eda_report.md` for full exploratory data analysis (class
balance, message length distributions, vocabulary statistics).

`dataset/` in this repo contains a **200-message held-out evaluation
set** (100 scam, 100 safe — see `dataset/README.md`) used for sanity
end-to-end testing, not the full training corpus. **The full raw and
processed training CSVs (used by `src/data/ingest.py` /
`src/data/clean.py` / `src/data/split.py`) need to be copied back in
from your local machine** — see "Restoring local files" below.

## 3. Methodology

| Phase | Description | Script |
|---|---|---|
| 1 | Data ingestion from raw sources | `src/data/ingest.py` |
| 2 | Cleaning & deduplication | `src/data/clean.py` |
| 3 | Train / validation / test split | `src/data/split.py` |
| 3 | Exploratory Data Analysis | `src/eda/eda.py` |
| 4 | Classical NLP baselines: TF-IDF + Naive Bayes / Logistic Regression / SVM | `src/models/train_nlp_baselines.py` |
| 5 | Deep learning: BiLSTM (PyTorch) | `src/models/train_lstm.py` |
| 5 | Deep learning: DistilBERT (fine-tuned) | `src/models/train_distilbert.py` |
| 6 | Comparative evaluation across all 5 models | `src/models/compare_models.py` |

## 4. Results (held-out test set, 7,899 samples)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Missed Scams (FN) | Latency | Size |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Naive Bayes | 0.9259 | 0.9088 | 0.8899 | 0.8993 | 0.9754 | 323 | 0.001 ms | 0.31 MB |
| Logistic Regression | 0.9599 | 0.9625 | 0.9281 | 0.9450 | 0.9921 | 211 | 0.001 ms | 0.08 MB |
| **Linear SVM** | **0.9691** | **0.9664** | 0.9499 | **0.9581** | **0.9944** | 147 | 0.006 ms | 0.38 MB |
| BiLSTM | 0.9571 | 0.9400 | 0.9448 | 0.9424 | 0.9899 | 162 | 1.014 ms | 12.35 MB |
| **DistilBERT** | 0.9606 | 0.9302 | **0.9666** | 0.9480 | 0.9933 | **98** | 0.401 ms | 256.1 MB |

**Key findings:**
- **Linear SVM** achieves the best F1-score (0.9581) and ROC-AUC
  (0.9944) with sub-millisecond inference and under 1MB total size —
  the strongest accuracy/efficiency trade-off of the five.
- **DistilBERT** achieves the highest recall (96.66%), missing only 98
  of 7,899 scam messages — the best choice when minimizing missed
  scams matters more than raw efficiency.
- Classical linear models (SVM, Logistic Regression) matched or beat
  the deep learning models here — a reflection of how lexically
  distinctive scam/phishing language tends to be (urgency phrases,
  links, financial terms), which TF-IDF features capture well without
  needing learned embeddings.

Full per-phase reports with additional detail: `reports/eda_report.md`,
`reports/nlp_baseline_report.md`, `reports/lstm_results.json`,
`reports/distilbert_results.json`, `reports/model_comparison_report.md`.

## 5. Project Structure

```
├── dataset/                  # small held-out evaluation set (200 msgs)
├── models/                   # trained model artifacts
│   ├── naive_bayes.joblib
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── lstm_tokenizer.json
│   ├── bilstm_model.pt
│   └── distilbert_scamshield/
├── reports/                  # EDA + evaluation reports (markdown + json)
├── src/
│   ├── config.py
│   ├── data/                 # ingest, clean, split, stats
│   ├── eda/                  # exploratory data analysis
│   └── models/               # training scripts + predictor.py (inference)
├── app.py                    # optional FastAPI demo for live predictions
├── frontend/
│   └── index.html            # visual web demo (talks to app.py's API)
├── demo_predict.py           # simple CLI demo — no server needed
└── requirements.txt
```

## 6. Restoring local files (optional — only for full re-training)

Everything needed to run the demo and view results is already included
in this package — trained weights for all 5 models (including
`models/bilstm_model.pt`) and all report figures (`reports/figures/`).

The only thing **not** included is the full raw/processed training
corpus (`dataset/raw/`, `dataset/processed/` — the MeAJOR, Enron,
Enron-Spam, and SMS Spam Collection CSVs). You only need these if you
want to re-run training end-to-end from scratch; copy them back in from
your local machine if so. They're not needed to run the demo or write
the report.

## 7. Running the demo

```bash
pip install -r requirements.txt

# Option A: quick CLI demo, no server
python demo_predict.py

# Option B: FastAPI demo with a /docs page for live testing
uvicorn app:app --reload
# then open http://127.0.0.1:8000/docs

# Option C: full visual dashboard (recommended for presenting to a professor)
# 1. In one terminal:
uvicorn app:app --reload
# 2. In a second terminal, from the project root:
python -m http.server 5500 --directory frontend
# 3. Open http://127.0.0.1:5500 in a browser
```

**The backend (`uvicorn app:app`) must be running before opening the
dashboard** — it's a static page that calls the FastAPI API for every
prediction.

### Dashboard features

**Scanner tab**
- Paste any message, pick a model (SVM / BiLSTM / DistilBERT), get a
  verdict with a segmented probability meter and risk level
- **Explainable AI for all 3 models** — not just SVM. SVM gets exact
  linear decomposition; BiLSTM and DistilBERT get gradient × input
  saliency (a real, standard attribution method — Simonyan et al. 2013 —
  clearly labeled "approximate" in the UI since these are non-linear
  models, unlike SVM's "exact" label). Every prediction is explained,
  whichever model you pick.
- **Compare All 3** button: runs the same message through SVM, BiLSTM,
  and DistilBERT at once and flags when they disagree. Model
  disagreement on ambiguous/borderline text (e.g. a legitimate-sounding
  bill reminder that also matches phishing phrasing) is a real,
  expected phenomenon worth discussing — not a bug — and this makes it
  visible at a glance instead of requiring three separate scans.
- **Decision threshold slider**: drag to see how the verdict changes as
  you move the classification cutoff away from 0.5 — a hands-on way to
  show the precision/recall trade-off without needing new backend calls.
- **Batch mode**: paste multiple messages (one per line) and classify
  all of them in one request via `/predict/batch`.
- **Session history**: every scan you run is logged in a table for the
  rest of the session — handy if your teacher asks to compare a few
  messages side by side.

**Model Comparison tab**
- Bar chart comparing all 5 trained models on any of Accuracy /
  Precision / Recall / F1 / ROC-AUC (click a metric to switch)
- Full metrics table with the best value per column highlighted
- Click any model row to see its confusion matrix (TN/FP/FN/TP) from
  the actual test-set evaluation

**Dataset & EDA tab**
- Dataset summary stats and the actual EDA charts generated during
  training (class distribution, message length distribution, samples
  by type/source, and the model comparison chart)

### New backend endpoints (added for the dashboard)

- `POST /explain` — prediction + per-token attribution for any of the 3
  backends (exact for SVM, gradient × input saliency for BiLSTM/DistilBERT)
- `GET /models/info` — metrics and confusion matrices for all 5 models,
  read from `reports/*.json`
- `POST /predict/batch` — classify a list of texts in one call

All of the above were tested with real HTTP requests (and the frontend
was tested end-to-end in a headless browser, including the SVM/BiLSTM
vs. DistilBERT disagreement scenario) while building this, not just
written and assumed to work.

### A note on model disagreement

You may see the 3 models disagree on ambiguous messages — e.g. a
routine-sounding bill reminder that also happens to match common
phishing phrasing (urgency + payment + link). This is expected, not a
bug: different architectures draw the decision boundary differently on
borderline cases. It's genuinely good material to discuss in your
report/demo — the "Compare All 3" button in the Scanner tab surfaces
this directly, and the per-model explainability panel shows *why* each
model landed where it did.

## 8. References

- MeAJOR Phishing Email Dataset
- Enron Email Dataset
- Enron-Spam Dataset
- SMS Spam Collection Dataset (UCI)
- Sanh et al., *DistilBERT, a distilled version of BERT* (2019)
- Scikit-learn: Pedregosa et al., *JMLR* (2011)
