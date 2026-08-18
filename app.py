import json
import os
import sys
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import BASE_DIR
from src.models.predictor import ScamPredictor

app = FastAPI(
    title="ScamShield API",
    description="Scam & phishing text detection using SVM (fast) and DistilBERT (high-accuracy) models.",
    version="1.0.0",
)

# Local demo only — allows the standalone frontend/index.html page to call
# this API from the browser regardless of which local port serves it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = ScamPredictor(backend='svm')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

# ── Explainability: load the SVM's underlying linear estimator directly ────
# clf is a CalibratedClassifierCV wrapping a LinearSVC. The calibration
# layer (Platt scaling) is what produces predict_proba(); the wrapped
# LinearSVC's .coef_ is the actual per-feature weight vector, so
# coef_[i] * tfidf_value[i] is the EXACT contribution of feature i to the
# model's raw decision score — not an approximation like LIME/SHAP, since
# the model itself is linear.
_svm_vectorizer = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib'))
_svm_clf = joblib.load(os.path.join(MODELS_DIR, 'svm.joblib'))
_svm_linear = _svm_clf.calibrated_classifiers_[0].estimator
_svm_coef = _svm_linear.coef_[0]
_svm_feature_names = np.array(_svm_vectorizer.get_feature_names_out())


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text message to classify")
    backend: Optional[str] = Field("svm", description="Model backend: 'svm', 'bilstm' or 'distilbert'")


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of texts to classify")
    backend: Optional[str] = Field("svm", description="Model backend: 'svm', 'bilstm' or 'distilbert'")


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text message to explain")
    backend: Optional[str] = Field("svm", description="Model backend: 'svm', 'bilstm' or 'distilbert'")
    top_k: Optional[int] = Field(8, ge=1, le=30, description="Number of top contributing tokens to return per direction")


BACKENDS = {"svm", "bilstm", "distilbert"}


@app.get("/")
def root():
    return {"service": "ScamShield API", "status": "running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(req: PredictRequest):
    backend = req.backend.lower()
    if backend not in BACKENDS:
        raise HTTPException(status_code=400, detail=f"Unknown backend '{req.backend}'. Choose from {sorted(BACKENDS)}.")
    try:
        return {"text": req.text, "backend": backend, **predictor.predict(req.text, backend=backend)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    backend = req.backend.lower()
    if backend not in BACKENDS:
        raise HTTPException(status_code=400, detail=f"Unknown backend '{req.backend}'. Choose from {sorted(BACKENDS)}.")
    if len(req.texts) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limited to 1000 texts per request.")
    try:
        results = predictor.predict(req.texts, backend=backend)
        return {"backend": backend, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/explain")
def explain(req: ExplainRequest):
    """
    Explainability for all 3 backends, using the method appropriate to
    each model's architecture:
      - svm:        EXACT — coefficient x TF-IDF weight per token, since
                     the model itself is linear (not an approximation).
      - bilstm / distilbert: gradient x input saliency — backprop the
                     predicted-class score to the input embeddings and
                     take grad . embedding per token. A standard, real
                     attribution method (Simonyan et al. 2013); not exact
                     like the SVM case since these are non-linear models,
                     but a genuine explanation, not a cosmetic stand-in.
    """
    backend = req.backend.lower()
    if backend not in BACKENDS:
        raise HTTPException(status_code=400, detail=f"Unknown backend '{req.backend}'. Choose from {sorted(BACKENDS)}.")

    try:
        if backend == "svm":
            X = _svm_vectorizer.transform([req.text])
            nz_idx = X.nonzero()[1]
            contributions = [
                {"token": str(_svm_feature_names[i]), "weight": round(float(X[0, i] * _svm_coef[i]), 5)}
                for i in nz_idx
            ]
            contributions.sort(key=lambda c: c["weight"], reverse=True)
            top_scam = [c for c in contributions if c["weight"] > 0][: req.top_k]
            top_safe = [c for c in contributions if c["weight"] < 0][-req.top_k:][::-1]
            method = "exact"
            note = ("Contributions are exact for the underlying LinearSVC decision "
                    "function (coefficient x TF-IDF weight per token) — not an "
                    "approximation, since the model is linear.")

        elif backend == "bilstm":
            (top_scam, top_safe), _prob = predictor.explain_bilstm(req.text, top_k=req.top_k)
            method = "gradient_x_input"
            note = ("Gradient x input saliency: backpropagates the predicted-class "
                    "score to the word-embedding layer and multiplies each token's "
                    "embedding by its gradient. A real attribution method (Simonyan "
                    "et al. 2013), but an approximation — not exact like the SVM "
                    "case, since BiLSTM is a non-linear model.")

        else:  # distilbert
            (top_scam, top_safe), _prob = predictor.explain_distilbert(req.text, top_k=req.top_k)
            method = "gradient_x_input"
            note = ("Gradient x input saliency on DistilBERT's token embeddings — "
                    "same method as BiLSTM, adapted for subword tokens. An "
                    "approximation, not exact, since DistilBERT is a non-linear "
                    "transformer model.")

        pred = predictor.predict(req.text, backend=backend)
        return {
            "text": req.text,
            "backend": backend,
            "method": method,
            **pred,
            "top_scam_tokens": top_scam,
            "top_safe_tokens": top_safe,
            "note": note,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain failed: {e}")


@app.get("/models/info")
def models_info():
    """Static comparison metrics for all 5 trained models, read from the
    saved evaluation reports (reports/model_comparison_results.json plus
    per-model confusion matrices)."""
    try:
        with open(os.path.join(REPORTS_DIR, "model_comparison_results.json")) as f:
            comparison = json.load(f)

        confusion = {}
        for fname, key_map in [
            ("nlp_baseline_results.json", {"Naive Bayes": "Naive Bayes",
                                            "Logistic Regression": "Logistic Regression",
                                            "SVM (Calibrated Linear)": "Linear SVM"}),
            ("lstm_results.json", {"BiLSTM": "BiLSTM"}),
            ("distilbert_results.json", {"DistilBERT": "DistilBERT"}),
        ]:
            path = os.path.join(REPORTS_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                for src_key, dst_key in key_map.items():
                    if src_key in data:
                        confusion[dst_key] = data[src_key].get("Confusion Matrix")

        models = []
        descriptions = {
            "Naive Bayes": "TF-IDF + Multinomial Naive Bayes — probabilistic bag-of-words baseline.",
            "Logistic Regression": "TF-IDF + Logistic Regression — linear baseline, smallest model.",
            "Linear SVM": "TF-IDF + Calibrated Linear SVM — best F1/ROC-AUC, sub-millisecond inference.",
            "BiLSTM": "PyTorch Bidirectional LSTM over learned word embeddings.",
            "DistilBERT": "Fine-tuned transformer — highest recall, catches the most scams.",
        }
        for name, metrics in comparison.items():
            models.append({
                "name": name,
                "description": descriptions.get(name, ""),
                "accuracy": metrics["Accuracy"],
                "precision": metrics["Precision"],
                "recall": metrics["Recall"],
                "f1": metrics["F1"],
                "roc_auc": metrics["ROC-AUC"],
                "latency_ms": metrics["Latency_ms"],
                "size_mb": metrics["Model_Size_MB"],
                "false_negatives": metrics["FN"],
                "confusion_matrix": confusion.get(name),
            })
        return {"models": models, "test_set_size": 7899}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load model info: {e}")
