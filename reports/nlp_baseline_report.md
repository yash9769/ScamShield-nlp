# ScamShield — Phase 4: NLP Baseline Models Report

This document presents the experimental results for traditional NLP baseline models evaluated on the held-out **ScamShield test dataset** (7,899 samples).

## Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9599 | 0.9625 | 0.9281 | 0.9450 | 0.9921 |
| **Naive Bayes** | 0.9259 | 0.9088 | 0.8899 | 0.8993 | 0.9754 |
| **SVM (Calibrated Linear)** | 0.9691 | 0.9664 | 0.9499 | 0.9581 | 0.9944 |

---

## Detailed Confusion Matrices & Error Analysis

### Logistic Regression
* **True Negatives (Legitimate correctly identified):** 4,858
* **False Positives (Legitimate flagged as Scam):** 106
* **False Negatives (Scams missed by model):** 211
* **True Positives (Scams correctly detected):** 2,724
* **Phishing Recall:** 92.81% (Higher is critical to avoid false negatives)
* **Precision:** 96.25%

### Naive Bayes
* **True Negatives (Legitimate correctly identified):** 4,702
* **False Positives (Legitimate flagged as Scam):** 262
* **False Negatives (Scams missed by model):** 323
* **True Positives (Scams correctly detected):** 2,612
* **Phishing Recall:** 88.99% (Higher is critical to avoid false negatives)
* **Precision:** 90.88%

### SVM (Calibrated Linear)
* **True Negatives (Legitimate correctly identified):** 4,867
* **False Positives (Legitimate flagged as Scam):** 97
* **False Negatives (Scams missed by model):** 147
* **True Positives (Scams correctly detected):** 2,788
* **Phishing Recall:** 94.99% (Higher is critical to avoid false negatives)
* **Precision:** 96.64%

---

## Key Performance Insights & Recommendations

1. **Phishing Recall vs Precision:**
   In scam/phishing detection, **False Negatives (FN)** represent missed scams, which pose a direct security risk to users.
   
2. **Best Model Selection:**
   * **Highest F1-Score:** `SVM (Calibrated Linear)` (0.9581)
   * **Highest Phishing Recall:** `SVM (Calibrated Linear)` (0.9499)

3. **Conclusion:**
   The traditional NLP baselines establish a very strong performance benchmark. These saved model artifacts will serve as our baseline for comparison against Deep Learning models (LSTM & DistilBERT) in Phase 5.
