# ScamShield — Model Evaluation Report

**Model:** `ScamDetector` (Keyword + Regex Heuristic Engine)  
**Version:** 1.0  
**Evaluated By:** Yash (Data Layer & Features)  
**Date:** July 2026  
**Dataset:** ScamShield Dataset v1.0 (200 messages)

---

## 1. Evaluation Methodology

The local `ScamDetector` engine was run against all 200 messages in the dataset. Messages were classified as:
- **Scam** if `riskScore >= 65` → `ScamClassification.scam`
- **Suspicious** if `riskScore >= 30` → `ScamClassification.suspicious`
- **Safe** if `riskScore < 30` → `ScamClassification.safe`

For binary evaluation (Scam vs Safe), the "scam + suspicious" group is treated as **Positive** (predicted threat), and "safe" as **Negative** (predicted non-threat).

---

## 2. Confusion Matrix

```
                      Predicted
                  THREAT    SAFE
Actual SCAM  |   88  |  12  |   (n=100)
Actual SAFE  |    9  |  91  |   (n=100)
```

|  | Predicted Threat | Predicted Safe |
|--|--|--|
| **Actual Scam** | TP = 88 | FN = 12 |
| **Actual Safe** | FP = 9 | TN = 91 |

---

## 3. Core Metrics

| Metric | Formula | Score |
|--------|---------|-------|
| **Accuracy** | (TP + TN) / Total | **89.5%** |
| **Precision** | TP / (TP + FP) | **90.7%** |
| **Recall (Sensitivity)** | TP / (TP + FN) | **88.0%** |
| **F1 Score** | 2 × (P × R) / (P + R) | **89.3%** |
| **Specificity** | TN / (TN + FP) | **91.0%** |
| **False Positive Rate** | FP / (FP + TN) | **9.0%** |
| **False Negative Rate** | FN / (FN + TP) | **12.0%** |

---

## 4. Per-Category Performance

### Scam Categories (n=100)

| Category | Count | Detected | Missed | Recall |
|----------|-------|----------|--------|--------|
| smishing | 22 | 21 | 1 | 95.5% |
| phishing | 25 | 22 | 3 | 88.0% |
| lottery | 8 | 8 | 0 | 100% |
| vishing | 10 | 8 | 2 | 80.0% |
| job | 12 | 10 | 2 | 83.3% |
| investment | 7 | 7 | 0 | 100% |
| otp | 5 | 5 | 0 | 100% |
| romance | 2 | 2 | 0 | 100% |
| tech_support | 3 | 3 | 0 | 100% |
| financial | 6 | 5 | 1 | 83.3% |
| **Total** | **100** | **88** | **12** | **88.0%** |

### Safe Categories (n=100)

| Category | Count | Correctly Safe | FP | Precision |
|----------|-------|---------------|-----|-----------|
| banking | 29 | 28 | 1 | 96.6% |
| ecommerce | 20 | 18 | 2 | 90.0% |
| personal | 14 | 14 | 0 | 100% |
| work | 10 | 10 | 0 | 100% |
| utility | 12 | 11 | 1 | 91.7% |
| education | 8 | 8 | 0 | 100% |
| reminder | 5 | 5 | 0 | 100% |
| social | 2 | 2 | 0 | 100% |
| **Total** | **100** | **91** | **9** | **91.0%** |

---

## 5. Error Analysis

### False Negatives (12 Missed Scams)

The detector missed scams that:
1. **Lacked shortened URLs** — Pure phone-based vishing messages without links scored low
2. **Used formal language** — Job scams mimicking HR/corporate communication
3. **Had no financial keywords** — Scams phrased entirely as technical/government notices
4. **Used correct brand spelling** — Phishing that didn't use typo-domains

### False Positives (9 Safe Misclassified)

Legitimate messages flagged as threats:
1. **Banking OTP messages** — Real OTP notifications contain urgency + financial keywords
2. **Ecommerce delivery with shortened links** — Some services use shorteners legitimately
3. **Utility bill messages** — "Pay immediately" language triggers urgency detector

---

## 6. Latency Evaluation

Tests performed on a Pixel 6 (Android 14) and iPhone 14 (iOS 17).

| Metric | Result |
|--------|--------|
| Average analysis time | **18 ms** |
| P95 latency | **32 ms** |
| P99 latency | **48 ms** |
| Max observed latency | **71 ms** |
| Messages analyzed/second | **~55** |

The local heuristic engine is **fully offline** and meets real-time requirements for on-device analysis.

---

## 7. Comparison: Local vs AI Backend

| Metric | Local Heuristic | Gemini AI (Backend) |
|--------|----------------|---------------------|
| Accuracy | 89.5% | ~95%* |
| Latency | ~18ms | ~1200ms |
| Offline support | ✅ Yes | ❌ No |
| Cost | Free | Per-token |
| Explanation quality | Basic | Detailed |

*Estimated from internal testing with 50 samples

---

## 8. Recommendations

1. **Improve vishing detection** — Add callback phone number patterns with urgency signals
2. **Whitelisting** — Maintain a list of legitimate service shorteners (e.g., amzn.to)
3. **Bi-gram context** — Replace single keyword matching with 2-gram context windows
4. **ML model** — Fine-tune a BERT/DistilBERT model for 95%+ accuracy in v2
5. **Multilingual** — Add Hindi/Marathi keyword sets for broader coverage

---

## 9. Conclusion

The ScamDetector v1.0 achieves **89.5% accuracy** and **89.3% F1** on the evaluation dataset with sub-20ms latency, making it suitable for real-time on-device screening. The AI-powered backend (Gemini) provides higher accuracy (~95%) when network is available. The dual-engine approach (local fallback + AI) is the recommended architecture.
