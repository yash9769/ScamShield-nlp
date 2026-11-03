# ScamShield — Code Walkthrough (viva prep)

Run order of the pipeline:
`ingest.py → clean.py → split.py → (stats.py, eda.py) → train_nlp_baselines.py → train_lstm.py → train_distilbert.py → compare_models.py → app.py / demo_predict.py`

All scripts do `sys.path.append(<project root>)` at the top so `from src.config import ...` works when run as a script.

---

## 1. `src/config.py` — shared settings
No functions, only constants.
- `BASE_DIR`: project root (two folders up from this file).
- `RAW_DIR`, `PROCESSED_DIR`, `RAW_*_PATH`, `PROCESSED_DATASET_PATH`: where raw CSVs and the cleaned CSV live.
- `RANDOM_SEED = 42`: used everywhere (sampling, splits, model init) so results are reproducible.
- `SUBSET_LIMIT_EMAIL = 5000`: **at most 5,000 emails per source** are kept (to keep training fast). `SUBSET_LIMIT_SMS = None`: keep all SMS.

> Viva point: the 52,660-message corpus is a *sampled* subset, not the full public datasets.

---

## 2. `src/data/ingest.py` — download raw data
- `ingest_datasets()`: uses Hugging Face `datasets.load_dataset` to download three datasets and save each as a CSV in `dataset/raw/`:
  1. `puyang2025/seven-phishing-email-datasets` (train + test concatenated)
  2. `simlab-vs/meajor_cleaned_preprocessed`
  3. `ucirvine/sms_spam`
  Each download is in a try/except; on failure it prints the error and exits.

## 3. `src/data/clean.py` — standardize + dedupe
- `clean_html(text)`: returns `""` if not a string; removes anything like `<...>` (HTML tags) with a regex, collapses whitespace.
- `format_text(subject, body)`: builds one string: `[SUBJECT] ... [BODY] ...` (handles missing subject or body).
- `process_seven_phishing()`: reads the raw CSV, optionally samples ≤5,000 rows **per `dataset_name`** (seeded), then for each row combines subject+body, cleans it, skips empty/unlabelled rows, and outputs a uniform table: `id, text, subject, label, source, type='email'`.
- `process_meajor()`: same idea for MeAJOR; groups by its `source` column, source name becomes `meajor_<source>`.
- `process_sms_spam()`: SMS has only a body (column `sms`); `type='sms'`, source `SMS_Spam_Collection`.
- `main()`: concatenates the three, **drops exact duplicate texts** (`duplicated(subset=['text'])`), saves `unified_text_dataset.csv`.

> Viva point: dedup is exact-string only; near-duplicates (same template, small edits) are not removed.

## 4. `src/data/split.py` — train/val/test split
- `split_dataset()`:
  - builds `strat_key = label + '_' + type` so the split keeps both the scam ratio **and** the email/SMS mix equal in every split;
  - 70% train / 30% temp, then temp → 50/50 → **15% val, 15% test**;
  - **leakage check**: computes set intersections of texts between train/val/test and `assert`s all are 0;
  - saves `train.csv`, `val.csv`, `test.csv` and prints the label ratio per split.

## 5. `src/data/stats.py` — dataset statistics
- `compute_statistics()`: prints totals, legit/phishing counts and %, counts per source and per type, and min/max/mean/median character length. Console only.

## 6. `src/eda/eda.py` — charts + EDA report
- `set_style()`: seaborn dark grid and font sizes.
- `generate_eda()`: makes 4 figures (class distribution bar, samples by source, email-vs-SMS donut, text-length histogram on log scale) into `reports/figures/`, then writes `reports/eda_report.md` using an f-string.

> Note: some numbers in that report text (`11 sources`, `1.69 : 1`, `47,500 emails`) are **typed in as constants**, not computed. The length table is computed.

---

## 7. `src/models/bilstm.py` — BiLSTM building blocks
- **`SimpleTokenizer`** (our own word tokenizer, no library):
  - `__init__`: vocab starts with `<pad>`=0 and `<unk>`=1; max vocab 20,000; max length 200.
  - `clean_text(text)`: lowercase, then regex `\b\w+\b` → list of words.
  - `fit_on_texts(texts)`: counts words on the **training** texts, keeps the 19,998 most common, assigns ids from 2.
  - `text_to_sequence(text)`: words → ids (unknown → 1), **post-pads with 0** or truncates to 200.
  - `save` / `load`: JSON round-trip so the API can reuse the exact same vocabulary.
- **`TextDataset`** (PyTorch `Dataset`): pre-converts all texts to id sequences; `__getitem__` returns `(LongTensor ids, FloatTensor label)`.
- **`ScamBiLSTM`** (the network):
  - `Embedding(vocab, 128, padding_idx=0)` → `LSTM(128→128, 2 layers, bidirectional, dropout 0.3)` → concat last forward + last backward hidden state (256) → `Linear(256→64)` → ReLU → Dropout → `Linear(64→1)`.
  - `forward(x)` returns a single **logit** per message (sigmoid is applied outside, in the loss/predictor).

## 8. `src/models/train_nlp_baselines.py` — classical models
- `train_and_evaluate()`:
  1. loads the splits, fills NaN text with `''`;
  2. fits **TfidfVectorizer** (`ngram_range=(1,2)`, `max_features=10000`, `min_df=2`, `sublinear_tf=True`) on **train only**, transforms val/test, saves `tfidf_vectorizer.joblib`;
  3. trains **Logistic Regression** (C=1), **Multinomial NB** (alpha=0.1), **Calibrated LinearSVC** (C=1);
  4. for each: predicts on test, gets probabilities (`predict_proba`), computes accuracy/precision/recall/F1/ROC-AUC + confusion matrix, saves the model `.joblib`;
  5. writes `nlp_baseline_results.json` and `nlp_baseline_report.md`.

> Viva point: **the validation set is not used to tune these models** (no grid search; defaults/fixed hyperparameters). It is only transformed, never scored.
> `CalibratedClassifierCV` default = 5-fold internal cross-validation: it trains 5 LinearSVCs and averages their calibrated probabilities.

## 9. `src/models/train_lstm.py` — train the BiLSTM
- `train_bilstm()`:
  - seeds torch/numpy; picks device (CUDA → Apple MPS → CPU);
  - fits the tokenizer on train, saves `lstm_tokenizer.json`;
  - `DataLoader`s: batch 64 (shuffle only for train);
  - loss `BCEWithLogitsLoss` (sigmoid + binary cross-entropy in one), optimizer `Adam(lr=1e-3)`;
  - per epoch: **train loop** (`zero_grad → forward → loss → backward → step`), then **validation loop** under `torch.no_grad()` with `model.eval()`;
  - **early stopping + checkpointing**: saves weights whenever validation loss improves; stops if no improvement for `patience=2` epochs; max 5 epochs;
  - reloads the **best** checkpoint, evaluates on the test set (threshold 0.5), saves `lstm_results.json`.

## 10. `src/models/train_distilbert.py` — fine-tune DistilBERT
- **`DistilBertDataset`**: tokenizes on the fly with the DistilBERT tokenizer (truncate/pad to **128 tokens**) and returns `input_ids`, `attention_mask`, `labels`.
- `train_distilbert()`:
  - **subsamples: 6,000 training and 1,500 validation messages** (to keep fine-tuning fast); the **full 7,899-message test set** is used for evaluation;
  - loads pretrained `distilbert-base-uncased` with a new 2-class head;
  - `AdamW(lr=2e-5, weight_decay=0.01)`, batch 16, **2 epochs**, gradient clipping at 1.0 (`clip_grad_norm_`) to stabilize training;
  - saves the model + tokenizer only when validation loss improves;
  - reloads the saved checkpoint, runs the test set with softmax probabilities, saves `distilbert_results.json`.

> Viva point (important): DistilBERT saw only ~6k training examples vs ~36.9k for the other models, and still reached the highest recall. Say this proactively; it makes the result more impressive but also means the comparison is not perfectly like-for-like.

## 11. `src/models/compare_models.py` — final comparison
- `DistilBertTestDataset`: same as the training dataset class, used for the latency batch.
- `get_file_size_mb(path)`: size of a file or (recursively) a folder in MB.
- `compare_models()`:
  1. reads stored metrics from the JSON files (it does **not** re-evaluate accuracy);
  2. loads each model again only to measure **latency** (ms/sample on 200 test messages) and **size on disk**;
  3. draws a 5-panel bar chart (`model_comparison.png`);
  4. generates `model_comparison_report.md` with winners per criterion (best F1, recall, AUC, fewest FN, fastest, smallest) and a recommendation;
  5. saves `model_comparison_results.json` (used by the API and dashboard).

> Viva point: latency numbers are **not perfectly comparable**: classical models time `predict` on 200 sparse rows (excluding vectorization), while BiLSTM and DistilBERT time one batch of 200. Treat them as rough orders of magnitude.

---

## 12. `src/models/predictor.py` — one class for inference
**`ScamPredictor(backend='svm', threshold=0.5)`**
- `__init__`: stores backend/threshold, chooses device, creates an empty model cache.
- `_load_svm()`, `_load_bilstm()`, `_load_distilbert()`: **lazy loading + caching** — a model is loaded from disk only the first time it is used. DistilBERT is loaded with `AutoTokenizer` (fast tokenizer from `tokenizer.json`).
- `_risk_label(prob)`: ≥0.8 → "high", ≥0.5 → "medium", else "low".
- `predict_svm(texts)`: TF-IDF transform → `predict_proba` → scam probability.
- `predict_bilstm(texts)`: text → id sequence → model logits → sigmoid.
- `predict_distilbert(texts)`: tokenizes in batches of 64 → softmax → probability of class 1.
- `explain_bilstm(text, top_k)`: **gradient × input saliency**. Embeds the text, keeps the gradient on the embeddings (`retain_grad`), runs the LSTM + head manually, backpropagates the scam probability, then per word computes `sum(grad * embedding)`. Positive = pushes toward scam, negative = toward safe.
- `explain_distilbert(text, top_k)`: same idea using the word-embedding layer; then **merges WordPiece pieces** (`##ing` joins the previous token) and skips `[CLS]/[SEP]/[PAD]`.
- `_tokens_from_saliency(words, saliency, top_k)`: sums scores of repeated words, sorts, returns `top_scam` (positive) and `top_safe` (negative) lists.
- `predict(text, backend)`: public entry point. Accepts one string or a list; routes to the right backend; converts probability → `label` using `threshold`; returns `{label, probability, risk, is_scam}` (a dict for one text, a list for many).

## 13. `app.py` — FastAPI backend
- At import: creates the app, enables **CORS for all origins** (demo only), creates one shared `ScamPredictor`, and loads the SVM vectorizer, classifier and coefficients for explanations (`_svm_clf.calibrated_classifiers_[0].estimator.coef_`).
- Pydantic models `PredictRequest`, `BatchPredictRequest`, `ExplainRequest`: validate input (non-empty text, `top_k` 1–30, batch ≥1).
- Endpoints:
  - `GET /` and `GET /health`: status.
  - `POST /predict`: validates backend (`svm|bilstm|distilbert`, else 400) → `predictor.predict`.
  - `POST /predict/batch`: same for a list, **max 1,000 texts**.
  - `POST /explain`: SVM → computes `tfidf_value × coef` for each non-zero feature and sorts (this is the "exact" explanation); BiLSTM/DistilBERT → saliency from the predictor; returns prediction + `top_scam_tokens` + `top_safe_tokens` + a text note on the method.
  - `GET /models/info`: reads the saved JSON reports and returns metrics + confusion matrices for the dashboard (`test_set_size` is hard-coded to 7899).
- Errors in the model code are caught and returned as HTTP 500.

> Viva nuance: `CalibratedClassifierCV` trains 5 internal LinearSVCs; the explanation uses the **first** one's coefficients. So the SVM explanation is exact for that underlying linear model, and a close (not literally identical) match to the averaged calibrated output. Mention it only if asked how "exact" it really is.

## 14. `demo_predict.py` — command-line demo
- `EXAMPLES`: five sample messages (scam and safe).
- `show(text, predictor)`: prints SVM, BiLSTM and DistilBERT verdicts + probability + risk (BiLSTM/DistilBERT wrapped in try/except so a missing weight file doesn't crash it).
- `main()`: loads the predictor, runs the examples, then an interactive loop (`quit`/`exit` to stop). Run it from the project root so `src` imports work.

## 15. `frontend/index.html` — dashboard (single file)
Plain HTML/CSS/JavaScript. `API_BASE` (line ~417) points at the FastAPI server. It uses `fetch` to call `/predict`, `/predict/batch`, `/explain` and `/models/info`, and renders the verdict, confidence meter, threshold slider, word highlighting, comparison tables/charts and history export.

---

## Quick answers to "what happens when I paste a message?"
1. Browser sends `POST /explain {text, backend}`.
2. `app.py` validates it → `predictor.predict` loads the chosen model once (cached) and returns probability + label + risk.
3. For the explanation: SVM multiplies each word's TF-IDF value by its learned coefficient; the neural models backpropagate to the embeddings.
4. JSON comes back; the frontend highlights the contributing words.

## Things to say proactively (they show you understand the limits)
- Data was **sampled** (≤5,000 emails per source); only exact duplicates were removed.
- DistilBERT was fine-tuned on **6,000** examples for 2 epochs; others used the full training split.
- Classical models used **no hyperparameter tuning**; validation was only used for early stopping/checkpointing of the two neural models.
- Latency numbers are rough and measured differently per model.
- Saliency explanations are approximate; the SVM one is exact for its underlying linear model.
