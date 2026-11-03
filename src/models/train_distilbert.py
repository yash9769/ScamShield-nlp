# =============================================================================
# FILE: src/models/train_distilbert.py   (PIPELINE STAGE 5b: FINE-TUNE DistilBERT)
# PURPOSE: Take a PRETRAINED DistilBERT language model and FINE-TUNE it to
#          classify scam vs legitimate text, then evaluate on the full test set.
#   Pretrained  = already learned English from huge text corpora.
#   Fine-tuning = continue training on OUR labelled data so it learns the scam task.
# INPUT:   dataset/processed/train.csv, val.csv, test.csv
# OUTPUT:  models/distilbert_scamshield/ (weights + tokenizer),
#          reports/distilbert_results.json
# IMPORTANT FOR VIVA: for speed, fine-tuning uses only a RANDOM SUBSET:
#          6,000 training and 1,500 validation messages. Evaluation uses ALL test messages.
# =============================================================================
import os
import sys
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW      # Adam with decoupled weight decay (standard for transformers)
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
import numpy as np
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
DISTILBERT_SAVE_DIR = os.path.join(MODELS_DIR, 'distilbert_scamshield')

class DistilBertDataset(Dataset):
    """PyTorch Dataset that tokenises each message ON THE FLY when it is requested."""
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len     # 128 tokens max: longer messages are cut (keeps speed/memory low)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # The DistilBERT tokenizer splits text into WordPiece sub-words, adds special tokens
        # [CLS] (start) and [SEP] (end), and returns:
        #   input_ids      - the token ids
        #   attention_mask - 1 for real tokens, 0 for padding (so the model ignores padding)
        # truncation=True cuts long text; padding='max_length' pads short text to 128.
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        # The tokenizer adds a batch dimension of size 1; squeeze(0) removes it
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def train_distilbert():
    """Fine-tune DistilBERT with a validation-based checkpoint, then evaluate on the test set."""
    print("Starting DistilBERT Transformer Fine-Tuning & Evaluation...")
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load dataset splits
    train_path = os.path.join(PROCESSED_DIR, 'train.csv')
    val_path = os.path.join(PROCESSED_DIR, 'val.csv')
    test_path = os.path.join(PROCESSED_DIR, 'test.csv')

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Transformers are slow on CPU, so we fine-tune on a random subset only
    # (seeded sample -> reproducible). The test set is NOT subsampled.
    # Sample subset for transformer fine-tuning to keep it fast & efficient
    train_sub = train_df.sample(n=min(len(train_df), 6000), random_state=RANDOM_SEED).reset_index(drop=True)
    val_sub = val_df.sample(n=min(len(val_df), 1500), random_state=RANDOM_SEED).reset_index(drop=True)

    X_train = train_sub['text'].fillna('').tolist()
    y_train = train_sub['label'].values

    X_val = val_sub['text'].fillna('').tolist()
    y_val = val_sub['label'].values

    X_test = test_df['text'].fillna('').tolist()
    y_test = test_df['label'].values

    print(f"Fine-tuning sizes -> Train: {len(X_train):,}, Val: {len(X_val):,}, Full Test Set: {len(X_test):,}")

    # ---- STEP 1: LOAD PRETRAINED MODEL ----
    # 'distilbert-base-uncased' = DistilBERT trained on lowercase English text.
    # DistilBertForSequenceClassification = DistilBERT + a new, randomly initialised
    # classification head with num_labels=2 outputs (legit / scam). Only this head is new;
    # the rest already "knows" English and gets gently adjusted during fine-tuning.
    # Load Tokenizer & Model
    model_name = "distilbert-base-uncased"
    print(f"\n1. Loading pretrained '{model_name}' tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Create DataLoaders  (smaller batch for training because transformers use a lot of memory)
    train_dataset = DistilBertDataset(X_train, y_train, tokenizer, max_len=128)
    val_dataset = DistilBertDataset(X_val, y_val, tokenizer, max_len=128)
    test_dataset = DistilBertDataset(X_test, y_test, tokenizer, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # lr=2e-5 is a very SMALL learning rate on purpose: big steps would destroy the
    # knowledge the model already has (catastrophic forgetting). weight_decay=0.01 is regularisation.
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    epochs = 2    # pretrained models need only a few passes
    best_val_loss = float('inf')
    os.makedirs(DISTILBERT_SAVE_DIR, exist_ok=True)

    print("\n2. Fine-Tuning DistilBERT...")
    for epoch in range(1, epochs + 1):
        # ---------------- TRAINING PASS ----------------
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            # Passing labels= makes the model compute the cross-entropy loss itself
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits     # shape (batch, 2): one raw score per class

            loss.backward()
            # Gradient clipping: rescale gradients if their total norm exceeds 1.0.
            # Prevents a single bad batch from causing a huge, destabilising weight update.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)     # class with the highest score
            correct_train += (preds == labels).sum().item()
            total_train += len(labels)

            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train

        # ---------------- VALIDATION PASS (no weight updates) ----------------
        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += len(labels)

        avg_val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val

        print(f"Epoch {epoch}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Checkpoint: keep only the epoch with the lowest validation loss
        # (save_pretrained writes weights + config in Hugging Face format)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(DISTILBERT_SAVE_DIR)
            tokenizer.save_pretrained(DISTILBERT_SAVE_DIR)
            print(f"  --> Saved new best DistilBERT checkpoint to {DISTILBERT_SAVE_DIR}")

    # ---- STEP 3: FINAL EVALUATION ON THE FULL TEST SET ----
    # Reload the SAVED best checkpoint rather than using the in-memory (last-epoch) model
    # Evaluate on Full Test Set
    print("\n3. Evaluating Fine-Tuned DistilBERT on Full Held-Out Test Set (7,899 samples)...")
    eval_model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_SAVE_DIR).to(device)
    eval_model.eval()

    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = eval_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # softmax turns the 2 logits into probabilities that sum to 1; [:, 1] = P(scam)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    acc = accuracy_score(y_test, all_preds)
    prec = precision_score(y_test, all_preds, pos_label=1)
    rec = recall_score(y_test, all_preds, pos_label=1)
    f1 = f1_score(y_test, all_preds, pos_label=1)
    auc = roc_auc_score(y_test, all_probs)
    cm = confusion_matrix(y_test, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\n--- DistilBERT Test Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f} (Phishing Recall)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Confusion Matrix [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")

    results = {
        "DistilBERT": {
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
            "Saved Directory": DISTILBERT_SAVE_DIR
        }
    }

    results_path = os.path.join(REPORTS_DIR, 'distilbert_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved DistilBERT test results to: {results_path}")

if __name__ == "__main__":
    train_distilbert()
