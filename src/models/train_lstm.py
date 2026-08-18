import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from src.models.bilstm import SimpleTokenizer, TextDataset, ScamBiLSTM

MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

def train_bilstm():
    print("Starting PyTorch BiLSTM Training & Evaluation...")
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
    
    X_train = train_df['text'].fillna('').tolist()
    y_train = train_df['label'].values
    
    X_val = val_df['text'].fillna('').tolist()
    y_val = val_df['label'].values
    
    X_test = test_df['text'].fillna('').tolist()
    y_test = test_df['label'].values
    
    # Fit Tokenizer
    print("\n1. Fitting Word Tokenizer on Training Data...")
    tokenizer = SimpleTokenizer(max_vocab_size=20000, max_length=200)
    tokenizer.fit_on_texts(X_train)
    
    tokenizer_path = os.path.join(MODELS_DIR, 'lstm_tokenizer.json')
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path} (Vocab size: {len(tokenizer.word2idx):,})")
    
    # Create Datasets & DataLoaders
    train_dataset = TextDataset(X_train, y_train, tokenizer)
    val_dataset = TextDataset(X_val, y_val, tokenizer)
    test_dataset = TextDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Instantiate Model
    model = ScamBiLSTM(vocab_size=len(tokenizer.word2idx), embedding_dim=128, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model_save_path = os.path.join(MODELS_DIR, 'bilstm_model.pt')
    best_val_loss = float('inf')
    epochs = 5
    patience = 2
    patience_counter = 0
    
    print("\n2. Training BiLSTM Neural Network...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * len(y_batch)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct_train += (preds == y_batch).sum().item()
            total_train += len(y_batch)
            
        avg_train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train
        
        # Validation Loop
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                total_val_loss += loss.item() * len(y_batch)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct_val += (preds == y_batch).sum().item()
                total_val += len(y_batch)
                
        avg_val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val
        
        print(f"Epoch {epoch}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Saved new best checkpoint to {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break
                
    # Load Best Model for Test Evaluation
    print("\n3. Evaluating Best BiLSTM Model on Held-Out Test Set...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
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
    
    print("\n--- BiLSTM Test Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f} (Phishing Recall)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Confusion Matrix [TN={tn}, FP={fp}, FN={fn}, TP={tp}]")
    
    results = {
        "BiLSTM": {
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
            "Saved File": model_save_path
        }
    }
    
    results_path = os.path.join(REPORTS_DIR, 'lstm_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved BiLSTM test results to: {results_path}")

if __name__ == "__main__":
    train_bilstm()
