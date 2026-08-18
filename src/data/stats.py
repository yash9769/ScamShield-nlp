import os
import pandas as pd
import numpy as np
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PROCESSED_DATASET_PATH

def compute_statistics():
    print("Computing statistics for the unified ScamShield dataset...")
    if not os.path.exists(PROCESSED_DATASET_PATH):
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_DATASET_PATH}. Please run clean.py first.")
        
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    
    # 1. Total samples
    total_samples = len(df)
    
    # 2. Legitimate vs Phishing counts and ratio
    label_counts = df['label'].value_counts()
    legit_count = label_counts.get(0, 0)
    phishing_count = label_counts.get(1, 0)
    legit_ratio = (legit_count / total_samples) * 100 if total_samples > 0 else 0
    phishing_ratio = (phishing_count / total_samples) * 100 if total_samples > 0 else 0
    
    # 3. Source counts
    source_counts = df['source'].value_counts()
    
    # 4. Type counts
    type_counts = df['type'].value_counts()
    
    # 5. Text length metrics
    # Handle possible empty text (should not happen if cleaned properly, but for safety)
    text_lengths = df['text'].apply(lambda x: len(str(x)))
    min_len = text_lengths.min()
    max_len = text_lengths.max()
    mean_len = text_lengths.mean()
    median_len = text_lengths.median()
    
    # Print the report
    print("\n" + "="*50)
    print("         SCAMSHIELD DATASET STATISTICS REPORT         ")
    print("="*50)
    print(f"Total Number of Samples: {total_samples:,}")
    print("-"*50)
    print(f"Legitimate Samples (0):  {legit_count:,} ({legit_ratio:.2f}%)")
    print(f"Phishing Samples (1):    {phishing_count:,} ({phishing_ratio:.2f}%)")
    print("-"*50)
    print("Samples per Source:")
    for src, cnt in source_counts.items():
        print(f"  - {src}: {cnt:,}")
    print("-"*50)
    print("Samples per Message Type:")
    for t, cnt in type_counts.items():
        print(f"  - {t}: {cnt:,}")
    print("-"*50)
    print("Text Character-Length Statistics:")
    print(f"  - Minimum Length: {min_len:,}")
    print(f"  - Maximum Length: {max_len:,}")
    print(f"  - Average Length: {mean_len:.2f}")
    print(f"  - Median Length:  {median_len:.1f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    compute_statistics()
