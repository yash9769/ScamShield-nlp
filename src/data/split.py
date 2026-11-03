# =============================================================================
# FILE: src/data/split.py             (PIPELINE STAGE 3 of 7: TRAIN/VAL/TEST SPLIT)
# PURPOSE: Divide the unified dataset into three NON-overlapping parts:
#            TRAIN (70%)       - the model learns from this
#            VALIDATION (15%)  - used during training to watch for overfitting /
#                                pick the best checkpoint (neural models only)
#            TEST (15%)        - touched ONLY at the end for the final scores
# INPUT:   dataset/processed/unified_text_dataset.csv
# OUTPUT:  dataset/processed/train.csv, val.csv, test.csv
# =============================================================================
import os
import pandas as pd
from sklearn.model_selection import train_test_split   # scikit-learn helper that splits data randomly
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PROCESSED_DATASET_PATH, PROCESSED_DIR, RANDOM_SEED

def split_dataset():
    """Create the stratified 70/15/15 split, verify no leakage, and save the 3 CSVs."""
    print("Starting leakage-free data splitting process...")
    if not os.path.exists(PROCESSED_DATASET_PATH):
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_DATASET_PATH}. Run clean.py first.")

    df = pd.read_csv(PROCESSED_DATASET_PATH)
    total_samples = len(df)
    print(f"Loaded unified dataset with {total_samples:,} records.")

    # STRATIFICATION = the split keeps the same class proportions in every part.
    # We stratify on label AND type together, giving 4 groups:
    # '0_email', '1_email', '0_sms', '1_sms'. Each group is divided 70/15/15, so the
    # scam ratio AND the email/SMS mix are the same in train, val and test.
    # Create stratification column combining label and message type
    df['strat_key'] = df['label'].astype(str) + '_' + df['type'].astype(str)

    # You cannot split into 3 parts in one call, so we split twice.
    # First split: 70% train, 30% temp (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,                 # 30% goes to temp
        random_state=RANDOM_SEED,       # reproducible shuffle
        stratify=df['strat_key']
    )

    # Second split: the 30% 'temp' is cut in half -> 15% of the original each
    # Second split: split temp into 50% val and 50% test (15% overall each)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df['strat_key']
    )

    # strat_key was only a helper; remove it. reset_index gives each file clean 0..N-1 row numbers.
    # Drop temporary strat_key column
    train_df = train_df.drop(columns=['strat_key']).reset_index(drop=True)
    val_df = val_df.drop(columns=['strat_key']).reset_index(drop=True)
    test_df = test_df.drop(columns=['strat_key']).reset_index(drop=True)

    # ---- DATA LEAKAGE CHECK ----
    # Put every message text of each split into a Python set, then intersect.
    # Any text found in two splits would mean the model is tested on something it
    # already saw in training. We expect zero overlap (duplicates were removed in clean.py).
    # NOTE: this only catches IDENTICAL texts, not near-duplicates.
    # Data Leakage Verification
    train_texts = set(train_df['text'])
    val_texts = set(val_df['text'])
    test_texts = set(test_df['text'])

    overlap_train_val = len(train_texts.intersection(val_texts))
    overlap_train_test = len(train_texts.intersection(test_texts))
    overlap_val_test = len(val_texts.intersection(test_texts))

    print("\n--- Data Leakage Check ---")
    print(f"Train vs Val text overlap: {overlap_train_val}")
    print(f"Train vs Test text overlap: {overlap_train_test}")
    print(f"Val vs Test text overlap: {overlap_val_test}")

    # `assert cond, msg` stops the program with an error if cond is False.
    # So a leaky split can never be saved silently.
    assert overlap_train_val == 0, "DATA LEAKAGE DETECTED between Train and Validation!"
    assert overlap_train_test == 0, "DATA LEAKAGE DETECTED between Train and Test!"
    assert overlap_val_test == 0, "DATA LEAKAGE DETECTED between Validation and Test!"
    print("PASSED: Zero text overlap across train, validation, and test splits.")

    # Save splits
    train_path = os.path.join(PROCESSED_DIR, 'train.csv')
    val_path = os.path.join(PROCESSED_DIR, 'val.csv')
    test_path = os.path.join(PROCESSED_DIR, 'test.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved Train set ({len(train_df):,} samples, {(len(train_df)/total_samples)*100:.1f}%): {train_path}")
    print(f"Saved Validation set ({len(val_df):,} samples, {(len(val_df)/total_samples)*100:.1f}%): {val_path}")
    print(f"Saved Test set ({len(test_df):,} samples, {(len(test_df)/total_samples)*100:.1f}%): {test_path}")

    # Print the scam share per split; the three percentages should be almost identical
    # Print label distribution across splits to confirm stratification
    print("\n--- Stratification Verification ---")
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        legit = (split['label'] == 0).sum()
        phish = (split['label'] == 1).sum()
        ratio = (phish / len(split)) * 100
        print(f"{name:5s} -> Legitimate (0): {legit:,} | Phishing (1): {phish:,} | Phishing Ratio: {ratio:.2f}%")

    print("\nData splitting process completed successfully!")

if __name__ == "__main__":
    split_dataset()
