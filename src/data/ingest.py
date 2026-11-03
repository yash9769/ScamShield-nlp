# =============================================================================
# FILE: src/data/ingest.py            (PIPELINE STAGE 1 of 7: DATA INGESTION)
# PURPOSE: Download the 3 raw public datasets from the Hugging Face Hub and
#          save each one unchanged as a CSV in dataset/raw/.
# RUN WITH: python -m src.data.ingest   (needs internet + the `datasets` package)
# OUTPUT:   dataset/raw/seven_phishing.csv, meajor.csv, sms_spam.csv
# NEXT STEP: src/data/clean.py reads these CSVs.
# =============================================================================
import os
import pandas as pd                       # DataFrame = in-memory table; used to convert + save CSV
from datasets import load_dataset         # Hugging Face library that downloads datasets by name
import sys

# When a script is run directly (python src/data/ingest.py) Python only knows about
# the script's own folder. We add the project root to sys.path so that the line
# `from src.config import ...` below can find the `src` package.
# dirname x3: ingest.py -> data -> src -> project root.
# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    RAW_DIR,
    RAW_SEVEN_PHISHING_PATH,
    RAW_MEAJOR_PATH,
    RAW_SMS_SPAM_PATH
)

def ingest_datasets():
    """Download all three datasets and write them to CSV. Exits the program on any failure."""
    print("Starting dataset ingestion from Hugging Face...")

    # exist_ok=True -> do not crash if the folder is already there
    # Create raw directory if it doesn't exist
    os.makedirs(RAW_DIR, exist_ok=True)

    # ---- DATASET 1: a bundle of several email corpora (this is where most of the
    # "11 sources" come from: TREC, Enron, CEAS-08, SpamAssassin, Ling, ...) ----
    # 1. Ingest Seven Phishing Email Datasets
    print("\n1/3. Downloading 'puyang2025/seven-phishing-email-datasets'...")
    try:
        # The dataset is published as separate 'train' and 'test' parts on the Hub.
        # We download both because we make our OWN split later (split.py), so the
        # original split is irrelevant to us.
        # Load train and test splits
        seven_train = load_dataset('puyang2025/seven-phishing-email-datasets', split='train')
        seven_test = load_dataset('puyang2025/seven-phishing-email-datasets', split='test')

        # Convert to pandas DataFrames
        df_seven_train = pd.DataFrame(seven_train)
        df_seven_test = pd.DataFrame(seven_test)

        # ignore_index=True renumbers rows 0..N-1 after stacking the two tables
        # Concatenate train and test splits so we have the entire raw collection in one place
        df_seven = pd.concat([df_seven_train, df_seven_test], ignore_index=True)
        print(f"Loaded Seven Phishing: {len(df_seven)} records.")

        # index=False -> do not write the row-number column into the CSV
        # Save raw to CSV
        df_seven.to_csv(RAW_SEVEN_PHISHING_PATH, index=False)
        print(f"Saved raw Seven Phishing to: {RAW_SEVEN_PHISHING_PATH}")
    except Exception as e:
        # If a download fails (no internet, dataset renamed, ...), stop with exit code 1
        # (non-zero = "error") instead of continuing with missing data.
        print(f"Error ingesting Seven Phishing dataset: {e}")
        sys.exit(1)

    # ---- DATASET 2: MeAJOR phishing e-mail corpus ----
    # 2. Ingest MeAJOR Phishing Email Corpus
    print("\n2/3. Downloading 'simlab-vs/meajor_cleaned_preprocessed'...")
    try:
        meajor = load_dataset('simlab-vs/meajor_cleaned_preprocessed', split='train')
        df_meajor = pd.DataFrame(meajor)
        print(f"Loaded MeAJOR: {len(df_meajor)} records.")

        # Save raw to CSV
        df_meajor.to_csv(RAW_MEAJOR_PATH, index=False)
        print(f"Saved raw MeAJOR to: {RAW_MEAJOR_PATH}")
    except Exception as e:
        print(f"Error ingesting MeAJOR dataset: {e}")
        sys.exit(1)

    # ---- DATASET 3: UCI SMS Spam Collection (short text messages; our only SMS data) ----
    # 3. Ingest SMS Spam Collection
    print("\n3/3. Downloading 'ucirvine/sms_spam'...")
    try:
        sms = load_dataset('ucirvine/sms_spam', split='train')
        df_sms = pd.DataFrame(sms)
        print(f"Loaded SMS Spam: {len(df_sms)} records.")

        # Save raw to CSV
        df_sms.to_csv(RAW_SMS_SPAM_PATH, index=False)
        print(f"Saved raw SMS Spam to: {RAW_SMS_SPAM_PATH}")
    except Exception as e:
        print(f"Error ingesting SMS Spam dataset: {e}")
        sys.exit(1)

    print("\nDataset ingestion completed successfully!")

# This guard means: run ingest_datasets() only when this file is executed directly,
# NOT when another file merely imports it.
if __name__ == "__main__":
    ingest_datasets()
