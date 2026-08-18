import os
import pandas as pd
from datasets import load_dataset
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    RAW_DIR,
    RAW_SEVEN_PHISHING_PATH,
    RAW_MEAJOR_PATH,
    RAW_SMS_SPAM_PATH
)

def ingest_datasets():
    print("Starting dataset ingestion from Hugging Face...")
    
    # Create raw directory if it doesn't exist
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # 1. Ingest Seven Phishing Email Datasets
    print("\n1/3. Downloading 'puyang2025/seven-phishing-email-datasets'...")
    try:
        # Load train and test splits
        seven_train = load_dataset('puyang2025/seven-phishing-email-datasets', split='train')
        seven_test = load_dataset('puyang2025/seven-phishing-email-datasets', split='test')
        
        # Convert to pandas DataFrames
        df_seven_train = pd.DataFrame(seven_train)
        df_seven_test = pd.DataFrame(seven_test)
        
        # Concatenate train and test splits so we have the entire raw collection in one place
        df_seven = pd.concat([df_seven_train, df_seven_test], ignore_index=True)
        print(f"Loaded Seven Phishing: {len(df_seven)} records.")
        
        # Save raw to CSV
        df_seven.to_csv(RAW_SEVEN_PHISHING_PATH, index=False)
        print(f"Saved raw Seven Phishing to: {RAW_SEVEN_PHISHING_PATH}")
    except Exception as e:
        print(f"Error ingesting Seven Phishing dataset: {e}")
        sys.exit(1)
        
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

if __name__ == "__main__":
    ingest_datasets()
