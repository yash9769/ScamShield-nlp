import os
import pandas as pd
import re
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import (
    RAW_SEVEN_PHISHING_PATH,
    RAW_MEAJOR_PATH,
    RAW_SMS_SPAM_PATH,
    PROCESSED_DATASET_PATH,
    PROCESSED_DIR,
    RANDOM_SEED,
    SUBSET_LIMIT_EMAIL,
    SUBSET_LIMIT_SMS
)

def clean_html(text):
    if not isinstance(text, str):
        return ""
    # Strip HTML tags but preserve text within them
    text = re.sub(r'<[^>]+>', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_text(subject, body):
    subject = str(subject).strip() if pd.notnull(subject) else ""
    body = str(body).strip() if pd.notnull(body) else ""
    
    if subject and body:
        return f"[SUBJECT] {subject} [BODY] {body}"
    elif subject:
        return f"[SUBJECT] {subject}"
    else:
        return f"[BODY] {body}"

def process_seven_phishing():
    print("Processing Seven Phishing Email Datasets...")
    if not os.path.exists(RAW_SEVEN_PHISHING_PATH):
        raise FileNotFoundError(f"Raw Seven Phishing dataset not found at {RAW_SEVEN_PHISHING_PATH}")
        
    df = pd.read_csv(RAW_SEVEN_PHISHING_PATH)
    
    # Standardize columns
    standardized = []
    
    # Sample subsets per dataset source to keep it manageable and balanced if configured
    if SUBSET_LIMIT_EMAIL is not None:
        sampled_dfs = []
        for source_name in df['dataset_name'].unique():
            df_source = df[df['dataset_name'] == source_name]
            sample_size = min(len(df_source), SUBSET_LIMIT_EMAIL)
            sampled_dfs.append(df_source.sample(n=sample_size, random_state=RANDOM_SEED))
        df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Sampled Seven Phishing to {len(df)} records ({SUBSET_LIMIT_EMAIL} per source).")

    for idx, row in df.iterrows():
        subj = row.get('subject', '')
        body = row.get('text', '')
        
        # Combine subject and body
        combined_text = format_text(subj, body)
        
        # Clean HTML/noise
        cleaned_text = clean_html(combined_text)
        
        if not cleaned_text or pd.isnull(row['label']):
            continue
            
        standardized.append({
            'id': f"seven_{idx}",
            'text': cleaned_text,
            'subject': str(subj) if pd.notnull(subj) else "",
            'label': int(row['label']),
            'source': str(row['dataset_name']),
            'type': 'email'
        })
        
    return pd.DataFrame(standardized)

def process_meajor():
    print("Processing MeAJOR Phishing Email Corpus...")
    if not os.path.exists(RAW_MEAJOR_PATH):
        raise FileNotFoundError(f"Raw MeAJOR dataset not found at {RAW_MEAJOR_PATH}")
        
    df = pd.read_csv(RAW_MEAJOR_PATH)
    
    # Standardize columns
    standardized = []
    
    # Sample subsets per source if configured
    if SUBSET_LIMIT_EMAIL is not None:
        sampled_dfs = []
        for source_name in df['source'].unique():
            df_source = df[df['source'] == source_name]
            sample_size = min(len(df_source), SUBSET_LIMIT_EMAIL)
            sampled_dfs.append(df_source.sample(n=sample_size, random_state=RANDOM_SEED))
        df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Sampled MeAJOR to {len(df)} records ({SUBSET_LIMIT_EMAIL} per source).")

    for idx, row in df.iterrows():
        subj = row.get('subject', '')
        body = row.get('body', '')
        
        # Combine subject and body
        combined_text = format_text(subj, body)
        
        # Clean HTML/noise
        cleaned_text = clean_html(combined_text)
        
        if not cleaned_text or pd.isnull(row['label']):
            continue
            
        standardized.append({
            'id': f"meajor_{idx}",
            'text': cleaned_text,
            'subject': str(subj) if pd.notnull(subj) else "",
            'label': int(float(row['label'])),
            'source': f"meajor_{row['source']}",
            'type': 'email'
        })
        
    return pd.DataFrame(standardized)

def process_sms_spam():
    print("Processing SMS Spam Collection...")
    if not os.path.exists(RAW_SMS_SPAM_PATH):
        raise FileNotFoundError(f"Raw SMS Spam dataset not found at {RAW_SMS_SPAM_PATH}")
        
    df = pd.read_csv(RAW_SMS_SPAM_PATH)
    
    # Standardize columns
    standardized = []
    
    # Sample if configured
    if SUBSET_LIMIT_SMS is not None:
        sample_size = min(len(df), SUBSET_LIMIT_SMS)
        df = df.sample(n=sample_size, random_state=RANDOM_SEED)
        print(f"Sampled SMS Spam to {len(df)} records.")

    for idx, row in df.iterrows():
        body = row.get('sms', '')
        cleaned_text = clean_html(body)
        
        if not cleaned_text or pd.isnull(row['label']):
            continue
            
        standardized.append({
            'id': f"sms_{idx}",
            'text': cleaned_text,
            'subject': "",
            'label': int(row['label']),
            'source': 'SMS_Spam_Collection',
            'type': 'sms'
        })
        
    return pd.DataFrame(standardized)

def main():
    print("Starting data cleaning and standardization pipeline...")
    
    # Process each raw dataset
    df_seven = process_seven_phishing()
    df_meajor = process_meajor()
    df_sms = process_sms_spam()
    
    # Integrate into one dataset
    print("\nIntegrating datasets...")
    unified_df = pd.concat([df_seven, df_meajor, df_sms], ignore_index=True)
    print(f"Total records before cleaning/deduplication: {len(unified_df)}")
    
    # Handle duplicates
    # Exact duplicate detection based on combined text content
    duplicates_mask = unified_df.duplicated(subset=['text'], keep='first')
    duplicate_count = duplicates_mask.sum()
    print(f"Found {duplicate_count} exact duplicate records.")
    
    # Remove duplicates
    unified_df = unified_df[~duplicates_mask].reset_index(drop=True)
    print(f"Total records after removing duplicates: {len(unified_df)}")
    
    # Write to CSV
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    unified_df.to_csv(PROCESSED_DATASET_PATH, index=False)
    print(f"Successfully saved unified dataset to: {PROCESSED_DATASET_PATH}")
    
if __name__ == "__main__":
    main()
