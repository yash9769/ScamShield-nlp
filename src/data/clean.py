# =============================================================================
# FILE: src/data/clean.py             (PIPELINE STAGE 2 of 7: CLEANING + MERGING)
# PURPOSE: The three raw datasets have DIFFERENT column names and formats.
#          This script converts each into ONE common table format, removes HTML
#          noise, samples a manageable amount, merges them, and removes duplicates.
# INPUT:   dataset/raw/*.csv                     (from ingest.py)
# OUTPUT:  dataset/processed/unified_text_dataset.csv
#          Columns: id, text, subject, label, source, type
#            label: 0 = legitimate, 1 = scam/phishing/spam
#            type : 'email' or 'sms'
# NEXT STEP: split.py
# =============================================================================
import os
import pandas as pd
import re                                  # regular expressions, used to strip HTML tags
import sys

# Ensure project root is in sys.path (so `src.config` can be imported when run as a script)
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
    """Remove HTML tags and extra whitespace from one string.

    Example: '<p>Hello   <b>world</b></p>'  ->  'Hello world'
    Phishing emails are often HTML; the tags are noise for a text classifier.
    """
    # If the value is not text (e.g. NaN/missing), return an empty string
    # so later code can safely skip it.
    if not isinstance(text, str):
        return ""
    # Regex '<[^>]+>' = a '<', then one or more characters that are not '>', then '>'.
    # i.e. it matches any HTML tag. We replace each tag by a space (not by nothing)
    # so words on either side of a tag do not get glued together.
    # Strip HTML tags but preserve text within them
    text = re.sub(r'<[^>]+>', ' ', text)
    # '\s+' = one or more whitespace characters (spaces, tabs, newlines) -> one space.
    # .strip() removes leading/trailing spaces.
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_text(subject, body):
    """Combine email subject and body into ONE string with marker tokens.

    The markers [SUBJECT] and [BODY] let the model know which part is which.
    Handles missing subject and/or missing body.
    """
    # pd.notnull(x) is False for NaN/None. Missing values become "".
    subject = str(subject).strip() if pd.notnull(subject) else ""
    body = str(body).strip() if pd.notnull(body) else ""

    if subject and body:
        return f"[SUBJECT] {subject} [BODY] {body}"
    elif subject:
        return f"[SUBJECT] {subject}"
    else:
        return f"[BODY] {body}"

def process_seven_phishing():
    """Convert the 'Seven Phishing' raw CSV into the common table format."""
    print("Processing Seven Phishing Email Datasets...")
    if not os.path.exists(RAW_SEVEN_PHISHING_PATH):
        raise FileNotFoundError(f"Raw Seven Phishing dataset not found at {RAW_SEVEN_PHISHING_PATH}")

    df = pd.read_csv(RAW_SEVEN_PHISHING_PATH)

    # Will collect one dict per message, turned into a DataFrame at the end
    # Standardize columns
    standardized = []

    # This dataset bundles several corpora; 'dataset_name' says which one a row came
    # from. We cap EACH corpus at SUBSET_LIMIT_EMAIL rows so no single huge corpus
    # (e.g. Enron) dominates, and so training stays fast.
    # Sample subsets per dataset source to keep it manageable and balanced if configured
    if SUBSET_LIMIT_EMAIL is not None:
        sampled_dfs = []
        for source_name in df['dataset_name'].unique():
            df_source = df[df['dataset_name'] == source_name]
            # min(...) so a corpus smaller than the limit is kept entirely
            sample_size = min(len(df_source), SUBSET_LIMIT_EMAIL)
            # random_state=RANDOM_SEED -> the same rows are picked every run (reproducible)
            sampled_dfs.append(df_source.sample(n=sample_size, random_state=RANDOM_SEED))
        df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"Sampled Seven Phishing to {len(df)} records ({SUBSET_LIMIT_EMAIL} per source).")

    # iterrows() yields (row_index, row). Slow for huge data but fine here.
    for idx, row in df.iterrows():
        subj = row.get('subject', '')
        body = row.get('text', '')

        # Combine subject and body
        combined_text = format_text(subj, body)

        # Clean HTML/noise
        cleaned_text = clean_html(combined_text)

        # Skip rows with no text or no label - they are useless for supervised learning
        if not cleaned_text or pd.isnull(row['label']):
            continue

        standardized.append({
            'id': f"seven_{idx}",                                   # unique id, prefix tells the origin file
            'text': cleaned_text,                                   # <- the ONLY column models are trained on
            'subject': str(subj) if pd.notnull(subj) else "",
            'label': int(row['label']),                             # 0 legit / 1 phishing
            'source': str(row['dataset_name']),                     # which original corpus
            'type': 'email'
        })

    return pd.DataFrame(standardized)

def process_meajor():
    """Convert the MeAJOR raw CSV into the common table format (same steps as above)."""
    print("Processing MeAJOR Phishing Email Corpus...")
    if not os.path.exists(RAW_MEAJOR_PATH):
        raise FileNotFoundError(f"Raw MeAJOR dataset not found at {RAW_MEAJOR_PATH}")

    df = pd.read_csv(RAW_MEAJOR_PATH)

    # Standardize columns
    standardized = []

    # Here the column that identifies the sub-corpus is called 'source'
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
        body = row.get('body', '')          # in MeAJOR the body column is called 'body' (in Seven it is 'text')

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
            # int(float(...)) because the label may be stored as '1.0' (a float) in the CSV
            'label': int(float(row['label'])),
            'source': f"meajor_{row['source']}",
            'type': 'email'
        })

    return pd.DataFrame(standardized)

def process_sms_spam():
    """Convert the SMS Spam raw CSV into the common table format. SMS has no subject."""
    print("Processing SMS Spam Collection...")
    if not os.path.exists(RAW_SMS_SPAM_PATH):
        raise FileNotFoundError(f"Raw SMS Spam dataset not found at {RAW_SMS_SPAM_PATH}")

    df = pd.read_csv(RAW_SMS_SPAM_PATH)

    # Standardize columns
    standardized = []

    # SUBSET_LIMIT_SMS is None in config, so ALL SMS are kept
    # Sample if configured
    if SUBSET_LIMIT_SMS is not None:
        sample_size = min(len(df), SUBSET_LIMIT_SMS)
        df = df.sample(n=sample_size, random_state=RANDOM_SEED)
        print(f"Sampled SMS Spam to {len(df)} records.")

    for idx, row in df.iterrows():
        body = row.get('sms', '')           # the SMS text column is called 'sms'
        cleaned_text = clean_html(body)

        if not cleaned_text or pd.isnull(row['label']):
            continue

        standardized.append({
            'id': f"sms_{idx}",
            'text': cleaned_text,
            'subject': "",                  # SMS has no subject line
            'label': int(row['label']),     # in this dataset 1 = spam
            'source': 'SMS_Spam_Collection',
            'type': 'sms'
        })

    return pd.DataFrame(standardized)

def main():
    """Run the whole cleaning pipeline: process each dataset, merge, de-duplicate, save."""
    print("Starting data cleaning and standardization pipeline...")

    # Process each raw dataset
    df_seven = process_seven_phishing()
    df_meajor = process_meajor()
    df_sms = process_sms_spam()

    # Stack the three tables on top of each other (they now share the same columns)
    # Integrate into one dataset
    print("\nIntegrating datasets...")
    unified_df = pd.concat([df_seven, df_meajor, df_sms], ignore_index=True)
    print(f"Total records before cleaning/deduplication: {len(unified_df)}")

    # Why remove duplicates? If the same message appeared in both the training and
    # the test set, the model could just memorise it and the test score would be
    # unrealistically high (data leakage). We only detect EXACT duplicate text;
    # near-duplicates (same template, small edits) are NOT removed - a known limitation.
    # keep='first' keeps the first occurrence and marks later ones as duplicates.
    # Handle duplicates
    # Exact duplicate detection based on combined text content
    duplicates_mask = unified_df.duplicated(subset=['text'], keep='first')
    duplicate_count = duplicates_mask.sum()
    print(f"Found {duplicate_count} exact duplicate records.")

    # '~' means NOT: keep only rows that are NOT duplicates
    # Remove duplicates
    unified_df = unified_df[~duplicates_mask].reset_index(drop=True)
    print(f"Total records after removing duplicates: {len(unified_df)}")

    # Write to CSV
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    unified_df.to_csv(PROCESSED_DATASET_PATH, index=False)
    print(f"Successfully saved unified dataset to: {PROCESSED_DATASET_PATH}")

if __name__ == "__main__":
    main()
