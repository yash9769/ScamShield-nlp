import os

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
RAW_DIR = os.path.join(BASE_DIR, 'dataset', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')

RAW_SEVEN_PHISHING_PATH = os.path.join(RAW_DIR, 'seven_phishing.csv')
RAW_MEAJOR_PATH = os.path.join(RAW_DIR, 'meajor.csv')
RAW_SMS_SPAM_PATH = os.path.join(RAW_DIR, 'sms_spam.csv')

PROCESSED_DATASET_PATH = os.path.join(PROCESSED_DIR, 'unified_text_dataset.csv')

# Processing configurations
RANDOM_SEED = 42

# Manageable subset limits (per dataset source) to avoid memory/speed issues during initial NLP baseline.
# Set to None to use full datasets.
SUBSET_LIMIT_EMAIL = 5000  # Take at most 5000 emails per source to keep baseline fast
SUBSET_LIMIT_SMS = None    # Take all SMS
