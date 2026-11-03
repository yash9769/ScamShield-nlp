# =============================================================================
# FILE: src/config.py
# PURPOSE: One central place for every path and setting shared by the project.
#          Every other script does `from src.config import ...` so that if we
#          ever change a folder name or the random seed, we change it ONCE here.
# THIS FILE HAS NO FUNCTIONS - only constants. It is imported, never "run".
# =============================================================================
import os

# __file__ is the path of THIS file (src/config.py).
# os.path.abspath(__file__)  -> full absolute path  (…/ScamShield-nlp/src/config.py)
# os.path.dirname(...) once  -> …/ScamShield-nlp/src
# os.path.dirname(...) twice -> …/ScamShield-nlp   <- the project root folder
# Using the file's own location (instead of a hard-coded "C:/..." path) means the
# project works no matter which computer/folder it is cloned into.
# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# os.path.join builds paths with the right separator for the OS (/ on Linux/Mac, \ on Windows).
# Dataset paths
RAW_DIR = os.path.join(BASE_DIR, 'dataset', 'raw')              # untouched downloaded CSVs
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')  # cleaned data + train/val/test splits

# One raw CSV per source dataset (created by src/data/ingest.py).
RAW_SEVEN_PHISHING_PATH = os.path.join(RAW_DIR, 'seven_phishing.csv')  # bundle of several phishing/legit email corpora
RAW_MEAJOR_PATH = os.path.join(RAW_DIR, 'meajor.csv')                  # MeAJOR phishing email corpus
RAW_SMS_SPAM_PATH = os.path.join(RAW_DIR, 'sms_spam.csv')              # UCI SMS Spam Collection

# The single merged + cleaned dataset (created by src/data/clean.py).
PROCESSED_DATASET_PATH = os.path.join(PROCESSED_DIR, 'unified_text_dataset.csv')

# Processing configurations
# A "random seed" fixes the random number generator so that sampling, splitting
# and neural-network weight initialisation give the SAME result on every run.
# That is what makes the experiment reproducible (42 is just a conventional choice).
RANDOM_SEED = 42

# Manageable subset limits (per dataset source) to avoid memory/speed issues during initial NLP baseline.
# Set to None to use full datasets.
# IMPORTANT FOR VIVA: this means the final corpus is a SAMPLE, not the full public datasets.
SUBSET_LIMIT_EMAIL = 5000  # Take at most 5000 emails per source to keep baseline fast
SUBSET_LIMIT_SMS = None    # Take all SMS (SMS is small, so no limit needed)
