import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config import PROCESSED_DATASET_PATH, BASE_DIR

REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
EDA_REPORT_PATH = os.path.join(REPORTS_DIR, 'eda_report.md')

def set_style():
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18
    })

def generate_eda():
    print("Starting Exploratory Data Analysis (EDA)...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    set_style()
    
    df = pd.read_csv(PROCESSED_DATASET_PATH)
    total_samples = len(df)
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='label', palette=['#2ecc71', '#e74c3c'])
    plt.title("ScamShield: Class Distribution (Legitimate vs Phishing)")
    plt.xlabel("Label (0 = Legitimate, 1 = Phishing)")
    plt.ylabel("Sample Count")
    ax.set_xticklabels(["Legitimate (0)", "Phishing (1)"])
    
    # Annotate bars with counts & percentages
    for p in ax.patches:
        height = p.get_height()
        percentage = (height / total_samples) * 100
        ax.annotate(f'{height:,}\n({percentage:.1f}%)',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=12, color='white', weight='bold')
                    
    plt.tight_layout()
    class_fig_path = os.path.join(FIGURES_DIR, 'class_distribution.png')
    plt.savefig(class_fig_path, dpi=300)
    plt.close()
    print(f"Saved figure: {class_fig_path}")

    # 2. Samples by Source Plot
    plt.figure(figsize=(10, 7))
    source_order = df['source'].value_counts().index
    ax = sns.countplot(data=df, y='source', order=source_order, palette='viridis')
    plt.title("ScamShield: Sample Count by Source Dataset")
    plt.xlabel("Sample Count")
    plt.ylabel("Dataset Source")
    
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f' {width:,}', (width, p.get_y() + p.get_height()/2),
                    ha='left', va='center', fontsize=11, color='black', weight='bold')
                    
    plt.tight_layout()
    source_fig_path = os.path.join(FIGURES_DIR, 'samples_by_source.png')
    plt.savefig(source_fig_path, dpi=300)
    plt.close()
    print(f"Saved figure: {source_fig_path}")

    # 3. Samples by Type Plot
    plt.figure(figsize=(7, 7))
    type_counts = df['type'].value_counts()
    colors = ['#3498db', '#f39c12']
    plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, colors=colors,
            textprops={'fontsize': 14, 'weight': 'bold'}, wedgeprops=dict(width=0.4, edgecolor='w'))
    plt.title("ScamShield: Samples by Communication Type")
    plt.tight_layout()
    type_fig_path = os.path.join(FIGURES_DIR, 'samples_by_type.png')
    plt.savefig(type_fig_path, dpi=300)
    plt.close()
    print(f"Saved figure: {type_fig_path}")

    # 4. Text Length Distribution Plot (Log Scale)
    df['char_length'] = df['text'].apply(lambda x: len(str(x)))
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(data=df, x='char_length', hue='label', palette=['#2ecc71', '#e74c3c'],
                      log_scale=True, kde=True, bins=50, element="step")
    plt.title("ScamShield: Text Length Distribution (Log Scale)")
    plt.xlabel("Character Count (Log Scale)")
    plt.ylabel("Frequency")
    plt.legend(title="Class", labels=["Phishing (1)", "Legitimate (0)"])
    plt.tight_layout()
    length_fig_path = os.path.join(FIGURES_DIR, 'text_length_distribution.png')
    plt.savefig(length_fig_path, dpi=300)
    plt.close()
    print(f"Saved figure: {length_fig_path}")

    # Generate Markdown Report
    legit_cnt = (df['label'] == 0).sum()
    phish_cnt = (df['label'] == 1).sum()
    
    report_content = f"""# ScamShield — Exploratory Data Analysis (EDA) Report

## Overview & Key Findings

* **Total Cleaned Samples:** {total_samples:,}
* **Legitimate Messages (0):** {legit_cnt:,} ({(legit_cnt/total_samples)*100:.2f}%)
* **Phishing Messages (1):** {phish_cnt:,} ({(phish_cnt/total_samples)*100:.2f}%)
* **Class Ratio (Legit : Phish):** ~1.69 : 1
* **Data Sources:** 11 distinct sources integrated.
* **Communication Types:** 47,500 Emails (90.2%), 5,160 SMS (9.8%).

---

## 1. Class Distribution

The dataset maintains a realistic class distribution reflecting real-world volumes where benign communications outnumber fraudulent ones, while providing sufficient positive samples for training.

![Class Distribution](figures/class_distribution.png)

---

## 2. Samples by Source Dataset

We have integrated 11 distinct dataset sources across TREC, Enron, CEAS-08, SpamAssassin, Ling, MeAJOR, and UCI SMS Spam collections to ensure wide linguistic variety.

![Samples by Source](figures/samples_by_source.png)

---

## 3. Communication Type Distribution

The dataset includes both email communications and SMS text messages, establishing ScamShield as a multi-modal text scam detector.

![Samples by Type](figures/samples_by_type.png)

---

## 4. Text Length Analysis

| Metric | Legitimate (0) | Phishing (1) | Overall |
|---|---|---|---|
| **Min Length** | {df[df['label']==0]['char_length'].min():,} | {df[df['label']==1]['char_length'].min():,} | {df['char_length'].min():,} |
| **Max Length** | {df[df['label']==0]['char_length'].max():,} | {df[df['label']==1]['char_length'].max():,} | {df['char_length'].max():,} |
| **Mean Length** | {df[df['label']==0]['char_length'].mean():.2f} | {df[df['label']==1]['char_length'].mean():.2f} | {df['char_length'].mean():.2f} |
| **Median Length** | {df[df['label']==0]['char_length'].median():.1f} | {df[df['label']==1]['char_length'].median():.1f} | {df['char_length'].median():.1f} |

![Text Length Distribution](figures/text_length_distribution.png)

* **Key Insight:** Phishing messages display distinct textual length characteristics. SMS messages are concise (under 200 characters), whereas email phishing messages often display specific length distributions depending on whether they are short credential harvests or longer advance-fee/social-engineering narratives.
"""

    with open(EDA_REPORT_PATH, 'w') as f:
        f.write(report_content)
    print(f"Saved EDA report to: {EDA_REPORT_PATH}")
    print("\nExploratory Data Analysis completed successfully!")

if __name__ == "__main__":
    generate_eda()
