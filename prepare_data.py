"""Download, filter, clean, and save train/val/test splits for Nepali news classification.

Pipeline:
    1. Stream the spandyie/nepali-news-dataset from HuggingFace Hub
    2. Filter to the top 6 categories by article count
    3. Clean text (remove HTML tags, URLs, truncate to 3 sentences)
    4. Deduplicate and balance across categories
    5. Split into train (80%), validation (10%), test (10%)
    6. Save as CSV files with a label mapping JSON

The dataset is streamed (not fully downloaded) to avoid loading ~500k+ articles
into memory at once. We stop after 600k rows, which is enough to find 15k+
articles per category.
"""

import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")               # Output directory for CSVs and labels.json
SAMPLES_PER_CATEGORY = 15000          # Target samples per category after balancing.
                                      # With 6 categories, total = 90,000 samples.
                                      # More data generally improves generalization.
RANDOM_SEED = 42                      # Reproducible shuffling and sampling
TRAIN_RATIO = 0.8                     # 80% of data for training
VAL_RATIO = 0.1                       # 10% for validation (hyperparameter tuning)
                                      # Remaining 10% goes to test (final evaluation)

# The top 6 categories by article count in the dataset.
# These cover the most common Nepali news topics.
TOP6_CATEGORIES = ["politics", "national", "health", "economy", "sports", "global"]


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean raw article text for model input.

    Steps:
        1. Remove HTML tags (e.g. <p>, <br>) that leak from web scraping
        2. Remove URLs (https://...) which add noise without semantic value
        3. Collapse multiple whitespace characters into single spaces
        4. Truncate to 3 sentences using Devanagari danda (।) as sentence delimiter

    Truncation to 3 sentences keeps the most informative lead sentences while
    staying within BERT's 256-token limit for most articles.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # Strip HTML tags
    text = re.sub(r"https?://\S+", " ", text)     # Strip URLs
    text = re.sub(r"\s+", " ", text).strip()       # Normalize whitespace
    sentences = text.split("\u0964")               # Split on Devanagari danda (।)
    text = "\u0964".join(sentences[:3])            # Keep first 3 sentences
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def data_exists():
    """Check if all required output files already exist."""
    return all((DATA_DIR / f).exists() for f in ["train.csv", "val.csv", "test.csv", "labels.json"])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    # Skip if data already prepared (idempotent)
    if data_exists():
        print(f"Data already exists in {DATA_DIR}/ — skipping download.")
        return

    # ---- Step 1: Stream and filter ----
    # Streaming avoids downloading the full dataset (~2GB) upfront.
    # We iterate through the HuggingFace dataset, keeping only articles
    # whose clean_categories matches one of our target 6.
    print("Loading spandyie/nepali-news-dataset (streaming) ...")
    ds = load_dataset("spandyie/nepali-news-dataset", streaming=True)

    rows = []
    for i, sample in enumerate(ds["train"]):
        if sample.get("clean_categories") in TOP6_CATEGORIES:
            # Combine heading + body text into a single input string
            text = f"{sample.get('heading', '')} {sample.get('text', '')}".strip()
            rows.append({
                "text": text,
                "category": sample["clean_categories"],
            })
        # Stop after 600k rows — enough to find 15k+ per category
        if i > 600_000:
            break

    # ---- Step 2: Clean and deduplicate ----
    df = pd.DataFrame(rows)
    df = df.dropna()                                    # Remove rows with missing text/category
    df = df[df["text"].str.len() > 10]                  # Remove very short texts (< 10 chars)
    df = df.drop_duplicates(subset="text")              # Remove duplicate articles

    print(f"Category counts after filtering:")
    print(df["category"].value_counts())

    # ---- Step 3: Build label mapping ----
    # Sort categories alphabetically for deterministic ID assignment.
    # This mapping is saved to labels.json and used by train.py and classify_cli.py.
    categories = sorted(df["category"].unique())
    label2id = {cat: i for i, cat in enumerate(categories)}

    # ---- Step 4: Balance categories ----
    # Sample exactly SAMPLES_PER_CATEGORY from each category (or all if fewer).
    # This prevents the model from being biased toward more frequent categories.
    sampled = []
    for cat in categories:
        sub = df[df["category"] == cat]
        sampled.append(sub.sample(n=min(SAMPLES_PER_CATEGORY, len(sub)), random_state=RANDOM_SEED))
    df = pd.concat(sampled).reset_index(drop=True)

    # ---- Step 5: Apply text cleaning ----
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 10]                  # Remove texts too short after cleaning
    df["label"] = df["category"].map(label2id)           # Convert category names to integer IDs

    # ---- Step 6: Split into train/val/test ----
    # Shuffle first, then slice sequentially. This gives reproducible splits
    # since the data is already balanced across categories.
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    # ---- Step 7: Save outputs ----
    DATA_DIR.mkdir(exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    # Save label mapping as JSON for other scripts to load
    with open(DATA_DIR / "labels.json", "w") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Labels: {label2id}")
    print("Done.")


if __name__ == "__main__":
    main()
