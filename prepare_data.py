"""Download, filter, clean, and save train/val/test splits for Nepali news classification."""

import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path("data")
SAMPLES_PER_CATEGORY = 6000
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TOP6_CATEGORIES = ["politics", "national", "health", "economy", "sports", "global"]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    sentences = text.split("\u0964")  # Devanagari danda (।)
    text = "\u0964".join(sentences[:3])
    return text


def data_exists():
    return all((DATA_DIR / f).exists() for f in ["train.csv", "val.csv", "test.csv", "labels.json"])


def main():
    if data_exists():
        print(f"Data already exists in {DATA_DIR}/ — skipping download.")
        return

    print("Loading spandyie/nepali-news-dataset (streaming) ...")
    ds = load_dataset("spandyie/nepali-news-dataset", streaming=True)

    rows = []
    for i, sample in enumerate(ds["train"]):
        if sample.get("clean_categories") in TOP6_CATEGORIES:
            text = f"{sample.get('heading', '')} {sample.get('text', '')}".strip()
            rows.append({
                "text": text,
                "category": sample["clean_categories"],
            })
        # Stop once we have enough per category
        if i > 600_000:
            break

    df = pd.DataFrame(rows)
    df = df.dropna()
    df = df[df["text"].str.len() > 10]
    df = df.drop_duplicates(subset="text")

    print(f"Category counts after filtering:")
    print(df["category"].value_counts())

    categories = sorted(df["category"].unique())
    label2id = {cat: i for i, cat in enumerate(categories)}

    sampled = []
    for cat in categories:
        sub = df[df["category"] == cat]
        sampled.append(sub.sample(n=min(SAMPLES_PER_CATEGORY, len(sub)), random_state=RANDOM_SEED))
    df = pd.concat(sampled).reset_index(drop=True)

    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 10]
    df["label"] = df["category"].map(label2id)

    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    DATA_DIR.mkdir(exist_ok=True)
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    with open(DATA_DIR / "labels.json", "w") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print(f"\nSplit sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Labels: {label2id}")
    print("Done.")


if __name__ == "__main__":
    main()
