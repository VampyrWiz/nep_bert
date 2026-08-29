"""Fine-tune NepaliBERT on prepared news data and evaluate on test set."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_NAME = "Rajan/NepaliBERT"
MAX_LEN = 128
BATCH_SIZE = 8
GRAD_ACCUM = 2
EPOCHS = 4
LR = 2e-5


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["text"].tolist(), df["label"].tolist()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


def main():
    with open(DATA_DIR / "labels.json") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")
    test_texts, test_labels = load_split("test")

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    train_ds = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds = NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_ds = NewsDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    MODEL_DIR.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
        save_total_limit=1,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Training ...")
    trainer.train()

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Model saved to {MODEL_DIR}")

    print("\n--- Test Evaluation ---")
    preds_output = trainer.predict(test_ds)
    logits, true_labels = preds_output.predictions, preds_output.label_ids
    pred_labels = np.argmax(logits, axis=-1)

    label_names = [id2label[i] for i in range(num_labels)]
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=label_names))
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))

    print("\n--- Qualitative Examples ---")
    correct = 0
    shown = 0
    for i in range(len(test_texts)):
        if pred_labels[i] == true_labels[i]:
            correct += 1
        if shown < 3 and pred_labels[i] == true_labels[i]:
            print(f"  [CORRECT] Text: {test_texts[i][:80]}...")
            print(f"            Predicted: {id2label[pred_labels[i]]}")
            shown += 1
    shown_wrong = 0
    for i in range(len(test_texts)):
        if pred_labels[i] != true_labels[i] and shown_wrong < 3:
            print(f"  [WRONG]   Text: {test_texts[i][:80]}...")
            print(f"            True: {id2label[true_labels[i]]} | Predicted: {id2label[pred_labels[i]]}")
            shown_wrong += 1


if __name__ == "__main__":
    main()
