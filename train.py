"""Fine-tune NepaliBERT on prepared news data and evaluate on test set.

This script loads preprocessed CSV data, fine-tunes a BERT-based sequence
classification model, and evaluates it on a held-out test set. It uses
HuggingFace Trainer for training loop management, mixed-precision (fp16)
when a GPU is available, gradient accumulation to simulate larger batch
sizes, linear warmup followed by linear decay for the learning rate
schedule, and early stopping to prevent overfitting.
"""

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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Paths
DATA_DIR = Path("data")          # Directory containing train.csv, val.csv, test.csv, labels.json
MODEL_DIR = Path("model")        # Directory where the fine-tuned model is saved

# Model
MODEL_NAME = "Rajan/NepaliBERT"  # Pretrained BERT model for Nepali language from HuggingFace Hub

# Training hyperparameters
MAX_LEN = 256        # Maximum token length; text is truncated or padded to this size.
                      # 256 captures ~3-4 sentences of Nepali text, enough for most news articles.
BATCH_SIZE = 32      # Samples per device per training step.
GRAD_ACCUM = 2       # Gradient accumulation steps; effective batch size = BATCH_SIZE * GRAD_ACCUM = 64.
                      # Larger effective batches stabilize gradient estimates but use more VRAM.
EPOCHS = 15          # Maximum training epochs. Early stopping may halt training before this.
LR = 1e-5            # Peak learning rate after warmup. Typical range for BERT fine-tuning: 1e-5 to 5e-5.
                      # Too high causes catastrophic forgetting; too low converges slowly.


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NewsDataset(Dataset):
    """PyTorch Dataset that tokenizes Nepali news text on-the-fly.

    Each sample returns:
        - input_ids:      Token IDs padded/truncated to max_len
        - attention_mask:  1 for real tokens, 0 for padding
        - labels:         Integer class label
    """

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
            padding="max_length",    # Pad to max_len so all batches have uniform shape
            truncation=True,         # Truncate text that exceeds max_len tokens
            return_tensors="pt",     # Return PyTorch tensors
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_split(name):
    """Load a train/val/test CSV and return (texts, labels) as parallel lists."""
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["text"].tolist(), df["label"].tolist()


def compute_metrics(eval_pred):
    """Compute accuracy from raw logits. Used by Trainer during evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main():
    # ---- Load label mappings ----
    # labels.json maps category name -> integer ID (e.g. {"economy": 0, "global": 1, ...})
    # We need both directions: label2id for the model config, id2label for predictions
    with open(DATA_DIR / "labels.json") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)

    # ---- Load pretrained model and tokenizer ----
    # AutoTokenizer picks the correct tokenizer class for the model architecture.
    # AutoModelForSequenceClassification adds a classification head on top of BERT,
    # outputting num_labels logits. The base BERT weights are loaded from Rajan/NepaliBERT;
    # the classification head is randomly initialized and will be trained from scratch.
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

    # ---- Device selection ----
    # Use CUDA GPU if available for ~3-10x speedup over CPU. fp16 flag in
    # TrainingArguments enables mixed-precision on CUDA, halving memory usage
    # and speeding up training on Ampere+ GPUs (RTX 30xx/40xx, A100, etc.)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load data splits ----
    train_texts, train_labels = load_split("train")
    val_texts, val_labels = load_split("val")
    test_texts, test_labels = load_split("test")

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # ---- Wrap in PyTorch datasets ----
    # Tokenization happens lazily in __getitem__, so the raw text stays in memory
    # until the DataLoader pulls a batch. This avoids tokenizing the full dataset upfront.
    train_ds = NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds = NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_ds = NewsDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    MODEL_DIR.mkdir(exist_ok=True)

    # ---- Training arguments ----
    # Key scheduling details:
    #   - warmup_steps=900:  LR ramps linearly from 0 to 1e-5 over the first 900 steps.
    #     This prevents large, destabilizing gradient updates at the start of training
    #     when the classification head is randomly initialized.
    #   - Default lr_scheduler_type="linear": After warmup, LR decays linearly from
    #     1e-5 to ~0 over the remaining steps. This lets the model settle into a
    #     sharp minimum without oscillating.
    #   - load_best_model_at_end + metric_for_best_model="accuracy": After training,
    #     reload the checkpoint with the highest validation accuracy (not the last one).
    #     This guards against overfitting in later epochs.
    #   - EarlyStoppingCallback(patience=3): Stop training if validation accuracy
    #     does not improve for 3 consecutive epochs. Combined with max 15 epochs,
    #     training can run 4-15 epochs depending on convergence.
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",          # Evaluate on validation set after each epoch
        save_strategy="epoch",          # Save checkpoint after each epoch
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        warmup_steps=900,               # Linear warmup: 0 -> LR over first 900 steps
        weight_decay=0.01,              # L2 regularization to reduce overfitting
        fp16=torch.cuda.is_available(), # Mixed-precision on CUDA GPUs
        load_best_model_at_end=True,    # Restore best checkpoint after training
        metric_for_best_model="accuracy",
        logging_steps=50,               # Log training loss every 50 steps
        report_to="none",               # Disable W&B, TensorBoard, etc.
        save_total_limit=1,             # Keep only the best checkpoint on disk
        dataloader_num_workers=0,       # 0 = main process loads data (safer on Windows)
    )

    # ---- Initialize Trainer ----
    # Trainer handles the training loop, gradient updates, evaluation, checkpointing,
    # and early stopping. We pass EarlyStoppingCallback to halt if val accuracy
    # plateaus for 3 consecutive epochs.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ---- Train ----
    print("Training ...")
    trainer.train()

    # ---- Save final model ----
    # Saves model weights + config (id2label, label2id) and tokenizer files.
    # These are needed by classify_cli.py to load the model later.
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Model saved to {MODEL_DIR}")

    # ---- Test set evaluation ----
    # Run inference on the held-out test set (never seen during training or validation).
    # This gives the final, unbiased performance estimate.
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

    # ---- Qualitative examples ----
    # Show a few correct and incorrect predictions for manual inspection.
    # Useful for understanding which categories confuse the model and why.
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
