"""Interactive CLI to classify Nepali news text using the fine-tuned model.

Loads the saved model from the model/ directory, accepts text input from
stdin, and prints the predicted category with confidence scores for the
top-3 predictions.

Usage:
    python classify_cli.py

The model must be trained first with `python train.py`.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("model")  # Directory containing the saved model and tokenizer


def main():
    # ---- Check model exists ----
    if not MODEL_DIR.exists():
        print(f"Error: Model directory '{MODEL_DIR}' not found.")
        print("Run 'python train.py' first to train and save the model.")
        sys.exit(1)

    # ---- Load model and tokenizer ----
    # The tokenizer converts raw text into token IDs that the model understands.
    # The model includes the classification head with id2label mapping from training.
    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    # Set to eval mode — disables dropout layers for deterministic inference
    model.eval()

    # Move to GPU if available for faster inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Extract the label mapping from model config (e.g. {0: "economy", 1: "global", ...})
    id2label = model.config.id2label
    print(f"Model loaded. Categories: {list(id2label.values())}")
    print("Type a Nepali headline/article to classify. Type 'quit' or 'exit' to stop.\n")

    # ---- Interactive classification loop ----
    while True:
        try:
            text = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        # Handle empty input and exit commands
        if not text:
            print("Empty input, try again.\n")
            continue
        if text.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        # ---- Tokenize input ----
        # max_length=128 for fast inference (shorter than training's 256 is fine
        # for single predictions; most classification signal is in the first sentence)
        enc = tokenizer(
            text,
            max_length=128,
            padding="max_length",     # Pad to fixed length for batch consistency
            truncation=True,          # Truncate if text exceeds 128 tokens
            return_tensors="pt",      # PyTorch tensors
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # ---- Run inference ----
        # torch.no_grad() disables gradient computation, reducing memory usage
        # and speeding up inference by ~2x.
        with torch.no_grad():
            logits = model(**enc).logits

        # ---- Convert logits to probabilities ----
        # Softmax converts raw logits (any real number) into a probability distribution
        # that sums to 1.0. Each value represents the model's confidence for that class.
        probs = torch.softmax(logits, dim=-1).squeeze()
        top_idx = probs.argmax().item()       # Index of the highest probability class
        confidence = probs[top_idx].item()    # The actual probability value

        # ---- Display results ----
        print(f"\n  Predicted category: {id2label[top_idx]} (confidence: {confidence:.2%})")
        print("  Top-3:")

        # Show the top 3 most probable categories for transparency.
        # This helps users understand when the model is uncertain between
        # similar categories (e.g., politics vs national).
        top3 = probs.topk(3)
        for rank, (prob, idx) in enumerate(zip(top3.values, top3.indices), 1):
            print(f"    {rank}. {id2label[idx.item()]}: {prob.item():.2%}")
        print()


if __name__ == "__main__":
    main()
