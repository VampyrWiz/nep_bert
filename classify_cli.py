"""Interactive CLI to classify Nepali news text using the fine-tuned model."""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path("model")


def main():
    if not MODEL_DIR.exists():
        print(f"Error: Model directory '{MODEL_DIR}' not found.")
        print("Run 'python train.py' first to train and save the model.")
        sys.exit(1)

    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    id2label = model.config.id2label
    print(f"Model loaded. Categories: {list(id2label.values())}")
    print("Type a Nepali headline/article to classify. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            text = input("Enter text: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not text:
            print("Empty input, try again.\n")
            continue
        if text.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        enc = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits

        probs = torch.softmax(logits, dim=-1).squeeze()
        top_idx = probs.argmax().item()
        confidence = probs[top_idx].item()

        print(f"\n  Predicted category: {id2label[top_idx]} (confidence: {confidence:.2%})")
        print("  Top-3:")
        top3 = probs.topk(3)
        for rank, (prob, idx) in enumerate(zip(top3.values, top3.indices), 1):
            print(f"    {rank}. {id2label[idx.item()]}: {prob.item():.2%}")
        print()


if __name__ == "__main__":
    main()
