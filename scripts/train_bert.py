"""
Fine-tunes bert-base-uncased for 5-class emotion classification, per the
Epic 2 T5 specification: 3 epochs, AdamW, learning rate 2e-5.

BERT fine-tuning script for the Emotion Detection & Learning Support Engine.

This script:
- Loads the cleaned emotion dataset.
- Fine-tunes bert-base-uncased for 5-class emotion classification.
- Evaluates the trained model.
- Saves the complete Hugging Face model and evaluation artifacts.

Outputs:
- models/bert_emotion_model_final/
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.preprocessing.label_mapping import TARGET_CLASSES

RANDOM_SEED = 42
DATA_PATH = Path("data/processed/cleaned_dataset.csv")
MODEL_DIR = Path("models/bert_emotion_model_final")
MAX_LENGTH = 128
VAL_SPLIT = 0.15          # matches scripts/train_bilstm.py exactly
BATCH_SIZE = 16
NUM_EPOCHS = 3            # per Epic 2 T5 specification
LEARNING_RATE = 2e-5      # per Epic 2 T5 specification


class EmotionTextDataset(Dataset):
    """Wraps pre-tokenized encodings + integer labels for a DataLoader."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for emotion classification.")
    parser.add_argument(
        "--model-name", type=str, default="bert-base-uncased",
        help="HuggingFace model id or local path (override only for testing).",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_labels(labels) -> np.ndarray:
    class_to_index = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    return np.array([class_to_index[label] for label in labels])


def plot_history(history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("BERT Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("BERT Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (BERT, validation set)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_epoch(model, loader, device, loss_fn, optimizer=None):
    """
    Runs one pass over `loader`. If `optimizer` is given, trains
    (backprop + step); otherwise evaluates under no_grad. Labels are
    popped from the batch and the loss is computed manually via
    `loss_fn` (rather than using the model's built-in loss) so that
    class weighting can be applied.

    Returns (avg_loss, accuracy, y_true, y_pred).
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    return avg_loss, accuracy, all_labels, all_preds


def main() -> None:
    args = parse_args()
    set_seeds(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("  [WARNING] No CUDA GPU detected -- BERT fine-tuning on CPU will be "
              "much slower than on the GPU. Check your PyTorch CUDA install "
              "if this is unexpected.")

    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows")

    # Same split as scripts/train_bilstm.py (same file, same row order,
    # same random_state/stratify/test_size) -- reproduces an identical
    # train/val split for a fair, direct BiLSTM-vs-BERT comparison.
    y = encode_labels(df["emotion"])
    train_idx, val_idx = train_test_split(
        np.arange(len(df)), test_size=VAL_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    train_texts = df["text"].iloc[train_idx].tolist()
    val_texts = df["text"].iloc[val_idx].tolist()
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"  Train: {len(train_texts):,} | Val: {len(val_texts):,}")

    num_classes = len(TARGET_CLASSES)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_train)
    print("\n== Automatically computed class weights ==")
    for idx, w in enumerate(class_weights):
        print(f"  {TARGET_CLASSES[idx]:<12}: {w:.3f}")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    print(f"\nLoading tokenizer and model ({args.model_name})...")
    id2label = {i: cls for i, cls in enumerate(TARGET_CLASSES)}
    label2id = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_classes, id2label=id2label, label2id=label2id
    ).to(device)

    print("Tokenizing...")
    train_encodings = tokenizer(
        train_texts, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt"
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt"
    )

    train_loader = DataLoader(EmotionTextDataset(train_encodings, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(EmotionTextDataset(val_encodings, y_val), batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
    y_true, y_pred = [], []  # will hold the FINAL epoch's validation results

    print("\nFine-tuning BERT...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, loss_fn, optimizer=optimizer)
        val_loss, val_acc, y_true, y_pred = run_epoch(model, val_loader, device, loss_fn, optimizer=None)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(macro_f1)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - train_loss: {train_loss:.4f} - "
            f"train_acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - "
            f"val_acc: {val_acc:.4f} - val_macro_f1: {macro_f1:.4f}"
        )

    print("\nFinal validation classification report:")
    report = classification_report(y_true, y_pred, target_names=TARGET_CLASSES, zero_division=0)
    print(report)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    print(f"\nSaving model + tokenizer to {MODEL_DIR}...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    (MODEL_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    (MODEL_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history, MODEL_DIR / "training_history.png")
    plot_confusion_matrix(cm, TARGET_CLASSES, MODEL_DIR / "confusion_matrix.png")

    print(f"\nSaved HuggingFace model suite (config.json, model.safetensors, tokenizer files) to {MODEL_DIR}")
    print("Saved classification_report.txt, history.json, training_history.png, confusion_matrix.png")
    print(
        "\nNote for later: load this model for inference with "
        "AutoModelForSequenceClassification.from_pretrained(path) and "
        "AutoTokenizer.from_pretrained(path) -- HuggingFace's save/load "
        "format is self-contained, no special flags needed."
    )


if __name__ == "__main__":
    main()
