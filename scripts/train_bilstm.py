"""
Trains the BiLSTM emotion classifier on the preprocessed, tokenized data
from Epic 2 T2.

Run:
    python scripts/train_bilstm.py

Outputs:
    models/bilstm/
        - model.keras
        - training_history.png
        - confusion_matrix.png
        - classification_report.txt
"""

import json
import os
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Project decision (Epic 1): TensorFlow stays CPU-only, PyTorch/CUDA is
# reserved for BERT. Setting this before importing tensorflow makes that
# explicit in code rather than incidental, and avoids CUDA-detection log
# noise on machines where a GPU is visible to the OS but not intended
# for TF's use.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import matplotlib
matplotlib.use("Agg")  # no display available; save plots directly to disk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.models.bilstm import build_bilstm_model
from src.models.focal_loss import compute_alpha_from_labels, sparse_categorical_focal_loss
from src.preprocessing.label_mapping import TARGET_CLASSES

RANDOM_SEED = 42
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/bilstm")
MAX_SEQ_LEN = 80
GAMMA = 2.0            # fixed per Epic 2 T3 specification
VAL_SPLIT = 0.15
BATCH_SIZE = 16
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 4


def set_seeds(seed: int) -> None:
    # Keras 3's set_random_seed seeds Python's random, numpy, and
    # tensorflow in one call -- more complete than seeding numpy/tf
    # separately (e.g. it also covers Python's own `random` module if
    # anything downstream ever uses it).
    tf.keras.utils.set_random_seed(seed)


def load_data():
    X = np.load(DATA_DIR / "X_padded.npy")
    y = np.load(DATA_DIR / "y_labels.npy")
    with open(DATA_DIR / "tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return X, y, tokenizer


def get_vocab_size(tokenizer, cap: int = 30000) -> int:
    """
    Effective vocabulary size is capped by the tokenizer's num_words, but
    the actual number of unique tokens found in a dataset this size will
    usually be smaller than that cap -- using the smaller of the two
    keeps the Embedding layer no larger than it needs to be.
    """
    return min(cap, len(tokenizer.word_index) + 1)


def plot_training_history(history, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Focal Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (validation set)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


class MacroF1Callback(tf.keras.callbacks.Callback):
    """
    Computes macro-F1 on the validation set at the end of every epoch and
    stores it in a plain list. This is purely for REPORTING -- it does
    not feed EarlyStopping or influence training in any way. Useful
    because aggregate accuracy can plateau while minority-class
    performance is still shifting underneath it.
    """

    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.macro_f1_history = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        macro_f1 = f1_score(self.y_val, y_pred, average="macro", zero_division=0)
        self.macro_f1_history.append(macro_f1)
        print(f"    val_macro_f1: {macro_f1:.4f}")


def main() -> None:
    set_seeds(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed data...")
    X, y, tokenizer = load_data()
    vocab_size = get_vocab_size(tokenizer)
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Effective vocab size: {vocab_size:,} (tokenizer cap: 30,000)")

    print(f"\nSplitting train/val ({int((1 - VAL_SPLIT) * 100)}/{int(VAL_SPLIT * 100)}, stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
    print("  Per-class counts in train / val:")
    for idx, cls in enumerate(TARGET_CLASSES):
        n_train = int((y_train == idx).sum())
        n_val = int((y_val == idx).sum())
        print(f"    {cls:<12}: train={n_train:>5} val={n_val:>4}")
        if n_val == 0:
            print(f"    [WARNING] {cls} has 0 validation examples -- its "
                  "val metrics will be undefined (see classification_report).")

    num_classes = len(TARGET_CLASSES)
    alpha = compute_alpha_from_labels(y_train, num_classes)

    print("\nBuilding model...")
    model = build_bilstm_model(vocab_size=vocab_size, max_seq_len=MAX_SEQ_LEN, num_classes=num_classes)
    model.compile(
        optimizer="adam",
        loss=sparse_categorical_focal_loss(alpha=alpha, gamma=GAMMA, num_classes=num_classes),
        metrics=["accuracy"],
    )
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )
    # Present in the Epic 2 T3 reference training code; halves the
    # learning rate when val_loss plateaus rather than only stopping
    # outright -- gives the optimizer a chance to fine-tune before
    # EarlyStopping gives up entirely.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=2, factor=0.5
    )
    macro_f1_callback = MacroF1Callback(X_val, y_val)

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping, reduce_lr, macro_f1_callback],
        verbose=2,
    )

    print("\nEvaluating on validation set...")
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    present_labels = sorted(set(y_val.tolist()) | set(y_pred.tolist()))
    present_names = [TARGET_CLASSES[i] for i in present_labels]
    report = classification_report(
        y_val, y_pred, labels=present_labels, target_names=present_names,
        zero_division=0,
    )
    print(report)

    cm = confusion_matrix(y_val, y_pred, labels=list(range(num_classes)))

    # --- Save everything ---
    model.save(MODEL_DIR / "model.keras")
    plot_training_history(history, MODEL_DIR / "training_history.png")
    plot_confusion_matrix(cm, TARGET_CLASSES, MODEL_DIR / "confusion_matrix.png")
    (MODEL_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    # Raw history numbers (not just the plot) -- useful for a report
    # appendix or for re-plotting later without re-running training.
    history_record = dict(history.history)
    history_record["val_macro_f1"] = macro_f1_callback.macro_f1_history
    (MODEL_DIR / "history.json").write_text(json.dumps(history_record, indent=2), encoding="utf-8")

    # Bundle the tokenizer + label order alongside the model so
    # models/bilstm/ is self-contained for the Streamlit app later,
    # instead of that script needing to reach back into data/processed/.
    shutil.copy(DATA_DIR / "tokenizer.pickle", MODEL_DIR / "tokenizer.pickle")
    shutil.copy(DATA_DIR / "label_encoder.json", MODEL_DIR / "label_encoder.json")

    print(f"\nSaved model to               {MODEL_DIR / 'model.keras'}")
    print(f"Saved training curves to     {MODEL_DIR / 'training_history.png'}")
    print(f"Saved confusion matrix to    {MODEL_DIR / 'confusion_matrix.png'}")
    print(f"Saved classification report to {MODEL_DIR / 'classification_report.txt'}")
    print(f"Saved raw history to         {MODEL_DIR / 'history.json'}")
    print(f"Bundled tokenizer + label order into {MODEL_DIR}")
    print(
        "\nNote for later: when loading this model for inference (e.g. in "
        "the Streamlit app), use tf.keras.models.load_model(path, "
        "compile=False) -- the custom focal loss doesn't need to be "
        "reconstructed for prediction, only for further training."
    )


if __name__ == "__main__":
    main()
