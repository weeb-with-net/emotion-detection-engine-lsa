"""
Fine-tunes the baseline BiLSTM model on real student interaction data,
once such data exists.

Usage:
    python scripts/finetune_bilstm.py --student-data path/to/student_data.csv

Expected CSV format:
    text, emotion

Notes:
- Reuses the tokenizer and label mapping from the baseline model.
- Intended for future domain adaptation after deployment.
- If no student dataset is provided, the script exits without modifying
  the existing model.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.preprocessing.label_mapping import TARGET_CLASSES
from src.preprocessing.text_cleaning import clean_text
from src.preprocessing.tokenization import encode_labels, texts_to_padded

RANDOM_SEED = 42
BASELINE_MODEL_PATH = Path("models/bilstm/model.keras")
BASELINE_TOKENIZER_PATH = Path("models/bilstm/tokenizer.pickle")
OUTPUT_DIR = Path("models/bilstm_student_adaptive")
VAL_SPLIT = 0.15
BATCH_SIZE = 32
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
FINE_TUNE_LR = 1e-4

# Below this many rows, any train/val split or reported metric would be
# noise rather than a meaningful fine-tuning result -- refuse rather
# than produce a number that looks like a result but isn't one.
MIN_ROWS_REQUIRED = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the baseline BiLSTM on real student interaction data."
    )
    parser.add_argument(
        "--student-data",
        type=str,
        default="data/student/student_interactions.csv",
        help=(
            "CSV with columns 'text' and 'emotion', matching the schema of "
            "data/processed/cleaned_dataset.csv. Point this at whatever file "
            "the deployed app's interaction logging eventually produces."
        ),
    )
    return parser.parse_args()


def plot_history(history, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Domain-Adaptive Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Domain-Adaptive Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (student-adaptive validation set)")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    student_csv = Path(args.student_data)

    if not student_csv.exists():
        print(f"No student data found at {student_csv}.")
        print(
            "This is expected pre-deployment: genuine 'student-specific' data "
            "can only come from real usage of the deployed app, which hasn't "
            "happened yet. This script is fully wired up and ready to run the "
            "moment such a file exists -- there's simply nothing to fine-tune "
            "on right now. Exiting without changing anything."
        )
        return

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"{BASELINE_MODEL_PATH} not found. Run scripts/train_bilstm.py first.")

    df = pd.read_csv(student_csv)
    missing_cols = {"text", "emotion"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{student_csv} is missing required columns: {missing_cols}")

    df = df.dropna(subset=["text", "emotion"]).reset_index(drop=True)

    unknown_labels = set(df["emotion"].unique()) - set(TARGET_CLASSES)
    if unknown_labels:
        raise ValueError(
            f"Found emotion labels not in TARGET_CLASSES {TARGET_CLASSES}: {unknown_labels}"
        )

    if len(df) < MIN_ROWS_REQUIRED:
        print(f"Only {len(df):,} rows found in {student_csv} (minimum {MIN_ROWS_REQUIRED} required).")
        print(
            "Any split or metric computed on a dataset this small would be "
            "noise, not a meaningful fine-tuning result. Exiting without "
            "training -- re-run once more real interactions have been logged."
        )
        return

    print(f"Loaded {len(df):,} rows of student interaction data from {student_csv}")

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("Cleaning text (same rules as the baseline pipeline)...")
    df["clean_text"] = df["text"].apply(clean_text)
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} rows that were empty after cleaning")

    print("Loading baseline tokenizer (NOT refitting -- must match the trained embedding layer)...")
    with open(BASELINE_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    X = texts_to_padded(tokenizer, df["clean_text"])
    y = encode_labels(df["emotion"])

    print(f"Splitting train/val ({int((1 - VAL_SPLIT) * 100)}/{int(VAL_SPLIT * 100)}, stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

    print("\nLoading baseline model and cloning for adaptation...")
    baseline_model = tf.keras.models.load_model(BASELINE_MODEL_PATH, compile=False)
    adaptive_model = tf.keras.models.clone_model(baseline_model)
    adaptive_model.set_weights(baseline_model.get_weights())

    # Freeze the embedding layer: protects word representations learned
    # from the much larger base corpus from being overwritten by a
    # comparatively small student dataset. Confirmed layers[0] is the
    # Embedding layer (the Input layer used to fix Keras 3's summary()
    # bug does not appear in .layers).
    adaptive_model.layers[0].trainable = False

    adaptive_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    adaptive_model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
    )

    print("\nFine-tuning on student domain...")
    history = adaptive_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping],
        verbose=2,
    )

    print("\nEvaluating on validation set...")
    y_pred = np.argmax(adaptive_model.predict(X_val, verbose=0), axis=1)
    present_labels = sorted(set(y_val.tolist()) | set(y_pred.tolist()))
    present_names = [TARGET_CLASSES[i] for i in present_labels]
    report = classification_report(
        y_val, y_pred, labels=present_labels, target_names=present_names, zero_division=0
    )
    print(report)

    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(TARGET_CLASSES))))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adaptive_model.save(OUTPUT_DIR / "bilstm_student_adaptive.keras")
    shutil.copy(BASELINE_TOKENIZER_PATH, OUTPUT_DIR / "tokenizer.pickle")
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    plot_history(history, OUTPUT_DIR / "training_history.png")
    plot_confusion_matrix(cm, TARGET_CLASSES, OUTPUT_DIR / "confusion_matrix.png")

    print(f"\nSaved adaptive model to   {OUTPUT_DIR / 'bilstm_student_adaptive.keras'}")
    print(f"Saved classification report to {OUTPUT_DIR / 'classification_report.txt'}")
    print(f"Saved training curves to  {OUTPUT_DIR / 'training_history.png'}")
    print(f"Saved confusion matrix to {OUTPUT_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
