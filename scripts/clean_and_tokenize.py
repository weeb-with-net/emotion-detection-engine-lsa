"""
Cleans the unified dataset's text and produces the padded token sequences
+ label array the BiLSTM will train on.

Run after scripts/build_dataset.py:
    python scripts/clean_and_tokenize.py

Writes:
    data/processed/cleaned_dataset.csv   -- unified_dataset.csv + a clean_text column
    data/processed/tokenizer.pickle
    data/processed/label_encoder.json    -- fixed class order (for reference/debugging)
    data/processed/X_padded.npy
    data/processed/y_labels.npy
"""

import json
import os
import sys
from pathlib import Path

# Allow `python scripts/clean_and_tokenize.py` to find the src/ package
# when run from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.preprocessing.label_mapping import TARGET_CLASSES
from src.preprocessing.text_cleaning import clean_text
from src.preprocessing.tokenization import (
    encode_labels,
    fit_tokenizer,
    save_pickle,
    texts_to_padded,
)

INPUT_CSV = Path("data/processed/unified_dataset.csv")
OUTPUT_DIR = Path("data/processed")


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run scripts/build_dataset.py first."
        )

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows from {INPUT_CSV}")

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Drop rows that became empty after cleaning (rare, but possible for
    # e.g. a comment that was only a URL).
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows that were empty after cleaning")

    cleaned_csv_path = OUTPUT_DIR / "cleaned_dataset.csv"
    df.to_csv(cleaned_csv_path, index=False)
    print(f"Saved cleaned dataset to {cleaned_csv_path}")

    print("\nFitting tokenizer...")
    tokenizer = fit_tokenizer(df["clean_text"])
    vocab_size_found = len(tokenizer.word_index)
    print(f"  Vocabulary found: {vocab_size_found:,} unique tokens (capped at 30,000 for the model)")

    print("Padding/truncating sequences to 80 tokens...")
    X_padded = texts_to_padded(tokenizer, df["clean_text"])
    print(f"  X_padded shape: {X_padded.shape}")

    print("Encoding labels...")
    y_labels = encode_labels(df["emotion"])
    print(f"  y_labels shape: {y_labels.shape}")
    print(f"  Class order (index -> label): {list(enumerate(TARGET_CLASSES))}")

    save_pickle(tokenizer, OUTPUT_DIR / "tokenizer.pickle")
    np.save(OUTPUT_DIR / "X_padded.npy", X_padded)
    np.save(OUTPUT_DIR / "y_labels.npy", y_labels)
    (OUTPUT_DIR / "label_encoder.json").write_text(
        json.dumps({"classes_in_order": TARGET_CLASSES}, indent=2), encoding="utf-8"
    )

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'tokenizer.pickle'}")
    print(f"  {OUTPUT_DIR / 'X_padded.npy'}")
    print(f"  {OUTPUT_DIR / 'y_labels.npy'}")
    print(f"  {OUTPUT_DIR / 'label_encoder.json'}")
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
