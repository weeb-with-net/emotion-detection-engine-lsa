"""
Runs BiLSTM over the validation set (same split used during training) and
saves raw per-class scores + some derived stats to a csv.

Kept as its own script (not combined with the BERT one) because TF and
torch/transformers segfault if loaded in the same process - same reason
train_bilstm.py and train_bert.py are separate scripts.

Run this from the project root:
    python analysis/run_bilstm_predictions.py
"""
import json
import sys
import time
from pathlib import Path

# so "python analysis/run_bilstm_predictions.py" works without -m syntax
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.inference.bilstm_predictor import BiLSTMPredictor
from src.preprocessing.label_mapping import TARGET_CLASSES

DATA_PATH = Path("data/processed/cleaned_dataset.csv")
RANDOM_SEED = 42
VAL_SPLIT = 0.15
OUT_PATH = Path("analysis/bilstm_val_predictions.csv")


def margin(scores: dict) -> float:
    # gap between the top pick and the runner up - how "sure" the model is
    top_two = sorted(scores.values(), reverse=True)[:2]
    return float(top_two[0] - top_two[1])


def entropy(scores: dict) -> float:
    # spread across all 5 classes, not just top pick. low = peaked, high = flat
    p = np.array(list(scores.values()))
    p = p[p > 0]  # avoid log(0)
    return float(-(p * np.log(p)).sum())


def main() -> None:
    
    overall_start = time.perf_counter()
    
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows")

    # same split logic as train_bilstm.py / train_bert.py (same seed, same
    # stratify, same test_size) so this reproduces their exact val set
    class_to_index = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    y = np.array([class_to_index[label] for label in df["emotion"]])
    _, val_idx = train_test_split(
        np.arange(len(df)), test_size=VAL_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    print(f"  Val set: {len(val_idx):,} rows")

    print("Loading BiLSTM predictor...")

    prediction_start = time.perf_counter()

    predictor = BiLSTMPredictor()

    rows = []
    for n, idx in enumerate(val_idx):
        text = df["text"].iloc[idx]
        true_label = TARGET_CLASSES[y[idx]]

        scores = predictor.predict(text)
        pred = max(scores, key=scores.get)

        rows.append({
            "row_id": int(idx),
            "text": text,
            "true_label": true_label,
            "bilstm_pred": pred,
            "bilstm_confidence": scores[pred],
            "bilstm_margin": margin(scores),
            "bilstm_entropy": entropy(scores),
            "bilstm_scores": json.dumps(scores),
        })

        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(val_idx)}")

    prediction_time = time.perf_counter() - prediction_start
    print(f"\nPrediction time: {prediction_time:.2f} seconds")
    print(f"Average per sample: {prediction_time / len(val_idx):.4f} seconds")

    out_df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out_df)} rows to {OUT_PATH}")
    print(f"BiLSTM raw accuracy on val set: {(out_df['bilstm_pred'] == out_df['true_label']).mean():.3f}")

    overall_time = time.perf_counter() - overall_start
    print(f"\nTotal script time: {overall_time:.2f} seconds")

if __name__ == "__main__":
    main()
