"""
Runs BERT over the validation set (same split as run_bilstm_predictions.py)
and saves BOTH raw softmax and boosted (class-weight + keyword) scores.

Separate process from the BiLSTM script on purpose - see that file's
docstring for why.

Run this from the project root:
    python analysis/run_bert_predictions.py
"""
import json
import sys
import time
from pathlib import Path

# so "python analysis/run_bert_predictions.py" works without -m syntax
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.inference.bert_predictor import BERTPredictor
from src.preprocessing.label_mapping import TARGET_CLASSES

DATA_PATH = Path("data/processed/cleaned_dataset.csv")
RANDOM_SEED = 42
VAL_SPLIT = 0.15
OUT_PATH = Path("analysis/bert_val_predictions.csv")


def margin(scores: dict) -> float:
    top_two = sorted(scores.values(), reverse=True)[:2]
    return float(top_two[0] - top_two[1])


def entropy(scores: dict) -> float:
    p = np.array(list(scores.values()))
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def main() -> None:\

    overall_start = time.perf_counter()

    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} rows")

    # identical split logic to run_bilstm_predictions.py / train scripts
    class_to_index = {cls: i for i, cls in enumerate(TARGET_CLASSES)}
    y = np.array([class_to_index[label] for label in df["emotion"]])
    _, val_idx = train_test_split(
        np.arange(len(df)), test_size=VAL_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    print(f"  Val set: {len(val_idx):,} rows")

    print("Loading BERT predictor...")
    
    prediction_start = time.perf_counter()

    predictor = BERTPredictor()

    rows = []
    for n, idx in enumerate(val_idx):
        text = df["text"].iloc[idx]
        true_label = TARGET_CLASSES[y[idx]]

        raw = predictor.predict_raw(text)
        boosted = predictor.predict(text)

        pred_raw = max(raw, key=raw.get)
        pred_boosted = max(boosted, key=boosted.get)

        rows.append({
            "row_id": int(idx),
            "true_label": true_label,
            "bert_pred_raw": pred_raw,
            "bert_confidence_raw": raw[pred_raw],
            "bert_margin_raw": margin(raw),
            "bert_entropy_raw": entropy(raw),
            "bert_scores_raw": json.dumps(raw),
            "bert_pred_boosted": pred_boosted,
            "bert_confidence_boosted": boosted[pred_boosted],
            "bert_margin_boosted": margin(boosted),
            "bert_entropy_boosted": entropy(boosted),
            "bert_scores_boosted": json.dumps(boosted),
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
    print(f"BERT raw accuracy on val set:     {(out_df['bert_pred_raw'] == out_df['true_label']).mean():.3f}")
    print(f"BERT boosted accuracy on val set: {(out_df['bert_pred_boosted'] == out_df['true_label']).mean():.3f}")
    
    overall_time = time.perf_counter() - overall_start
    print(f"\nTotal script time: {overall_time:.2f} seconds")

if __name__ == "__main__":
    main()
