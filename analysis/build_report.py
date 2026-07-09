"""
Combines bilstm_val_predictions.csv + bert_val_predictions.csv and builds
the actual analysis report the decision engine design is based on.

No TF/torch needed here, just pandas/numpy/sklearn - run this after both
run_bilstm_predictions.py and run_bert_predictions.py have finished.

Run this from the project root:
    python analysis/build_report.py
"""
import json
import sys
import time
from pathlib import Path

# so "python analysis/build_report.py" works without needing -m syntax
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing.label_mapping import TARGET_CLASSES

BILSTM_PATH = Path("analysis/bilstm_val_predictions.csv")
BERT_PATH = Path("analysis/bert_val_predictions.csv")
MERGED_OUT = Path("analysis/merged_val_predictions.csv")
REPORT_OUT = Path("analysis/decision_signals_report.txt")


def load_and_merge() -> pd.DataFrame:
    bilstm_df = pd.read_csv(BILSTM_PATH)
    bert_df = pd.read_csv(BERT_PATH)
    # true_label and row_id are in both files, only keep one copy
    bert_df = bert_df.drop(columns=["true_label"])
    merged = bilstm_df.merge(bert_df, on="row_id", how="inner")
    assert len(merged) == len(bilstm_df) == len(bert_df), "row counts don't match, something's off with the split"
    return merged


def section(title: str) -> str:
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}\n"


def confusion_block(df: pd.DataFrame, pred_col: str, model_name: str) -> str:
    out = section(f"Confusion matrix / classification report - {model_name}")
    y_true = df["true_label"]
    y_pred = df[pred_col]
    out += classification_report(y_true, y_pred, labels=TARGET_CLASSES, zero_division=0)
    out += "\n\nConfusion matrix (rows = true, cols = predicted):\n"
    cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
    header = "".join(f"{c:>12}" for c in TARGET_CLASSES)
    out += " " * 14 + header + "\n"
    for true_cls, row in zip(TARGET_CLASSES, cm):
        out += f"{true_cls:>12}  " + "".join(f"{v:>12}" for v in row) + "\n"
    return out


def agreement_block(df: pd.DataFrame) -> str:
    # comparing bilstm vs bert's ACTUAL live output (boosted) since that's
    # what the pipeline uses right now
    agree = df["bilstm_pred"] == df["bert_pred_boosted"]
    out = section("Agreement between BiLSTM and BERT (boosted, live behavior)")
    out += f"Agreement rate: {agree.mean():.3f} ({agree.sum()}/{len(df)} rows)\n\n"

    agree_acc = (df.loc[agree, "bilstm_pred"] == df.loc[agree, "true_label"]).mean()
    disagree_acc_bilstm = (df.loc[~agree, "bilstm_pred"] == df.loc[~agree, "true_label"]).mean()
    disagree_acc_bert = (df.loc[~agree, "bert_pred_boosted"] == df.loc[~agree, "true_label"]).mean()

    out += f"When they AGREE: accuracy = {agree_acc:.3f}  (n={agree.sum()})\n"
    out += f"When they DISAGREE:\n"
    out += f"  BiLSTM was right: {disagree_acc_bilstm:.3f}  (n={(~agree).sum()})\n"
    out += f"  BERT was right:   {disagree_acc_bert:.3f}  (n={(~agree).sum()})\n"

    both_wrong = (~agree) & (df["bilstm_pred"] != df["true_label"]) & (df["bert_pred_boosted"] != df["true_label"])
    out += f"  Neither right:    {both_wrong.sum()} rows\n"
    return out


def disagreement_per_class_block(df: pd.DataFrame) -> str:
    agree = df["bilstm_pred"] == df["bert_pred_boosted"]
    disagree_df = df.loc[~agree]
    out = section("Disagreements broken down by true class")
    out += f"{'True class':<14}{'n':>6}{'BiLSTM right':>16}{'BERT right':>14}{'Neither':>10}\n"
    for cls in TARGET_CLASSES:
        sub = disagree_df[disagree_df["true_label"] == cls]
        if len(sub) == 0:
            out += f"{cls:<14}{0:>6}{'-':>16}{'-':>14}{'-':>10}\n"
            continue
        bilstm_right = (sub["bilstm_pred"] == cls).sum()
        bert_right = (sub["bert_pred_boosted"] == cls).sum()
        neither = len(sub) - bilstm_right - bert_right
        out += f"{cls:<14}{len(sub):>6}{bilstm_right:>16}{bert_right:>14}{neither:>10}\n"
    return out


def margin_entropy_block(df: pd.DataFrame) -> str:
    out = section("Margin / entropy: correct vs incorrect predictions")
    specs = [
        ("BiLSTM", "bilstm_pred", "bilstm_margin", "bilstm_entropy"),
        ("BERT raw", "bert_pred_raw", "bert_margin_raw", "bert_entropy_raw"),
        ("BERT boosted", "bert_pred_boosted", "bert_margin_boosted", "bert_entropy_boosted"),
    ]
    for name, pred_col, margin_col, entropy_col in specs:
        correct = df[pred_col] == df["true_label"]
        out += f"\n{name}:\n"
        out += f"  correct   (n={correct.sum():>4}): margin mean={df.loc[correct, margin_col].mean():.3f}  entropy mean={df.loc[correct, entropy_col].mean():.3f}\n"
        out += f"  incorrect (n={(~correct).sum():>4}): margin mean={df.loc[~correct, margin_col].mean():.3f}  entropy mean={df.loc[~correct, entropy_col].mean():.3f}\n"
    return out


def calibration_block(df: pd.DataFrame) -> str:
    # coarse bins since val set is small (~770 rows) - fine bins would be
    # mostly noise, especially for Bored with only 2 samples total
    bins = [0.0, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["0.0-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

    out = section("Calibration: does confidence X% mean X% correct?")
    specs = [
        ("BiLSTM", "bilstm_pred", "bilstm_confidence"),
        ("BERT raw", "bert_pred_raw", "bert_confidence_raw"),
        ("BERT boosted", "bert_pred_boosted", "bert_confidence_boosted"),
    ]
    for name, pred_col, conf_col in specs:
        out += f"\n{name}:\n"
        out += f"  {'bucket':<10}{'n':>6}{'avg conf':>12}{'actual acc':>14}\n"
        correct = (df[pred_col] == df["true_label"]).astype(int)
        bucket_idx = pd.cut(df[conf_col], bins=bins, labels=bin_labels, right=False)
        for label in bin_labels:
            mask = bucket_idx == label
            n = mask.sum()
            if n == 0:
                out += f"  {label:<10}{0:>6}{'-':>12}{'-':>14}\n"
                continue
            avg_conf = df.loc[mask, conf_col].mean()
            actual_acc = correct[mask].mean()
            flag = "  <- low n, treat with caution" if n < 15 else ""
            out += f"  {label:<10}{n:>6}{avg_conf:>12.3f}{actual_acc:>14.3f}{flag}\n"
    return out


def fusion_rescue_block(df: pd.DataFrame) -> str:
    # for rows where BOTH top-1 picks (bilstm, bert boosted) are wrong:
    # would averaging bilstm's raw scores + bert's RAW (not boosted) scores
    # have found the correct answer? this is the direct test of whether
    # the fusion fallback is worth keeping
    both_wrong = (df["bilstm_pred"] != df["true_label"]) & (df["bert_pred_boosted"] != df["true_label"])
    sub = df.loc[both_wrong]

    out = section("Fusion rescue test (both top-1 predictions wrong)")
    out += f"Rows where both models' live top-1 pick is wrong: {len(sub)}\n\n"

    if len(sub) == 0:
        out += "No rows to test - both models never wrong at the same time in this val set.\n"
        return out

    rescued = 0
    for _, row in sub.iterrows():
        bilstm_scores = json.loads(row["bilstm_scores"])
        bert_raw_scores = json.loads(row["bert_scores_raw"])
        fused = {cls: (bilstm_scores[cls] + bert_raw_scores[cls]) / 2 for cls in TARGET_CLASSES}
        fused_pred = max(fused, key=fused.get)
        if fused_pred == row["true_label"]:
            rescued += 1

    out += f"Rescued by fusing BiLSTM + BERT-raw distributions: {rescued}/{len(sub)}\n"
    if len(sub) > 0:
        out += f"Rescue rate: {rescued / len(sub):.3f}\n"
    return out


def main() -> None:

    overall_start = time.perf_counter()

    print("Loading and merging prediction files...")
    df = load_and_merge()
    print(f"  {len(df)} rows merged")
    df.to_csv(MERGED_OUT, index=False)
    print(f"  Saved merged csv to {MERGED_OUT}")

    report = section("DECISION ENGINE ANALYSIS REPORT")
    report += f"Validation set size: {len(df)}\n"
    report += f"Per-class support:\n{df['true_label'].value_counts().reindex(TARGET_CLASSES).to_string()}\n"

    report += confusion_block(df, "bilstm_pred", "BiLSTM")
    report += confusion_block(df, "bert_pred_raw", "BERT (raw softmax)")
    report += confusion_block(df, "bert_pred_boosted", "BERT (boosted, live)")
    report += agreement_block(df)
    report += disagreement_per_class_block(df)
    report += margin_entropy_block(df)
    report += calibration_block(df)
    report += fusion_rescue_block(df)

    overall_time = time.perf_counter() - overall_start

    REPORT_OUT.write_text(report)
    print(f"\nFull report saved to {REPORT_OUT}")
    print("\n--- quick preview ---")
    print(report[:2000])
    
    print(f"\nReport generation time: {overall_time:.2f} seconds")

if __name__ == "__main__":
    main()
