"""
Verifies that all three Epic 2 datasets are present and loadable.

Run this from the project root, after download_datasets.py:
    python scripts/verify_datasets.py

Prints row counts, column names, and a sample row for each dataset so you
can confirm nothing is corrupted or misplaced before moving on to
preprocessing.
"""

from pathlib import Path

import pandas as pd

DATA_ROOT = Path("data/raw")


def check_goemotions() -> bool:
    print("\n== GoEmotions ==")
    folder = DATA_ROOT / "goemotions"
    files = sorted(folder.glob("goemotions_*.csv"))
    if len(files) != 3:
        print(f"  [FAIL] Expected 3 CSV files, found {len(files)} in {folder}")
        return False

    frames = [pd.read_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    print(f"  [OK] {len(files)} files loaded, {len(combined)} total rows")
    print(f"  Columns: {list(combined.columns)}")
    print(f"  Sample text: {combined.iloc[0]['text']!r}")
    return True


def check_empathetic_dialogues() -> bool:
    print("\n== EmpatheticDialogues ==")
    folder = DATA_ROOT / "empathetic_dialogues" / "empatheticdialogues"
    train_csv = folder / "train.csv"
    if not train_csv.exists():
        print(f"  [FAIL] {train_csv} not found")
        return False

    # This dataset's CSV has a quirky, inconsistent field count in places,
    # so it's read with the python engine and bad lines skipped.
    df = pd.read_csv(train_csv, engine="python", on_bad_lines="skip")
    print(f"  [OK] train.csv loaded, {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample utterance: {df.iloc[0].get('utterance', df.iloc[0, -1])!r}")
    return True


def check_isear() -> bool:
    print("\n== ISEAR ==")
    csv_path = DATA_ROOT / "isear" / "ISEAR.csv"
    if not csv_path.exists():
        print(f"  [FAIL] {csv_path} not found")
        return False
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["emotion", "text", "_unused"]
    )
    # Remove the empty third column
    df = df.drop(columns=["_unused"], errors="ignore")
    print(f"  [OK] ISEAR.csv loaded, {len(df)} rows")
    print(f"  Emotion labels found: {sorted(df['emotion'].unique())}")
    print(f"  Sample text: {df.iloc[0]['text']!r}")
    
    return True

def main() -> None:
    results = {
        "GoEmotions": check_goemotions(),
        "EmpatheticDialogues": check_empathetic_dialogues(),
        "ISEAR": check_isear(),
    }

    print("\n== Summary ==")
    all_ok = True
    for name, ok in results.items():
        status = "OK" if ok else "MISSING/BROKEN"
        print(f"  {name}: {status}")
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll three datasets verified. Ready for preprocessing.")
    else:
        print("\nSome datasets are missing or broken. Fix before proceeding.")


if __name__ == "__main__":
    main()
