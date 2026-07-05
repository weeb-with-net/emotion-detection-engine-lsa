"""
Builds the unified, single-label emotion dataset from the three approved
raw sources (GoEmotions, EmpatheticDialogues, ISEAR) and writes:

    data/processed/unified_dataset.csv        -- columns: text, emotion, source_dataset
    data/processed/label_mapping_report.txt   -- full log of every mapping decision + counts

Run from the project root:
    python scripts/build_dataset.py

Prerequisite: scripts/download_datasets.py must be present in
data/raw/. All mapping rules and rationale live in
src/preprocessing/label_mapping.py -- edit there, not here, if you want
to change any decision.
"""

import os
import sys
from pathlib import Path

# Allow `python scripts/build_dataset.py` to find the src/ package when
# run from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.dataset_builder import build_unified_dataset  # noqa: E402

OUTPUT_DIR = Path("data/processed")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    unified_df, report_text = build_unified_dataset()

    csv_path = OUTPUT_DIR / "unified_dataset.csv"
    unified_df.to_csv(csv_path, index=False)

    report_path = OUTPUT_DIR / "label_mapping_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"\nSaved unified dataset to: {csv_path}")
    print(f"Saved mapping report to : {report_path}")


if __name__ == "__main__":
    main()
