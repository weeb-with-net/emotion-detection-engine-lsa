"""
Downloads the three Epic 2 datasets into data/raw/.

Run this from the project root:
    python scripts/download_datasets.py

Datasets:
    1. GoEmotions           -> data/raw/goemotions/
    2. EmpatheticDialogues  -> data/raw/empathetic_dialogues/
    3. ISEAR                -> data/raw/isear/

If a source URL ever goes down, see the notes printed at the end of this
file's docstring in the chat message for a backup source.
"""

import os
import tarfile
import urllib.request
from pathlib import Path

DATA_ROOT = Path("data/raw")

GOEMOTIONS_FILES = [
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv",
    "https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv",
]

EMPATHETIC_DIALOGUES_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"

ISEAR_URL = "https://raw.githubusercontent.com/PoorvaRane/Emotion-Detector/master/ISEAR.csv"


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  [downloading] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  [done] {dest.name} ({size_mb:.2f} MB)")


def download_goemotions() -> None:
    print("\n== GoEmotions ==")
    out_dir = DATA_ROOT / "goemotions"
    for url in GOEMOTIONS_FILES:
        filename = url.split("/")[-1]
        download_file(url, out_dir / filename)


def download_empathetic_dialogues() -> None:
    print("\n== EmpatheticDialogues ==")
    out_dir = DATA_ROOT / "empathetic_dialogues"
    archive_path = out_dir / "empatheticdialogues.tar.gz"
    download_file(EMPATHETIC_DIALOGUES_URL, archive_path)

    # Extract only if the CSVs aren't already there
    expected = out_dir / "empatheticdialogues" / "train.csv"
    if expected.exists():
        print(f"  [skip] already extracted")
        return

    print("  [extracting] empatheticdialogues.tar.gz")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(out_dir)
    print("  [done] extracted to", out_dir / "empatheticdialogues")


def download_isear() -> None:
    print("\n== ISEAR ==")
    out_dir = DATA_ROOT / "isear"
    dest = out_dir / "ISEAR.csv"
    try:
        download_file(ISEAR_URL, dest)
    except Exception as e:
        print(f"  [FAILED] Could not download ISEAR automatically: {e}")
        print("  Manual fallback: download from")
        print("  https://www.kaggle.com/datasets/faisalsanto007/isear-dataset")
        print(f"  and place the CSV at: {dest}")


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    download_goemotions()
    download_empathetic_dialogues()
    download_isear()
    print("\nAll downloads attempted. Run scripts/verify_datasets.py next.")


if __name__ == "__main__":
    main()
