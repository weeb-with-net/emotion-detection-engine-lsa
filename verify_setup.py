"""
Epic 1 / T5: Verify Model and Data Directories.

Run this after installing dependencies and setting up .env to confirm
the project is ready for Epic 2/3 work:

    python verify_setup.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = ["GOOGLE_API_KEY"]

# (path, is_directory, required_now)
# required_now=False means "expected later" (e.g. after Epic 2 training) —
# we warn instead of fail so this script stays usable before models exist.
CHECKS = [
    ("data", True, True),
    ("data/emotion_text_dataset.csv", False, False),
    ("data/GoEmotions", True, False),
    ("data/empatheticdialogues", True, False),
    ("models/bltsm", True, True),
    ("models/bert_emotion_model_final", True, True),
    ("logs", True, True),
]


def _is_effectively_empty(path: Path) -> bool:
    if not path.is_dir():
        return False
    contents = [p for p in path.iterdir() if p.name != ".gitkeep"]
    return len(contents) == 0


def main() -> int:
    ok = True
    print("=== Environment variables ===")
    for var in REQUIRED_ENV_VARS:
        val = os.getenv(var)
        if not val or "your_gemini_api_key_here" in val:
            print(f"  [MISSING] {var} is not set in .env")
            ok = False
        else:
            masked = val[:4] + "..." + val[-2:] if len(val) > 8 else "****"
            print(f"  [OK] {var} = {masked}")

    print("\n=== Folder structure ===")
    root = Path(__file__).parent
    for rel_path, is_dir, required_now in CHECKS:
        path = root / rel_path
        exists = path.is_dir() if is_dir else path.is_file()

        if required_now:
            # Required items just need to exist (as a folder or file) at this stage —
            # e.g. "logs/" existing as an empty folder is fine before the app has run.
            if exists:
                print(f"  [OK] {rel_path}")
            else:
                print(f"  [MISSING] {rel_path} (required now)")
                ok = False
        else:
            # Pending items (datasets/model weights from Epic 2) need real content,
            # not just a placeholder .gitkeep.
            has_content = exists and not (is_dir and _is_effectively_empty(path))
            if has_content:
                print(f"  [OK] {rel_path}")
            else:
                print(f"  [PENDING] {rel_path} (needed before Epic 2/3 model loading, not required yet)")

    print("\n=== Result ===")
    if ok:
        print("Core structure and .env look good. You're set for Epic 1.")
    else:
        print("Some required items are missing — fix the items marked [MISSING] above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
