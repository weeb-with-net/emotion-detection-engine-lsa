import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.persistence.csv_logger import EXAMPLES_PATH, MAPPING_PATH, log_interaction

if __name__ == "__main__":
    # Same (emotion, response) pair logged twice on purpose, to check dedup.
    log_interaction(
        text="I don't understand recursion",
        emotion="Confused",
        confidence=0.78,
        response="Let's break recursion down step by step.",
        field="Computer Science",
    )
    log_interaction(
        text="Recursion makes no sense to me",
        emotion="Confused",
        confidence=0.65,
        response="Let's break recursion down step by step.",
        field="Computer Science",
    )
    log_interaction(
        text="This is amazing, I get it now!",
        emotion="Confident",
        confidence=0.91,
        response="Great work, you've got this!",
        field="Mathematics",
    )

    print(f"--- {EXAMPLES_PATH} ---")
    print(EXAMPLES_PATH.read_text())
    print(f"--- {MAPPING_PATH} (should have 2 rows, not 3 — dedup check) ---")
    print(MAPPING_PATH.read_text())
