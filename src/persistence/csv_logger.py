"""
CSV persistence for interaction logging.

- emotion_response_examples.csv: one row per interaction (full history)
- emotion_response_mapping.csv: deduplicated (emotion, response) pairs only
  — full analytics fields already live in the examples file, so this
  stays a plain lookup table.
"""
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

EXAMPLES_PATH = Path("logs/emotion_response_examples.csv")
MAPPING_PATH = Path("logs/emotion_response_mapping.csv")

EXAMPLES_HEADER = ["text", "emotion", "confidence", "response", "field", "timestamp"]
MAPPING_HEADER = ["emotion", "response"]


def _ensure_header(path: Path, header: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def log_interaction(text: str, emotion: str, confidence: float, response: str, field: str) -> None:
    """Append one interaction, then update the emotion->response mapping."""
    _ensure_header(EXAMPLES_PATH, EXAMPLES_HEADER)
    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")

    with open(EXAMPLES_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([text, emotion, confidence, response, field, timestamp])

    _update_mapping(emotion, response)


def _update_mapping(emotion: str, response: str) -> None:
    _ensure_header(MAPPING_PATH, MAPPING_HEADER)

    with open(MAPPING_PATH, "r", newline="", encoding="utf-8") as f:
        existing = {(row["emotion"], row["response"]) for row in csv.DictReader(f)}

    if (emotion, response) not in existing:
        with open(MAPPING_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([emotion, response])


def count_examples() -> int:
    """How many rows are in the examples csv right now - for the sidebar.
    Returns 0 if the file doesn't exist yet (nobody's submitted anything
    this session, or it's a fresh clone with no logs/ folder)."""
    if not EXAMPLES_PATH.exists():
        return 0
    with open(EXAMPLES_PATH, "r", newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))
