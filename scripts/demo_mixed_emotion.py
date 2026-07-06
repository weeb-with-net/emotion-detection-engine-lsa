"""
Sanity check for src/inference/mixed_emotion.py.

Runs a few handcrafted emotion score dictionaries to verify:
- Single emotion detection
- Mixed emotion detection
- Inclusive 15% threshold (>= 0.15)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.mixed_emotion import detect_mixed_emotions

CASES = [
    ("Clean single emotion", {
        "Bored": 0.05, "Confident": 0.03, "Confused": 0.08,
        "Curious": 0.04, "Frustrated": 0.80,
    }),
    ("Two above threshold", {
        "Bored": 0.05, "Confident": 0.05, "Confused": 0.35,
        "Curious": 0.05, "Frustrated": 0.50,
    }),
    ("Three above threshold", {
        "Bored": 0.05, "Confident": 0.05, "Confused": 0.30,
        "Curious": 0.30, "Frustrated": 0.30,
    }),
    ("Exact boundary (0.15)", {
        "Bored": 0.05, "Confident": 0.05, "Confused": 0.15,
        "Curious": 0.15, "Frustrated": 0.60,
    }),
]

if __name__ == "__main__":
    for label, scores in CASES:
        result = detect_mixed_emotions(scores)
        print(f"\n{label}: {scores}")
        print(f"  Primary: {result['primary_emotion']} ({result['primary_confidence']:.2f})")
        print(f"  Is mixed: {result['is_mixed']}")
        print(f"  Secondary: {result['secondary_emotions']}")
