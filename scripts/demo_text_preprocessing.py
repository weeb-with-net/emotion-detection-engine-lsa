"""
Sanity check for src/inference/text_preprocessing.py — no trained model
needed. Feeds a fake uniform probability distribution through the keyword
booster for one sample sentence per emotion.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.text_preprocessing import (
    boost_probabilities,
    clean_for_bert,
    clean_for_bilstm,
    score_keywords,
)
from src.preprocessing.label_mapping import TARGET_CLASSES

SAMPLES = [
    "I'm so bored, this lecture is way too repetitive.",
    "This is amazing, I finally understand clearly!",
    "I don't understand, this doesn't make sense to me.",
    "I'm curious, what happens if we change this variable?",
    "This is so frustrating, I keep getting the wrong answer.",
]

if __name__ == "__main__":
    for text in SAMPLES:
        print(f"\nInput: {text!r}")
        print(f"  BiLSTM clean: {clean_for_bilstm(text)!r}")
        print(f"  BERT clean:   {clean_for_bert(text)!r}")

        uniform_probs = np.full(len(TARGET_CLASSES), 1 / len(TARGET_CLASSES))
        scores = score_keywords(text)
        boosted = boost_probabilities(uniform_probs, scores)

        print(f"  Keyword scores: {scores}")
        for cls, p in zip(TARGET_CLASSES, boosted):
            print(f"    {cls}: {p:.3f}")
