"""
Sanity check for src/inference/bert_predictor.py using the real fine-tuned
model. Run locally (needs torch + transformers + the model files on disk):
    python scripts/demo_bert_predictor.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.bert_predictor import BERTPredictor

SAMPLES = [
    "I'm so bored, this lecture is way too repetitive.",
    "This is amazing, I finally understand clearly!",
    "I don't understand, this doesn't make sense to me.",
    "I'm curious, what happens if we change this variable?",
    "This is so frustrating, I keep getting the wrong answer.",
]

if __name__ == "__main__":
    predictor = BERTPredictor()

    for text in SAMPLES:
        raw = predictor.predict_raw(text)
        final = predictor.predict(text)

        print(f"\nInput: {text!r}")
        print(f"  Raw top:   {max(raw, key=raw.get)} ({max(raw.values()):.3f})")
        print(f"  Final top: {max(final, key=final.get)} ({max(final.values()):.3f})")
        print("  Raw scores:  ", {k: round(v, 3) for k, v in raw.items()})
        print("  Final scores:", {k: round(v, 3) for k, v in final.items()})
