"""
Sanity check for src/inference/bilstm_predictor.py using the real trained
model. Run locally (needs TensorFlow + the model files on disk):
    python scripts/demo_bilstm_predictor.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.bilstm_predictor import BiLSTMPredictor

SAMPLES = [
    "I'm so bored, this lecture is way too repetitive.",
    "This is amazing, I finally understand clearly!",
    "I don't understand, this doesn't make sense to me.",
    "I'm curious, what happens if we change this variable?",
    "This is so frustrating, I keep getting the wrong answer.",
]

if __name__ == "__main__":
    predictor = BiLSTMPredictor()

    for text in SAMPLES:
        probs = predictor.predict(text)
        top = max(probs, key=probs.get)
        print(f"\nInput: {text!r}")
        print(f"  Predicted: {top} ({probs[top]:.3f})")
        for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"    {cls}: {p:.3f}")
