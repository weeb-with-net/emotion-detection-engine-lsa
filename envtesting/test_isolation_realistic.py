"""
Realistic test for the TF/PyTorch segfault - loads BOTH real predictors
and runs predict() on each, back to back, in one process. This mirrors
exactly what cached_loaders.py + app.py currently do.

Run from the project root (needs your real model weights in place):
    python test_isolation_realistic.py

If test_isolation_minimal.py passed but this one crashes, the conflict
is triggered by model construction/inference, not import alone - still
means we need isolation, just narrows down where.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAMPLE = "I don't understand this concept at all, can you explain it again?"

print("Step 1: loading BiLSTM predictor (TensorFlow)...")
from src.inference.bilstm_predictor import BiLSTMPredictor
bilstm = BiLSTMPredictor()
print("  OK - BiLSTM loaded")

print("Step 2: loading BERT predictor (PyTorch/transformers)...")
from src.inference.bert_predictor import BERTPredictor
bert = BERTPredictor()
print("  OK - BERT loaded")

print("Step 3: running BiLSTM.predict()...")
bilstm_result = bilstm.predict(SAMPLE)
print(f"  OK - {bilstm_result}")

print("Step 4: running BERT.predict()...")
bert_result = bert.predict(SAMPLE)
print(f"  OK - {bert_result}")

print("\nALL STEPS OK - both models loaded and predicted in the same process without crashing.")
