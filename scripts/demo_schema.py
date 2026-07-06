"""
Demo for src/inference/schema.py. The generic function is tested with
real data. The BiLSTM/BERT wrappers are exercised using a fake predictor
(same .predict() interface as the real ones) since TensorFlow/PyTorch
aren't available in every environment — swap in the real predictors
locally to test against actual model output.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.schema import bert_schema, bilstm_schema, build_prediction_schema


class FakePredictor:
    """Stands in for BiLSTMPredictor / BERTPredictor — same predict(text) interface."""

    def __init__(self, fixed_scores: dict):
        self.fixed_scores = fixed_scores

    def predict(self, text: str) -> dict:
        return self.fixed_scores


if __name__ == "__main__":
    scores = {"Bored": 0.05, "Confident": 0.05, "Confused": 0.6, "Curious": 0.2, "Frustrated": 0.1}
    print("Generic schema:", build_prediction_schema(scores, "example cleaned text"))

    fake_scores = {"Bored": 0.1, "Confident": 0.1, "Confused": 0.1, "Curious": 0.1, "Frustrated": 0.6}
    fake = FakePredictor(fake_scores)
    text = "I'm SO frustrated!!! this doesn't work"

    print("\nBiLSTM schema:", bilstm_schema(text, fake))
    print("\nBERT schema:  ", bert_schema(text, fake))
