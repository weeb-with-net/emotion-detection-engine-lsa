"""
Internal worker for verify_model_integration.py -- NOT meant to be run
directly. Loads the BERT stack (PyTorch/transformers) and runs one
prediction, printing the result as JSON to stdout.

Kept in a separate process from TensorFlow for the same reason as
scripts/_integration_check_bilstm.py -- see that file's docstring.
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BERT_DIR = Path("models/bert_emotion_model_final")
SAMPLE_TEXT = sys.argv[1] if len(sys.argv) > 1 else "I don't understand this concept at all."


def main() -> None:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)

        inputs = tokenizer(SAMPLE_TEXT, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=0).numpy()
        predicted_idx = int(probs.argmax())

        id2label = model.config.id2label
        label_order = [id2label[i] for i in range(len(id2label))]

        result = {
            "status": "ok",
            "predicted_class": label_order[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "label_order": label_order,
        }
    except Exception as e:
        result = {"status": "fail", "error": f"{type(e).__name__}: {e}"}

    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
