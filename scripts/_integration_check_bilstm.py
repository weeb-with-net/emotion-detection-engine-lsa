"""
Internal worker for verify_model_integration.py -- NOT meant to be run
directly. Loads the BiLSTM stack (TensorFlow) and runs one prediction,
printing the result as JSON to stdout.

Kept in a separate process from any PyTorch/transformers code: importing
tensorflow and transformers (which pulls in torch) in the SAME process
caused a hard segfault when this was first tested (verified directly --
reproduced even if neither library's model was ever touched, just from
both being imported). This is a native-library conflict between the two
frameworks, not a bug in this project's code.

Re-tested since then (see test_isolation_minimal.py / test_isolation_
realistic.py at the repo root) on Windows, Python 3.12.10, tensorflow-
cpu 2.16.1, torch 2.11.0+cu128, transformers 4.44.2 - both import-only
and full load+predict passed 3/3 runs with no crash in-process. So the
original finding stands as something that DID happen on some earlier
combination of versions/environment, but isn't currently reproducing
with these pinned versions on this OS. Not proof it's gone for good on
every platform though - see the deployment checklist in
DEPLOYMENT_NOTES.md before trusting this same in-process setup on a
different OS (e.g. the Linux container Hugging Face Spaces uses).
"""


import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # keep stdout clean for JSON parsing

import numpy as np
import tensorflow as tf

from src.preprocessing.text_cleaning import clean_text
from src.preprocessing.tokenization import texts_to_padded

BILSTM_DIR = Path("models/bilstm")
SAMPLE_TEXT = sys.argv[1] if len(sys.argv) > 1 else "I don't understand this concept at all."


def main() -> None:
    try:
        model = tf.keras.models.load_model(BILSTM_DIR / "model.keras", compile=False)
        with open(BILSTM_DIR / "tokenizer.pickle", "rb") as f:
            tokenizer = pickle.load(f)
        label_order = json.loads((BILSTM_DIR / "label_encoder.json").read_text())["classes_in_order"]

        cleaned = clean_text(SAMPLE_TEXT)
        X = texts_to_padded(tokenizer, [cleaned])
        probs = model.predict(X, verbose=0)[0]
        predicted_idx = int(np.argmax(probs))

        result = {
            "status": "ok",
            "predicted_class": label_order[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "label_order": label_order,
        }
    except Exception as e:
        result = {"status": "fail", "error": f"{type(e).__name__}: {e}"}

    # Print ONLY the JSON on the last line -- TF/Keras write assorted
    # log lines to stdout before this that we don't control.
    print("RESULT_JSON:" + json.dumps(result))


if __name__ == "__main__":
    main()
