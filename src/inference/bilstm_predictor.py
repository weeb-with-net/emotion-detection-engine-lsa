"""
Loads the trained BiLSTM model + tokenizer and runs inference.
Returns raw model probabilities only — keyword boosting (see
src/inference/text_preprocessing.py) is applied as a separate step.
"""
import pickle
from pathlib import Path

from tensorflow.keras.models import load_model

from src.inference.text_preprocessing import prepare_bilstm_input
from src.preprocessing.label_mapping import TARGET_CLASSES

MODEL_PATH = Path("models/bilstm/model.keras")
TOKENIZER_PATH = Path("models/bilstm/tokenizer.pickle")


class BiLSTMPredictor:
    def __init__(self, model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH):
        try:
            self.model = load_model(model_path)
        except Exception:
            # Custom loss (focal loss) isn't needed for inference — skip compiling.
            self.model = load_model(model_path, compile=False)

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

    def predict(self, text: str) -> dict:
        padded = prepare_bilstm_input(text, self.tokenizer)
        probs = self.model.predict(padded, verbose=0)[0]
        return {cls: float(p) for cls, p in zip(TARGET_CLASSES, probs)}
