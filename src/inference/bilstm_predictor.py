"""
Loads the trained BiLSTM model + tokenizer and runs inference.
Returns raw model probabilities only — keyword boosting (see
src/inference/text_preprocessing.py) is applied as a separate step.

models/ is gitignored (weights are big binaries, not meant to live in
git) - a fresh clone (like whatever a deploy platform does) just has
the .gitkeep placeholders, no actual weights. If HF_MODEL_REPO_ID is
set, pull the two files from there first. If it's not set, this just
does exactly what it did before - falls through to the same
FileNotFoundError from load_model(), no behavior change for local dev.
"""
import os
import pickle
from pathlib import Path

from tensorflow.keras.models import load_model

from src.inference.text_preprocessing import prepare_bilstm_input
from src.preprocessing.label_mapping import TARGET_CLASSES

MODEL_PATH = Path("models/bilstm/model.keras")
TOKENIZER_PATH = Path("models/bilstm/tokenizer.pickle")



def _resolve_path(local_path: Path, hf_filename: str) -> Path:
    hf_model_repo_id = os.getenv("HF_MODEL_REPO_ID")

    if local_path.exists() or not hf_model_repo_id:
        return local_path
    from huggingface_hub import hf_hub_download
    return Path(
        hf_hub_download(
            repo_id=hf_model_repo_id,
            filename=hf_filename,
        )
    )


class BiLSTMPredictor:
    def __init__(self, model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH):
        model_path = _resolve_path(model_path, "bilstm/model.keras")
        tokenizer_path = _resolve_path(tokenizer_path, "bilstm/tokenizer.pickle")

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
