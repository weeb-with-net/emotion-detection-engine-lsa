"""
Loads the fine-tuned BERT model and runs inference. Exposes both the raw
softmax output (predict_raw) and the class-weighted + keyword-adjusted
final output (predict), per the T3 spec.
"""
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.inference.text_preprocessing import EMOTION_KEYWORDS, clean_for_bert
from src.preprocessing.label_mapping import TARGET_CLASSES

MODEL_PATH = Path("models/bert_emotion_model_final")

# same deal as bilstm_predictor.py - models/ is gitignored, so a fresh
# deploy clone won't have this folder at all. from_pretrained() can take
# a HF Hub repo id directly instead of a local path, so this needs way
# less code than BiLSTM's per-file download. HF_MODEL_REPO_ID unset ->
# same local-path behavior as before, same eventual error if it's
# missing and there's nowhere else to get it from.
HF_SUBFOLDER = "bert_emotion_model_final"

# Order matches TARGET_CLASSES: Bored, Confident, Confused, Curious, Frustrated.
CLASS_WEIGHTS = np.array([1.2, 1.8, 0.6, 1.0, 1.4])

CONFIDENCE_BOOST = 2.5
CONFUSION_BOOST = 2.0


class BERTPredictor:
    def __init__(self, model_path=MODEL_PATH):
        hf_model_repo_id = os.getenv("HF_MODEL_REPO_ID")
        if model_path.exists():
            source, kwargs = model_path, {}
        elif hf_model_repo_id:
            source, kwargs = hf_model_repo_id, {"subfolder": HF_SUBFOLDER}    
        else:
            source, kwargs = model_path, {}  # same not-found error as before

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs)
        self.model.to(self.device)
        self.model.eval()

    def predict_raw(self, text: str) -> dict:
        cleaned = clean_for_bert(text)
        inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        return {cls: float(p) for cls, p in zip(TARGET_CLASSES, probs)}

    def predict(self, text: str) -> dict:
        raw = self.predict_raw(text)
        probs = np.array([raw[cls] for cls in TARGET_CLASSES])

        weighted = probs * CLASS_WEIGHTS
        weighted = weighted / weighted.sum()

        text_lower = text.lower()
        confident_idx = TARGET_CLASSES.index("Confident")
        confused_idx = TARGET_CLASSES.index("Confused")

        if any(kw in text_lower for kw in EMOTION_KEYWORDS["Confident"]):
            weighted[confident_idx] *= CONFIDENCE_BOOST
        elif any(kw in text_lower for kw in EMOTION_KEYWORDS["Confused"]):
            weighted[confused_idx] *= CONFUSION_BOOST

        weighted = weighted / weighted.sum()
        return {cls: float(p) for cls, p in zip(TARGET_CLASSES, weighted)}
