"""
Standardized prediction schema — identical shape for BiLSTM and BERT so
downstream consumers (CSV logging, Streamlit, analytics) don't need
per-model handling. Deliberately just the 4 documented fields; mixed-
emotion info stays separate (see src/inference/mixed_emotion.py) and can
be combined by the caller when needed.
"""
from src.inference.text_preprocessing import clean_for_bert, clean_for_bilstm


def build_prediction_schema(scores: dict, cleaned_text: str) -> dict:
    emotion = max(scores, key=scores.get)
    return {
        "emotion": emotion,
        "confidence": scores[emotion],
        "scores": scores,
        "cleaned_text": cleaned_text,
    }


def bilstm_schema(text: str, predictor) -> dict:
    scores = predictor.predict(text)
    return build_prediction_schema(scores, clean_for_bilstm(text))


def bert_schema(text: str, predictor) -> dict:
    scores = predictor.predict(text)
    return build_prediction_schema(scores, clean_for_bert(text))
