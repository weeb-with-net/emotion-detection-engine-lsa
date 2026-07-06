"""
Inference-time preprocessing: cleans raw student input for both models
and applies keyword-based boosting to the resulting emotion probabilities.
"""
import re

import numpy as np

from src.preprocessing.label_mapping import TARGET_CLASSES
from src.preprocessing.text_cleaning import clean_text

_MULTI_WHITESPACE = re.compile(r"\s+")

# Explicit single-word emotion names get the highest weight; everything
# else is a supporting phrase. Starting point adapted from the college
# reference implementation (Activity 3.1) for our 5 target classes.
EMOTION_KEYWORDS = {
    "Frustrated": ["frustrated", "frustrating", "annoying", "angry", "hate",
                   "difficult", "stuck", "wrong answer", "keep getting",
                   "unnecessarily complicated", "tried everything", "give up"],
    "Curious": ["curious", "wonder", "interested", "want to know", "explore",
                "what happens if", "how does", "why does", "intuition",
                "tell me more", "could we"],
    "Confident": ["confident", "easy", "amazing", "great", "excellent", "good",
                  "awesome", "perfect", "solved", "got it", "clear now",
                  "finally", "understand clearly", "makes sense now"],
    "Bored": ["bored", "boring", "tired", "repetitive", "dull", "not engaging",
              "didn't feel engaging", "not interesting", "too basic",
              "losing interest", "same thing again"],
    "Confused": ["confused", "lost", "unclear", "don't understand",
                 "doesn't make sense", "not fully confident", "missing",
                 "incomplete", "unsure", "no idea"],
}

# Words in this set count as an explicit emotion mention (10x weight);
# everything else in EMOTION_KEYWORDS is a supporting phrase (2x weight).
_EXPLICIT_WORDS = {"frustrated", "frustrating", "curious", "confident",
                    "bored", "boring", "confused"}


def clean_for_bilstm(text: str) -> str:
    return clean_text(text)


def clean_for_bert(text: str) -> str:
    """Light clean: BERT's own tokenizer handles casing/punctuation."""
    text = str(text) if text is not None else ""
    return _MULTI_WHITESPACE.sub(" ", text).strip()


def prepare_bilstm_input(text: str, tokenizer) -> np.ndarray:
    from src.preprocessing.tokenization import texts_to_padded  # TF import, kept local

    cleaned = clean_for_bilstm(text)
    return texts_to_padded(tokenizer, [cleaned])


def score_keywords(text: str) -> dict:
    """Score each emotion by keyword presence in the (lowercased) text."""
    text_lower = text.lower()
    scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                score += 10 if keyword in _EXPLICIT_WORDS else 2
        scores[emotion] = score
    return scores


def boost_probabilities(probs: np.ndarray, keyword_scores: dict) -> np.ndarray:
    """
    Boost the model's probability distribution using keyword scores, then
    renormalize. `probs` must be ordered per TARGET_CLASSES.
    """
    probs = np.array(probs, dtype=float)
    max_score = max(keyword_scores.values())

    if max_score > 0:
        winning_emotions = [e for e, s in keyword_scores.items() if s == max_score]
        for i, emotion in enumerate(TARGET_CLASSES):
            if emotion in winning_emotions:
                probs[i] *= (1 + max_score * 3.0)
            elif max_score >= 5:
                probs[i] *= 0.01

    return probs / probs.sum()
