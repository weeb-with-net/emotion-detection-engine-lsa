"""
Detects secondary emotions above a confidence threshold (>=15% by default).
Works on any {emotion: probability} dict — model-agnostic by design, so it
can be called on BiLSTM scores, BERT scores, or a future combined score.
"""

DEFAULT_THRESHOLD = 0.15


def detect_mixed_emotions(scores: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary_emotion, primary_confidence = sorted_emotions[0]

    secondary_emotions = [
        {"emotion": emotion, "confidence": score}
        for emotion, score in sorted_emotions[1:]
        if score >= threshold
    ]

    return {
        "primary_emotion": primary_emotion,
        "primary_confidence": primary_confidence,
        "secondary_emotions": secondary_emotions,
        "is_mixed": len(secondary_emotions) > 0,
    }
