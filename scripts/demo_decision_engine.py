"""
Sanity check for src/orchestration/decision_engine.py.

Runs handcrafted bilstm/bert result dicts to check all 4 branches:
- agreement + high bert confidence -> HIGH
- disagreement + high bert confidence -> MEDIUM
- low bert confidence -> LOW (regardless of agreement)
- bert unavailable -> LOW, bilstm fallback
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.decision_engine import decide_emotion


def fake_result(emotion: str, confidence: float) -> dict:
    return {"emotion": emotion, "confidence": confidence, "scores": {}, "cleaned_text": ""}


CASES = [
    (
        "Agreement, high bert confidence",
        fake_result("Frustrated", 0.55),
        fake_result("Frustrated", 0.91),
    ),
    (
        "Disagreement, high bert confidence",
        fake_result("Frustrated", 0.60),
        fake_result("Confident", 0.85),
    ),
    (
        "Low bert confidence, models agree",
        fake_result("Curious", 0.40),
        fake_result("Curious", 0.55),
    ),
    (
        "Low bert confidence, models disagree",
        fake_result("Bored", 0.35),
        fake_result("Curious", 0.62),
    ),
    (
        "Bert unavailable",
        fake_result("Confused", 0.50),
        None,
    ),
]

if __name__ == "__main__":
    for label, bilstm_result, bert_result in CASES:
        decision = decide_emotion(bilstm_result, bert_result)
        print(f"\n{label}")
        print(f"  bilstm: {bilstm_result['emotion']} ({bilstm_result['confidence']:.2f})")
        if bert_result:
            print(f"  bert:   {bert_result['emotion']} ({bert_result['confidence']:.2f})")
        else:
            print("  bert:   unavailable")
        print(f"  -> emotion={decision['emotion']}  confidence={decision['confidence']:.2f}  "
              f"trust={decision['trust_level']}  reason={decision['reason']}")
