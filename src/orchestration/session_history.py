"""
Keeps track of interactions in Streamlit session state (for showing
recent history in the UI later) and optionally logs each one to CSV.

Takes the decision dict from decide_emotion() now instead of just
grabbing bilstm_result and doing its own thing. Used to log bilstm AND
bert as two separate rows with two separate emotion/confidence values,
which doesn't match what the decision engine actually picked and made
"Total Interactions" count double. One entry per interaction now, using
whatever decide_emotion() decided - same value that generate_response()
used, so history actually matches what the user saw on screen.

To figure out which model's scores to use for the mixed-emotion check
(and which cleaned_text to log to CSV), we look at decision["reason"] -
bert_unavailable is the only branch where bilstm is the one that
actually drove the pick, every other branch in decide_emotion() goes
with bert. Using the reason code for this instead of re-guessing from
scratch, since that's exactly why decision_engine.py keeps those as
machine-readable strings.

Uses detect_mixed_emotions() from mixed_emotion.py instead of writing
our own threshold check again - no reason to have two different mixed-
emotion implementations that could drift apart.

CSV logging reuses the existing log_interaction() from csv_logger.py
as-is (already handles the logs/ folder, IST timestamps, dedup - no
reason to rewrite any of that). Logs the decision's emotion/confidence
and whichever model's cleaned_text drove it, so the CSV row matches what
actually generated the AI response instead of always being bilstm's
guess.
"""
from datetime import datetime

import streamlit as st

from src.inference.mixed_emotion import detect_mixed_emotions
from src.orchestration.decision_engine import REASON_BERT_UNAVAILABLE
from src.persistence.csv_logger import log_interaction


def init_session_history() -> None:
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []


def _emotion_label(scores: dict, primary_emotion: str) -> str:
    mixed = detect_mixed_emotions(scores)
    if not mixed["is_mixed"]:
        return primary_emotion
    secondary = " + ".join(e["emotion"] for e in mixed["secondary_emotions"])
    return f"{primary_emotion} + {secondary}"


def _driving_result(decision: dict, bilstm_result: dict, bert_result: dict | None) -> dict:
    """Which model's raw result actually drove decision['emotion']."""
    if bert_result is None or decision["reason"] == REASON_BERT_UNAVAILABLE:
        return bilstm_result
    return bert_result


def record_interaction(
    field: str,
    problem: str,
    decision: dict,
    ai_response: str,
    bilstm_result: dict,
    bert_result: dict = None,
    save_to_csv: bool = True,
) -> None:
    init_session_history()

    driving = _driving_result(decision, bilstm_result, bert_result)

    st.session_state.emotion_history.append({
        "timestamp": datetime.now(),
        "field": field,
        "problem": problem,
        "emotion": _emotion_label(driving["scores"], decision["emotion"]),
        "confidence": decision["confidence"],
        "trust_level": decision["trust_level"],
        "reason": decision["reason"],
        "driving_model": "bilstm" if driving is bilstm_result else "bert",
        "ai_response": ai_response,
        "all_scores": driving["scores"],
        "bilstm_result": bilstm_result,
        "bert_result": bert_result,
    })

    if save_to_csv:
        log_interaction(
            text=driving["cleaned_text"],
            emotion=decision["emotion"],
            confidence=decision["confidence"],
            response=ai_response,
            field=field,
        )
