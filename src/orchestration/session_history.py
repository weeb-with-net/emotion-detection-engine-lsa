"""
Keeps track of interactions in Streamlit session state (for showing
recent history in the UI later) and optionally logs each one to CSV.

Uses detect_mixed_emotions() from mixed_emotion.py instead of writing
our own threshold check again - no reason to have two different mixed-
emotion implementations that could drift apart.

CSV logging reuses the existing log_interaction() from csv_logger.py
as-is (already handles the logs/ folder, IST timestamps, dedup - no
reason to rewrite any of that). Only logs ONE row per interaction using
whichever result actually drove the AI response (BiLSTM for now, see
analysis_pipeline.py) - this file is basically training data
("continuous learning" per the task name), so it stays one clean label
per row instead of a "Confused + Frustrated" mixed string. The mixed-
emotion label is only for the in-app session history display, not CSV.
"""
from datetime import datetime

import streamlit as st

from src.inference.mixed_emotion import detect_mixed_emotions
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


def record_interaction(
    field: str,
    problem: str,
    bilstm_result: dict,
    ai_response: str,
    bert_result: dict = None,
    save_to_csv: bool = True,
) -> None:
    init_session_history()

    st.session_state.emotion_history.append({
        "timestamp": datetime.now(),
        "field": field,
        "problem": problem,
        "emotion": _emotion_label(bilstm_result["scores"], bilstm_result["emotion"]),
        "confidence": bilstm_result["confidence"],
        "ai_response": ai_response,
        "all_scores": bilstm_result["scores"],
        "model": "BiLSTM",
    })

    if bert_result:
        st.session_state.emotion_history.append({
            "timestamp": datetime.now(),
            "field": field,
            "problem": problem,
            "emotion": _emotion_label(bert_result["scores"], bert_result["emotion"]),
            "confidence": bert_result["confidence"],
            "ai_response": ai_response,
            "all_scores": bert_result["scores"],
            "model": "BERT",
        })

    if save_to_csv:
        log_interaction(
            text=bilstm_result["cleaned_text"],
            emotion=bilstm_result["emotion"],
            confidence=bilstm_result["confidence"],
            response=ai_response,
            field=field,
        )
