"""
Demo for analysis_pipeline.py (runs both models + decision engine) +
session_history.py. Updated for the decision-engine version of
record_interaction() - it now takes result["decision"] instead of just
bilstm_result, so this passes the same three pieces app.py will.

Session state doesn't fully work outside `streamlit run` (you'll see a
warning about that - it's harmless for this quick logic check, but the
real test is wiring this into app.py like the other UI pieces needed).

Needs your real BiLSTM + BERT weights to mean anything - same situation
as demo_analysis_pipeline.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.orchestration.analysis_pipeline import run_analysis
from src.orchestration.session_history import record_interaction

if __name__ == "__main__":
    result = run_analysis("Computer Science", "I don't understand recursion", ai_enabled=False)

    print("--- bilstm_result ---")
    print(result["bilstm_result"])
    print("--- bert_result (None if BERT weights aren't loaded) ---")
    print(result["bert_result"])
    print("--- decision ---")
    print(result["decision"])

    record_interaction(
        "Computer Science",
        "I don't understand recursion",
        result["decision"],
        result["ai_response"],
        result["bilstm_result"],
        bert_result=result["bert_result"],
        save_to_csv=True,
    )

    print("\n--- session_state.emotion_history ---")
    for entry in st.session_state.emotion_history:
        print(f"  {entry['driving_model'].upper()}: {entry['emotion']} ({entry['confidence']:.1%}) - trust={entry['trust_level']}")

    print("\nCheck logs/emotion_response_examples.csv - should have one new row.")
