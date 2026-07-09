"""
Demo for T4: analysis_pipeline.py (now runs both models) + session_history.py.
Session state doesn't fully work outside `streamlit run` (you'll see a
warning about that - it's harmless for this quick logic check, but the
real test is wiring this into test_ui.py like ai_toggle.py needed).

Needs your real BiLSTM + BERT weights to mean anything - same situation
as demo_analysis_pipeline.py from T3.
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

    record_interaction(
        "Computer Science",
        "I don't understand recursion",
        result["bilstm_result"],
        result["ai_response"],
        bert_result=result["bert_result"],
        save_to_csv=True,
    )

    print("\n--- session_state.emotion_history ---")
    for entry in st.session_state.emotion_history:
        print(f"  {entry['model']}: {entry['emotion']} ({entry['confidence']:.1%})")

    print("\nCheck logs/emotion_response_examples.csv - should have one new row.")
