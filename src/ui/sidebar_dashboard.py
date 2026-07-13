"""
Sidebar dashboard - model status, interaction counts, and the last 3
interactions for a quick glance. Just renders off session state and
csv_logger, no logic of its own lives here (same pure-UI approach as
field_problem_capture.py etc).
"""
import streamlit as st

from src.persistence.csv_logger import count_examples


def render_sidebar(bert_ready: bool) -> None:
    with st.sidebar:
        st.header("📊 Dashboard")

        status = "BiLSTM + BERT" if bert_ready else "BiLSTM only (BERT unavailable)"
        st.write(f"Models: {status}")
        st.write(f"Total Interactions: {len(st.session_state.emotion_history)}")
        st.write(f"CSV Examples: {count_examples()}")

        if st.button("Clear History"):
            st.session_state.emotion_history = []
            st.session_state.last_result = None
            st.rerun()

        if st.session_state.emotion_history:
            st.subheader("Recent Sessions")
            recent = st.session_state.emotion_history[-3:]
            for item in reversed(recent):
                st.write(f"• {item['field']}: {item['emotion']} ({item['confidence']:.1%})")
        else:
            st.caption("No interactions yet - submit a problem to see it here.")
