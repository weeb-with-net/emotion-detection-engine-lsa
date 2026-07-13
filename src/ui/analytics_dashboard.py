"""
Analytics dashboard skeleton - T3's job is just the tab structure
(Emotions / Fields / Summary), gated on session history existing. Left
genuinely empty inside each tab on purpose - T4 ("Visualize Scores and
Plotly Charts with Caching") is the story that fills these in, not this
one. Matches the mockup exactly rather than guessing at placeholder
content.
"""
import pandas as pd
import streamlit as st


def render_analytics_dashboard() -> None:
    if not st.session_state.emotion_history:
        return

    st.markdown("---")
    st.header("📈 Learning Analytics")

    df = pd.DataFrame(st.session_state.emotion_history)

    tab1, tab2, tab3 = st.tabs(["Emotions", "Fields", "Summary"])
