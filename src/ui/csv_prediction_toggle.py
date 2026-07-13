"""
Just the "CSV-based prediction" checkbox. The actual BEHAVIOR this
implies (predicting from saved examples instead of running BiLSTM/BERT)
is intentionally NOT wired up - there's no existing lookup/similarity
pipeline anywhere in the project to hook this into, and the mockup
doesn't specify how a match would even be chosen (same field? closest
text? most recent?). Building that would be a new inference pipeline,
not a UI control - see EPIC5_POLISH_BACKLOG.md.

Disabled on purpose, not just left clickable-but-inert - a checkbox you
can check with an info message that then does nothing reads as broken,
not as deferred. Disabled + labeled "planned feature" is honest about
what's actually there.
"""
import streamlit as st


def capture_csv_prediction_toggle() -> bool:
    use_csv_prediction = st.checkbox(
        "CSV-based prediction (planned feature)",
        value=False,
        disabled=True,
    )
    st.caption("Uses previously saved examples for retrieval-assisted prediction. Not implemented in this project version.")

    return use_csv_prediction
