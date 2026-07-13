"""
Just the "show analysis details" checkbox. Controls whether the Model
Comparison + How I Decided sections (built in T2) show up at all, or
just the plain Decision + AI Guidance. Returns a bool, nothing else -
same one-control-per-file pattern as ai_toggle.py/save_data_toggle.py.
"""
import streamlit as st


def capture_show_details_toggle() -> bool:
    return st.checkbox("Show analysis details", value=False)
