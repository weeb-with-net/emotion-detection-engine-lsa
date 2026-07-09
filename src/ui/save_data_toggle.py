"""
Just the save-to-CSV toggle checkbox. Same pattern as ai_toggle.py -
one control, returns a bool, nothing else.
"""
import streamlit as st


def capture_save_data_toggle() -> bool:
    return st.checkbox("Save to CSV for learning", value=True)
