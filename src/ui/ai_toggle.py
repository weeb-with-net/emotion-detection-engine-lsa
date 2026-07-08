"""
Just the AI response toggle checkbox. Returns a bool, nothing else -
save_data and show_details checkboxes from the mockup are NOT this
file's job (those are T4 / Epic 5), so keeping this to one control.
"""
import streamlit as st


def capture_ai_toggle() -> bool:
    return st.checkbox("Use AI Response (Gemini)", value=True)
