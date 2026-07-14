"""
Main function for generating the AI response: builds the prompt, calls
whichever LLM provider is active (see llm_client.py), and falls back to
the template responses if anything goes wrong. Disabled / provider down
/ error - all three cases just fall back to templates, matches what the
T2 doc actually wants (the T1 mockup had a different generic error
message but that got replaced by this).
"""
import streamlit as st

from src.generation.llm_client import generate_ai_response
from src.generation.prompt_builder import build_gemini_prompt
from src.generation.template_fallback import get_fallback_response


def generate_response(field: str, problem: str, emotion: str, confidence: float, enabled: bool = True) -> str:
    if not enabled:
        return get_fallback_response(emotion)

    try:
        prompt = build_gemini_prompt(field, problem, emotion, confidence)
        return generate_ai_response(prompt)
    except Exception as e:
        # TEMP DEBUG - remove once the real OpenRouter failure is found.
        # Still falls back gracefully either way, this just stops
        # hiding what actually went wrong.
        st.error(f"AI call failed, using fallback. {type(e).__name__}: {e}")
        return get_fallback_response(emotion)
