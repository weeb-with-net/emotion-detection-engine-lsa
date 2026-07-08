"""
Main function for generating the AI response: builds the prompt, calls
whichever LLM provider is active (see llm_client.py), and falls back to
the template responses if anything goes wrong. Disabled / provider down
/ error - all three cases just fall back to templates, matches what the
T2 doc actually wants (the T1 mockup had a different generic error
message but that got replaced by this).
"""
from src.generation.llm_client import generate_ai_response
from src.generation.prompt_builder import build_gemini_prompt
from src.generation.template_fallback import get_fallback_response


def generate_response(field: str, problem: str, emotion: str, confidence: float, enabled: bool = True) -> str:
    if not enabled:
        return get_fallback_response(emotion)

    try:
        prompt = build_gemini_prompt(field, problem, emotion, confidence)
        return generate_ai_response(prompt)
    except Exception:
        return get_fallback_response(emotion)
