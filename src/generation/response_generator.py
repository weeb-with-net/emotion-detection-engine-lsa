"""
Orchestrates response generation: build the prompt, call the active LLM
provider (see llm_client.py), and fall back to a predefined template on
any failure. Disabled, provider-unavailable, and exception-during-call
all land on the same template fallback per T2's spec — this supersedes
an earlier mockup (T1) that returned a generic "AI response unavailable"
string on error instead.
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
