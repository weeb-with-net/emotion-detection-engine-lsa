"""
Dispatches to whichever LLM provider LLM_PROVIDER selects, so callers
(response_generator.py) don't need to know or care which one is active.
Defaults to "gemini" — the project's documented/primary provider — so
the app behaves identically for anyone who clones the repo without this
env var set. LLM_PROVIDER=openrouter is a documented contingency for the
external Gemini auth issue (see gemini_client.py), not a permanent
architecture change.

call_openrouter is imported lazily (inside the branch, not at module
top) so openrouter_client.py's client init — which raises if
OPENROUTER_API_KEY is unset — never runs unless openrouter is actually
selected. Without this, anyone using the gemini default without an
OpenRouter key would crash on import alone.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from src.generation.gemini_client import call_gemini


def generate_ai_response(prompt: str) -> str:
    provider = os.getenv("LLM_PROVIDER", "gemini")

    if provider == "openrouter":
        from src.generation.openrouter_client import call_openrouter

        return call_openrouter(prompt)
    return call_gemini(prompt)

def get_active_provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini")