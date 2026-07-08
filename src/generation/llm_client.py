"""
Picks which LLM provider to use based on LLM_PROVIDER env var, so
response_generator.py doesn't need to know or care which one is active.
 
Defaults to "gemini" (the main provider for this project) if the env
var isn't set, so nothing breaks for anyone who clones the repo without
knowing about this. LLM_PROVIDER=openrouter is just a backup option for
when Gemini auth is acting up (see gemini_client.py), not a real switch
in the project's architecture.
 
call_openrouter only gets imported inside the if-branch, not at the top
of the file. Reason: openrouter_client.py sets up its API client as soon
as it's imported, and that crashes if OPENROUTER_API_KEY isn't set. If
we imported it at the top always, then even people just using the
default Gemini setup (no OpenRouter key at all) would get a crash. So
it only imports when actually needed.
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