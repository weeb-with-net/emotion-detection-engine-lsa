"""
Quick regression check that the active LLM provider (see llm_client.py,
LLM_PROVIDER env var) actually works end to end, without the try/except
in response_generator.py hiding the real error if something breaks.

Started as a one-off diagnostic for the httpx/openai 'proxies' crash
(see requirements.txt's httpx==0.27.2 pin) - keeping it around since
that kind of dependency break can silently resurface after a future
pip install --upgrade, and the app itself would just quietly fall back
to template_fallback.py without telling you why.

Run from the project root:
    python scripts/verify_llm_connectivity.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.generation.llm_client import get_active_provider
from src.generation.prompt_builder import build_gemini_prompt

FIELD = "Physics"
PROBLEM = "how do I use kirchoff's rule in a complex circuit?! I am so fed up, Ughhhhh!"
EMOTION = "Curious"
CONFIDENCE = 0.665847181593709

provider = get_active_provider()
print(f"Active provider (from LLM_PROVIDER env var): {provider}")

prompt = build_gemini_prompt(FIELD, PROBLEM, EMOTION, CONFIDENCE)
print(f"\nCalling {provider} directly, no try/except - if this fails, the full traceback prints below.\n")

if provider == "openrouter":
    from src.generation.openrouter_client import call_openrouter
    response = call_openrouter(prompt)
else:
    from src.generation.gemini_client import call_gemini
    response = call_gemini(prompt)

print("--- SUCCESS - real LLM response, not the fallback template ---")
print(response)
