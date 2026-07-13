"""
Diagnostic only - traces one real LLM call end to end WITHOUT
swallowing the exception, so we see exactly why it's falling back to
the template instead of guessing. Doesn't change any app code -
response_generator.py's try/except still silently falls back in the
real app; this just bypasses that on purpose for one test call so the
real traceback shows up instead of being caught and hidden.

Uses the exact same field/emotion/confidence that were actually logged
for the Kirchhoff's-law input (see logs/emotion_response_examples.csv),
so this tests the real failing case, not a generic "hello world" prompt
that might behave differently.

Run from the project root:
    python test_llm_call_trace.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.generation.llm_client import get_active_provider
from src.generation.prompt_builder import build_gemini_prompt

# exact values from the logged CSV row for this input
FIELD = "Physics"
PROBLEM = "how do I use kirchoff's rule in a complex circuit?! I am so fed up, Ughhhhh!"
EMOTION = "Curious"
CONFIDENCE = 0.665847181593709

provider = get_active_provider()
print(f"Active provider (from LLM_PROVIDER env var): {provider}")

prompt = build_gemini_prompt(FIELD, PROBLEM, EMOTION, CONFIDENCE)
print("\n--- Exact prompt being sent ---")
print(prompt)
print("--- end prompt ---\n")

print(f"Calling {provider} directly, no try/except - if this fails, the full traceback prints below.\n")

if provider == "openrouter":
    from src.generation.openrouter_client import call_openrouter
    response = call_openrouter(prompt)
else:
    from src.generation.gemini_client import call_gemini
    response = call_gemini(prompt)

print("--- SUCCESS - this is the real LLM response, not the fallback template ---")
print(response)
