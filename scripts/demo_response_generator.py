"""
Demo for src/generation/response_generator.py. The enabled=False (template
fallback) path is fully exercised here with no external dependencies.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.response_generator import generate_response

from src.generation.llm_client import get_active_provider
print(f"Using LLM provider: {get_active_provider()}")

if __name__ == "__main__":
    print("--- Template fallback (enabled=False) ---")
    for emotion in ["Confused", "Frustrated", "Confident", "Bored", "Curious"]:
        print(f"{emotion}: {generate_response('Computer Science', 'sample problem', emotion, 0.8, enabled=False)}")

    print("\n--- Real LLM call (enabled=True) ---")
    print(generate_response(
    "Computer Science",
    "I don't understand recursion",
    "Confused",
    0.82,
    enabled=True
))
