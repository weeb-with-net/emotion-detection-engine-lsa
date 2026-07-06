"""
Demo for src/generation/prompt_builder.py. Pure function, no real
predictor or Streamlit needed. field_problem_capture.py isn't exercised
here since it needs an actual Streamlit run — verify that one with
`streamlit run` against a throwaway page instead.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.prompt_builder import build_gemini_prompt

if __name__ == "__main__":
    prompt = build_gemini_prompt(
        field="Computer Science",
        problem="I don't understand how recursion actually terminates.",
        emotion="Confused",
        confidence=0.82,
    )
    print(prompt)
