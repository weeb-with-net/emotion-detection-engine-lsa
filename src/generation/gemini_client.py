"""
Thin wrapper around the Gemini SDK. Deliberately does not build the
prompt or handle fallback — those live in prompt_builder.py and
response_generator.py respectively, so this stays swappable/mockable in
isolation and only owns the actual API call.

Uses the deprecated google-generativeai package on purpose (see project
notes) — google-genai migration is deferred as known technical debt,
not planned for this project.

Loads .env itself via load_dotenv() so this module works standalone
without depending on the caller (e.g. app.py) having loaded it first.
"""
import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-2.5-flash")


def call_gemini(prompt: str) -> str:
    response = _model.generate_content(prompt)
    return response.text.strip()
