"""
Wrapper for calling the Gemini API. Doesn't build the prompt itself
(that's in prompt_builder.py) and doesn't handle fallback (that's in
response_generator.py) - keeping it separate so it's easy to test/swap.
 
Using the old google-generativeai package here even though it's
deprecated now, since it was already set up in Epic 1 and switching
SDKs mid-project isn't worth the risk this close to deadline.
 
NOTE: Gemini API key keeps throwing 401 ACCESS_TOKEN_TYPE_UNSUPPORTED.
Pretty sure this isn't our bug - Google recently changed how API keys
work (old AIza keys -> new AQ keys) and a bunch of people are hitting
the same error on Google's own dev forum. Tested the same setup with
OpenRouter + the official OpenAI SDK and that worked fine, so it seems
to be a Google-side issue with the new key format, not something wrong
in this file. Either way, response_generator.py already falls back to
the template responses if this call fails, so the app still works.
 
Loads .env itself so this file works even if nothing else loaded it
first.
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
