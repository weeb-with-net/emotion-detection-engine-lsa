"""
Wrapper for calling an LLM through OpenRouter (OpenAI-compatible API).
Same shape as gemini_client.py - no prompt building, no fallback here,
just the actual API call.
 
This is only a backup provider for when Gemini auth is being weird (see
gemini_client.py), not the main provider for this project.
 
Using a free-tier model (tencent/hy3:free) through a separate OpenRouter
account made for this project, so testing doesn't eat into personal
API credits since multiple people (graders etc) might run this.
 
Loads .env itself so this file works even if nothing else loaded it
first.
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def call_openrouter(prompt: str) -> str:
    response = _client.chat.completions.create(
        model="tencent/hy3:free",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
