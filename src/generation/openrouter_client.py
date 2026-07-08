"""
Thin wrapper around OpenRouter, via the OpenAI-compatible SDK. Same
shape as gemini_client.py — no prompt-building, no fallback logic, just
the API call — so llm_client.py can treat both providers identically.

Contingency provider only (see llm_client.py) for the current external
Gemini auth rollout issue — not the project's primary/documented
provider. Model pinned to deepseek/deepseek-chat-v3.1, the one verified
working during diagnosis.

Loads .env itself via load_dotenv() so this module works standalone
without depending on the caller having loaded it first.
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
