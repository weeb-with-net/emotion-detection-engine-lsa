"""
Builds the text prompt sent to Gemini. Deliberately pure — no API calls,
no Streamlit, no model imports. `emotion`/`confidence` are plain args
rather than a schema dict so this works with BiLSTM, BERT, or any future
combined score without caring where they came from (Epic 4 T2 owns the
actual API call; see src/generation/gemini_client.py once that exists).
"""


def build_gemini_prompt(field: str, problem: str, emotion: str, confidence: float) -> str:
    return f"""You are a helpful learning assistant. A student studying {field} is feeling {emotion} (confidence: {confidence:.1%}) about this problem:

"{problem}"

Provide a clear, supportive response with:
1. Brief acknowledgment of their feeling
2. One specific tip or strategy for {field}
3. One encouraging next step

Use simple, clear language. Keep each point to 1-2 sentences. No markdown formatting."""
