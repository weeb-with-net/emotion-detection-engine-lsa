"""
Predefined empathetic responses used when Gemini is disabled, unavailable,
or errors out (see response_generator.py). emoji/action are included for
a future UI story to display alongside the response text; only 'response'
is consumed for now.
"""

EMOTION_RESPONSES = {
    "Confused": {
        "emoji": "😕",
        "response": "I see you might be confused. Let me break this down step-by-step...",
        "action": "Show detailed explanation",
    },
    "Frustrated": {
        "emoji": "😤",
        "response": "I understand this is frustrating! Let's try a simpler approach...",
        "action": "Suggest alternative learning path",
    },
    "Confident": {
        "emoji": "💪",
        "response": "Great! You're making excellent progress! Ready for the next challenge?",
        "action": "Suggest advanced content",
    },
    "Bored": {
        "emoji": "😐",
        "response": "Let's make this more engaging. Here are some interactive exercises...",
        "action": "Show interactive content",
    },
    "Curious": {
        "emoji": "🤔",
        "response": "Excellent question! Here's more in-depth information...",
        "action": "Provide research papers & advanced materials",
    },
}


def get_fallback_response(emotion: str) -> str:
    return EMOTION_RESPONSES[emotion]["response"]
