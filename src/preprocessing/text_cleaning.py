"""
Text cleaning utilities for the BiLSTM preprocessing pipeline.

Normalizes text while preserving punctuation that may carry
emotional information.
"""
import re

import contractions

_URL_PATTERN = re.compile(r"http\S+|www\S+")
_MENTION_PATTERN = re.compile(r"@\w+")
_MASK_TOKEN_PATTERN = re.compile(r"\[name\]|\[religion\]")
_REPEATED_BANG = re.compile(r"!{2,}")
_REPEATED_QUESTION = re.compile(r"\?{2,}")
_REPEATED_ELLIPSIS = re.compile(r"\.{4,}")
_NON_ALLOWED_CHARS = re.compile(r"[^a-z0-9\s!?.,']")
_MULTI_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean a text string for tokenization.

    Returns an empty string for missing values.
    """
    if text is None or (isinstance(text, float)):
        return ""

    text = str(text).lower()

    # Restore literal commas.
    text = text.replace("_comma_", ",")

    text = _URL_PATTERN.sub(" ", text)
    text = _MENTION_PATTERN.sub(" ", text)
    text = _MASK_TOKEN_PATTERN.sub(" ", text)  # GoEmotions' masked tokens

    try:
        text = contractions.fix(text)
    except Exception:
        pass    # Ignore malformed text.

    # Limit repeated punctuation.
    text = _REPEATED_BANG.sub("!!!", text)
    text = _REPEATED_QUESTION.sub("???", text)
    text = _REPEATED_ELLIPSIS.sub("...", text)

    # Strip everything else except letters, digits, whitespace, and the emotion-carrying punctuation kept above.
    text = _NON_ALLOWED_CHARS.sub(" ", text)
    text = _MULTI_WHITESPACE.sub(" ", text).strip()

    return text
