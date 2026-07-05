"""
Regex-based text cleaning for the unified emotion dataset.

Design goal: normalize text for the BiLSTM tokenizer while preserving
punctuation that carries emotional signal (repeated "!" or "?", trailing
"..."). Stripping all punctuation would throw away information a
frustrated or excited student's phrasing actually encodes.
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
    Cleans a single string for tokenization. Safe to call with pandas'
    .apply(). Returns an empty string for null/NaN input rather than
    raising, since a handful of source rows can be empty after upstream
    filtering.
    """
    if text is None or (isinstance(text, float)):
        return ""

    text = str(text).lower()

    # EmpatheticDialogues-specific artifact; harmless no-op for the
    # other two sources.
    text = text.replace("_comma_", ",")

    text = _URL_PATTERN.sub(" ", text)
    text = _MENTION_PATTERN.sub(" ", text)
    text = _MASK_TOKEN_PATTERN.sub(" ", text)  # GoEmotions' masked tokens

    try:
        text = contractions.fix(text)
    except Exception:
        # contractions.fix can occasionally choke on unusual unicode;
        # fall back to the un-expanded text rather than failing the
        # whole pipeline over a handful of rows.
        pass

    # Bound (but don't erase) emphasis punctuation -- "!!!!!" and "???"
    # both signal something the model should be able to use, but we cap
    # the run length so the tokenizer doesn't treat "!!" and "!!!!!!!" as
    # totally distinct "words".
    text = _REPEATED_BANG.sub("!!!", text)
    text = _REPEATED_QUESTION.sub("???", text)
    text = _REPEATED_ELLIPSIS.sub("...", text)

    # Strip everything else except letters, digits, whitespace, and the
    # emotion-carrying punctuation kept above.
    text = _NON_ALLOWED_CHARS.sub(" ", text)
    text = _MULTI_WHITESPACE.sub(" ", text).strip()

    return text
