"""
Central configuration for mapping dataset-specific emotion labels
to the project's five target classes.

Modify the mappings here and rebuild the dataset if needed.
"""

TARGET_CLASSES = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

RANDOM_SEED = 42  # kept for any future step that needs determinism


# ---------------------------------------------------------------------------
# GoEmotions (27 emotions + neutral, multi-label, rater-level raw CSVs)
# ---------------------------------------------------------------------------
# GoEmotions is multi-label. Ambiguous mappings are filtered
# during dataset construction.
GOEMOTIONS_MAP = {
    "Confused": ["confusion"],     # exact semantic match
    "Curious": ["curiosity"],      # exact semantic match
    "Confident": [],                # No suitable GoEmotions equivalent.
    "Frustrated": ["annoyance"],   # Uses "annoyance" only.
}

# Candidate labels for boredom keyword matching.
BORED_SEARCH_SOURCE_LABELS = ["neutral", "disappointment"]

# Literal keyword match used to weak-label "Bored" from real text only.
# No synthetic data is generated anywhere in this pipeline.
BORED_KEYWORDS = [
    "bored", "boring", "so dull", "tedious", "nothing to do",
    "same old", "monotonous", "uninteresting", "dragging on",
    "snooze", "yawn", "mind numbing", "mind-numbing",
]


# ---------------------------------------------------------------------------
# EmpatheticDialogues (32 emotions, single label per conversation)
# ---------------------------------------------------------------------------
# Single-label dataset.
EMPATHETIC_MAP = {
    "Confident": ["confident"],   # exact match
    "Frustrated": ["annoyed"],    # "angry"/"furious" excluded for the
                                   # same intensity reason as GoEmotions'
                                   # "anger" above.
}


# ---------------------------------------------------------------------------
# ISEAR (7 emotions: joy, sadness, anger, fear, shame, disgust, guilt)
# ---------------------------------------------------------------------------
# Deliberately empty. None of ISEAR's 7 categories map faithfully to any
# of our 5 target classes without reusing "anger" -- already rejected
# above for the intensity-mismatch reason. Rather than force a bad
# mapping just to use all three datasets, ISEAR is validated as present
# (per Epic 2 T1) but contributes zero rows to the unified dataset. This
# is a documented, inspected decision -- not a missed opportunity.
ISEAR_MAP = {}

# Normalize known label typos.
ISEAR_TYPO_FIXES = {"guit": "guilt"}
