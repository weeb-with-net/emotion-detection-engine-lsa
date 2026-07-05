"""
Central definitions for mapping each dataset's native emotion labels to
the project's five target classes:

    Bored, Confident, Confused, Curious, Frustrated

Nothing in this file executes anything -- it's pure configuration -- so
every mapping decision lives in exactly one place and is documented
inline. See data/processed/label_mapping_report.txt (written by
scripts/build_dataset.py) for the actual counts these rules produce on
the real downloaded data.

None of these decisions are final/sacred -- if you disagree with a
mapping, change it here and re-run scripts/build_dataset.py.
"""

TARGET_CLASSES = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]

RANDOM_SEED = 42  # kept for any future step that needs determinism


# ---------------------------------------------------------------------------
# GoEmotions (27 emotions + neutral, multi-label, rater-level raw CSVs)
# ---------------------------------------------------------------------------
# GoEmotions is multi-label: a single comment can have several emotions
# agreed upon simultaneously. Because our downstream classifier is
# single-label (one of 5 classes per text), dataset_builder.py enforces
# mutual exclusivity: a comment is only accepted for a target class if
# NONE of the other mapped source labels below are also agreed-upon for
# that same comment. Comments that hit more than one target class are
# dropped rather than arbitrarily assigned to either one.
GOEMOTIONS_MAP = {
    "Confused": ["confusion"],     # exact semantic match
    "Curious": ["curiosity"],      # exact semantic match
    "Confident": [],                # deliberately empty: GoEmotions' closest
                                    # candidate is "pride", which is
                                    # post-achievement satisfaction, not the
                                    # anticipatory self-assurance "confident"
                                    # implies. Confident is sourced only from
                                    # EmpatheticDialogues (see EMPATHETIC_MAP)
                                    # to avoid diluting the class with a
                                    # semantically different emotion.
    "Frustrated": ["annoyance"],   # "anger" deliberately excluded --
                                    # anger is full-blown rage, much more
                                    # intense than a student stuck on a
                                    # problem. Mixing them in would skew
                                    # the class toward the wrong register.
}

# Lowest-arousal GoEmotions categories, used only as the candidate pool
# for the Bored keyword search (see BORED_KEYWORDS below). Restricting
# the search to these categories avoids pulling in text that already has
# a strong, unrelated emotional signal (e.g. searching inside "joy" rows
# for the word "bored" would mostly find sarcasm, not real boredom).
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
# Single-label at the conversation level, so no mutual-exclusivity
# handling is needed here (unlike GoEmotions).
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

# Known data-quality issue in the public ISEAR mirror: one row has the
# emotion mistyped as "guit" instead of "guilt". Normalized regardless
# of whether ISEAR ends up contributing rows, since it's a correctness
# fix, not a mapping decision.
ISEAR_TYPO_FIXES = {"guit": "guilt"}
