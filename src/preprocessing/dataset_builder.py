"""
Builds the unified emotion dataset from the approved raw datasets.

This module:
- Loads the raw datasets.
- Maps source labels to the project's target classes.
- Removes duplicate samples.
- Produces the unified dataset used for model training.
"""

from pathlib import Path

import pandas as pd

from src.preprocessing.label_mapping import (
    BORED_KEYWORDS,
    BORED_SEARCH_SOURCE_LABELS,
    EMPATHETIC_MAP,
    GOEMOTIONS_MAP,
    ISEAR_MAP,
    ISEAR_TYPO_FIXES,
)

DATA_RAW = Path("data/raw")

GOEMOTIONS_EMOTION_COLUMNS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

AGREEMENT_THRESHOLD = 2  # Minimum annotator agreement required for a label.


# ---------------------------------------------------------------------------
# GoEmotions
# ---------------------------------------------------------------------------
def load_goemotions_raw() -> pd.DataFrame:
    """Loads and concatenates the 3 raw GoEmotions CSVs (rater-level rows)."""
    folder = DATA_RAW / "goemotions"
    files = sorted(folder.glob("goemotions_*.csv"))
    if len(files) != 3:
        raise FileNotFoundError(
            f"Expected 3 GoEmotions CSVs in {folder}, found {len(files)}. "
            "Run scripts/download_datasets.py first."
        )
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def deduplicate_goemotions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse GoEmotions from rater-level annotations into one row per
    comment using the configured agreement threshold.
    """

    n_raw_rows = len(raw_df)
    n_unique_ids = raw_df["id"].nunique()

    grouped = raw_df.groupby("id")
    agg = grouped[GOEMOTIONS_EMOTION_COLUMNS].sum()
    agreed = (agg >= AGREEMENT_THRESHOLD).astype(int)

    text_lookup = raw_df.groupby("id")["text"].first()
    agreed["text"] = text_lookup
    agreed = agreed.reset_index()

    print("\n== GoEmotions deduplication ==")
    print(f"  Raw rater-level rows : {n_raw_rows:,}")
    print(f"  Unique comments (ids): {n_unique_ids:,}")
    print(f"  Avg raters / comment : {n_raw_rows / n_unique_ids:.2f}")
    print(f"  Agreement threshold  : >= {AGREEMENT_THRESHOLD} raters")

    return agreed


def map_goemotions_to_target(agreed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map GoEmotions labels to the target emotion classes.

    Rows matching multiple target classes are excluded.
    """
    rows = []
    claimed_mask = pd.Series(False, index=agreed_df.index)

    # Build a per-row "which target classes does this hit" list first,
    # so we can drop ambiguous rows before assigning anything.
    hits_per_row = pd.DataFrame(index=agreed_df.index)
    for target_class, source_labels in GOEMOTIONS_MAP.items():
        if not source_labels:
            continue
        hit = agreed_df[source_labels].any(axis=1)
        hits_per_row[target_class] = hit

    n_hits = hits_per_row.sum(axis=1)
    unambiguous = n_hits == 1
    ambiguous_count = int((n_hits > 1).sum())

    for target_class in hits_per_row.columns:
        selected = agreed_df[unambiguous & hits_per_row[target_class]]
        for text in selected["text"]:
            rows.append({"text": text, "emotion": target_class, "source_dataset": "GoEmotions"})
        claimed_mask |= (unambiguous & hits_per_row[target_class]).reindex(claimed_mask.index, fill_value=False)

    print("\n== GoEmotions label mapping ==")
    for target_class in hits_per_row.columns:
        count = sum(1 for r in rows if r["emotion"] == target_class)
        print(f"  {target_class:<12}: {count:,} rows (source: {GOEMOTIONS_MAP[target_class]})")
    print(f"  Dropped (ambiguous, hit >1 class): {ambiguous_count:,}")

    mapped_df = pd.DataFrame(rows)
    unclaimed_df = agreed_df[~claimed_mask].copy()
    return mapped_df, unclaimed_df


def weak_label_bored(unclaimed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify boredom-related samples using keyword matching on
    eligible GoEmotions entries.
    """
    pool_mask = unclaimed_df[BORED_SEARCH_SOURCE_LABELS].any(axis=1)
    pool = unclaimed_df[pool_mask]

    keyword_pattern = "|".join(BORED_KEYWORDS)
    text_lower = pool["text"].str.lower()
    keyword_mask = text_lower.str.contains(keyword_pattern, regex=True, na=False)
    bored_rows = pool[keyword_mask]

    print("\n== Bored weak-labeling (GoEmotions neutral/disappointment only) ==")
    print(f"  Candidate pool size      : {len(pool):,}")
    print(f"  Matched boredom keywords : {len(bored_rows):,}")
    if len(bored_rows) < 200:
        print(
            "  [LIMITATION] Bored class is small and derived purely from "
            "literal keyword matches on real text. No synthetic data was "
            "generated to pad this class, per project decision. See "
            "label_mapping_report.txt for full detail."
        )

    return pd.DataFrame({
        "text": bored_rows["text"],
        "emotion": "Bored",
        "source_dataset": "GoEmotions",
    })


# ---------------------------------------------------------------------------
# EmpatheticDialogues
# ---------------------------------------------------------------------------
def load_empathetic_dialogues_raw() -> pd.DataFrame:
    """Loads and concatenates train/valid/test splits (utterance-level rows)."""
    folder = DATA_RAW / "empathetic_dialogues" / "empatheticdialogues"
    frames = []
    for split in ("train.csv", "valid.csv", "test.csv"):
        path = folder / split
        if not path.exists():
            print(f"  [WARNING] {path} not found, skipping this split")
            continue
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No EmpatheticDialogues splits found in {folder}")
    return pd.concat(frames, ignore_index=True)


def deduplicate_empathetic_dialogues(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate conversations while preserving one prompt
    per conversation.
    """
    n_raw_rows = len(raw_df)
    deduped = raw_df.drop_duplicates(subset="conv_id", keep="first")[["conv_id", "context", "prompt"]].copy()

    # The raw text uses "_comma_" as a literal-comma placeholder (a known
    # quirk of this dataset, documented by the original authors).
    deduped["prompt"] = deduped["prompt"].str.replace("_comma_", ",", regex=False)

    print("\n== EmpatheticDialogues deduplication ==")
    print(f"  Raw utterance-level rows : {n_raw_rows:,}")
    print(f"  Unique conversations     : {len(deduped):,}")

    return deduped


def map_empathetic_to_target(deduped_df: pd.DataFrame) -> pd.DataFrame:
    context_lower = deduped_df["context"].str.lower()
    rows = []
    print("\n== EmpatheticDialogues label mapping ==")
    for target_class, source_labels in EMPATHETIC_MAP.items():
        mask = context_lower.isin(source_labels)
        matched = deduped_df[mask]
        print(f"  {target_class:<12}: {len(matched):,} rows (source: {source_labels})")
        for text in matched["prompt"]:
            rows.append({"text": text, "emotion": target_class, "source_dataset": "EmpatheticDialogues"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ISEAR
# ---------------------------------------------------------------------------
def load_isear_raw() -> pd.DataFrame:
    path = DATA_RAW / "isear" / "ISEAR.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_datasets.py first.")
    df = pd.read_csv(path, header=None, names=["emotion", "text"])
    df["emotion"] = df["emotion"].str.strip().str.lower().replace(ISEAR_TYPO_FIXES)
    return df


def map_isear_to_target(df: pd.DataFrame) -> pd.DataFrame:
    print("\n== ISEAR label mapping ==")
    if not ISEAR_MAP:
        print(
            "  ISEAR contributes 0 rows by design: none of its 7 categories "
            "(joy, sadness, anger, fear, shame, disgust, guilt) map "
            "faithfully to Bored/Confident/Confused/Curious/Frustrated "
            "without reusing 'anger', which was rejected for the same "
            "intensity-mismatch reason documented for GoEmotions/ED. "
            "Dataset was validated as present (Epic 2 T1) but is excluded "
            "here as a deliberate, inspected decision."
        )
        return pd.DataFrame(columns=["text", "emotion", "source_dataset"])

    rows = []
    for target_class, source_labels in ISEAR_MAP.items():
        matched = df[df["emotion"].isin(source_labels)]
        for text in matched["text"]:
            rows.append({"text": text, "emotion": target_class, "source_dataset": "ISEAR"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build_unified_dataset() -> tuple[pd.DataFrame, str]:
    """
    Runs the full pipeline and returns (unified_dataframe, report_text).
    The report text is also written to disk by scripts/build_dataset.py.
    """
    report_lines = []

    def log(line: str = ""):
        print(line)
        report_lines.append(line)

    log("=" * 70)
    log("UNIFIED DATASET BUILD REPORT")
    log("=" * 70)

    # --- GoEmotions ---
    ge_raw = load_goemotions_raw()
    ge_agreed = deduplicate_goemotions(ge_raw)
    ge_mapped, ge_unclaimed = map_goemotions_to_target(ge_agreed)
    ge_bored = weak_label_bored(ge_unclaimed)

    # --- EmpatheticDialogues ---
    ed_raw = load_empathetic_dialogues_raw()
    ed_deduped = deduplicate_empathetic_dialogues(ed_raw)
    ed_mapped = map_empathetic_to_target(ed_deduped)

    # --- ISEAR ---
    isear_raw = load_isear_raw()
    isear_mapped = map_isear_to_target(isear_raw)

    unified = pd.concat([ge_mapped, ge_bored, ed_mapped, isear_mapped], ignore_index=True)

    # Global safety net: drop any exact-duplicate text that might have
    # slipped through (e.g. the same comment quoted verbatim in two
    # different subreddits).
    before = len(unified)
    unified = unified.drop_duplicates(subset="text").reset_index(drop=True)
    after = len(unified)

    log("\n" + "=" * 70)
    log("FINAL UNIFIED DATASET")
    log("=" * 70)
    log(f"Rows before final dedup: {before:,}")
    log(f"Rows after final dedup : {after:,}")
    log("\nClass distribution:")
    counts = unified["emotion"].value_counts()
    for cls in ["Bored", "Confident", "Confused", "Curious", "Frustrated"]:
        n = int(counts.get(cls, 0))
        pct = 100 * n / after if after else 0
        log(f"  {cls:<12}: {n:>7,} ({pct:5.1f}%)")

    log("\nSource contribution:")
    for src, n in unified["source_dataset"].value_counts().items():
        log(f"  {src:<20}: {n:,}")

    bored_count = int(counts.get("Bored", 0))
    bored_pct = 100 * bored_count / after if after else 0
    log("\n" + "-" * 70)
    log("LIMITATION -- Bored class")
    log("-" * 70)
    log(
        f"Bored has {bored_count:,} rows ({bored_pct:.1f}% of the dataset), "
        "derived entirely from literal keyword matching on GoEmotions "
        "neutral/disappointment text. No other dataset here has a "
        "boredom-equivalent label, and no synthetic data was generated to "
        "pad this class, per project decision. This class is expected to "
        "be the smallest and most narrowly-worded of the five -- the "
        "model will likely generalize to boredom expressed in similar "
        "phrasing better than to boredom expressed in novel ways. This is "
        "a known, documented limitation, not an implementation bug."
    )
    log("-" * 70)

    return unified, "\n".join(report_lines)
