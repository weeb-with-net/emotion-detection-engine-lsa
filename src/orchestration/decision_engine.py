"""
Decides what emotion actually gets sent to generate_response(), and how
much we should trust that pick.

Every number here comes from analysis/decision_signals_report.txt (770
row validation set). Quick recap of why it's built this way, not just
"pick BERT":
- BERT alone: 83% acc. BiLSTM alone: 53%. When they disagree BERT is
  right 79% of the time vs BiLSTM's 9% - so BiLSTM doesn't get to
  override BERT's pick. Tested a bunch of override ideas (per-class
  precision, the Confused/Curious mixup) and all of them lost more than
  they gained.
- BUT agreement between the two models IS a useful signal on its own -
  when they agree, accuracy jumps to 86%. So BiLSTM still earns a role,
  just as a "second opinion" instead of a competitor.
- BERT's own confidence is badly calibrated below ~0.8 - in that zone
  it's basically a coinflip (see calibration table in the report). So
  low bert confidence means LOW trust no matter what bilstm says.
- Fusion (averaging both models' distributions) was tested too - only
  rescued 3/100 rows where both were wrong. Not worth the complexity,
  cut.

This stays a pure function on purpose - no streamlit, no file/network
calls - so it's easy to unit test and easy to swap out later if the
models get retrained.
"""

# from calibration table: bert conf >= 0.8 -> ~88% actual accuracy,
# below that it drops to ~50% (basically a coinflip)
BERT_TRUST_THRESHOLD = 0.8

# reason codes, kept as short machine-friendly strings instead of full
# sentences so logging/ui code can branch on them later if needed
REASON_BERT_UNAVAILABLE = "bert_unavailable"
REASON_LOW_BERT_CONFIDENCE = "low_bert_confidence"
REASON_AGREEMENT_HIGH_CONFIDENCE = "agreement_high_confidence"
REASON_DISAGREEMENT_HIGH_CONFIDENCE = "disagreement_high_confidence"

TRUST_HIGH = "HIGH"
TRUST_MEDIUM = "MEDIUM"
TRUST_LOW = "LOW"


def decide_emotion(bilstm_result: dict, bert_result: dict | None) -> dict:
    """
    bilstm_result / bert_result are the standard schema dicts from
    schema.py ({emotion, confidence, scores, cleaned_text}). bert_result
    can be None if bert isn't available (same graceful-degrade pattern
    as cached_loaders.py).

    Returns {emotion, confidence, trust_level, reason}.

    Branch order matters here - low trust checks come first so they
    always win, even if the models happen to agree.
    """
    # bert missing entirely - fall back to bilstm, low trust since
    # bilstm alone is only ~53% accurate
    if bert_result is None:
        return {
            "emotion": bilstm_result["emotion"],
            "confidence": bilstm_result["confidence"],
            "trust_level": TRUST_LOW,
            "reason": REASON_BERT_UNAVAILABLE,
        }

    # bert is unsure - per the calibration table this is close to a
    # coinflip, so low trust regardless of what bilstm says
    if bert_result["confidence"] < BERT_TRUST_THRESHOLD:
        return {
            "emotion": bert_result["emotion"],
            "confidence": bert_result["confidence"],
            "trust_level": TRUST_LOW,
            "reason": REASON_LOW_BERT_CONFIDENCE,
        }

    # bert is confident and bilstm agrees - best case, highest trust
    if bilstm_result["emotion"] == bert_result["emotion"]:
        return {
            "emotion": bert_result["emotion"],
            "confidence": bert_result["confidence"],
            "trust_level": TRUST_HIGH,
            "reason": REASON_AGREEMENT_HIGH_CONFIDENCE,
        }

    # bert is confident but bilstm disagrees - still go with bert
    # (79% right on disagreements vs bilstm's 9%) but flag it as medium
    # since a disagreement is still a bit of a warning sign
    return {
        "emotion": bert_result["emotion"],
        "confidence": bert_result["confidence"],
        "trust_level": TRUST_MEDIUM,
        "reason": REASON_DISAGREEMENT_HIGH_CONFIDENCE,
    }
