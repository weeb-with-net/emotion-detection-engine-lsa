"""
Ties together the model predictions and the response generation for one
button click. Only calls each predictor once, so whatever gets shown on
screen and whatever gets sent to generate_response() are always the
exact same values - no risk of predicting twice and getting mismatched
numbers.

Both BiLSTM and BERT run every time now and get returned separately
(bert_result is None if BERT isn't available, same graceful-degrade
pattern as cached_loaders.py already uses). This file does NOT pick a
winner between them - that's decision_engine.py's job, coming right
after Epic 4. Response generation still uses BiLSTM's result for now,
but that's just a temporary placeholder until the decision engine
exists, not a permanent "BiLSTM wins" choice.
"""
from src.generation.response_generator import generate_response
from src.inference.cached_loaders import get_bert_predictor, get_bilstm_predictor
from src.inference.schema import bert_schema, bilstm_schema


def run_analysis(field: str, problem: str, ai_enabled: bool) -> dict:
    bilstm_result = bilstm_schema(problem, get_bilstm_predictor())

    bert_predictor = get_bert_predictor()
    bert_result = bert_schema(problem, bert_predictor) if bert_predictor else None

    # TEMPORARY: still using BiLSTM to drive the actual response since
    # decision_engine.py doesn't exist yet. Not a final call on which
    # model is "better" - just what's wired up until that module lands.
    ai_response = generate_response(
        field,
        problem,
        bilstm_result["emotion"],
        bilstm_result["confidence"],
        enabled=ai_enabled,
    )

    return {
        "bilstm_result": bilstm_result,
        "bert_result": bert_result,
        "ai_response": ai_response,
    }
