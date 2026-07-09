"""
Ties together the model predictions and the response generation for one
button click. Only calls each predictor once, so whatever gets shown on
screen and whatever gets sent to generate_response() are always the
exact same values - no risk of predicting twice and getting mismatched
numbers.

Both BiLSTM and BERT run every time now and get returned separately
(bert_result is None if BERT isn't available, same graceful-degrade
pattern as cached_loaders.py already uses). decide_emotion() picks
which one actually drives the response and how much to trust it - see
src/orchestration/decision_engine.py for the reasoning behind that.
"""
from src.generation.response_generator import generate_response
from src.inference.cached_loaders import get_bert_predictor, get_bilstm_predictor
from src.inference.schema import bert_schema, bilstm_schema
from src.orchestration.decision_engine import decide_emotion


def run_analysis(field: str, problem: str, ai_enabled: bool) -> dict:
    bilstm_result = bilstm_schema(problem, get_bilstm_predictor())

    bert_predictor = get_bert_predictor()
    bert_result = bert_schema(problem, bert_predictor) if bert_predictor else None

    decision = decide_emotion(bilstm_result, bert_result)

    ai_response = generate_response(
        field,
        problem,
        decision["emotion"],
        decision["confidence"],
        enabled=ai_enabled,
    )

    return {
        "bilstm_result": bilstm_result,
        "bert_result": bert_result,
        "decision": decision,
        "ai_response": ai_response,
    }
