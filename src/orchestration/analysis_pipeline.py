"""
Ties together the BiLSTM prediction and the response generation for one
button click. Only calls bilstm_schema() once, so the emotion/confidence
shown on screen and the ones sent to generate_response() are always the
exact same values - no risk of predicting twice and getting mismatched
numbers between what's displayed and what the AI response is based on.

BERT isn't in here on purpose - BiLSTM is still the model driving the
actual response (matches T1's approach), BERT comparison is an Epic 6
thing, not this task.
"""
from src.generation.response_generator import generate_response
from src.inference.cached_loaders import get_bilstm_predictor
from src.inference.schema import bilstm_schema


def run_analysis(field: str, problem: str, ai_enabled: bool) -> dict:
    predictor = get_bilstm_predictor()
    prediction = bilstm_schema(problem, predictor)

    ai_response = generate_response(
        field,
        problem,
        prediction["emotion"],
        prediction["confidence"],
        enabled=ai_enabled,
    )

    return {**prediction, "ai_response": ai_response}
