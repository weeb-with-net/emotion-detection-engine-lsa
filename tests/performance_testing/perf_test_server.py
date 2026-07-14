"""
Performance-testing wrapper — NOT part of the app itself.

Streamlit communicates over WebSocket, not plain HTTP, so a load tool
like Locust can't directly simulate someone clicking "Analyze" in the
browser. But the actual computational cost of an analysis (BiLSTM +
BERT inference, the decision engine, the LLM call) all lives inside
run_analysis() - the same function app.py calls - and that part has no
Streamlit-specific dependency other than the @st.cache_resource-decorated
loaders, which already work fine outside a live session (see
scripts/demo_analysis_pipeline.py, which does exactly this).

This wrapper exposes run_analysis() as a real POST endpoint so Locust
can drive real concurrent load against the real model/LLM pipeline.
The Streamlit UI layer itself is not the bottleneck here and isn't
worth simulating separately.

Usage:
    pip install fastapi uvicorn
    python perf_test_server.py
    # then point Locust at http://127.0.0.1:8500

To test "degraded mode" (matching the public Streamlit Cloud deployment),
set the same env flags the real app checks, before starting this script:
    set DISABLE_BERT=true
    set DISABLE_AI_RESPONSE=true
    python perf_test_server.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel

from src.orchestration.analysis_pipeline import run_analysis

app = FastAPI()


class AnalyzeRequest(BaseModel):
    field: str
    problem: str
    ai_enabled: bool = True


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    start = time.perf_counter()
    result = run_analysis(req.field, req.problem, req.ai_enabled)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "elapsed_ms": elapsed_ms,
        "emotion": result["decision"]["emotion"],
        "trust_level": result["decision"]["trust_level"],
        "bert_available": result["bert_result"] is not None,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    # single worker on purpose - this mirrors one Streamlit server process
    # handling requests; Locust supplies the concurrency, not extra workers
    uvicorn.run(app, host="127.0.0.1", port=8500, workers=1)
