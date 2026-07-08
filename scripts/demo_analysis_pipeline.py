"""
Demo for src/orchestration/analysis_pipeline.py. Needs your real trained
BiLSTM weights (models/bilstm/) to actually mean anything - this is the
one file in this task I couldn't verify myself, since my sandbox doesn't
have your model files. Run this on your machine to confirm run_analysis()
works end to end with the real predictor, not a mocked one.

Note: get_bilstm_predictor() uses @st.cache_resource, which is a
Streamlit thing. Running this as a plain script (not through
`streamlit run`) should still work fine, just might print a harmless
"missing ScriptRunContext" warning - ignore that if you see it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.analysis_pipeline import run_analysis

if __name__ == "__main__":
    print("--- ai_enabled=False (template fallback, no LLM call) ---")
    result = run_analysis("Computer Science", "I don't understand recursion", ai_enabled=False)
    print(result)

    print("\n--- ai_enabled=True (real BiLSTM prediction + real LLM call) ---")
    result = run_analysis("Computer Science", "I don't understand recursion", ai_enabled=True)
    print(result)

    print("\nCheck: 'emotion' and 'confidence' above should match whatever your")
    print("BiLSTM model actually predicts for this text - that's the whole point")
    print("of this task, same values get used for display and for the AI response.")
