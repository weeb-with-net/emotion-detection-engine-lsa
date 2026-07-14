"""
Cached model loading — each predictor loads once per Streamlit session
instead of reloading weights on every rerun.

BiLSTM is required (matches project priority: BiLSTM is primary, BERT is
a stretch goal), so a BiLSTM load failure is not swallowed. If BERT's weights are missing or broken, get_bert_predictor()
returns None so the app still works with BiLSTM alone.
"""
import os

import streamlit as st


@st.cache_resource
def get_bilstm_predictor():
    from src.inference.bilstm_predictor import BiLSTMPredictor

    return BiLSTMPredictor()


@st.cache_resource
def get_bert_predictor():
    if os.getenv("DISABLE_BERT") == "true":
        # opt-in escape hatch for memory-constrained deployment (e.g.
        # Streamlit Cloud's 1GB free tier) - skips importing torch/
        # transformers entirely, not just skipping the weights, since
        # torch's own import costs real memory before any model is
        # even touched. BiLSTM-only is already a fully supported mode,
        # not a hack - this just makes it reachable without BERT ever
        # being attempted at all.
        return None
    try:
        from src.inference.bert_predictor import BERTPredictor

        return BERTPredictor()
    except Exception:
        return None
