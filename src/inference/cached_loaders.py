"""
Cached model loading — each predictor loads once per Streamlit session
instead of reloading weights on every rerun.

BiLSTM is required (matches project priority: BiLSTM is primary, BERT is
a stretch goal), so a BiLSTM load failure is not swallowed. BERT is
optional — if its weights are missing or broken, get_bert_predictor()
returns None so the app still works with BiLSTM alone.
"""
import streamlit as st


@st.cache_resource
def get_bilstm_predictor():
    from src.inference.bilstm_predictor import BiLSTMPredictor

    return BiLSTMPredictor()


@st.cache_resource
def get_bert_predictor():
    try:
        from src.inference.bert_predictor import BERTPredictor

        return BERTPredictor()
    except Exception:
        return None
