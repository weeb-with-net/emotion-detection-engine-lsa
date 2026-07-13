"""
Model comparison display - T2. Side-by-side BiLSTM vs BERT (single
column if BERT isn't available), mixed-emotion label with emoji when
more than one emotion clears the threshold, confidence bars for every
score in descending order.

T2's mockup calls get_mixed_emotions(scores) and treats the result as a
list of (emotion, score) tuples - that function doesn't exist. What
exists is detect_mixed_emotions() in mixed_emotion.py (already used by
session_history.py), which returns a dict instead: {primary_emotion,
primary_confidence, secondary_emotions, is_mixed}. Using that here
instead of writing a second mixed-emotion function - same reason
session_history.py already gives for not duplicating this logic.

One shared column-renderer instead of the mockup's copy-pasted block
for each model - same output, one place to fix if the layout changes.
"""
import streamlit as st

from src.generation.template_fallback import EMOTION_RESPONSES
from src.inference.mixed_emotion import detect_mixed_emotions


def _render_model_column(name: str, result: dict) -> None:
    st.write(f"**{name}**")

    mixed = detect_mixed_emotions(result["scores"])

    if mixed["is_mixed"]:
        parts = [f"{EMOTION_RESPONSES[mixed['primary_emotion']]['emoji']} {mixed['primary_emotion']}"]
        parts += [f"{EMOTION_RESPONSES[e['emotion']]['emoji']} {e['emotion']}" for e in mixed["secondary_emotions"]]
        st.metric("Mixed Emotions", " + ".join(parts), f"Primary: {mixed['primary_confidence']:.1%}")
    else:
        emoji = EMOTION_RESPONSES[result["emotion"]]["emoji"]
        st.metric("Emotion", f"{emoji} {result['emotion']}", f"{result['confidence']:.1%}")

    for emotion_name, score in sorted(result["scores"].items(), key=lambda x: x[1], reverse=True):
        st.progress(score, text=f"{emotion_name}: {score:.1%}")


def render_model_comparison(bilstm_result: dict, bert_result: dict = None) -> None:
    st.subheader("📊 Model Predictions Comparison")

    if bert_result:
        col1, col2 = st.columns(2)
        with col1:
            _render_model_column("BiLSTM Student Adaptive", bilstm_result)
        with col2:
            _render_model_column("BERT Transformer", bert_result)
    else:
        _render_model_column("BiLSTM Student Adaptive", bilstm_result)
