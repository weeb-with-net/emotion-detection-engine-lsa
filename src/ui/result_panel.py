"""
The parts of the results screen T2's mockup doesn't cover but that make
the analysis flow read as one thing instead of a pile of separate
widgets: the headline decision, the actual AI guidance text, and a
"how did you decide that" section for anyone curious about the raw
signals.

render_how_i_decided() used to wrap its content in st.expander("How I
decided"). Confirmed (real repro, isolated by commenting the call out
entirely) that this specific expander was causing a white-screen crash
- browser console: "Cannot read properties of undefined (reading
'vertical')" - triggered by changing the field dropdown after one
completed analysis with show_details on, no re-Analyze needed. Root
cause inside Streamlit's expander reconciliation wasn't fully chased
down, but since show_details already gates whether this whole section
renders at all, the expander's own collapse-by-default behavior was
redundant anyway - removed it instead of fighting the widget further.

Trust-adapted phrasing is intentionally NOT here yet - held back on
purpose until the plain version is verified working end to end, see
EPIC5_POLISH_BACKLOG.md. This just states the trust level and reason
code as-is, no rewording based on them.
"""
import streamlit as st

from src.generation.template_fallback import EMOTION_RESPONSES


def render_decision_summary(decision: dict) -> None:
    emoji = EMOTION_RESPONSES[decision["emotion"]]["emoji"]
    st.metric(
        "Detected Emotion",
        f"{emoji} {decision['emotion']}",
        f"{decision['confidence']:.1%} confidence",
    )


def render_ai_guidance(ai_response: str) -> None:
    st.subheader("💡 Your Guidance")
    st.write(ai_response)


def render_how_i_decided(decision: dict, bilstm_result: dict, bert_result: dict = None) -> None:
    st.markdown("**How I decided**")
    st.write(f"**Trust level:** {decision['trust_level']}")
    st.write(f"**Reason code:** `{decision['reason']}`")
    st.write(f"**BiLSTM said:** {bilstm_result['emotion']} ({bilstm_result['confidence']:.1%})")
    if bert_result:
        st.write(f"**BERT said:** {bert_result['emotion']} ({bert_result['confidence']:.1%})")
    else:
        st.write("**BERT:** unavailable this run")
