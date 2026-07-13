"""
Main Streamlit app - Epic 5 Story 1 (layout + session state) plus
Story 2 (Analyze flow, model comparison, AI guidance, How I decided).
Results panel follows Input -> Analyze -> Decision -> AI Guidance ->
Model Comparison -> Explainability, so it reads as one flow instead of
a pile of separate widgets. Trust-adapted phrasing intentionally not
here yet - see EPIC5_POLISH_BACKLOG.md for what's held back and why.
"""
import streamlit as st

from src.inference.cached_loaders import get_bert_predictor, get_bilstm_predictor
from src.orchestration.analysis_pipeline import run_analysis
from src.orchestration.session_history import init_session_history, record_interaction
from src.ui.ai_toggle import capture_ai_toggle
from src.ui.field_problem_capture import capture_field_and_problem
from src.ui.model_comparison import render_model_comparison
from src.ui.result_panel import render_ai_guidance, render_decision_summary, render_how_i_decided
from src.ui.save_data_toggle import capture_save_data_toggle
from src.ui.sidebar_dashboard import render_sidebar
from src.ui.welcome_splash import show_welcome_splash

st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="wide")

show_welcome_splash()

init_session_history()
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# loading here (not just at analyze-time) so the sidebar can show real
# model status right away - both are @st.cache_resource so this only
# actually loads weights once per session, not on every rerun
get_bilstm_predictor()
bert_predictor = get_bert_predictor()

render_sidebar(bert_ready=bert_predictor is not None)

st.title("🎓 AI Learning Assistant")

st.subheader("📝 Tell us about your learning challenge")
field, problem = capture_field_and_problem()
ai_enabled = capture_ai_toggle()
save_data = capture_save_data_toggle()

if st.button("🔍 Analyze"):
    if not problem.strip():
        st.warning("Describe your problem first - the box above is empty.")
    else:
        with st.spinner("Analyzing your learning challenge..."):
            result = run_analysis(field, problem, ai_enabled)
            record_interaction(
                field,
                problem,
                result["decision"],
                result["ai_response"],
                result["bilstm_result"],
                bert_result=result["bert_result"],
                save_to_csv=save_data,
            )
        st.session_state.last_result = result

if st.session_state.last_result:
    result = st.session_state.last_result
    st.divider()
    render_decision_summary(result["decision"])
    render_ai_guidance(result["ai_response"])
    render_model_comparison(result["bilstm_result"], result["bert_result"])
    render_how_i_decided(result["decision"], result["bilstm_result"], result["bert_result"])

