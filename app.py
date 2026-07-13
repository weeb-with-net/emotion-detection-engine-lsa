"""
Main Streamlit app - Epic 5 Story 1 (Responsive Layout and Session State
Management) only. Wires together pieces already built in earlier epics:
session state init, sidebar dashboard, and the input section. No
Analyze button here on purpose - running run_analysis() and showing
results is Epic 5 Story 2, not this one.
"""
import streamlit as st

from src.inference.cached_loaders import get_bert_predictor, get_bilstm_predictor
from src.orchestration.session_history import init_session_history
from src.ui.ai_toggle import capture_ai_toggle
from src.ui.field_problem_capture import capture_field_and_problem
from src.ui.save_data_toggle import capture_save_data_toggle
from src.ui.sidebar_dashboard import render_sidebar
from src.ui.welcome_splash import show_welcome_splash

st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="wide")

show_welcome_splash()

init_session_history()

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

# TODO (Epic 5 Story 2): Analyze button goes here - run_analysis(field,
# problem, ai_enabled) then record_interaction(field, problem,
# result["decision"], result["ai_response"], result["bilstm_result"],
# bert_result=result["bert_result"], save_to_csv=save_data). Left out
# on purpose - not this story's scope.
