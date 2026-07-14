"""
Main Streamlit app - Epic 5 Story 1 (layout + session state), Story 2
(Analyze flow, model comparison, AI guidance, How I decided), and
Story 3 (settings panel restructure, show_details toggle, analytics
tabs skeleton). Results panel follows Input -> Analyze -> Decision ->
AI Guidance -> Model Comparison -> Explainability, so it reads as one
flow instead of a pile of separate widgets. Trust-adapted phrasing
intentionally not here yet - see EPIC5_POLISH_BACKLOG.md for what's
held back and why.
"""
import os

import streamlit as st

from src.inference.cached_loaders import get_bert_predictor, get_bilstm_predictor
from src.orchestration.analysis_pipeline import run_analysis
from src.orchestration.session_history import init_session_history, record_interaction
from src.ui.ai_toggle import capture_ai_toggle
from src.ui.analytics_dashboard import render_analytics_dashboard
from src.ui.csv_prediction_toggle import capture_csv_prediction_toggle
from src.ui.field_problem_capture import capture_field_and_problem
from src.ui.model_comparison import render_model_comparison
from src.ui.result_panel import render_ai_guidance, render_decision_summary, render_how_i_decided
from src.ui.save_data_toggle import capture_save_data_toggle
from src.ui.show_details_toggle import capture_show_details_toggle
from src.ui.sidebar_dashboard import render_sidebar
from src.ui.welcome_splash import show_welcome_splash

st.set_page_config(page_title="AI Learning Assistant", page_icon="🎓", layout="wide")


def _load_cloud_secrets_into_env() -> None:
    """Local dev uses .env (gitignored, loaded by dotenv in llm_client.py
    etc) - that file just doesn't exist on Streamlit Cloud, secrets go
    through st.secrets there instead. Bridging the ones we actually use
    into os.environ so every os.getenv(...) call elsewhere in the
    codebase keeps working unchanged, regardless of which one supplied
    it. Wrapped in try/except because st.secrets raises FileNotFoundError
    if no secrets.toml exists anywhere - which is the normal case for
    local dev, not an error.

    Has to run AFTER st.set_page_config() - accessing st.secrets counts
    as a Streamlit command, and set_page_config has to be the very first
    one or Streamlit throws (found this the hard way in Claude's own
    dry run, not guessed).
    """
    try:
        for key in ("LLM_PROVIDER", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "HF_MODEL_REPO_ID"):
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
    except FileNotFoundError:
        pass


_load_cloud_secrets_into_env()

if os.getenv("HF_MODEL_REPO_ID"):
    # HF_MODEL_REPO_ID only gets set via Streamlit Cloud secrets - local
    # dev never has it, so this only fires on deployment. Streamlit
    # Cloud has no GPU at all, but an unpinned torch install can still
    # pull in a full CUDA build, and torch.cuda.is_available() spends
    # real time (sometimes a lot) probing for hardware that isn't
    # there. Telling it up front there are zero visible devices skips
    # that probing completely instead of hoping it fails fast.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Tell us about your learning challenge")
    field, problem = capture_field_and_problem()

with col2:
    st.subheader("⚙️ Settings")
    ai_enabled = capture_ai_toggle()
    save_data = capture_save_data_toggle()
    show_details = capture_show_details_toggle()
    use_csv_prediction = capture_csv_prediction_toggle()  # currently no effect on Analyze - see its own docstring

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
    if show_details:
        render_model_comparison(result["bilstm_result"], result["bert_result"])
        render_how_i_decided(result["decision"], result["bilstm_result"], result["bert_result"])

render_analytics_dashboard()


