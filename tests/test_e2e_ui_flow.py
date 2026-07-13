"""
Epic 6 T1 - automated smoke test for the UI flow, using Streamlit's
AppTest instead of a real browser. This is step 1 of two for T1 - a
manual click-through pass in an actual browser still has to happen
after this (see EPIC6_T1_VALIDATION_REPORT.md), because AppTest can't
"click" a tab and can't inspect what a Plotly figure actually looks
like - see the notes on that further down.

Needs real trained weights on disk (models/bilstm/, models/
bert_emotion_model_final/) and LLM_PROVIDER=openrouter in .env to
actually exercise the real pipeline - this can't run in a sandbox with
no weights, has to run on a real dev machine. Run from the project
root (not from inside tests/) so the relative "logs/" path csv_logger.py
uses resolves the same way it does for a normal `streamlit run app.py`:

    pytest tests/test_e2e_ui_flow.py -v

Every test below leaves "Save to CSV" unchecked on purpose. That
checkbox writes straight to the same logs/emotion_response_examples.csv
and logs/emotion_response_mapping.csv used for real analytics/history -
there's no separate test path for it - so running this a bunch of times
while debugging won't add junk rows to real data.

Known AppTest limitations (from Streamlit's own "cheat sheet" docs,
limitations section) that shape what's tested here:
- st.expander isn't supported at all - not an issue for us, it's
  already been removed from result_panel.py for unrelated reasons (see
  that file's own docstring).
- st.plotly_chart isn't a typed element AppTest understands - shows up
  as an UnknownElement via at.get("plotly_chart"). So the chart checks
  below can only prove the right NUMBER of charts got created without
  raising an exception, not what they actually look like. Confirming
  the pie/line/bar charts render correctly and look like the mockups is
  the manual pass's job, not this script's.
"""
import os
import re
import sys

# same path trick scripts/test_ui.py already uses - makes sure "src" is
# importable regardless of which directory pytest gets run from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

from src.generation.template_fallback import get_fallback_response
from src.inference.mixed_emotion import detect_mixed_emotions

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

SAMPLE_PROBLEM_1 = "I don't get how recursion works, I've read the same paragraph five times"
SAMPLE_PROBLEM_2 = "debugging this is so frustrating, nothing I try is working"


def _launch() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception, f"app crashed on startup: {at.exception}"
    return at


def _fill_and_analyze(at, problem=SAMPLE_PROBLEM_1, ai_enabled=True, save_data=False, show_details=True):
    """Fills the form and clicks Analyze once. save_data defaults to
    False on purpose - see the module docstring above."""
    at.main.text_area[0].set_value(problem)
    at.main.checkbox[0].set_value(ai_enabled)    # "Use AI Response"
    at.main.checkbox[1].set_value(save_data)     # "Save to CSV for learning"
    at.main.checkbox[2].set_value(show_details)  # "Show analysis details"
    at.main.button[0].click()
    at.run()
    assert not at.exception, f"analyze crashed: {at.exception}"
    return at


def _sidebar_text(at) -> str:
    return " ".join(w.value for w in at.sidebar.markdown)


# ---------------------------------------------------------------------
# Checkpoint 1: startup & model loading
# ---------------------------------------------------------------------

def test_startup_loads_without_error_and_shows_fresh_state():
    at = _launch()

    sidebar_text = _sidebar_text(at)
    assert "Models:" in sidebar_text
    assert "Total Interactions:" in sidebar_text
    assert "CSV Examples:" in sidebar_text
    assert len(at.session_state.emotion_history) == 0

    # nothing analyzed yet - the results panel shouldn't exist at all
    assert len(at.main.metric) == 0
    assert not any("Your Guidance" in s.value for s in at.main.subheader)


# ---------------------------------------------------------------------
# Input validation (empty problem shouldn't crash or run inference)
# ---------------------------------------------------------------------

def test_empty_problem_shows_warning_and_does_not_run_analysis():
    at = _launch()
    at.main.button[0].click()
    at.run()

    assert not at.exception
    assert len(at.main.warning) == 1
    assert at.session_state.last_result is None
    assert len(at.session_state.emotion_history) == 0


# ---------------------------------------------------------------------
# Checkpoint 2: full input -> inference -> results pipeline
# ---------------------------------------------------------------------

def test_full_pipeline_updates_all_sections():
    at = _launch()
    _fill_and_analyze(at, show_details=True)

    result = at.session_state.last_result
    assert result is not None
    assert len(at.session_state.emotion_history) == 1

    # Decision summary
    assert len(at.main.metric) >= 1
    assert at.main.metric[0].label == "Detected Emotion"
    assert result["decision"]["emotion"] in at.main.metric[0].value

    # AI guidance section
    assert any("Your Guidance" in s.value for s in at.main.subheader)
    assert result["ai_response"].strip() != ""


# ---------------------------------------------------------------------
# Checkpoint 3: mixed emotion detection matches actual model output
# (not hardcoded - checks whatever this run's real scores say)
# ---------------------------------------------------------------------

def _model_metric_matches(metric, mixed: dict, result: dict) -> bool:
    if mixed["is_mixed"]:
        if metric.label != "Mixed Emotions":
            return False
        if mixed["primary_emotion"] not in metric.value:
            return False
        return all(e["emotion"] in metric.value for e in mixed["secondary_emotions"])
    return metric.label == "Emotion" and result["emotion"] in metric.value


def test_mixed_emotion_rendering_matches_detection_logic():
    at = _launch()
    _fill_and_analyze(at, show_details=True)
    result = at.session_state.last_result

    # metric[0] is always the top-level decision summary - the
    # per-model ones from model_comparison.py start right after it
    model_metrics = at.main.metric[1:]

    bilstm_mixed = detect_mixed_emotions(result["bilstm_result"]["scores"])
    assert _model_metric_matches(model_metrics[0], bilstm_mixed, result["bilstm_result"])

    if result["bert_result"]:
        bert_mixed = detect_mixed_emotions(result["bert_result"]["scores"])
        assert _model_metric_matches(model_metrics[1], bert_mixed, result["bert_result"])


# ---------------------------------------------------------------------
# Checkpoint 4: Model Predictions Comparison section
# ---------------------------------------------------------------------

def test_model_comparison_section_structure():
    at = _launch()
    _fill_and_analyze(at, show_details=True)
    result = at.session_state.last_result

    assert any("Model Predictions Comparison" in s.value for s in at.main.subheader)

    # st.progress isn't a typed AppTest element (same deal as
    # plotly_chart) - shows up as UnknownElement via .get(), so this
    # only confirms the right COUNT of bars got created, not the
    # percentages drawn on them
    num_models_shown = 2 if result["bert_result"] else 1
    assert len(at.main.get("progress")) == num_models_shown * 5  # 5 emotion classes each


def test_model_comparison_hidden_when_show_details_off():
    at = _launch()
    _fill_and_analyze(at, show_details=False)

    assert not any("Model Predictions Comparison" in s.value for s in at.main.subheader)
    assert len(at.main.get("progress")) == 0


# ---------------------------------------------------------------------
# Checkpoint 5: AI response section - and confirm it's a real LLM call,
# not a silent template-fallback (this is exactly the failure mode the
# httpx pin issue caused before - see EPIC5 handoff notes)
# ---------------------------------------------------------------------

def test_ai_response_is_real_not_template_fallback():
    at = _launch()
    _fill_and_analyze(at, ai_enabled=True, show_details=False)
    result = at.session_state.last_result

    canned = get_fallback_response(result["decision"]["emotion"])
    assert result["ai_response"].strip() != canned.strip()


def test_ai_toggle_off_uses_template_fallback_on_purpose():
    """Sanity check on the comparison above - confirms the fallback
    path is reachable at all and matches template_fallback.py exactly
    when AI is deliberately turned off, so we know the "!=" check above
    is actually meaningful and not just always-true."""
    at = _launch()
    _fill_and_analyze(at, ai_enabled=False, show_details=False)
    result = at.session_state.last_result

    canned = get_fallback_response(result["decision"]["emotion"])
    assert result["ai_response"].strip() == canned.strip()


# ---------------------------------------------------------------------
# Checkpoint 6: Analytics Dashboard tab navigation
# ---------------------------------------------------------------------

def test_analytics_dashboard_tabs_render_after_first_interaction():
    at = _launch()
    # analytics_dashboard.py returns early with nothing rendered until
    # there's at least one interaction - can't check this at pure
    # startup, only after one Analyze
    _fill_and_analyze(at, show_details=False)

    assert len(at.tabs) == 3  # Emotions, Fields, Summary

    emotions_tab, fields_tab, summary_tab = at.tabs[0], at.tabs[1], at.tabs[2]

    assert len(emotions_tab.get("plotly_chart")) == 2  # pie + line
    assert len(fields_tab.get("plotly_chart")) == 1    # bar

    # Summary tab is intentionally empty (see handoff_epic5.md /
    # EPIC5_POLISH_BACKLOG.md, no scope defined for it yet) - this is a
    # known, already-decided gap, not a bug. Just confirming it exists
    # and doesn't blow up.
    assert len(summary_tab.get("plotly_chart")) == 0


# ---------------------------------------------------------------------
# Session-state test: multiple interactions + Clear History
# ---------------------------------------------------------------------

def test_multi_interaction_and_clear_history_resets_session_only():
    at = _launch()
    csv_before = re.search(r"CSV Examples: (\d+)", _sidebar_text(at)).group(1)

    _fill_and_analyze(at, problem=SAMPLE_PROBLEM_1)
    assert len(at.session_state.emotion_history) == 1

    _fill_and_analyze(at, problem=SAMPLE_PROBLEM_2)
    assert len(at.session_state.emotion_history) == 2

    # analytics dashboard should reflect both interactions now
    assert len(at.tabs) == 3

    at.sidebar.button[0].click()  # "Clear History"
    at.run()
    assert not at.exception

    assert at.session_state.emotion_history == []
    assert at.session_state.last_result is None

    sidebar_after = _sidebar_text(at)
    assert "Total Interactions: 0" in sidebar_after

    # Clear History only touches session state (confirmed by reading
    # sidebar_dashboard.py directly) - it never writes to or clears the
    # CSV, and we never checked "Save to CSV" above, so this must be
    # completely unchanged
    csv_after = re.search(r"CSV Examples: (\d+)", sidebar_after).group(1)
    assert csv_after == csv_before
