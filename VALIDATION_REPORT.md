# Epic 6 T1 - Validate UI Flow End-to-End

Two-part validation, per the plan: automated smoke test first (`tests/
test_e2e_ui_flow.py`, Streamlit AppTest), then a manual click-through
pass to catch what AppTest can't see (tab clicks, chart visuals, layout).

## Environment

Windows, Python 3.12.10, real trained weights (BiLSTM + BERT),
`LLM_PROVIDER=openrouter`.

## Part 1 - Automated (AppTest)

`pytest tests/test_e2e_ui_flow.py -v` - **10/10 passed**, real weights,
real OpenRouter calls, no mocking. ~224s total (model loading is
cached once per session; most of the time is the real LLM calls +
BERT/BiLSTM inference across 10 tests). Two harmless deprecation
warnings from `google-generativeai`'s internals, unrelated to our code.

| # | Test | Checkpoint |
|---|---|---|
| 1 | `test_startup_loads_without_error_and_shows_fresh_state` | Startup & model loading |
| 2 | `test_empty_problem_shows_warning_and_does_not_run_analysis` | Input validation |
| 3 | `test_full_pipeline_updates_all_sections` | Full input->inference->results |
| 4 | `test_mixed_emotion_rendering_matches_detection_logic` | Mixed emotion detection |
| 5 | `test_model_comparison_section_structure` | Model Predictions Comparison |
| 6 | `test_model_comparison_hidden_when_show_details_off` | show_details gating |
| 7 | `test_ai_response_is_real_not_template_fallback` | AI Response (real LLM, not silent fallback) |
| 8 | `test_ai_toggle_off_uses_template_fallback_on_purpose` | Fallback path sanity check |
| 9 | `test_analytics_dashboard_tabs_render_after_first_interaction` | Analytics Dashboard tabs |
| 10 | `test_multi_interaction_and_clear_history_resets_session_only` | Session state + Clear History |

**Known AppTest limitations** (not bugs, just what the framework can't
see - see the script's own docstring): `st.expander` isn't supported at
all (moot - already removed from `result_panel.py`); `st.plotly_chart`
and `st.progress` aren't typed elements, so those tests can only prove
the right *count* was created without an exception, not what's
actually drawn. That gap is exactly what Part 2 covers.

## Part 2 - Manual click-through pass

All 6 checkpoints confirmed working in a real browser: startup, full
pipeline, mixed emotion display, Model Comparison layout, AI Response
text, and Analytics Dashboard (all three tabs navigate correctly - pie
+ line charts on Emotions, bar chart on Fields, Summary confirmed
empty-but-not-broken, consistent with its documented, intentionally-
undefined scope).

## Known, already-decided gaps (not defects)

- **Summary tab intentionally empty** - no scope was ever defined for
  it (see `handoff_epic5.md` / `EPIC5_POLISH_BACKLOG.md`). Confirmed
  empty and stable, not crashing. Held as a clean slot for a later
  dashboard-metrics pass, per Aditya's decision to not build it now.
- **Fields tab doesn't facet by model**, unlike the original T1 mockup
  screenshot (which showed separate BERT/BiLSTM columns). Already a
  deliberate, documented deviation from an earlier story (session
  history consolidation removed the per-model row split this would
  have needed) - not something T1 introduced or needs to fix.

## Outstanding handoff item - now closed

The handoff doc flagged confirming `EPIC5_POLISH_BACKLOG.md`,
`DEPLOYMENT_NOTES.md`, and the `result_panel.py` expander-removal fix
were actually pushed before starting Epic 6.

- `DEPLOYMENT_NOTES.md` - confirmed pushed.
- `result_panel.py` expander fix - confirmed pushed (no `st.expander`
  anywhere in the file).
- `EPIC5_POLISH_BACKLOG.md` - **was missing from the repo entirely**
  (checked full git history across all branches, never committed).
  Fixed: committed and pushed, re-verified via a fresh clone.

## Status

**T1 complete.** All 6 checkpoints pass both automated and manual
validation. No open bugs. Ready to move to Epic 6 Story 2
(Optimization and Deployment Readiness).
