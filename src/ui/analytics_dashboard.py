"""
Analytics dashboard - T3 built the tab skeleton (empty on purpose), T4
fills tab1 (pie + line charts) and tab2 (bar chart) with real Plotly
visualizations. Summary tab (tab3) stays untouched/empty - T4's doc
doesn't cover it at all, and per direction we're not inventing analytics
beyond what's documented. Clean slot for a later dashboard-metrics pass.

Bar chart doesn't facet by model like the mockup showed - the mockup's
'model' column doesn't exist anymore (T1 consolidated two rows per
interaction into one, renamed to driving_model), and faceting on
driving_model wouldn't be a meaningful BiLSTM-vs-BERT comparison anyway,
since it reads "bert" for almost every row whenever BERT's available.
Always uses the simple field+emotion groupby instead - same as the
mockup's own fallback branch, just without the (no-longer-meaningful)
facet check in front of it.

Chart builders are @st.cache_data'd on df - Streamlit hashes the actual
dataframe content, so a chart only recomputes when emotion_history has
actually changed, not on every rerun (typing in the text area, toggling
a checkbox, etc).
"""
import pandas as pd
import plotly.express as px
import streamlit as st


#@st.cache_data
def _build_emotion_pie(df: pd.DataFrame):
    emotion_counts = df["emotion"].value_counts()
    return px.pie(
        values=emotion_counts.values,
        names=emotion_counts.index,
        title="Emotion Distribution",
    )


#@st.cache_data
def _build_confidence_timeline(df: pd.DataFrame):
    df_copy = df.copy()
    df_copy["time"] = df_copy["timestamp"].dt.strftime("%H:%M:%S")
    return px.line(
        df_copy,
        x="time",
        y="confidence",
        color="emotion",
        title="Emotional Journey",
        markers=True,
    )


#@st.cache_data
def _build_field_emotion_bar(df: pd.DataFrame):
    field_emotion = df.groupby(["field", "emotion"]).size().reset_index(name="count")
    return px.bar(
        field_emotion,
        x="field",
        y="count",
        color="emotion",
        title="Emotions by Study Field",
    )


def render_analytics_dashboard() -> None:
    if not st.session_state.emotion_history:
        return

    st.markdown("---")
    st.header("📈 Learning Analytics")

    df = pd.DataFrame(st.session_state.emotion_history)
    # only the plain columns charts actually use - bilstm_result/
    # bert_result/all_scores are dicts, which pandas can't hash, which
    # was forcing st.cache_data into a slow pickle-based fallback
    # instead of actually caching. Slimming this down here fixes that.
    chart_df = df[["field", "emotion", "confidence", "timestamp"]]

    tab1, tab2, tab3 = st.tabs(["Emotions", "Fields", "Summary"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(_build_emotion_pie(chart_df), use_container_width=True)
        with col2:
            st.plotly_chart(_build_confidence_timeline(chart_df), use_container_width=True)

    with tab2:
        st.plotly_chart(_build_field_emotion_bar(chart_df), use_container_width=True)
