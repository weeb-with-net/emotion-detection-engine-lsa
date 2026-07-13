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

IMPORTANT: only the data transformations (value_counts, groupby, the
timestamp formatting) are @st.cache_data'd - NOT the Plotly Figure
objects themselves. Found the hard way: st.plotly_chart() mutates the
Figure in place while serializing it for the browser (filling in layout
defaults etc), so caching the actual Figure and reusing that same
object on a later rerun hands the frontend a partially-mutated object
from last time, not a clean one - caused a real white-screen crash
(browser console: "Cannot read properties of undefined (reading
'vertical')"), triggered by just changing the field dropdown after one
completed analysis, no re-Analyze needed. Data transforms are plain
DataFrames/Series (hashable, safe to cache) - Figures get built fresh
from that cached data on every single render instead.
"""
import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def _emotion_counts(df: pd.DataFrame) -> pd.Series:
    return df["emotion"].value_counts()


@st.cache_data
def _timeline_data(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy["time"] = df_copy["timestamp"].dt.strftime("%H:%M:%S")
    return df_copy


@st.cache_data
def _field_emotion_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["field", "emotion"]).size().reset_index(name="count")


def render_analytics_dashboard() -> None:
    if not st.session_state.emotion_history:
        return

    st.markdown("---")
    st.header("📈 Learning Analytics")

    df = pd.DataFrame(st.session_state.emotion_history)
    # only the plain columns charts actually use - bilstm_result/
    # bert_result/all_scores are dicts, which pandas can't hash, which
    # would force st.cache_data into a slow pickle-based fallback
    # instead of actually caching. Slimming this down here fixes that.
    chart_df = df[["field", "emotion", "confidence", "timestamp"]]

    tab1, tab2, tab3 = st.tabs(["Emotions", "Fields", "Summary"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            emotion_counts = _emotion_counts(chart_df)
            fig1 = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Emotion Distribution",
            )
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            timeline_df = _timeline_data(chart_df)
            fig2 = px.line(
                timeline_df,
                x="time",
                y="confidence",
                color="emotion",
                title="Emotional Journey",
                markers=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        field_emotion = _field_emotion_counts(chart_df)
        fig3 = px.bar(
            field_emotion,
            x="field",
            y="count",
            color="emotion",
            title="Emotions by Study Field",
        )
        st.plotly_chart(fig3, use_container_width=True)
