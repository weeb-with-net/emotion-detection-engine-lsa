"""
One-time welcome splash - fades in, holds a moment, fades out on its
own via CSS, then never shows again this session. Session-state gated
so a later rerun (like changing the field dropdown) doesn't replay it.

Pure CSS animation, no time.sleep() - that would block the whole
Streamlit server thread, not just this one user's view. The overlay
animates its own opacity and keeps pointer-events off throughout, so it
never blocks a click even mid-fade.
"""
import streamlit as st

_SPLASH_HTML = """
<style>
@keyframes splash-fade {
    0%   { opacity: 0; }
    15%  { opacity: 1; }
    70%  { opacity: 1; }
    100% { opacity: 0; }
}
.splash-overlay {
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #0e1117;
    pointer-events: none;
    animation: splash-fade 2.5s ease-in-out forwards;
}
.splash-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #fafafa;
}
.splash-tagline {
    margin-top: 0.5rem;
    font-size: 1rem;
    color: #a3a8b8;
}
</style>
<div class="splash-overlay">
    <div class="splash-title">🎓 AI Learning Assistant</div>
    <div class="splash-tagline">Understanding how you feel about what you're learning</div>
</div>
"""


def show_welcome_splash() -> None:
    if st.session_state.get("splash_shown"):
        return

    st.session_state.splash_shown = True
    st.markdown(_SPLASH_HTML, unsafe_allow_html=True)
