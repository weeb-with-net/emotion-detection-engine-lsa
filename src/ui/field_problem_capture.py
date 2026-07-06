"""
Streamlit inputs for the student's field of study and problem description.
Returns plain values to the caller — no session-state handling or
downstream logic lives here, keeping this a pure UI concern.
"""
import streamlit as st

FIELD_OPTIONS = [
    "Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
    "Engineering", "Business", "Literature", "History", "Psychology", "Other",
]


def capture_field_and_problem() -> tuple[str, str]:
    field = st.selectbox(
        "What field are you studying?",
        FIELD_OPTIONS,
        help="Select your area of study for personalized responses",
    )

    problem = st.text_area(
        f"Describe your {field} problem or challenge:",
        placeholder=f"e.g., 'I'm struggling with algorithms in {field}' or 'This concept is confusing'",
        height=120,
    )

    return field, problem
