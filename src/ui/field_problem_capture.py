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

# one real example per field instead of subbing the field name into the
# same generic sentence every time - actually gives the student an idea
# of what kind of problem to type
FIELD_PLACEHOLDERS = {
    "Computer Science": "e.g., 'I don't get how recursion works' or 'my code keeps throwing an IndexError'",
    "Mathematics": "e.g., 'I can't solve this integration problem' or 'matrices confuse me'",
    "Physics": "e.g., 'I don't understand torque' or 'this circuit problem makes no sense'",
    "Chemistry": "e.g., 'balancing equations is confusing' or 'I don't get molarity'",
    "Biology": "e.g., 'I can't remember the steps of the Krebs cycle' or 'genetics problems confuse me'",
    "Engineering": "e.g., 'I don't understand this circuit diagram' or 'stuck on a design problem'",
    "Business": "e.g., 'I don't get how to read a balance sheet' or 'this case study is confusing'",
    "Literature": "e.g., 'I can't figure out the theme of this poem' or 'this novel's symbolism is confusing'",
    "History": "e.g., 'I can't remember the dates for this era' or 'this cause-and-effect chain is confusing'",
    "Psychology": "e.g., 'I don't get classical vs operant conditioning' or 'this theory feels confusing'",
    "Other": "e.g., 'I'm struggling with this concept' or 'this doesn't make sense to me'",
}


def capture_field_and_problem() -> tuple[str, str]:
    field = st.selectbox(
        "What field are you studying?",
        FIELD_OPTIONS,
        help="Select your area of study for personalized responses",
    )

    problem = st.text_area(
        f"Describe your {field} problem or challenge:",
        placeholder=FIELD_PLACEHOLDERS[field],
        height=120,
    )

    return field, problem
