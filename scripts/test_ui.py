import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import streamlit as st
from src.ui.field_problem_capture import capture_field_and_problem

field, problem = capture_field_and_problem()

st.write(field)
st.write(problem)