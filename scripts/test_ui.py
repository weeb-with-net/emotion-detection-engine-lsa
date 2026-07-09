import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import streamlit as st
from src.ui.field_problem_capture import capture_field_and_problem
from src.ui.ai_toggle import capture_ai_toggle
from src.ui.save_data_toggle import capture_save_data_toggle

field, problem = capture_field_and_problem()
ai_enabled = capture_ai_toggle()
save_data = capture_save_data_toggle()

st.write(field)
st.write(problem)
st.write("AI enabled:", ai_enabled)
st.write("Save data:", save_data)