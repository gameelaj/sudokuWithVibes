import streamlit as st
from app.ui import run
from app.digit_recognizer import load_model

st.set_page_config(page_title="SudokuWithVibes", layout="wide")

# Load model once at startup so the first scan does not pay the disk load cost.
if "model" not in st.session_state:
	st.session_state.model = load_model()

run()