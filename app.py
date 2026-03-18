import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Insight Engine",
    page_icon=":material/insights:",
    layout="wide"
)

st.title(":material/insights: Insight Engine")
st.markdown("Upload your data and get instant analysis, visualizations, and AI-powered insights.")

# ── Session State Init ───────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "notes" not in st.session_state:
    st.session_state.notes = []

st.info(":material/arrow_back: Start by uploading your file from the sidebar.")

