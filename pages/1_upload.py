import streamlit as st
import pandas as pd
from utils.notepad import render_notepad
render_notepad()

st.set_page_config(page_title="Upload Data", page_icon=":material/upload_file:", layout="wide")

st.title(":material/upload_file: Upload Your Data")
st.markdown("Supports **CSV** and **Excel** files.")

# ── File Upload ──────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df
        st.session_state.df_original = df.copy()  # 👈 save original once
        st.session_state.filename = uploaded_file.name
        st.success(f":material/check_circle: `{uploaded_file.name}` uploaded successfully!")

    except Exception as e:
        st.error(f":material/error: Error reading file: {e}")

# ── Show preview if data exists in session ───────────────────
if st.session_state.get("df") is not None:
    df = st.session_state.df

    if st.session_state.get("filename"):
        st.info(f":material/dataset: **{st.session_state.filename}** is loaded.")

    # ── Quick Stats ──────────────────────────────────────────
    st.subheader(":material/bar_chart: Quick Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicate Rows", df.duplicated().sum())

    st.divider()

    # ── Data Preview ─────────────────────────────────────────
    st.subheader(":material/preview: Data Preview")
    col1, col2 = st.columns([1, 3])
    with col1:
        preview_type = st.radio("Show", ["Head", "Tail", "Sample"])
        n_rows = st.slider("Rows", 5, min(100, len(df)), 10)
    with col2:
        if preview_type == "Head":
            st.dataframe(df.head(n_rows), use_container_width=True)
        elif preview_type == "Tail":
            st.dataframe(df.tail(n_rows), use_container_width=True)
        elif preview_type == "Sample":
            st.dataframe(df.sample(n_rows), use_container_width=True)

    st.divider()

    # ── DataFrame Info ────────────────────────────────────────
    st.subheader(":material/info: DataFrame Info")
    info_df = pd.DataFrame({
        "Column"        : df.columns,
        "Dtype"         : df.dtypes.values,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count"    : df.isnull().sum().values,
        "Null %"        : (df.isnull().sum().values / len(df) * 100).round(2),
        "Unique Values" : df.nunique().values,
        "Memory (KB)"   : (df.memory_usage(deep=True)[1:].values / 1024).round(3)
    })
    st.dataframe(info_df, use_container_width=True)

    total_mem = df.memory_usage(deep=True).sum() / 1024
    st.caption(f":material/memory: Total memory usage: **{total_mem:.2f} KB**")

else:
    st.info(":material/upload: Upload a CSV or Excel file to get started.")