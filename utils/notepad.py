import streamlit as st
import pandas as pd

def render_notepad():
    with st.sidebar:
        st.divider()
        st.markdown("### :material/edit_note: My Notes")

        # ── Init ─────────────────────────────────────────────
        if "notepad_content" not in st.session_state:
            st.session_state.notepad_content = ""

        # ── Notepad ──────────────────────────────────────────
        st.session_state.notepad_content = st.text_area(
            "Write your observations",
            value=st.session_state.notepad_content,
            placeholder="Start writing your insights here...\n\ne.g. Age column is skewed\nSalary has outliers\n...",
            height=300,
            label_visibility="collapsed",
            key="note_textarea"
        )

        # ✅ ADD THIS LINE 👇 (IMPORTANT)
        st.session_state["user_notes"] = st.session_state.notepad_content

        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.notepad_content.strip():
                st.download_button(
                    label="Export",
                    data=st.session_state.notepad_content,
                    file_name="my_insights.txt",
                    mime="text/plain",
                    icon=":material/download:",
                    use_container_width=True
                )
        with col2:
            if st.session_state.notepad_content.strip():
                if st.button("Clear", icon=":material/delete:", use_container_width=True):
                    st.session_state.notepad_content = ""
                    st.session_state["user_notes"] = ""   # ✅ ALSO CLEAR HERE
                    st.rerun()