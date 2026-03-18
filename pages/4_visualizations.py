import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from utils.notepad import render_notepad
render_notepad()

st.set_page_config(page_title="Visualizations", page_icon=":material/bar_chart:", layout="wide")

st.title(":material/bar_chart: Visualizations")

# ── Guard ────────────────────────────────────────────────────
if st.session_state.get("df") is None:
    st.warning(":material/warning: No data loaded. Please upload a file first.")
    st.stop()

df = st.session_state.df
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols     = df.select_dtypes(include="object").columns.tolist()

# ── Report Charts Init ────────────────────────────────────────
if "report_charts" not in st.session_state:
    st.session_state.report_charts = []

def add_to_report(fig, title):
    import plotly.io as pio
    img_bytes = pio.to_image(fig, format="png", scale=1.5)
    st.session_state.report_charts.append({
        "title": title,
        "img"  : img_bytes
    })
    st.session_state.msg_chart_added = f":material/check_circle: **{title}** added to report — {len(st.session_state.report_charts)} chart(s) queued!"

# ── Chart Added Message ───────────────────────────────────────
chart_added_msg = st.empty()
if st.session_state.get("msg_chart_added"):
    chart_added_msg.success(st.session_state.msg_chart_added)
    st.session_state.msg_chart_added = ""

# ── Report Queue Status ───────────────────────────────────────
if st.session_state.report_charts:
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.info(f":material/picture_as_pdf: **{len(st.session_state.report_charts)}** chart(s) queued for PDF report")
    with col_b:
        if st.button("Clear Queue", icon=":material/delete:", use_container_width=True):
            st.session_state.report_charts = []
            st.rerun()

# ── Chart Selector ────────────────────────────────────────────
st.subheader(":material/palette: Choose a Chart Type")
chart_type = st.selectbox("Chart Type", [
    "Histogram",
    "Box Plot",
    "Scatter Plot",
    "Bar Chart",
    "Correlation Heatmap",
    "Pie Chart",
    "Line Chart"
])

st.divider()

# ── Histogram ────────────────────────────────────────────────
if chart_type == "Histogram":
    st.markdown(":material/bar_chart: **Histogram Settings**")
    col = st.selectbox("Select numeric column", numeric_cols)
    bins = st.slider("Number of bins", 5, 100, 20)
    color = st.selectbox("Color by (optional)", ["None"] + cat_cols)
    fig = px.histogram(
        df, x=col, nbins=bins,
        color=None if color == "None" else color,
        marginal="box", template="plotly_white",
        title=f"Distribution of {col}"
    )
    st.plotly_chart(fig, use_container_width=True)
    if st.button("Add to Report", icon=":material/picture_as_pdf:", key="add_hist"):
        add_to_report(fig, f"Histogram — {col}")
        st.rerun()

# ── Box Plot ─────────────────────────────────────────────────
elif chart_type == "Box Plot":
    st.markdown(":material/candlestick_chart: **Box Plot Settings**")
    col = st.selectbox("Select numeric column", numeric_cols)
    group = st.selectbox("Group by (optional)", ["None"] + cat_cols)
    fig = px.box(
        df, y=col,
        x=None if group == "None" else group,
        color=None if group == "None" else group,
        template="plotly_white",
        title=f"Box Plot of {col}"
    )
    st.plotly_chart(fig, use_container_width=True)
    if st.button("Add to Report", icon=":material/picture_as_pdf:", key="add_box"):
        add_to_report(fig, f"Box Plot — {col}")
        st.rerun()

# ── Scatter Plot ─────────────────────────────────────────────
elif chart_type == "Scatter Plot":
    st.markdown(":material/scatter_plot: **Scatter Plot Settings**")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X axis", numeric_cols)
    with col2:
        y_col = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1))
    color = st.selectbox("Color by (optional)", ["None"] + cat_cols)
    trendline = st.checkbox("Show trendline", value=True)
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=None if color == "None" else color,
        trendline="ols" if trendline else None,
        template="plotly_white",
        title=f"{x_col} vs {y_col}"
    )
    st.plotly_chart(fig, use_container_width=True)
    if st.button("Add to Report", icon=":material/picture_as_pdf:", key="add_scatter"):
        add_to_report(fig, f"Scatter — {x_col} vs {y_col}")
        st.rerun()

# ── Bar Chart ────────────────────────────────────────────────
elif chart_type == "Bar Chart":
    st.markdown(":material/bar_chart: **Bar Chart Settings**")
    if not cat_cols:
        st.warning(":material/warning: No categorical columns found.")
    else:
        cat = st.selectbox("Categorical column (X)", cat_cols)
        val = st.selectbox("Numeric column (Y)", numeric_cols)
        agg = st.selectbox("Aggregation", ["mean", "sum", "count", "median", "max", "min"])
        top_n = st.slider("Top N categories", 5, 30, 10)

        agg_df = df.groupby(cat)[val].agg(agg).reset_index()
        agg_df.columns = [cat, val]
        agg_df = agg_df.nlargest(top_n, val)

        fig = px.bar(
            agg_df, x=cat, y=val,
            template="plotly_white",
            title=f"{agg.capitalize()} of {val} by {cat}"
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Add to Report", icon=":material/picture_as_pdf:", key="add_bar"):
            add_to_report(fig, f"Bar Chart — {agg} of {val} by {cat}")
            st.rerun()

# ── Correlation Heatmap ──────────────────────────────────────
elif chart_type == "Correlation Heatmap":
    st.markdown(":material/grid_on: **Correlation Heatmap Settings**")
    if len(numeric_cols) < 2:
        st.warning(":material/warning: Need at least 2 numeric columns.")
    else:
        corr = df[numeric_cols].corr().round(2)
        fig = px.imshow(
            corr, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Add to Report", icon=":material/picture_as_pdf:", key="add_heatmap"):
            add_to_report(fig, "Correlation Heatmap")
            st.rerun()

# ── Pie Chart ────────────────────────────────────────────────
elif chart_type == "Pie Chart":
    st.markdown(":material/pie_chart: **Pie Chart Settings**")
    if not cat_cols:
        st.warning(":material/warning: No categorical columns found.")
    else:
        cat = st.selectbox("Categorical column", cat_cols)
        top_n = st.slider("Top N categories", 3, 15, 7)
        vc = df[cat].value_counts().nlargest(top_n).reset_index()
        vc.columns = [cat, "Count"]
        fig = px.pie(
            vc, names=cat, values="Count",
            title=f"Distribution of {cat}",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Add to Report", icon=":material/picture_as_pdf:", key="add_pie"):
            add_to_report(fig, f"Pie Chart — {cat}")
            st.rerun()

# ── Line Chart ───────────────────────────────────────────────
elif chart_type == "Line Chart":
    st.markdown(":material/show_chart: **Line Chart Settings**")
    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X axis", df.columns.tolist())
    with col2:
        y_col = st.selectbox("Y axis", numeric_cols)
    color = st.selectbox("Color by (optional)", ["None"] + cat_cols)
    fig = px.line(
        df, x=x_col, y=y_col,
        color=None if color == "None" else color,
        template="plotly_white",
        title=f"{y_col} over {x_col}"
    )
    st.plotly_chart(fig, use_container_width=True)
    if st.button("Add to Report", icon=":material/picture_as_pdf:", key="add_line"):
        add_to_report(fig, f"Line Chart — {y_col} over {x_col}")
        st.rerun()

# ── Queued Charts Preview ─────────────────────────────────────
if st.session_state.report_charts:
    st.divider()
    st.subheader(":material/picture_as_pdf: Charts Queued for Report")
    st.caption(f"{len(st.session_state.report_charts)} chart(s) will be included in your PDF report.")

    for i, chart in enumerate(st.session_state.report_charts):
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.markdown(f":material/bar_chart: **{chart['title']}**")
        with col_b:
            if st.button("Remove", key=f"remove_{i}", icon=":material/delete:"):
                st.session_state.report_charts.pop(i)
                st.rerun()