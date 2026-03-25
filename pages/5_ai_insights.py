import streamlit as st
import numpy as np
import pandas as pd
from utils.ai import build_dataset_summary, build_prompt, stream_insights
from utils.notepad import render_notepad

render_notepad()
st.set_page_config(page_title="AI Insights", page_icon=":material/auto_awesome:", layout="wide")

st.title(":material/auto_awesome: AI-Powered Insights")
st.markdown("Get intelligent analysis and recommendations.")

# Guard
if st.session_state.get("df") is None:
    st.warning(":material/warning: No data loaded. Please upload a file first.")
    st.stop()

df = st.session_state.df

st.divider()

# Decision Support
st.subheader(":material/support_agent: Decision Support")
st.caption("Automated recommendations and readiness assessment based on your dataset.")

if st.button("Run Decision Support Analysis", icon=":material/rocket_launch:", type="primary"):

    df = st.session_state.df
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols     = df.select_dtypes(include="object").columns.tolist()
    total_cells  = df.shape[0] * df.shape[1]
    missing_pct  = df.isnull().sum().sum() / total_cells * 100
    dup_pct      = df.duplicated().sum() / len(df) * 100

    # 1. Data Quality Checklist
    st.markdown("#### :material/checklist: Data Quality Checklist")

    checks = [
        {
            "check" : "No missing values",
            "passed" : df.isnull().sum().sum() == 0,
            "detail" : f"{df.isnull().sum().sum()} missing values ({missing_pct:.1f}%)"
        },
        {
            "check" : "No duplicate rows",
            "passed" : df.duplicated().sum() == 0,
            "detail" : f"{df.duplicated().sum()} duplicate rows ({dup_pct:.1f}%)"
        },
        {
            "check" : "Sufficient rows (≥ 100)",
            "passed" : len(df) >= 100,
            "detail" : f"{len(df)} rows"
        },
        {
            "check" : "Has numeric columns",
            "passed" : len(numeric_cols) > 0,
            "detail" : f"{len(numeric_cols)} numeric columns"
        },
        {
            "check" : "Has categorical columns",
            "passed" : len(cat_cols) > 0,
            "detail" : f"{len(cat_cols)} categorical columns"
        },
        {
            "check" : "Low missing % (< 5%)",
            "passed" : missing_pct < 5,
            "detail" : f"{missing_pct:.1f}% missing"
        },
        {
            "check" : "No high cardinality columns (< 50 unique)",
            "passed" : all(df[c].nunique() < 50 for c in cat_cols) if cat_cols else True,
            "detail" : ", ".join([f"{c} ({df[c].nunique()})" for c in cat_cols if df[c].nunique() >= 50]) or "All good"
        },
        {
            "check" : "No constant columns",
            "passed" : all(df[c].nunique() > 1 for c in df.columns),
            "detail" : ", ".join([c for c in df.columns if df[c].nunique() == 1]) or "None found"
        },
    ]

    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)

    score_pct = int(passed / total * 100)
    col1, col2 = st.columns([1, 3])
    col1.metric("Quality Score", f"{score_pct}%")
    with col2:
        if score_pct >= 80:
            st.success(f":material/check_circle: Good quality dataset — {passed}/{total} checks passed")
        elif score_pct >= 50:
            st.warning(f":material/warning: Moderate quality — {passed}/{total} checks passed")
        else:
            st.error(f":material/error: Poor quality — {passed}/{total} checks passed. Clean before analysis.")

    checklist_df = pd.DataFrame({
        "Check" : [c["check"]  for c in checks],
        "Status" : ["✅ Pass" if c["passed"] else "❌ Fail" for c in checks],
        "Detail" : [c["detail"] for c in checks],
    })
    st.dataframe(checklist_df, use_container_width=True)

    st.divider()

    # 2. Column-Level Recommendations
    st.markdown("#### :material/table_chart: Column-Level Recommendations")

    col_recs = []
    for col in df.columns:
        recs = []
        dtype = df[col].dtype
        null_pct = df[col].isnull().sum() / len(df) * 100

        if null_pct > 0:
            strategy = "mean/median" if pd.api.types.is_numeric_dtype(dtype) else "mode or 'Unknown'"
            recs.append(f"Fill {null_pct:.1f}% missing values with {strategy}")

        if pd.api.types.is_numeric_dtype(dtype):
            skew = float(df[col].skew())
            if abs(skew) > 1:
                recs.append(f"Highly skewed (skew={skew:.2f}) — consider log transform")
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = len(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)])
            if outliers > 0:
                recs.append(f"{outliers} outliers detected — consider removal or robust scaling")
            recs.append("Apply Min-Max or Z-Score scaling before ML")

        elif dtype == object:
            nunique = df[col].nunique()
            if nunique == len(df):
                recs.append("All values unique — likely an ID column, consider dropping")
            elif nunique == 1:
                recs.append("Constant column — drop it, adds no information")
            elif nunique <= 10:
                recs.append(f"Low cardinality ({nunique} values) — good for One-Hot Encoding")
            else:
                recs.append(f"High cardinality ({nunique} values) — use Label Encoding")

        if not recs:
            recs.append("Looks good — no issues detected")

        col_recs.append({
            "Column" : col,
            "Type" : str(dtype),
            "Recommendations" : " · ".join(recs)
        })

    st.dataframe(pd.DataFrame(col_recs), use_container_width=True)

    st.divider()

    # 3. Smart Next-Step Suggestions
    st.markdown("#### :material/lightbulb: Smart Next-Step Suggestions")

    steps = []

    if df.isnull().sum().sum() > 0:
        steps.append(("High", ":material/find_replace:", "Handle missing values",
                       f"{df.isnull().sum().sum()} missing values found — go to Cleaning page"))

    if df.duplicated().sum() > 0:
        steps.append(("High", ":material/content_copy:", "Remove duplicate rows",
                       f"{df.duplicated().sum()} duplicates found — go to Cleaning page"))

    skewed = [c for c in numeric_cols if abs(float(df[c].skew())) > 1]
    if skewed:
        steps.append(("Medium", ":material/auto_fix_high:", "Fix skewed columns",
                       f"{', '.join(skewed[:3])}{'...' if len(skewed) > 3 else ''} — apply log transform or robust scaling"))

    if len(numeric_cols) >= 2:
        steps.append(("Medium", ":material/straighten:", "Scale numeric columns",
                       "Apply Min-Max or Z-Score scaling before any ML modeling"))

    if cat_cols:
        steps.append(("Medium", ":material/tag:", "Encode categorical columns",
                       f"{len(cat_cols)} categorical columns — apply Label or One-Hot Encoding for ML"))

    if len(numeric_cols) >= 2:
        steps.append(("Low", ":material/hub:", "Run K-Means Clustering",
                       "Find natural groupings in your data — go to Analysis page"))

    if len(numeric_cols) >= 2:
        steps.append(("Low", ":material/star:", "Check Feature Importance",
                       "Identify which columns matter most — go to Analysis page"))

    steps.append(("Low", ":material/auto_awesome:", "Generate AI Insights",
                   "Ask Claude to give a full analysis of your dataset"))

    for priority, icon, title, detail in steps:
        if priority == "High":
            st.error(f"{icon} **{title}** — {detail}")
        elif priority == "Medium":
            st.warning(f"{icon} **{title}** — {detail}")
        else:
            st.info(f"{icon} **{title}** — {detail}")

    st.divider()

    # 4. ML Readiness Score
    st.markdown("#### :material/model_training: ML Readiness Score")

    ml_checks = [
        {
            "criterion": "Sufficient data (≥ 100 rows)",
            "score" : min(len(df) / 1000, 1.0),
            "passed" : len(df) >= 100,
            "detail" : f"{len(df)} rows — {'good' if len(df) >= 500 else 'minimum viable' if len(df) >= 100 else 'too few'}"
        },
        {
            "criterion": "Low missing values (< 5%)",
            "score" : max(0, 1 - missing_pct / 100),
            "passed" : missing_pct < 5,
            "detail" : f"{missing_pct:.1f}% missing"
        },
        {
            "criterion": "No duplicates",
            "score" : max(0, 1 - dup_pct / 100),
            "passed" : dup_pct < 1,
            "detail" : f"{dup_pct:.1f}% duplicates"
        },
        {
            "criterion": "Has numeric features",
            "score" : min(len(numeric_cols) / 5, 1.0),
            "passed" : len(numeric_cols) >= 2,
            "detail" : f"{len(numeric_cols)} numeric columns"
        },
        {
            "criterion": "No constant columns",
            "score" : 1.0 if all(df[c].nunique() > 1 for c in df.columns) else 0.5,
            "passed" : all(df[c].nunique() > 1 for c in df.columns),
            "detail" : "All columns have variance"
        },
        {
            "criterion": "Reasonable feature count (≤ 50)",
            "score" : 1.0 if len(df.columns) <= 50 else 0.5,
            "passed" : len(df.columns) <= 50,
            "detail" : f"{len(df.columns)} columns"
        },
    ]

    ml_score = round(sum(c["score"] for c in ml_checks) / len(ml_checks) * 100)

    col1, col2 = st.columns([1, 3])
    col1.metric("ML Readiness", f"{ml_score}%")
    with col2:
        if ml_score >= 80:
            st.success(":material/check_circle: Dataset is **ML ready** — go ahead and model!")
        elif ml_score >= 50:
            st.warning(":material/warning: Dataset needs **some preparation** before modeling")
        else:
            st.error(":material/error: Dataset **not ready** for ML — clean and transform first")

    ml_df = pd.DataFrame({
        "Criterion" : [c["criterion"] for c in ml_checks],
        "Status" : ["✅" if c["passed"] else "❌" for c in ml_checks],
        "Score" : [f"{round(c['score']*100)}%" for c in ml_checks],
        "Detail" : [c["detail"] for c in ml_checks],
    })
    st.dataframe(ml_df, use_container_width=True)

st.divider()

# Insight Type Selector
st.subheader(":material/lightbulb: What would you like to know?")

insight_type = st.selectbox("Choose an insight type", [
    "Full Dataset Report",
    "Data Quality Assessment",
    "Key Trends & Patterns",
    "Cleaning Recommendations",
    "Feature Importance Suggestions",
    "Custom Question"
])

custom_question = ""
if insight_type == "Custom Question":
    custom_question = st.text_area(
        "Ask anything about your data",
        placeholder="e.g. What columns are most important for predicting sales?"
    )

response_length = st.select_slider(
    "Response length",
    options=["Short", "Medium", "Detailed"],
    value="Medium"
)

length_map = {
    "Short"   : 500,
    "Medium"  : 1000,
    "Detailed": 2000
}

# Generate Insights
st.divider()
if st.button("Generate AI Insights", type="primary", icon=":material/rocket_launch:"):
    if insight_type == "Custom Question" and not custom_question.strip():
        st.warning(":material/warning: Please enter your custom question.")
    else:
        summary = build_dataset_summary(df)
        prompt = build_prompt(
            insight_type,
            summary,
            df.columns.tolist(),
            custom_question
        )

        with st.spinner("Analyzing your data..."):
            try:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""

                    for text in stream_insights(api_key=st.secrets.get("NVIDIA API KEY",""), prompt=prompt, max_tokens=length_map[response_length]):
                        full_response += text
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                    st.session_state["latest_insight"] = full_response

                if "insight_history" not in st.session_state:
                    st.session_state.insight_history = []

                st.session_state.insight_history.append({
                    "type": insight_type,
                    "question": custom_question if insight_type == "Custom Question" else "",
                    "response": full_response
                })
                st.success(":material/check_circle: Insight generated and saved to history!")

            except Exception as e:
                st.error(f":material/error: API Error: {e}")

st.divider()

# Insight History
if st.session_state.get("insight_history"):
    st.subheader(":material/history: Previous Insights")

    for i, item in enumerate(reversed(st.session_state.insight_history)):
        actual_index = len(st.session_state.insight_history) - 1 - i

        label = f"{item['type']} — #{actual_index + 1}"
        if item.get("question"):
            label += f" · _{item['question'][:50]}_"

        with st.expander(label):
            st.markdown(item["response"])

            if st.button("Remove", key=f"remove_insight_{actual_index}", icon=":material/delete:"):
                st.session_state.insight_history.pop(actual_index)
                st.rerun()

    if st.button("Clear All History", icon=":material/delete_forever:"):
        st.session_state.insight_history = []
        st.rerun()
else:
    st.info(":material/info: No insights generated yet. Hit **Generate AI Insights** to get started!")

st.divider()

# PDF Report Export
st.subheader(":material/picture_as_pdf: Export PDF Report")
st.caption("Generate a complete analysis report of your dataset as a downloadable PDF.")

report_title = st.text_input("Report Title", value=f"Data Analysis Report — {st.session_state.get('filename', 'dataset')}")
include_notes = st.checkbox("Include my notes from notepad", value=True)

if st.button("Generate PDF Report", icon=":material/picture_as_pdf:", type="primary"):
    import io
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, PageBreak, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import Image as RLImage
    from datetime import datetime

    df = st.session_state.df
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    # Styles
    styles  = getSampleStyleSheet()
    style_title = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=22, spaceAfter=6,
        textColor=colors.HexColor("#1D9E75"),
        alignment=TA_CENTER
    )
    style_h1 = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=14, spaceBefore=14, spaceAfter=4,
        textColor=colors.HexColor("#0F6E56")
    )
    style_h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=11, spaceBefore=8, spaceAfter=3,
        textColor=colors.HexColor("#444441")
    )
    style_body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9, spaceAfter=4, leading=14
    )
    style_caption = ParagraphStyle(
        "Caption", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#888780"),
        spaceAfter=4, alignment=TA_CENTER
    )

    def section_divider():
        return HRFlowable(
            width="100%", thickness=0.5,
            color=colors.HexColor("#D3D1C7"),
            spaceAfter=6, spaceBefore=6
        )

    def make_table(data, col_widths=None):
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0),  colors.HexColor("#1D9E75")),
            ("TEXTCOLOR", (0,0), (-1,0),  colors.white),
            ("FONTNAME", (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0),  8),
            ("FONTSIZE", (0,1), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F1EFE8")]),
            ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#D3D1C7")),
            ("LEFTPADDING", (0,0), (-1,-1), 5),
            ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0), (-1,-1), 3),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        return t

    story = []
    W = A4[0] - 4*cm

    story.append(Spacer(1, 3*cm))
    story.append(Paragraph(report_title, style_title))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="60%", thickness=1.5, color=colors.HexColor("#1D9E75"), hAlign="CENTER"))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", style_caption))
    story.append(Paragraph(f"File: {st.session_state.get('filename', 'Unknown')}", style_caption))
    story.append(Spacer(1, 1*cm))

    cover_data = [
        ["Metric", "Value"],
        ["Total Rows", str(df.shape[0])],
        ["Total Columns", str(df.shape[1])],
        ["Numeric Columns", str(len(numeric_cols))],
        ["Categorical Columns", str(len(cat_cols))],
        ["Missing Values", str(df.isnull().sum().sum())],
        ["Duplicate Rows", str(df.duplicated().sum())],
        ["Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"],
    ]
    story.append(make_table(cover_data, col_widths=[W*0.5, W*0.5]))
    story.append(PageBreak())

    # 1. Dataset Overview
    story.append(Paragraph("1. Dataset Overview", style_h1))
    story.append(section_divider())

    story.append(Paragraph("Column Information", style_h2))
    col_data = [["Column", "Type", "Non-Null", "Null %", "Unique"]]
    for col in df.columns:
        null_pct = round(df[col].isnull().sum() / len(df) * 100, 1)
        col_data.append([
            col,
            str(df[col].dtype),
            str(df[col].notnull().sum()),
            f"{null_pct}%",
            str(df[col].nunique())
        ])
    story.append(make_table(col_data, col_widths=[W*0.3, W*0.15, W*0.15, W*0.15, W*0.15]))
    story.append(Spacer(1, 0.3*cm))

    # Sample data
    story.append(Paragraph("Sample Data (first 5 rows)", style_h2))
    sample = df.head(5)
    sample_cols = list(sample.columns[:8])  # max 8 cols to fit page
    sample_data  = [sample_cols]
    for _, row in sample[sample_cols].iterrows():
        sample_data.append([str(v)[:15] for v in row.values])
    col_w = W / len(sample_cols)
    story.append(make_table(sample_data, col_widths=[col_w]*len(sample_cols)))
    story.append(PageBreak())

    # 2. Descriptive Statistics
    if numeric_cols:
        story.append(Paragraph("2. Descriptive Statistics", style_h1))
        story.append(section_divider())

        desc = df[numeric_cols].describe().T.round(3)
        desc["skewness"] = df[numeric_cols].skew().round(3)
        desc = desc.reset_index()
        desc.columns = ["Column"] + list(desc.columns[1:])

        stat_cols = ["Column", "count", "mean", "std", "min", "25%", "50%", "75%", "max", "skewness"]
        stat_data = [stat_cols]
        for _, row in desc.iterrows():
            stat_data.append([str(row.get(c, ""))[:10] for c in stat_cols])

        cw = W / len(stat_cols)
        story.append(make_table(stat_data, col_widths=[cw*1.5] + [cw*0.85]*(len(stat_cols)-1)))
        story.append(PageBreak())

    # 3. Data Quality Checklist
    story.append(Paragraph("3. Data Quality Checklist", style_h1))
    story.append(section_divider())

    total_cells = df.shape[0] * df.shape[1]
    missing_pct = df.isnull().sum().sum() / total_cells * 100
    dup_pct = df.duplicated().sum() / len(df) * 100

    checks = [
        ("No missing values", df.isnull().sum().sum() == 0, f"{df.isnull().sum().sum()} missing ({missing_pct:.1f}%)"),
        ("No duplicate rows", df.duplicated().sum() == 0, f"{df.duplicated().sum()} duplicates ({dup_pct:.1f}%)"),
        ("Sufficient rows (>= 100)", len(df) >= 100, f"{len(df)} rows"),
        ("Has numeric columns", len(numeric_cols) > 0, f"{len(numeric_cols)} numeric cols"),
        ("Has categorical columns", len(cat_cols) > 0, f"{len(cat_cols)} categorical cols"),
        ("Low missing % (< 5%)", missing_pct < 5, f"{missing_pct:.1f}% missing"),
        ("No constant columns", all(df[c].nunique() > 1 for c in df.columns), "All columns have variance"),
    ]

    passed = sum(1 for _, p, _ in checks if p)
    story.append(Paragraph(f"Quality Score: {round(passed/len(checks)*100)}% ({passed}/{len(checks)} checks passed)", style_body))
    story.append(Spacer(1, 0.2*cm))

    check_data = [["Check", "Status", "Detail"]]
    for name, passed_val, detail in checks:
        check_data.append([name, "PASS" if passed_val else "FAIL", detail])

    t = Table(check_data, colWidths=[W*0.45, W*0.1, W*0.45], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0),  colors.HexColor("#1D9E75")),
        ("TEXTCOLOR", (0,0), (-1,0),  colors.white),
        ("FONTNAME", (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("GRID", (0,0), (-1,-1), 0.3, colors.HexColor("#D3D1C7")),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F1EFE8")]),
    ]))
    for i, (_, passed_val, _) in enumerate(checks):
        color = colors.HexColor("#EAF3DE") if passed_val else colors.HexColor("#FCEBEB")
        t.setStyle(TableStyle([("BACKGROUND", (1, i+1), (1, i+1), color)]))
    story.append(t)
    story.append(PageBreak())

    # 4. Column Recommendations
    story.append(Paragraph("4. Column-Level Recommendations", style_h1))
    story.append(section_divider())

    rec_data = [["Column", "Type", "Recommendations"]]
    for col in df.columns:
        recs = []
        dtype = df[col].dtype
        null_pct = df[col].isnull().sum() / len(df) * 100
        if null_pct > 0:
            strategy = "mean/median" if pd.api.types.is_numeric_dtype(dtype) else "mode"
            recs.append(f"Fill {null_pct:.1f}% nulls with {strategy}")
        if pd.api.types.is_numeric_dtype(dtype):
            skew = float(df[col].skew())
            if abs(skew) > 1:
                recs.append(f"Skewed ({skew:.2f}) — log transform")
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = len(df[(df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)])
            if outliers > 0:
                recs.append(f"{outliers} outliers — check IQR")
            recs.append("Scale before ML")
        elif dtype == object:
            nunique = df[col].nunique()
            if nunique == len(df):
                recs.append("Unique ID — consider dropping")
            elif nunique <= 10:
                recs.append(f"Low cardinality ({nunique}) — One-Hot")
            else:
                recs.append(f"High cardinality ({nunique}) — Label Encode")
        if not recs:
            recs.append("No issues detected")
        rec_data.append([col, str(dtype), " | ".join(recs)])

    story.append(make_table(rec_data, col_widths=[W*0.2, W*0.12, W*0.68]))
    story.append(PageBreak())

    # 5. Correlation Matrix
    if len(numeric_cols) >= 2:
        story.append(Paragraph("5. Correlation Matrix", style_h1))
        story.append(section_divider())

        corr = df[numeric_cols].corr().round(3)
        corr_cols = list(corr.columns)
        corr_data = [[""] + corr_cols]
        for col in corr_cols:
            corr_data.append([col] + [str(corr[col][r]) for r in corr_cols])

        cw = W / (len(corr_cols) + 1)
        t = Table(corr_data, colWidths=[cw]*(len(corr_cols)+1), repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#1D9E75")),
            ("BACKGROUND",  (0,0), (0,-1),  colors.HexColor("#1D9E75")),
            ("TEXTCOLOR",   (0,0), (-1,0),  colors.white),
            ("TEXTCOLOR",   (0,0), (0,-1),  colors.white),
            ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTNAME",    (0,0), (0,-1),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 7),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#D3D1C7")),
            ("ALIGN",       (0,0), (-1,-1), "CENTER"),
            ("LEFTPADDING",  (0,0), (-1,-1), 3),
            ("RIGHTPADDING", (0,0), (-1,-1), 3),
            ("TOPPADDING",   (0,0), (-1,-1), 2),
            ("BOTTOMPADDING",(0,0), (-1,-1), 2),
            ("ROWBACKGROUNDS", (1,1), (-1,-1), [colors.white, colors.HexColor("#F1EFE8")]),
        ]))
        for i, r in enumerate(corr_cols):
            for j, c in enumerate(corr_cols):
                val = corr[c][r]
                if i != j:
                    if abs(val) >= 0.7:
                        bg = colors.HexColor("#C0DD97") if val > 0 else colors.HexColor("#F7C1C1")
                        t.setStyle(TableStyle([("BACKGROUND", (j+1, i+1), (j+1, i+1), bg)]))
        story.append(t)
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph("Green = strong positive (>= 0.7) · Red = strong negative (<= -0.7)", style_caption))
        story.append(PageBreak())

    # 6. Charts
    story.append(Paragraph("6. Visualizations", style_h1))
    story.append(section_divider())

    if st.session_state.get("report_charts"):
        for chart in st.session_state.report_charts:
            import io as _io
            story.append(Paragraph(chart["title"], style_h2))
            img_buf = _io.BytesIO(chart["img"])
            story.append(RLImage(img_buf, width=W, height=W*0.55))
            story.append(Paragraph(chart["title"], style_caption))
            story.append(Spacer(1, 0.4*cm))
            story.append(PageBreak())
    else:
        story.append(Paragraph(
            "No charts were added to the report. Go to the Visualizations page and click 'Add to Report' on any chart.",
            style_body
        ))
    story.append(PageBreak())

    # 7. AI Insights Summary
    if st.session_state.get("insight_history"):
        story.append(Paragraph("7. AI Insights Summary", style_h1))
        story.append(section_divider())

        for item in st.session_state.insight_history[-3:]:  # last 3 insights
            story.append(Paragraph(f"{item['type']}", style_h2))
            if item.get("question"):
                story.append(Paragraph(f"Question: {item['question']}", style_body))
            # Clean markdown for PDF
            clean = item["response"].replace("**", "").replace("##", "").replace("#", "").replace("*", "")
            for line in clean.split("\n"):
                line = line.strip()
                if line:
                    story.append(Paragraph(line[:500], style_body))
            story.append(Spacer(1, 0.3*cm))
        story.append(PageBreak())

    # 8. User Notes
    if include_notes and st.session_state.get("notepad_content", "").strip():
        story.append(Paragraph("8. My Notes", style_h1))
        story.append(section_divider())
        notes = st.session_state.notepad_content
        for line in notes.split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line, style_body))
        story.append(PageBreak())

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    st.session_state.pdf_buffer = buffer.getvalue()
    st.session_state.msg_pdf = ":material/check_circle: PDF report generated successfully!"
    st.rerun()

pdf_msg = st.empty()
if st.session_state.get("msg_pdf"):
    pdf_msg.success(st.session_state.msg_pdf)
    st.session_state.msg_pdf = ""

if st.session_state.get("pdf_buffer"):
    st.download_button(
        label="Download PDF Report",
        data=st.session_state.pdf_buffer,
        file_name=f"report_{st.session_state.get('filename', 'data').replace('.csv','').replace('.xlsx','')}.pdf",
        mime="application/pdf",
        icon=":material/download:",
        type="primary",
        use_container_width=False
    )
