import streamlit as st
import pandas as pd
import numpy as np
from utils.notepad import render_notepad
render_notepad()

st.set_page_config(page_title="Data Cleaning", page_icon=":material/cleaning_services:", layout="wide")

st.title(":material/cleaning_services: Data Cleaning")

# ── Guard ────────────────────────────────────────────────────
if st.session_state.get("df") is None:
    st.warning(":material/warning: No data loaded. Please upload a file first.")
    st.stop()

# ── Undo History Init ────────────────────────────────────────
if "df_history" not in st.session_state:
    st.session_state.df_history = []

def save_snapshot():
    st.session_state.df_history.append(st.session_state.df.copy())

def undo():
    if st.session_state.df_history:
        st.session_state.df = st.session_state.df_history.pop()

# ── Meta Info ────────────────────────────────────────────────
df = st.session_state.df.copy()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

st.info(f":material/dataset: Working on: **{st.session_state.get('filename', 'dataset')}** — {df.shape[0]} rows × {df.shape[1]} columns")

# ── Two Column Layout ────────────────────────────────────────
left, right = st.columns([1.2, 1.8], gap="large")

with right:
    def show_live_preview():
        current_df = st.session_state.df
        st.subheader(":material/visibility: Live Dataset Preview")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", current_df.shape[0])
        m2.metric("Columns", current_df.shape[1])
        m3.metric("Missing", current_df.isnull().sum().sum())
        m4.metric("Duplicates", current_df.duplicated().sum())
        st.dataframe(current_df, use_container_width=True, height=500)
    show_live_preview()

with left:
    # ── Cleaning Controls Header + Undo ──────────────────────
    st.subheader(":material/build: Cleaning Controls")

    undo_steps = len(st.session_state.df_history)
    col_undo, col_info = st.columns([1, 2])
    with col_undo:
        if st.button("Undo", icon=":material/undo:", disabled=undo_steps == 0):
            undo()
            st.rerun()
    with col_info:
        if undo_steps > 0:
            st.caption(f":material/history: **{undo_steps}** action(s) can be undone")
        else:
            st.caption(":material/block: Nothing to undo")

    st.divider()

    # ── 1. Remove Duplicates ─────────────────────────────────
    st.markdown("#### :material/content_copy: Duplicate Rows")
    dup_count = st.session_state.df.duplicated().sum()
    st.write(f"Found **{dup_count}** duplicate rows.")
    if dup_count > 0:
        if st.button("Remove Duplicates", icon=":material/delete:"):
            save_snapshot()
            before = len(st.session_state.df)
            st.session_state.df = st.session_state.df.drop_duplicates()
            after = len(st.session_state.df)
            st.session_state.msg_dup = f":material/check_circle: Removed **{dup_count}** duplicate rows — {before} rows → {after} rows"
            st.rerun()
    dup_msg = st.empty()
    if st.session_state.get("msg_dup"):
        dup_msg.success(st.session_state.msg_dup)
        st.session_state.msg_dup = ""

    st.divider()

    # ── 2. Handle Missing Values ─────────────────────────────
    st.markdown("#### :material/find_replace: Handle Missing Values")
    missing = st.session_state.df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success(":material/celebration: No missing values found!")
    else:
        st.write(f"Found missing values in **{len(missing)}** columns.")
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Select column", missing.index.tolist())
        with col2:
            col_type = st.session_state.df[target_col].dtype
            if pd.api.types.is_numeric_dtype(col_type):
                strategy = st.selectbox("Strategy", ["Mean", "Median", "Mode", "Fill with 0", "Drop Rows"])
            else:
                strategy = st.selectbox("Strategy", ["Mode", "Fill with 'Unknown'", "Drop Rows"])

        if st.button("Apply Missing Value Fix", icon=":material/check_circle:"):
            before_nulls = int(st.session_state.df[target_col].isnull().sum())
            save_snapshot()
            df_temp = st.session_state.df.copy()
            if strategy == "Mean":
                df_temp[target_col].fillna(df_temp[target_col].mean(), inplace=True)
            elif strategy == "Median":
                df_temp[target_col].fillna(df_temp[target_col].median(), inplace=True)
            elif strategy == "Mode":
                df_temp[target_col].fillna(df_temp[target_col].mode()[0], inplace=True)
            elif strategy == "Fill with 0":
                df_temp[target_col].fillna(0, inplace=True)
            elif strategy == "Fill with 'Unknown'":
                df_temp[target_col].fillna("Unknown", inplace=True)
            elif strategy == "Drop Rows":
                df_temp = df_temp.dropna(subset=[target_col])
            st.session_state.df = df_temp
            st.session_state.msg_missing = f":material/check_circle: Fixed **{before_nulls}** missing values in **{target_col}** using **{strategy}**"
            st.rerun()
        missing_msg = st.empty()
        if st.session_state.get("msg_missing"):
            missing_msg.success(st.session_state.msg_missing)
            st.session_state.msg_missing = ""

    st.divider()

    # ── 3. Fix Data Types ─────────────────────────────────────
    st.markdown("#### :material/transform: Fix Column Data Types")
    col1, col2 = st.columns(2)
    with col1:
        type_col = st.selectbox("Select column", st.session_state.df.columns.tolist(), key="type_col")
        st.caption(f"Current type: `{st.session_state.df[type_col].dtype}`")
    with col2:
        new_type = st.selectbox("Convert to", ["int", "float", "string", "datetime", "category"])

    if st.button("Convert Type", icon=":material/sync:"):
        try:
            old_type = str(st.session_state.df[type_col].dtype)
            save_snapshot()
            df_temp = st.session_state.df.copy()
            if new_type == "int":
                df_temp[type_col] = pd.to_numeric(df_temp[type_col], errors="coerce").astype("Int64")
            elif new_type == "float":
                df_temp[type_col] = pd.to_numeric(df_temp[type_col], errors="coerce")
            elif new_type == "string":
                df_temp[type_col] = df_temp[type_col].astype(str)
            elif new_type == "datetime":
                df_temp[type_col] = pd.to_datetime(df_temp[type_col], errors="coerce")
            elif new_type == "category":
                df_temp[type_col] = df_temp[type_col].astype("category")
            st.session_state.df = df_temp
            st.session_state.msg_type = f":material/check_circle: Converted **{type_col}** — `{old_type}` → `{new_type}`"
            st.rerun()
        except Exception as e:
            st.session_state.df_history.pop()
            st.error(f":material/error: Conversion failed: {e}")
    type_msg = st.empty()
    if st.session_state.get("msg_type"):
        type_msg.success(st.session_state.msg_type)
        st.session_state.msg_type = ""

    st.divider()

    # ── 4. Standardize ────────────────────────────────────────
    st.markdown("#### :material/auto_fix_high: Standardize Column")

    std_col = st.selectbox("Select column to standardize", st.session_state.df.columns.tolist(), key="std_col")
    std_method = st.selectbox("Standardization method", [
        "Lowercase", "Uppercase", "Title Case",
        "Strip Whitespace", "Remove Extra Spaces",
        "Remove Special Characters", "Replace Value",
        "Date Format (strftime)",
    ])

    find_val, replace_val, input_fmt, output_fmt = "", "", "", ""

    if std_method == "Replace Value":
        col1, col2 = st.columns(2)
        with col1:
            find_val = st.text_input("Find value", placeholder="e.g. N/A")
        with col2:
            replace_val = st.text_input("Replace with", placeholder="e.g. Unknown")
    elif std_method == "Date Format (strftime)":
        col1, col2 = st.columns(2)
        with col1:
            input_fmt = st.text_input("Input format", placeholder="e.g. %d/%m/%Y")
        with col2:
            output_fmt = st.text_input("Output format", placeholder="e.g. %Y-%m-%d")
        st.caption(":material/info: `%d` day · `%m` month · `%Y` 4-digit year · `%H` hour · `%M` minute")

    def apply_std(series):
        if std_method == "Lowercase":
            return series.str.lower()
        elif std_method == "Uppercase":
            return series.str.upper()
        elif std_method == "Title Case":
            return series.str.title()
        elif std_method == "Strip Whitespace":
            return series.str.strip()
        elif std_method == "Remove Extra Spaces":
            return series.str.replace(r"\s+", " ", regex=True).str.strip()
        elif std_method == "Remove Special Characters":
            return series.str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
        elif std_method == "Replace Value":
            return series.replace(find_val, replace_val)
        elif std_method == "Date Format (strftime)":
            return pd.to_datetime(series, format=input_fmt, errors="coerce").dt.strftime(output_fmt)
        return series

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Preview", icon=":material/preview:", use_container_width=True):
            try:
                preview_series = apply_std(st.session_state.df[std_col].copy())
                preview_df = pd.DataFrame({
                    "Before": st.session_state.df[std_col].head(10).values,
                    "After" : preview_series.head(10).values
                })
                st.dataframe(preview_df, use_container_width=True)
            except Exception as e:
                st.error(f":material/error: Preview failed: {e}")
    with col2:
        if st.button("Apply", icon=":material/check_circle:", use_container_width=True):
            try:
                if std_method == "Replace Value" and not find_val.strip():
                    st.warning(":material/warning: Please enter a value to find.")
                elif std_method == "Date Format (strftime)" and (not input_fmt or not output_fmt):
                    st.warning(":material/warning: Please enter both input and output formats.")
                else:
                    save_snapshot()
                    df_temp = st.session_state.df.copy()
                    before_sample = str(df_temp[std_col].iloc[0]) if len(df_temp) > 0 else ""
                    df_temp[std_col] = apply_std(df_temp[std_col])
                    after_sample = str(df_temp[std_col].iloc[0]) if len(df_temp) > 0 else ""
                    st.session_state.df = df_temp
                    st.session_state.msg_std = f":material/check_circle: Applied **{std_method}** to **{std_col}** — e.g. `{before_sample}` → `{after_sample}`"
                    st.rerun()
            except Exception as e:
                st.session_state.df_history.pop()
                st.error(f":material/error: Standardization failed: {e}")
    std_msg = st.empty()
    if st.session_state.get("msg_std"):
        std_msg.success(st.session_state.msg_std)
        st.session_state.msg_std = ""

    st.divider()

    # ── 5. Rename Columns ─────────────────────────────────────
    st.markdown("#### :material/edit: Rename Column")
    col1, col2 = st.columns(2)
    with col1:
        rename_col = st.selectbox("Select column", st.session_state.df.columns.tolist(), key="rename_col")
    with col2:
        new_name = st.text_input("New name", value=rename_col)

    if st.button("Rename Column", icon=":material/edit:"):
        if new_name and new_name != rename_col:
            save_snapshot()
            st.session_state.df = st.session_state.df.rename(columns={rename_col: new_name})
            st.session_state.msg_rename = f":material/check_circle: Renamed **{rename_col}** → **{new_name}**"
            st.rerun()
        else:
            st.warning(":material/warning: Please enter a different name.")
    rename_msg = st.empty()
    if st.session_state.get("msg_rename"):
        rename_msg.success(st.session_state.msg_rename)
        st.session_state.msg_rename = ""

    st.divider()

    # ── 6. Drop Columns ───────────────────────────────────────
    st.markdown("#### :material/remove_circle: Drop Columns")
    cols_to_drop = st.multiselect("Select columns to drop", st.session_state.df.columns.tolist())
    if cols_to_drop:
        if st.button("Drop Selected Columns", icon=":material/delete:"):
            save_snapshot()
            before_cols = st.session_state.df.shape[1]
            st.session_state.df = st.session_state.df.drop(columns=cols_to_drop)
            after_cols = st.session_state.df.shape[1]
            st.session_state.msg_drop = f":material/check_circle: Dropped **{len(cols_to_drop)}** columns: `{', '.join(cols_to_drop)}` — {before_cols} cols → {after_cols} cols"
            st.rerun()
    drop_msg = st.empty()
    if st.session_state.get("msg_drop"):
        drop_msg.success(st.session_state.msg_drop)
        st.session_state.msg_drop = ""

    st.divider()

    # ── 7. Outlier Removal ────────────────────────────────────
    st.markdown("#### :material/troubleshoot: Outlier Removal (IQR)")
    current_numeric = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    if not current_numeric:
        st.warning(":material/warning: No numeric columns found.")
    else:
        out_col = st.selectbox("Select numeric column", current_numeric)
        Q1 = st.session_state.df[out_col].quantile(0.25)
        Q3 = st.session_state.df[out_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = st.session_state.df[
            (st.session_state.df[out_col] < lower) | (st.session_state.df[out_col] > upper)
        ]
        st.write(f"Detected **{len(outliers)}** outliers (range: `{lower:.2f}` – `{upper:.2f}`)")
        if len(outliers) > 0:
            if st.button("Remove Outliers", icon=":material/delete:"):
                save_snapshot()
                before_rows = len(st.session_state.df)
                st.session_state.df = st.session_state.df[
                    (st.session_state.df[out_col] >= lower) &
                    (st.session_state.df[out_col] <= upper)
                ]
                after_rows = len(st.session_state.df)
                st.session_state.msg_outlier = f":material/check_circle: Removed **{before_rows - after_rows}** outliers from **{out_col}** — {before_rows} rows → {after_rows} rows"
                st.rerun()
        outlier_msg = st.empty()
        if st.session_state.get("msg_outlier"):
            outlier_msg.success(st.session_state.msg_outlier)
            st.session_state.msg_outlier = ""

    st.divider()

    # ── 8. Numeric Scaling ────────────────────────────────────
    st.markdown("#### :material/straighten: Numeric Scaling")

    scaling_method = st.selectbox("Scaling method", [
        "Min-Max Scaling", "Z-Score Standardization", "Robust Scaling",
    ], key="scaling_method")

    current_numeric = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    if not current_numeric:
        st.warning(":material/warning: No numeric columns found.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            scale_col = st.selectbox("Select numeric column", current_numeric, key="scale_col")
        with col2:
            new_col_name = st.text_input("Output column name", value=f"{scale_col}_scaled")

        if scaling_method == "Min-Max Scaling":
            st.caption(":material/info: `(x - min) / (max - min)` → range 0 to 1")
        elif scaling_method == "Z-Score Standardization":
            st.caption(":material/info: `(x - mean) / std` → mean=0, std=1")
        elif scaling_method == "Robust Scaling":
            st.caption(":material/info: `(x - median) / IQR` → robust to outliers")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Preview", icon=":material/preview:", use_container_width=True, key="scale_preview"):
                try:
                    series = st.session_state.df[scale_col].copy()
                    if scaling_method == "Min-Max Scaling":
                        scaled = (series - series.min()) / (series.max() - series.min())
                    elif scaling_method == "Z-Score Standardization":
                        scaled = (series - series.mean()) / series.std()
                    elif scaling_method == "Robust Scaling":
                        q1, q3 = series.quantile(0.25), series.quantile(0.75)
                        iqr = q3 - q1
                        scaled = (series - series.median()) / iqr if iqr != 0 else series * 0
                    preview_df = pd.DataFrame({
                        "Original"   : series.head(8).values,
                        "Transformed": scaled.head(8).round(4).values
                    })
                    st.dataframe(preview_df, use_container_width=True)
                    col_a, col_b = st.columns(2)
                    col_a.metric("Original mean", round(float(series.mean()), 3))
                    col_b.metric("Scaled mean",   round(float(scaled.mean()), 3))
                    col_a.metric("Original std",  round(float(series.std()), 3))
                    col_b.metric("Scaled std",    round(float(scaled.std()), 3))
                except Exception as e:
                    st.error(f":material/error: Preview failed: {e}")
        with col2:
            if st.button("Apply Scaling", icon=":material/check_circle:", use_container_width=True, key="scale_apply"):
                try:
                    save_snapshot()
                    df_temp = st.session_state.df.copy()
                    series = df_temp[scale_col]
                    old_min = round(float(series.min()), 3)
                    old_max = round(float(series.max()), 3)
                    if scaling_method == "Min-Max Scaling":
                        df_temp[new_col_name] = (series - series.min()) / (series.max() - series.min())
                    elif scaling_method == "Z-Score Standardization":
                        df_temp[new_col_name] = (series - series.mean()) / series.std()
                    elif scaling_method == "Robust Scaling":
                        q1, q3 = series.quantile(0.25), series.quantile(0.75)
                        iqr = q3 - q1
                        df_temp[new_col_name] = (series - series.median()) / iqr if iqr != 0 else series * 0
                    new_min = round(float(df_temp[new_col_name].min()), 3)
                    new_max = round(float(df_temp[new_col_name].max()), 3)
                    st.session_state.df = df_temp
                    st.session_state.msg_scale = f":material/check_circle: **{scaling_method}** on **{scale_col}** → **{new_col_name}** | Range: `[{old_min}, {old_max}]` → `[{new_min}, {new_max}]`"
                    st.rerun()
                except Exception as e:
                    st.session_state.df_history.pop()
                    st.error(f":material/error: Scaling failed: {e}")
        scale_msg = st.empty()
        if st.session_state.get("msg_scale"):
            scale_msg.success(st.session_state.msg_scale)
            st.session_state.msg_scale = ""

    st.divider()

    # ── 9. Encoding ───────────────────────────────────────────
    st.markdown("#### :material/tag: Encoding")

    encoding_method = st.selectbox("Encoding method", [
        "Label Encoding", "One-Hot Encoding",
    ], key="encoding_method")

    current_cat = st.session_state.df.select_dtypes(include="object").columns.tolist()
    if not current_cat:
        st.warning(":material/warning: No categorical columns found.")
    else:
        encode_col = st.selectbox("Select categorical column", current_cat, key="encode_col")
        unique_vals = st.session_state.df[encode_col].dropna().unique()
        st.caption(f":material/info: {len(unique_vals)} unique values: `{', '.join(map(str, unique_vals[:8]))}{'...' if len(unique_vals) > 8 else ''}`")

        if encoding_method == "Label Encoding":
            st.caption(":material/info: Assigns a unique number to each category. Original column is replaced.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Preview", icon=":material/preview:", use_container_width=True, key="label_preview"):
                    series = st.session_state.df[encode_col]
                    categories = sorted(series.dropna().unique())
                    mapping = {cat: i for i, cat in enumerate(categories)}
                    preview_df = pd.DataFrame({
                        "Original": series.head(8).values,
                        "Encoded" : series.head(8).map(mapping).values
                    })
                    st.dataframe(preview_df, use_container_width=True)
                    st.write("**Mapping:**")
                    st.dataframe(
                        pd.DataFrame({"Category": list(mapping.keys()), "Code": list(mapping.values())}),
                        use_container_width=True
                    )
            with col2:
                if st.button("Apply Encoding", icon=":material/check_circle:", use_container_width=True, key="label_apply"):
                    try:
                        save_snapshot()
                        df_temp = st.session_state.df.copy()
                        categories = sorted(df_temp[encode_col].dropna().unique())
                        mapping = {cat: i for i, cat in enumerate(categories)}
                        df_temp[encode_col] = df_temp[encode_col].map(mapping)
                        st.session_state.df = df_temp
                        st.session_state.msg_encode = f":material/check_circle: Label encoded **{encode_col}** — {len(mapping)} categories mapped to 0–{len(mapping)-1}"
                        st.rerun()
                    except Exception as e:
                        st.session_state.df_history.pop()
                        st.error(f":material/error: Encoding failed: {e}")
            encode_msg = st.empty()
            if st.session_state.get("msg_encode"):
                encode_msg.success(st.session_state.msg_encode)
                st.session_state.msg_encode = ""

        elif encoding_method == "One-Hot Encoding":
            st.caption(":material/info: Creates a new binary column per category. Original column is dropped.")
            if len(unique_vals) > 15:
                st.warning(f":material/warning: **{len(unique_vals)}** unique values — will add {len(unique_vals)} new columns. Consider label encoding instead.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Preview", icon=":material/preview:", use_container_width=True, key="ohe_preview"):
                    try:
                        sample = st.session_state.df[[encode_col]].head(8)
                        encoded = pd.get_dummies(sample, columns=[encode_col], prefix=encode_col)
                        st.dataframe(encoded, use_container_width=True)
                        st.caption(f"Will add **{len(encoded.columns)}** new columns.")
                    except Exception as e:
                        st.error(f":material/error: Preview failed: {e}")
            with col2:
                if st.button("Apply Encoding", icon=":material/check_circle:", use_container_width=True, key="ohe_apply"):
                    try:
                        save_snapshot()
                        before_cols = st.session_state.df.shape[1]
                        df_temp = st.session_state.df.copy()
                        df_temp = pd.get_dummies(df_temp, columns=[encode_col], prefix=encode_col)
                        after_cols = df_temp.shape[1]
                        st.session_state.df = df_temp
                        st.session_state.msg_ohe = f":material/check_circle: One-Hot encoded **{encode_col}** — {before_cols} cols → {after_cols} cols (+{after_cols - before_cols} new columns)"
                        st.rerun()
                    except Exception as e:
                        st.session_state.df_history.pop()
                        st.error(f":material/error: Encoding failed: {e}")
            ohe_msg = st.empty()
            if st.session_state.get("msg_ohe"):
                ohe_msg.success(st.session_state.msg_ohe)
                st.session_state.msg_ohe = ""

    st.divider()

    # ── 10. Skewness Reduction ────────────────────────────────
    st.markdown("#### :material/show_chart: Skewness Reduction")

    current_numeric = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    if not current_numeric:
        st.warning(":material/warning: No numeric columns found.")
    else:
        skew_df = pd.DataFrame({
            "Column"  : current_numeric,
            "Skewness": [round(float(st.session_state.df[c].skew()), 3) for c in current_numeric],
        })
        skew_df["Severity"] = skew_df["Skewness"].abs().apply(
            lambda x: "🔴 High" if x > 1 else ("🟡 Moderate" if x > 0.5 else "🟢 Low")
        )
        skew_df = skew_df.sort_values("Skewness", key=abs, ascending=False)
        st.caption(":material/info: Skewness overview — treat columns with High or Moderate skewness")
        st.dataframe(skew_df, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            skew_col = st.selectbox("Select column to treat", current_numeric, key="skew_col")
            current_skew = round(float(st.session_state.df[skew_col].skew()), 3)
            st.caption(f"Current skewness: `{current_skew}`")
        with col2:
            skew_method = st.selectbox("Transform method", [
                "Log Transform", "Square Root Transform", "Reciprocal Transform",
                "Box-Cox Transform", "Yeo-Johnson Transform",
            ], key="skew_method")

        hints = {
            "Log Transform"         : "`log(x)` — best for right-skewed data. Requires all values > 0.",
            "Square Root Transform" : "`√x` — milder than log. Requires all values ≥ 0.",
            "Reciprocal Transform"  : "`1/x` — for extreme right skew. Requires no zeros.",
            "Box-Cox Transform"     : "Finds best power automatically. Requires all values > 0.",
            "Yeo-Johnson Transform" : "Like Box-Cox but works with negative values too.",
        }
        st.caption(f":material/info: {hints[skew_method]}")

        new_skew_col = st.text_input("Output column name", value=f"{skew_col}_transformed", key="new_skew_col")

        def apply_skew_transform(series, method):
            from scipy import stats
            if method == "Log Transform":
                if (series <= 0).any():
                    raise ValueError("Log Transform requires all values > 0.")
                return np.log(series)
            elif method == "Square Root Transform":
                if (series < 0).any():
                    raise ValueError("Square Root requires all values ≥ 0.")
                return np.sqrt(series)
            elif method == "Reciprocal Transform":
                if (series == 0).any():
                    raise ValueError("Reciprocal Transform requires no zero values.")
                return 1 / series
            elif method == "Box-Cox Transform":
                if (series <= 0).any():
                    raise ValueError("Box-Cox requires all values > 0.")
                transformed, _ = stats.boxcox(series)
                return pd.Series(transformed, index=series.index)
            elif method == "Yeo-Johnson Transform":
                transformed, _ = stats.yeojohnson(series)
                return pd.Series(transformed, index=series.index)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Preview", icon=":material/preview:", use_container_width=True, key="skew_preview"):
                try:
                    series = st.session_state.df[skew_col].dropna()
                    transformed = apply_skew_transform(series, skew_method)
                    preview_df = pd.DataFrame({
                        "Original"   : series.head(8).values,
                        "Transformed": transformed.head(8).round(4).values
                    })
                    st.dataframe(preview_df, use_container_width=True)
                    col_a, col_b = st.columns(2)
                    col_a.metric("Skewness before", round(float(series.skew()), 3))
                    col_b.metric("Skewness after",  round(float(transformed.skew()), 3),
                                 delta=round(float(transformed.skew()) - float(series.skew()), 3))
                    import plotly.figure_factory as ff
                    fig = ff.create_distplot(
                        [series.tolist(), transformed.tolist()],
                        group_labels=["Original", "Transformed"],
                        show_hist=False, show_rug=False
                    )
                    fig.update_layout(title="Distribution: Before vs After", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                except ValueError as e:
                    st.error(f":material/error: {e}")
                except Exception as e:
                    st.error(f":material/error: Preview failed: {e}")
        with col2:
            if st.button("Apply Transform", icon=":material/check_circle:", use_container_width=True, key="skew_apply"):
                try:
                    series = st.session_state.df[skew_col].dropna()
                    transformed = apply_skew_transform(series, skew_method)
                    save_snapshot()
                    df_temp = st.session_state.df.copy()
                    df_temp[new_skew_col] = transformed
                    new_skew = round(float(transformed.skew()), 3)
                    st.session_state.df = df_temp
                    st.session_state.msg_skew = f":material/check_circle: **{skew_method}** on **{skew_col}** → **{new_skew_col}** | Skewness: `{current_skew}` → `{new_skew}`"
                    st.rerun()
                except ValueError as e:
                    st.error(f":material/error: {e}")
                except Exception as e:
                    if st.session_state.df_history:
                        st.session_state.df_history.pop()
                    st.error(f":material/error: Transform failed: {e}")
        skew_msg = st.empty()
        if st.session_state.get("msg_skew"):
            skew_msg.success(st.session_state.msg_skew)
            st.session_state.msg_skew = ""

    st.divider()

    # ── 11. Download ──────────────────────────────────────────
    st.markdown("#### :material/download: Download Cleaned Dataset")
    cleaned_csv = st.session_state.df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Cleaned CSV",
        data=cleaned_csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
        icon=":material/download:"
    )

    st.divider()

    # ── 12. Save / Reset ──────────────────────────────────────
    st.markdown("#### :material/save: Save Cleaned Data")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply & Replace", icon=":material/check_circle:", type="primary", use_container_width=True):
            st.session_state.df_checkpoint = st.session_state.df.copy()
            st.session_state.df_history = []
            st.session_state.msg_save = ":material/check_circle: Cleaned data saved as checkpoint!"
            st.rerun()
    with col2:
        original_exists = st.session_state.get("df_original") is not None
        if st.button("Reset to Original", icon=":material/restart_alt:", disabled=not original_exists, use_container_width=True):
            orig_shape = st.session_state.df_original.shape
            st.session_state.df = st.session_state.df_original.copy()
            st.session_state.df_history = []
            st.session_state.msg_save = f":material/restart_alt: Reset complete — restored to original dataset ({orig_shape[0]} rows × {orig_shape[1]} cols)"
            st.rerun()
    save_msg = st.empty()
    if st.session_state.get("msg_save"):
        save_msg.success(st.session_state.msg_save)
        st.session_state.msg_save = ""

    if st.session_state.get("df_original") is not None:
        orig = st.session_state.df_original
        curr = st.session_state.df
        r_diff = orig.shape[0] - curr.shape[0]
        c_diff = orig.shape[1] - curr.shape[1]
        st.caption(
            f":material/compare_arrows: vs Original — "
            f"Rows: **{orig.shape[0]} → {curr.shape[0]}** "
            f"({'−' if r_diff >= 0 else '+'}{abs(r_diff)}) | "
            f"Columns: **{orig.shape[1]} → {curr.shape[1]}** "
            f"({'−' if c_diff >= 0 else '+'}{abs(c_diff)})"
        )