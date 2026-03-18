import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
from utils.notepad import render_notepad
render_notepad()

st.set_page_config(page_title="Auto EDA", page_icon=":material/search:", layout="wide")

st.title(":material/search: Auto Exploratory Data Analysis")

# ── Guard ────────────────────────────────────────────────────
if st.session_state.get("df") is None:
    st.warning(":material/warning: No data loaded. Please upload a file first.")
    st.stop()

df = st.session_state.df

# ── Column type lists ─────────────────────────────────────────
numeric_df   = df.select_dtypes(include=np.number)
cat_df       = df.select_dtypes(include="object")
numeric_cols = numeric_df.columns.tolist()
cat_cols     = cat_df.columns.tolist()

# ── Numeric Analysis ─────────────────────────────────────────
if not numeric_df.empty:
    st.subheader(":material/pin: Numeric Columns — Descriptive Statistics")
    desc = numeric_df.describe().T
    desc["skewness"] = numeric_df.skew().round(3)
    desc["kurtosis"] = numeric_df.kurt().round(3)
    st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)

    st.subheader(":material/warning: Skewness Alerts")
    skewed = numeric_df.skew().abs()
    highly_skewed = skewed[skewed > 1].index.tolist()
    if highly_skewed:
        st.warning(f"Highly skewed columns (|skew| > 1): **{', '.join(highly_skewed)}** — consider transformations.")
    else:
        st.success(":material/check_circle: No highly skewed columns detected!")

st.divider()

# ── Categorical Analysis ─────────────────────────────────────
if not cat_df.empty:
    st.subheader(":material/abc: Categorical Columns — Value Counts")
    selected_cat = st.selectbox("Select a categorical column", cat_df.columns)

    vc = df[selected_cat].value_counts().reset_index()
    vc.columns = [selected_cat, "Count"]
    vc["Percentage"] = (vc["Count"] / len(df) * 100).round(2)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(vc, use_container_width=True)
    with col2:
        st.metric("Unique Values",   df[selected_cat].nunique())
        st.metric("Most Frequent",   vc.iloc[0][selected_cat])
        st.metric("Least Frequent",  vc.iloc[-1][selected_cat])

st.divider()

# ── Missing Values ───────────────────────────────────────────
st.subheader(":material/find_replace: Missing Values Analysis")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    st.success(":material/celebration: No missing values found in the dataset!")
else:
    miss_df = pd.DataFrame({
        "Column"       : missing.index,
        "Missing Count": missing.values,
        "Missing %"    : (missing.values / len(df) * 100).round(2)
    })
    st.dataframe(miss_df, use_container_width=True)

st.divider()

# ── Correlation Matrix ───────────────────────────────────────
if not numeric_df.empty and numeric_df.shape[1] > 1:
    st.subheader(":material/grid_on: Correlation Matrix")
    corr = numeric_df.corr().round(3)
    st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)

    st.subheader(":material/link: Strong Correlations (|r| > 0.7)")
    strong = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            val = corr.iloc[i, j]
            if abs(val) > 0.7:
                strong.append({
                    "Column A"   : corr.columns[i],
                    "Column B"   : corr.columns[j],
                    "Correlation": round(val, 3)
                })
    if strong:
        st.dataframe(pd.DataFrame(strong), use_container_width=True)
    else:
        st.info(":material/info: No strong correlations found.")

st.divider()

# ── Statistical Analysis ─────────────────────────────────────
st.subheader(":material/science: Statistical Analysis")

stat_test = st.selectbox("Choose a test", [
    "Normality Test (Shapiro-Wilk)",
    "T-Test",
    "ANOVA",
    "Chi-Square",
    "Correlation Significance",
], key="stat_test")

st.divider()

if stat_test == "Normality Test (Shapiro-Wilk)":
    st.markdown("##### :material/query_stats: Normality Test (Shapiro-Wilk)")
    st.caption("Tests whether a numeric column follows a normal distribution.")

    if not numeric_df.empty:
        norm_col = st.selectbox("Select numeric column", numeric_cols, key="norm_col")
        if st.button("Run Normality Test", icon=":material/science:", type="primary"):
            series = df[norm_col].dropna()
            if len(series) < 3:
                st.error(":material/error: Need at least 3 values.")
            else:
                if len(series) > 5000:
                    st.warning(":material/warning: Large sample — using first 5000 rows.")
                    series = series.sample(5000, random_state=42)
                stat_val, p_value = stats.shapiro(series)
                col1, col2, col3 = st.columns(3)
                col1.metric("W Statistic", round(float(stat_val), 4))
                col2.metric("P-Value",     round(float(p_value), 4))
                col3.metric("Sample Size", len(series))
                st.divider()
                if p_value > 0.05:
                    st.success(f":material/check_circle: **{norm_col}** appears **normally distributed** (p={round(float(p_value),4)} > 0.05)")
                else:
                    st.warning(f":material/warning: **{norm_col}** is **NOT normally distributed** (p={round(float(p_value),4)} ≤ 0.05)")
                st.caption(":material/info: p > 0.05 → normal · p ≤ 0.05 → not normal")
    else:
        st.warning(":material/warning: No numeric columns found.")

elif stat_test == "T-Test":
    st.markdown("##### :material/compare_arrows: T-Test (Independent)")
    st.caption("Tests whether the means of a numeric column differ significantly between two groups.")

    if not numeric_df.empty and not cat_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            ttest_num = st.selectbox("Numeric column", numeric_cols, key="ttest_num")
        with col2:
            ttest_cat = st.selectbox("Group column (categorical)", cat_cols, key="ttest_cat")

        unique_groups = df[ttest_cat].dropna().unique()
        if len(unique_groups) < 2:
            st.warning(":material/warning: Need at least 2 groups.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                group_a = st.selectbox("Group A", unique_groups, key="group_a")
            with col2:
                group_b = st.selectbox("Group B", unique_groups, index=1 if len(unique_groups) > 1 else 0, key="group_b")

            if st.button("Run T-Test", icon=":material/science:", type="primary"):
                if group_a == group_b:
                    st.error(":material/error: Please select two different groups.")
                else:
                    a = df[df[ttest_cat] == group_a][ttest_num].dropna()
                    b = df[df[ttest_cat] == group_b][ttest_num].dropna()
                    t_stat, p_value = stats.ttest_ind(a, b)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("T Statistic",  round(float(t_stat), 4))
                    col2.metric("P-Value",       round(float(p_value), 4))
                    col3.metric("Significance",  "Yes ✓" if p_value <= 0.05 else "No ✗")
                    col1, col2 = st.columns(2)
                    col1.metric(f"Mean ({group_a})", round(float(a.mean()), 3))
                    col2.metric(f"Mean ({group_b})", round(float(b.mean()), 3))
                    col1.metric(f"Std ({group_a})",  round(float(a.std()), 3))
                    col2.metric(f"Std ({group_b})",  round(float(b.std()), 3))
                    st.divider()
                    if p_value <= 0.05:
                        st.success(f":material/check_circle: Significant difference between **{group_a}** and **{group_b}** (p={round(float(p_value),4)})")
                    else:
                        st.info(f":material/info: No significant difference between **{group_a}** and **{group_b}** (p={round(float(p_value),4)})")
                    st.caption(":material/info: p ≤ 0.05 → significant · p > 0.05 → not significant")
    else:
        st.warning(":material/warning: Need at least one numeric and one categorical column.")

elif stat_test == "ANOVA":
    st.markdown("##### :material/bar_chart: ANOVA (One-Way)")
    st.caption("Tests whether means of a numeric column differ across 3 or more groups.")

    if not numeric_df.empty and not cat_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            anova_num = st.selectbox("Numeric column", numeric_cols, key="anova_num")
        with col2:
            anova_cat = st.selectbox("Group column (categorical)", cat_cols, key="anova_cat")

        groups = df[anova_cat].dropna().unique()
        st.caption(f":material/info: Found **{len(groups)}** groups: `{', '.join(map(str, groups[:6]))}{'...' if len(groups) > 6 else ''}`")

        if len(groups) < 3:
            st.warning(":material/warning: ANOVA needs at least 3 groups. Use T-Test for 2 groups.")
        else:
            if st.button("Run ANOVA", icon=":material/science:", type="primary"):
                group_data = [df[df[anova_cat] == g][anova_num].dropna() for g in groups]
                group_data = [g for g in group_data if len(g) > 0]
                f_stat, p_value = stats.f_oneway(*group_data)
                col1, col2, col3 = st.columns(3)
                col1.metric("F Statistic", round(float(f_stat), 4))
                col2.metric("P-Value",     round(float(p_value), 4))
                col3.metric("Groups",      len(group_data))
                summary = pd.DataFrame({
                    "Group": groups,
                    "Count": [len(df[df[anova_cat] == g][anova_num].dropna()) for g in groups],
                    "Mean":  [round(float(df[df[anova_cat] == g][anova_num].dropna().mean()), 3) for g in groups],
                    "Std":   [round(float(df[df[anova_cat] == g][anova_num].dropna().std()), 3) for g in groups],
                })
                st.dataframe(summary, use_container_width=True)
                st.divider()
                if p_value <= 0.05:
                    st.success(f":material/check_circle: Significant difference across groups in **{anova_num}** (p={round(float(p_value),4)})")
                else:
                    st.info(f":material/info: No significant difference across groups in **{anova_num}** (p={round(float(p_value),4)})")
                st.caption(":material/info: p ≤ 0.05 → at least one group differs · p > 0.05 → no significant difference")
    else:
        st.warning(":material/warning: Need at least one numeric and one categorical column.")

elif stat_test == "Chi-Square":
    st.markdown("##### :material/grid_on: Chi-Square Test")
    st.caption("Tests whether two categorical columns are independent of each other.")

    if len(cat_cols) < 2:
        st.warning(":material/warning: Need at least 2 categorical columns.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            chi_col1 = st.selectbox("Categorical column A", cat_cols, key="chi_col1")
        with col2:
            chi_col2 = st.selectbox("Categorical column B", cat_cols, index=1 if len(cat_cols) > 1 else 0, key="chi_col2")

        if st.button("Run Chi-Square Test", icon=":material/science:", type="primary"):
            if chi_col1 == chi_col2:
                st.error(":material/error: Please select two different columns.")
            else:
                contingency = pd.crosstab(df[chi_col1], df[chi_col2])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                col1, col2, col3 = st.columns(3)
                col1.metric("Chi² Statistic",     round(float(chi2), 4))
                col2.metric("P-Value",             round(float(p_value), 4))
                col3.metric("Degrees of Freedom",  int(dof))
                st.markdown("**Contingency Table:**")
                st.dataframe(contingency, use_container_width=True)
                st.divider()
                if p_value <= 0.05:
                    st.success(f":material/check_circle: **{chi_col1}** and **{chi_col2}** are **dependent** (p={round(float(p_value),4)}) — relationship exists")
                else:
                    st.info(f":material/info: **{chi_col1}** and **{chi_col2}** are **independent** (p={round(float(p_value),4)}) — no significant relationship")
                st.caption(":material/info: p ≤ 0.05 → dependent · p > 0.05 → independent")

elif stat_test == "Correlation Significance":
    st.markdown("##### :material/link: Correlation Significance Test")
    st.caption("Tests whether the correlation between two numeric columns is statistically significant.")

    if len(numeric_cols) < 2:
        st.warning(":material/warning: Need at least 2 numeric columns.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            corr_col1 = st.selectbox("Numeric column A", numeric_cols, key="corr_col1")
        with col2:
            corr_col2 = st.selectbox("Numeric column B", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="corr_col2")

        corr_method = st.radio("Correlation method", ["Pearson", "Spearman"], horizontal=True)

        if st.button("Run Correlation Test", icon=":material/science:", type="primary"):
            if corr_col1 == corr_col2:
                st.error(":material/error: Please select two different columns.")
            else:
                common = df[[corr_col1, corr_col2]].dropna()
                a, b   = common[corr_col1], common[corr_col2]
                if corr_method == "Pearson":
                    corr_val, p_value = stats.pearsonr(a, b)
                else:
                    corr_val, p_value = stats.spearmanr(a, b)
                col1, col2, col3 = st.columns(3)
                col1.metric("Correlation (r)", round(float(corr_val), 4))
                col2.metric("P-Value",          round(float(p_value), 4))
                col3.metric("Sample Size",       len(common))
                abs_corr  = abs(corr_val)
                strength  = "Strong" if abs_corr >= 0.7 else "Moderate" if abs_corr >= 0.4 else "Weak"
                direction = "positive" if corr_val > 0 else "negative"
                st.divider()
                if p_value <= 0.05:
                    st.success(f":material/check_circle: **{strength} {direction} correlation** between **{corr_col1}** and **{corr_col2}** (r={round(float(corr_val),4)}, p={round(float(p_value),4)})")
                else:
                    st.info(f":material/info: Correlation between **{corr_col1}** and **{corr_col2}** is **not significant** (p={round(float(p_value),4)})")
                st.caption(":material/info: |r| ≥ 0.7 strong · 0.4–0.7 moderate · < 0.4 weak")

st.divider()

# ── Pattern Detection ─────────────────────────────────────────
st.subheader(":material/pattern: Pattern Detection")

pattern_type = st.selectbox("Choose a method", [
    "K-Means Clustering & Segmentation",
    "Anomaly Detection (Isolation Forest)",
    "Feature Importance (Random Forest)",
], key="pattern_type")

st.divider()

# ── K-Means Clustering & Segmentation ────────────────────────
if pattern_type == "K-Means Clustering & Segmentation":
    st.markdown("##### :material/hub: K-Means Clustering & Segmentation")

    if len(numeric_cols) < 2:
        st.warning(":material/warning: Need at least 2 numeric columns.")
    else:
        seg_check_data = df[numeric_cols].dropna()

        # ── Auto Detection ────────────────────────────────────
        st.markdown("**:material/auto_awesome: Should you segment this data?**")

        with st.spinner("Analyzing dataset for segmentation readiness..."):
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            from sklearn.cluster import KMeans
            from scipy.stats import variation
            from scipy.signal import find_peaks

            signals     = []
            score_total = 0

            # Signal 1: Dataset size
            n_rows = len(seg_check_data)
            if n_rows >= 500:
                signals.append(("pass", "Sufficient data size", f"{n_rows} rows — large enough for meaningful segments"))
                score_total += 2
            elif n_rows >= 100:
                signals.append(("warn", "Moderate data size", f"{n_rows} rows — segmentation possible but may not be stable"))
                score_total += 1
            else:
                signals.append(("fail", "Insufficient data size", f"Only {n_rows} rows — need at least 100 for segmentation"))

            # Signal 2: High variance
            cv_scores     = {}
            for col in numeric_cols:
                series = seg_check_data[col].dropna()
                if series.mean() != 0:
                    cv = abs(float(variation(series)))
                    cv_scores[col] = round(cv, 3)
            high_var_cols = [c for c, v in cv_scores.items() if v > 0.3]
            if len(high_var_cols) >= 2:
                signals.append(("pass", "High variance detected", f"{len(high_var_cols)} columns have high variation: `{', '.join(high_var_cols[:3])}`"))
                score_total += 2
            elif len(high_var_cols) == 1:
                signals.append(("warn", "Moderate variance", f"Only 1 column has high variation: `{high_var_cols[0]}`"))
                score_total += 1
            else:
                signals.append(("fail", "Low variance", "Columns are too similar — segmentation may not reveal useful groups"))

            # Signal 3: Multimodality
            multimodal_cols = []
            for col in numeric_cols[:8]:
                try:
                    series   = seg_check_data[col].dropna()
                    hist, _  = np.histogram(series, bins=20)
                    peaks, _ = find_peaks(hist, height=len(series) * 0.05)
                    if len(peaks) >= 2:
                        multimodal_cols.append(col)
                except Exception:
                    pass
            if len(multimodal_cols) >= 2:
                signals.append(("pass", "Multiple distribution peaks", f"`{', '.join(multimodal_cols[:3])}` show multimodal distributions — natural groups likely exist"))
                score_total += 3
            elif len(multimodal_cols) == 1:
                signals.append(("warn", "One multimodal column", f"`{multimodal_cols[0]}` shows multiple peaks"))
                score_total += 1
            else:
                signals.append(("info", "No multimodal distributions", "Columns appear unimodal — groups may be less distinct"))

            # Signal 4: Silhouette score
            try:
                scaler  = StandardScaler()
                scaled  = scaler.fit_transform(seg_check_data[numeric_cols[:6]])
                km      = KMeans(n_clusters=3, random_state=42, n_init=10)
                labels  = km.fit_predict(scaled)
                sil     = round(float(silhouette_score(scaled, labels)), 3)
                if sil >= 0.5:
                    signals.append(("pass", "Strong cluster structure", f"Silhouette score = {sil} (≥ 0.5) — clear natural clusters exist"))
                    score_total += 3
                elif sil >= 0.25:
                    signals.append(("warn", "Moderate cluster structure", f"Silhouette score = {sil} (0.25–0.5) — some structure exists"))
                    score_total += 2
                else:
                    signals.append(("fail", "Weak cluster structure", f"Silhouette score = {sil} (< 0.25) — clusters may not be meaningful"))
            except Exception:
                signals.append(("info", "Silhouette score unavailable", "Skipped"))

            # Signal 5: Numeric richness
            if len(numeric_cols) >= 4:
                signals.append(("pass", "Rich numeric features", f"{len(numeric_cols)} numeric columns — more dimensions = better segments"))
                score_total += 1
            elif len(numeric_cols) >= 2:
                signals.append(("warn", "Minimal numeric features", f"Only {len(numeric_cols)} numeric columns — segmentation will be limited"))
            else:
                signals.append(("fail", "Insufficient numeric features", "Need at least 2 numeric columns"))

        # ── Verdict ───────────────────────────────────────────
        max_score = 11
        pct       = round(score_total / max_score * 100)

        col1, col2 = st.columns([1, 3])
        col1.metric("Readiness", f"{pct}%", f"{score_total}/{max_score}")
        with col2:
            if pct >= 70:
                st.success(":material/check_circle: **Highly recommended** — strong signs of natural groupings!")
            elif pct >= 40:
                st.warning(":material/warning: **Possibly useful** — some signals suggest groups exist.")
            else:
                st.error(":material/error: **Not recommended** — no clear signs of natural groups.")

        with st.expander(":material/checklist: See signal breakdown"):
            for status, title, detail in signals:
                if status == "pass":
                    st.success(f":material/check_circle: **{title}** — {detail}")
                elif status == "warn":
                    st.warning(f":material/warning: **{title}** — {detail}")
                elif status == "fail":
                    st.error(f":material/error: **{title}** — {detail}")
                else:
                    st.info(f":material/info: **{title}** — {detail}")

        st.divider()

        # ── Elbow Method ──────────────────────────────────────
        with st.expander(":material/search: Find optimal K (Elbow Method)"):
            col1, col2 = st.columns([1, 3])
            with col1:
                max_k = st.slider("Max K to test", 3, 15, 10, key="elbow_max_k")
                if st.button("Run Elbow Method", icon=":material/science:", use_container_width=True, key="elbow_btn"):
                    scaler   = StandardScaler()
                    scaled   = scaler.fit_transform(seg_check_data)
                    inertias = []
                    for k in range(2, max_k + 1):
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        km.fit(scaled)
                        inertias.append(km.inertia_)
                    st.session_state.elbow_df = pd.DataFrame({
                        "K"      : list(range(2, max_k + 1)),
                        "Inertia": inertias
                    })
            with col2:
                if st.session_state.get("elbow_df") is not None:
                    fig = px.line(
                        st.session_state.elbow_df, x="K", y="Inertia",
                        markers=True, template="plotly_white",
                        title="Elbow Method — pick K where curve bends"
                    )
                    fig.update_traces(line_color="#1D9E75", marker_color="#1D9E75")
                    fig.update_layout(xaxis=dict(tickmode="linear", dtick=1))
                    st.plotly_chart(fig, use_container_width=True)

        # ── Column Selection + K ──────────────────────────────
        st.markdown("**:material/hub: Run Segmentation**")
        col1, col2 = st.columns(2)
        with col1:
            cluster_cols = st.multiselect(
                "Select columns", numeric_cols,
                default=numeric_cols[:3], key="cluster_cols"
            )
        with col2:
            n_clusters = st.slider("Number of segments (K)", 2, 10, 3, key="n_clusters")

        seg_name_prefix = st.text_input("Segment label prefix", value="Segment", key="seg_prefix")

        if len(cluster_cols) < 2:
            st.warning(":material/warning: Select at least 2 columns.")
        else:
            if st.button("Run Segmentation", icon=":material/rocket_launch:", type="primary", key="run_seg"):
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler

                cluster_df         = df[cluster_cols].dropna()
                scaler             = StandardScaler()
                scaled             = scaler.fit_transform(cluster_df)
                kmeans             = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels             = kmeans.fit_predict(scaled)
                cluster_df         = cluster_df.copy()
                cluster_df["_lbl"] = labels
                seg_means          = cluster_df.groupby("_lbl")[cluster_cols].mean()
                seg_means["_rank"] = seg_means.mean(axis=1)
                seg_means          = seg_means.sort_values("_rank")
                rank_map           = {old: f"{seg_name_prefix} {i+1}" for i, old in enumerate(seg_means.index)}
                cluster_df["Segment"] = cluster_df["_lbl"].map(rank_map)

                st.session_state.seg_result    = cluster_df
                st.session_state.seg_cols_used = cluster_cols
                st.rerun()

            # ── Results ───────────────────────────────────────
            if st.session_state.get("seg_result") is not None:
                seg_result    = st.session_state.seg_result
                seg_cols_used = st.session_state.seg_cols_used
                segments      = sorted(seg_result["Segment"].unique())

                st.divider()
                st.markdown("**:material/bar_chart: Results**")

                # Distribution
                dist = seg_result["Segment"].value_counts().reset_index()
                dist.columns    = ["Segment", "Count"]
                dist["Percentage"] = (dist["Count"] / len(seg_result) * 100).round(2)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(dist, use_container_width=True)
                with col2:
                    fig = px.pie(dist, names="Segment", values="Count",
                                 template="plotly_white", title="Segment Distribution")
                    st.plotly_chart(fig, use_container_width=True)

                # Profiles
                st.markdown("**Segment Profiles (mean values):**")
                profile = seg_result.groupby("Segment")[seg_cols_used].mean().round(3)
                st.dataframe(
                    profile.style.background_gradient(cmap="YlGn", axis=0),
                    use_container_width=True
                )

                # Box plot
                chart_col = st.selectbox("Compare column across segments", seg_cols_used, key="seg_chart_col")
                fig = px.box(
                    seg_result, x="Segment", y=chart_col,
                    color="Segment", template="plotly_white",
                    title=f"{chart_col} across Segments"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Scatter
                col1, col2 = st.columns(2)
                with col1:
                    sx = st.selectbox("Scatter X", seg_cols_used, key="sx")
                with col2:
                    sy = st.selectbox("Scatter Y", seg_cols_used, index=min(1, len(seg_cols_used)-1), key="sy")
                fig = px.scatter(
                    seg_result, x=sx, y=sy, color="Segment",
                    template="plotly_white", title=f"Segments — {sx} vs {sy}"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Deep dive
                st.markdown("**Segment Deep Dive:**")
                sel_seg    = st.selectbox("Select segment", segments, key="deep_seg")
                subset     = seg_result[seg_result["Segment"] == sel_seg][seg_cols_used]
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(subset.describe().T[["mean", "std", "min", "max"]].round(3), use_container_width=True)
                with col2:
                    overall = seg_result[seg_cols_used].mean()
                    seg_m   = subset.mean()
                    diff_df = pd.DataFrame({
                        "Column"       : seg_cols_used,
                        "Segment Mean" : seg_m.round(3).values,
                        "Overall Mean" : overall.round(3).values,
                        "Difference"   : (seg_m - overall).round(3).values,
                    })
                    st.dataframe(diff_df, use_container_width=True)

                # Save + Export
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Add Segment column to dataset", icon=":material/add:", use_container_width=True, key="add_seg"):
                        df_temp = st.session_state.df.copy()
                        df_temp.loc[seg_result.index, "Segment"] = seg_result["Segment"]
                        st.session_state.df = df_temp
                        st.success(":material/check_circle: Segment column added to dataset!")
                with col2:
                    export_seg = st.selectbox("Export segment", segments, key="export_seg")
                    seg_csv    = seg_result[seg_result["Segment"] == export_seg].drop(columns=["_lbl"]).to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"Download {export_seg}",
                        data=seg_csv,
                        file_name=f"{export_seg.replace(' ', '_')}.csv",
                        mime="text/csv",
                        icon=":material/download:",
                        use_container_width=True
                    )
                all_csv = seg_result.drop(columns=["_lbl"]).to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download All Segments",
                    data=all_csv,
                    file_name="all_segments.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    use_container_width=True
                )

# ── Anomaly Detection ─────────────────────────────────────────
elif pattern_type == "Anomaly Detection (Isolation Forest)":
    st.markdown("##### :material/troubleshoot: Anomaly Detection (Isolation Forest)")
    st.caption("Detects rows that don't fit the normal pattern of the data.")

    if len(numeric_cols) < 1:
        st.warning(":material/warning: Need at least 1 numeric column.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            anomaly_cols = st.multiselect(
                "Select numeric columns", numeric_cols,
                default=numeric_cols[:3], key="anomaly_cols"
            )
        with col2:
            contamination = st.slider("Expected anomaly %", min_value=1, max_value=20, value=5, key="contamination")
            st.caption("Approx % of rows expected to be anomalies")

        if not anomaly_cols:
            st.warning(":material/warning: Select at least 1 column.")
        else:
            if st.button("Run Anomaly Detection", icon=":material/science:", type="primary"):
                from sklearn.ensemble import IsolationForest

                anomaly_df            = df[anomaly_cols].dropna()
                clf                   = IsolationForest(contamination=contamination / 100, random_state=42)
                preds                 = clf.fit_predict(anomaly_df)
                scores                = clf.decision_function(anomaly_df)
                anomaly_df            = anomaly_df.copy()
                anomaly_df["Anomaly"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
                anomaly_df["Anomaly Score"] = scores.round(4)

                n_anomalies = (preds == -1).sum()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Rows", len(anomaly_df))
                col2.metric("Anomalies",  int(n_anomalies))
                col3.metric("Anomaly %",  f"{round(n_anomalies / len(anomaly_df) * 100, 2)}%")

                st.markdown("**Detected Anomalies:**")
                anomaly_rows = anomaly_df[anomaly_df["Anomaly"] == "Anomaly"].sort_values("Anomaly Score")
                st.dataframe(anomaly_rows, use_container_width=True)

                if len(anomaly_cols) >= 2:
                    fig = px.scatter(
                        anomaly_df, x=anomaly_cols[0], y=anomaly_cols[1],
                        color="Anomaly",
                        color_discrete_map={"Normal": "#1D9E75", "Anomaly": "#D85A30"},
                        template="plotly_white", title="Anomaly Detection Results"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if st.button("Add Anomaly column to dataset", icon=":material/add:"):
                    df_temp = st.session_state.df.copy()
                    df_temp.loc[anomaly_df.index, "Anomaly"]       = anomaly_df["Anomaly"]
                    df_temp.loc[anomaly_df.index, "Anomaly Score"] = anomaly_df["Anomaly Score"]
                    st.session_state.df = df_temp
                    st.success(":material/check_circle: Anomaly columns added to dataset!")
                    st.rerun()

# ── Feature Importance ────────────────────────────────────────
elif pattern_type == "Feature Importance (Random Forest)":
    st.markdown("##### :material/star: Feature Importance (Random Forest)")
    st.caption("Ranks which columns are most useful for predicting a target column.")

    if len(numeric_cols) < 2:
        st.warning(":material/warning: Need at least 2 numeric columns.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox("Target column (what to predict)", numeric_cols, key="fi_target")
        with col2:
            feature_cols = st.multiselect(
                "Feature columns",
                [c for c in numeric_cols if c != target_col],
                default=[c for c in numeric_cols if c != target_col][:5],
                key="fi_features"
            )

        if not feature_cols:
            st.warning(":material/warning: Select at least 1 feature column.")
        else:
            if st.button("Run Feature Importance", icon=":material/science:", type="primary"):
                from sklearn.ensemble import RandomForestRegressor

                fi_df          = df[feature_cols + [target_col]].dropna()
                X, y           = fi_df[feature_cols], fi_df[target_col]
                rf             = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                importance_df  = pd.DataFrame({
                    "Feature"   : feature_cols,
                    "Importance": rf.feature_importances_.round(4)
                }).sort_values("Importance", ascending=False)
                importance_df["Importance %"] = (importance_df["Importance"] * 100).round(2)

                col1, col2 = st.columns(2)
                col1.metric("Features analyzed", len(feature_cols))
                col2.metric("Top feature",        importance_df.iloc[0]["Feature"])

                st.markdown("**Feature Importance Ranking:**")
                st.dataframe(importance_df, use_container_width=True)

                fig = px.bar(
                    importance_df, x="Importance %", y="Feature",
                    orientation="h", template="plotly_white",
                    title=f"Feature Importance for predicting '{target_col}'",
                    color="Importance %", color_continuous_scale="teal"
                )
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(":material/info: Higher % = more important for predicting the target column")