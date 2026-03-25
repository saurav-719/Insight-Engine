import json
import numpy as np
import pandas as pd
import requests
import streamlit as st


def get_config():
    try:
        return {
            "model": st.secrets["NVIDIA_MODEL"],
            "base_url": st.secrets["NVIDIA_BASE_URL"],
            "api_key": st.secrets["NVIDIA_API_KEY"]
        }
    except Exception:
        return {
            "model": "",
            "base_url": "",
            "api_key": ""
        }


def build_dataset_summary(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    summary = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "columns": {},
        "missing_values": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    for col in df.columns:
        col_info = {"dtype": str(df[col].dtype)}

        if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isnull().all():
            col_info.update({
                "mean": round(float(df[col].mean()), 3),
                "std": round(float(df[col].std()), 3),
                "min": round(float(df[col].min()), 3),
                "max": round(float(df[col].max()), 3),
                "skewness": round(float(df[col].skew()), 3),
            })
        else:
            col_info.update({
                "unique_values": int(df[col].nunique()),
                "top_values": df[col].value_counts().head(3).to_dict(),
            })

        summary["columns"][col] = col_info

    # Correlation analysis
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().round(3)
        strong_corr = []

        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    strong_corr.append({
                        "col_a": corr.columns[i],
                        "col_b": corr.columns[j],
                        "correlation": round(val, 3)
                    })

        summary["strong_correlations"] = strong_corr

    return summary


def build_prompt(insight_type, summary, df_columns, custom_question=""):
    base = f"""You are a senior data scientist analyzing a dataset.

Here is the dataset summary in JSON:

{json.dumps(summary, indent=2)}

Columns: {', '.join(df_columns[:10])}
"""

    prompts = {
        "Full Dataset Report": base + """
Provide a comprehensive data analysis report including:
1. Dataset overview and structure
2. Data quality issues
3. Key statistical findings
4. Notable patterns and correlations
5. Actionable recommendations
Format with clear sections and bullet points.
""",

        "Data Quality Assessment": base + """
Perform a thorough data quality assessment covering:
1. Missing value analysis and impact
2. Duplicate records
3. Data type issues
4. Outlier concerns
5. Consistency problems
6. Fix recommendations.
""",

        "Key Trends & Patterns": base + """
Identify and explain:
1. Interesting statistical patterns
2. Strong correlations
3. Skewed distributions
4. Dominant categories
5. Any anomalies.
""",

        "Cleaning Recommendations": base + """
Give a prioritized data cleaning plan:
1. Critical fixes
2. Important fixes
3. Optional improvements
4. Missing value strategies
5. Columns to drop/merge.
""",

        "Feature Importance Suggestions": base + """
From ML perspective:
1. Strong predictors
2. Redundant columns
3. Target variable suggestions
4. Feature engineering ideas
5. Data leakage risks.
""",

        "Custom Question": base + f"\n{custom_question}"
    }

    return prompts.get(insight_type, base)

def stream_insights(api_key, prompt, max_tokens=1000):
    config = get_config()

    NVIDIA_MODEL = config["model"]
    NVIDIA_BASE_URL = config["base_url"]
    NVIDIA_API_KEY = config["api_key"]

    # Safety check
    if not NVIDIA_MODEL or not NVIDIA_BASE_URL:
        raise Exception("Missing NVIDIA configuration. Check Streamlit secrets.")

    # Length control
    length_instruction = {
        500: """Respond SHORT:
- 4-5 bullet summary
- One-line points""",

        1000: """Respond MEDIUM:
- Summary (5 bullets)
- Key insights""",

        2000: """Respond DETAILED:
- Summary
- Insights
- Detailed explanation"""
    }.get(max_tokens, "Be clear and structured.")

    prompt_with_length = f"{length_instruction}\n\n{prompt}"

    key = api_key if api_key else NVIDIA_API_KEY

    headers = {
        "Authorization": f"Bearer {key}",
        "Accept": "text/event-stream"
    }

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [{"role": "user", "content": prompt_with_length}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.7,
        "stream": True
    }

    response = requests.post(
        NVIDIA_BASE_URL,
        headers=headers,
        json=payload,
        stream=True
    )

    if response.status_code != 200:
        raise Exception(f"{response.status_code} - {response.text}")

    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8")

            if decoded.startswith("data: "):
                data_str = decoded[6:]

                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"].get("content", "")

                    if delta:
                        yield delta

                except json.JSONDecodeError:
                    continue