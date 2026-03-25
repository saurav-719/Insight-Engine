import json
import numpy as np
import pandas as pd
import requests

# Config
NVIDIA_MODEL    = st.secrets.get("NVIDIA_MODEL","")
NVIDIA_BASE_URL = st.secrets.get("NVIDIA_BASE_URL","")


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
                "mean" : round(float(df[col].mean()), 3),
                "std" : round(float(df[col].std()), 3),
                "min" : round(float(df[col].min()), 3),
                "max" : round(float(df[col].max()), 3),
                "skewness": round(float(df[col].skew()), 3),
            })
        else:
            col_info.update({
                "unique_values": int(df[col].nunique()),
                "top_values"   : df[col].value_counts().head(3).to_dict(),
            })
        summary["columns"][col] = col_info

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().round(3)
        strong_corr = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    strong_corr.append({
                        "col_a"      : corr.columns[i],
                        "col_b"      : corr.columns[j],
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
Format with clear sections and bullet points.""",

        "Data Quality Assessment": base + """
Perform a thorough data quality assessment covering:
1. Missing value analysis and impact
2. Duplicate records
3. Data type issues
4. Outlier concerns
5. Consistency problems
6. Specific fix recommendations for each issue.""",

        "Key Trends & Patterns": base + """
Identify and explain:
1. The most interesting statistical patterns
2. Strong correlations and what they might mean
3. Skewed distributions and their implications
4. Dominant categories in categorical columns
5. Any anomalies worth investigating.""",

        "Cleaning Recommendations": base + """
Give a prioritized data cleaning action plan:
1. Critical fixes (must do)
2. Important fixes (should do)
3. Optional improvements
4. Specific strategies for each missing value column
5. Columns that could be dropped or merged.""",

        "Feature Importance Suggestions": base + """
From a machine learning perspective:
1. Which columns are likely strong predictors?
2. Which columns might be redundant?
3. Suggest potential target variables
4. Recommend feature engineering ideas
5. Flag any data leakage risks.""",

        "Custom Question": base + f"\n{custom_question}"
    }
    return prompts.get(insight_type, base)


def stream_insights(api_key, prompt, max_tokens=1000):
    """Stream insights from NVIDIA API."""

    # Length instruction based on max_tokens
    length_instruction = {
    500: """Respond in SHORT format:
- Give only Summary (4-5 bullet points)
- Keep each point 1 line
- No long paragraphs
""",

    1000: """Respond in MEDIUM format:
- Section 1: Summary (5 bullets)
- Section 2: Key Insights (grouped bullets)
- Avoid long paragraphs
""",

    2000: """Respond in DETAILED format:
- Section 1: Summary
- Section 2: Key Insights
- Section 3: Detailed Analysis
- Include explanations and reasoning
"""
}.get(max_tokens, "Be clear and structured.")

    prompt_with_length = f"{length_instruction}\n\n{prompt}"

    # Use passed api_key if provided, else fall back to hardcoded
    key = api_key if api_key else NVIDIA_API_KEY

    headers = {
        "Authorization": f"Bearer {key}",
        "Accept"       : "text/event-stream"
    }
    payload = {
        "model" : NVIDIA_MODEL,
        "messages" : [{"role": "user", "content": prompt_with_length}],
        "max_tokens" : max_tokens,
        "temperature" : 0.20,
        "top_p" : 0.70,
        "frequency_penalty": 0.00,
        "presence_penalty" : 0.00,
        "stream" : True
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
                    data  = json.loads(data_str)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue
