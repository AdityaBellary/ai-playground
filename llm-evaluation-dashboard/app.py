import os

import pandas as pd
import streamlit as st

from evaluate import RESULTS_PATH, run_evaluation

st.title("LLM Evaluation Dashboard")

# Read saved results so the dashboard doesn't call the LLM on every rerun
if st.button("Run evaluation"):
    with st.spinner("Running both prompts through the model..."):
        run_evaluation()

if not os.path.exists(RESULTS_PATH):
    st.info("No results yet. Click **Run evaluation** or run `python evaluate.py`.")
    st.stop()

comparison = pd.read_csv(RESULTS_PATH)

col1, col2 = st.columns(2)
col1.metric("Average Quality", f"{comparison['quality'].mean():.2f}")
col2.metric("Average Latency", f"{comparison['latency'].mean():.2f}s")

st.subheader("Quality by Prompt")
st.bar_chart(comparison.groupby("prompt")["quality"].mean())

st.subheader("Latency by Prompt")
st.bar_chart(comparison.groupby("prompt")["latency"].mean())

st.subheader("All Results")
st.dataframe(comparison)
