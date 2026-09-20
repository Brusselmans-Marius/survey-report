"""Streamlit app: visualises coded survey results as simple charts.

Everything runs client-side / in-session - the uploaded response file is
held only in memory for this browser session and is never written to disk
or sent anywhere else. The question/answer translation table
(lisasurvey_codes.xlsx) ships with the app itself; see metadata.py for how
to update it.
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from metadata import load_metadata
from report import build_report

st.set_page_config(page_title="Survey results", layout="wide")
st.title("Attitudes towards anesthesia for pregnant women: a survey [results]")

metadata, countries = load_metadata()

if "report_items" not in st.session_state:
    st.session_state.report_items = None

with st.sidebar:
    uploaded_file = st.file_uploader("Drag the survey responses .xlsx here", type=["xlsx"])
    full_only = st.checkbox("Full responses only", value=False)
    run_clicked = st.button("Get results", type="primary")
    status = st.empty()

if run_clicked:
    if uploaded_file is None:
        status.warning("Please upload a survey responses file first.")
    else:
        with st.spinner("Building report..."):
            try:
                df = pd.read_excel(uploaded_file, sheet_name=0)
            except Exception:
                st.session_state.report_items = None
                status.error("Couldn't read that file - please upload a valid .xlsx survey export.")
            else:
                try:
                    st.session_state.report_items = build_report(df, metadata, countries, full_only)
                    status.empty()
                except Exception as exc:
                    st.session_state.report_items = None
                    status.error(f"Error: {exc}")

items = st.session_state.report_items
if items:
    for item in items:
        st.subheader(item["title"])
        if item["type"] == "error":
            st.error(f"Couldn't render this question: {item['message']}")
        elif item["type"] == "table":
            st.dataframe(item["obj"], hide_index=True, use_container_width=True)
        else:
            st.pyplot(item["obj"])
            plt.close(item["obj"])
