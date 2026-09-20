"""Loads the bundled question/answer code translations and country lookup.

`lisasurvey_codes.xlsx` ships alongside this file rather than being uploaded
by end users - it's the same fixed metadata for every response file. To
update the translations (add a question, fix a typo, add a country), just
edit or replace that .xlsx and restart the app - no code changes needed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

CODES_FILE = Path(__file__).resolve().parent / "lisasurvey_codes.xlsx"


@st.cache_data
def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (metadata, countries).

    metadata columns: QuestionCode, Category, Question, AnswerCode, Meaning, Type
    countries columns: Country, Code
    """
    if not CODES_FILE.exists():
        raise FileNotFoundError(
            f"Missing bundled codes file: {CODES_FILE.name}. It must sit "
            "next to app.py."
        )

    metadata = pd.read_excel(CODES_FILE, sheet_name="Codes")
    countries = pd.read_excel(CODES_FILE, sheet_name="Countries")

    # Codes can come back as numbers or strings depending on how the sheet
    # was edited (e.g. "01" vs 1); force trimmed strings here once so every
    # later comparison is string-to-string.
    metadata["QuestionCode"] = metadata["QuestionCode"].astype(str).str.strip()
    metadata["AnswerCode"] = metadata["AnswerCode"].astype(str).str.strip()
    countries["Code"] = countries["Code"].astype(str).str.strip()

    return metadata, countries


def lookup_meaning(question_metadata: pd.DataFrame, code) -> str | None:
    """Human-readable meaning of one answer code, within one question's
    metadata rows. Returns None if the code isn't recognised."""
    code = str(code).strip()
    hit = question_metadata.loc[question_metadata["AnswerCode"] == code, "Meaning"]
    return hit.iloc[0] if len(hit) else None
