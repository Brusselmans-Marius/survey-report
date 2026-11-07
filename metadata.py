"""Loads the bundled question/answer code translations and country lookup.

Two codes files ship alongside this file rather than being uploaded by end
users - it's the same fixed metadata for every response file:

- `lisasurvey_codes.xlsx` (compact view): short question titles ("Question")
  and short answer labels ("Meaning").
- `lisasurvey_codes_extended.xlsx` (extended view): the full question text
  ("FullQuestion") and full answer text ("CleanAnswer"), as respondents saw
  them.

Each view reads only its own file, so edit the file of the view you want to
change. To update the translations (add a question, fix a typo, add a
country), just edit or replace the .xlsx and restart the app - no code
changes needed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
VIEWS = {
    # view: (file, column holding the question text, column holding answer text)
    "compact": (HERE / "lisasurvey_codes.xlsx", "Question", "Meaning"),
    "extended": (HERE / "lisasurvey_codes_extended.xlsx", "FullQuestion", "CleanAnswer"),
}


# Display names for the codes file's Category column, used as the topic
# headings in the Explore dropdowns. A category not listed here is shown
# as written in the codes file.
CATEGORY_NAMES = {
    "Intro": "About the respondent",
    "Pulmonary aspiration": "Pulmonary aspiration",
    "Positioning": "Positioning",
    "Monitoring": "Monitoring",
    "Airway": "Airway management",
    "Ventilation": "Ventilation",
    "BPmanagement": "Blood pressure management",
    "DrugChoice": "Drug choice",
    "Tocoloysis": "Tocolysis",
}


@st.cache_data
def load_metadata(view: str = "compact") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (metadata, countries) for the "compact" or "extended" view.

    metadata columns: QuestionCode, Category, Question, AnswerCode, Meaning, Type
    (in the extended view, Question/Meaning hold the full texts)
    countries columns: Country, Code
    """
    codes_file, question_col, answer_col = VIEWS[view]
    if not codes_file.exists():
        raise FileNotFoundError(
            f"Missing bundled codes file: {codes_file.name}. It must sit "
            "next to app.py."
        )

    # By default pandas treats text like "None", "NA" or "null" as a missing
    # value - which silently erased real labels (G01Q07's answer "None").
    # Only truly empty cells count as missing here.
    read = dict(keep_default_na=False, na_values=[""])
    metadata = pd.read_excel(codes_file, sheet_name="Codes", **read)
    countries = pd.read_excel(codes_file, sheet_name="Countries", **read)

    # The rest of the app reads "Question" and "Meaning"; point them at this
    # view's text columns. A blank full text falls back to the short one.
    if question_col != "Question":
        metadata["Question"] = metadata[question_col].fillna(metadata["Question"])
    if answer_col != "Meaning":
        metadata["Meaning"] = metadata[answer_col].fillna(metadata["Meaning"])

    # Codes can come back as numbers or strings depending on how the sheet
    # was edited (e.g. "01" vs 1); force trimmed strings here once so every
    # later comparison is string-to-string.
    metadata["QuestionCode"] = metadata["QuestionCode"].astype(str).str.strip()
    metadata["AnswerCode"] = metadata["AnswerCode"].astype(str).str.strip()
    countries["Code"] = countries["Code"].astype(str).str.strip()
    # Stray spaces in labels (e.g. "Academic hospital ") would show up in
    # legends and break label matching.
    metadata["Meaning"] = metadata["Meaning"].str.strip()
    metadata["Question"] = metadata["Question"].str.strip()
    category = metadata["Category"].str.strip()
    metadata["Category"] = category.map(lambda c: CATEGORY_NAMES.get(c, c))

    return metadata, countries


def lookup_meaning(question_metadata: pd.DataFrame, code) -> str | None:
    """Human-readable meaning of one answer code, within one question's
    metadata rows. Returns None if the code isn't recognised."""
    code = str(code).strip()
    hit = question_metadata.loc[question_metadata["AnswerCode"] == code, "Meaning"]
    return hit.iloc[0] if len(hit) else None
