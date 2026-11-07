"""Builds the full report: matches an uploaded response file's columns to
the bundled question metadata, translates codes to text, and produces one
chart/table item per question.

Reading the raw columns lives in variables.py; this module only summarises
each question for its ungrouped chart. Each question is processed
independently and wrapped in its own try/except - a malformed or unexpected
question shows its own inline error instead of breaking the rest of the
report.
"""
from __future__ import annotations

import pandas as pd

import charts
from variables import Variable, list_questions, read_variable

# Last real survey question. Everything after this column in a LimeSurvey
# export is timing/bookkeeping metadata (interviewtime, "...Time" columns),
# and its presence is used as a sanity check that the upload is the right
# kind of file.
ANCHOR_QUESTION = "G10Q01"


def prepare_responses(df: pd.DataFrame, only_full_responses: bool) -> pd.DataFrame:
    """Sanity-checks the upload and applies the "full responses only" filter."""
    if ANCHOR_QUESTION not in df.columns:
        raise ValueError(
            "This doesn't look like a survey export - couldn't find the "
            f"expected '{ANCHOR_QUESTION}' column."
        )
    if only_full_responses:
        if "submitdate" not in df.columns:
            raise ValueError("Can't filter to full responses - no 'submitdate' column in this file.")
        df = df[df["submitdate"].notna()]
    if df.empty:
        raise ValueError("No responses left after filtering.")
    return df


def _discrete_exclusive(var: Variable) -> dict:
    labels = var.values.dropna()
    if labels.empty:
        raise ValueError("no valid responses for this question")
    return {"type": "plot", "obj": charts.discrete_exclusive_chart(labels, var.options)}


def _discrete_inclusive(var: Variable) -> dict:
    ticked = var.values[var.answered]
    if ticked.empty:
        raise ValueError("no valid responses for this question")
    props = ticked.astype(float).mean()
    return {"type": "plot", "obj": charts.discrete_inclusive_chart(props)}


def _continuous(var: Variable) -> dict:
    values = var.values.dropna()
    if values.empty:
        raise ValueError("no valid numeric responses for this question")
    return {"type": "plot", "obj": charts.continuous_chart(values)}


def _ranking(var: Variable) -> dict:
    ranks = var.values[var.answered]
    n = len(ranks)
    if n == 0:
        raise ValueError("no valid responses for this question")
    n_ranks = len(var.options)
    pivot = pd.DataFrame(
        {opt: [(ranks[opt] == r).sum() / n for r in range(1, n_ranks + 1)] for opt in var.options},
        index=[f"Ranked as #{r}" for r in range(1, n_ranks + 1)],
    )
    return {"type": "plot", "obj": charts.ranking_chart(pivot)}


def _country(var: Variable) -> dict:
    names = var.values.dropna()
    if names.empty:
        raise ValueError("no valid responses for this question")
    counts = names.value_counts()
    table = pd.DataFrame({
        "Country": counts.index,
        "Number of responses": counts.values,
    }).sort_values("Number of responses", ascending=False).reset_index(drop=True)
    return {"type": "table", "obj": table}


_BUILDERS = {
    "DiscreteExclusive": _discrete_exclusive,
    "DiscreteInclusive": _discrete_inclusive,
    "Continuous": _continuous,
    "Ranking": _ranking,
    "Country": _country,
}


def build_question_item(df: pd.DataFrame, question: dict, countries: pd.DataFrame) -> dict:
    """The ungrouped chart/table for one question, or an error item."""
    try:
        var = read_variable(df, question, countries)
        item = _BUILDERS[question["kind"]](var)
        item["title"] = question["title"]
        item["other_texts"] = var.other_texts
        if var.notes:
            item["caption"] = " ".join(var.notes)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, isolates one bad question
        item = {
            "type": "error",
            "title": f"{question['code']} ({question['kind']})",
            "message": str(exc),
        }
    return item


def build_report(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    countries: pd.DataFrame,
    only_full_responses: bool,
) -> list[dict]:
    return build_report_items(prepare_responses(df, only_full_responses), metadata, countries)


def build_report_items(df: pd.DataFrame, metadata: pd.DataFrame, countries: pd.DataFrame) -> list[dict]:
    """Like build_report, for responses already passed through prepare_responses."""
    return [build_question_item(df, q, countries) for q in list_questions(metadata, df.columns)]
