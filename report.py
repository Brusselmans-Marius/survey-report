"""Builds the full report: matches an uploaded response file's columns to
the bundled question metadata, translates codes to text, and produces one
chart/table item per question.

Ports app.R's build_report() and its per-type helpers. Each question is
processed independently and wrapped in its own try/except - a malformed or
unexpected question shows its own inline error instead of breaking the rest
of the report, same as the R version.
"""
from __future__ import annotations

import pandas as pd

import charts
from metadata import lookup_meaning

# Last real survey question. Everything after this column in a LimeSurvey
# export is timing/bookkeeping metadata (interviewtime, "...Time" columns),
# and its presence is used as a sanity check that the upload is the right
# kind of file.
ANCHOR_QUESTION = "G10Q01"


def _matches_question(column: str, question_code: str) -> bool:
    """True if `column` belongs to `question_code`: either the exact code
    itself, or a bracketed sub-answer/rank column, e.g. "G05Q01[A1]" or
    "G07Q04[1]" for question "G05Q01"/"G07Q04". "...Time" bookkeeping
    columns never match either form."""
    return column == question_code or column.startswith(f"{question_code}[")


def _has_columns(columns, question_code: str) -> bool:
    return any(_matches_question(c, question_code) for c in columns)


def _primary_column(columns, question_code: str) -> str | None:
    """The single data column for a question that has exactly one answer
    value: the exact code if present, otherwise its lone bracketed
    sub-answer column. LimeSurvey sometimes implements what's conceptually
    a single question (a number, a country code) as a one-item array, which
    gives it a "[A1]"-style column instead of a bare one - both forms show
    up in this survey for otherwise-identical "Continuous" questions."""
    if question_code in columns:
        return question_code
    matches = [c for c in columns if _matches_question(c, question_code)]
    return matches[0] if matches else None


def _discrete_exclusive(df: pd.DataFrame, question_code: str, qmeta: pd.DataFrame) -> dict:
    col = _primary_column(df.columns, question_code)
    if col is None:
        raise ValueError(f"expected column '{question_code}' not found in the uploaded file")

    raw = df[col].dropna().astype(str).str.strip()
    # LimeSurvey records a literal "-oth-" sentinel when the respondent chose
    # a free-text "Other" option instead of a coded answer - exclude those
    # from the coded-answer bar chart.
    raw = raw[raw != "-oth-"]

    ordered_labels = [lookup_meaning(qmeta, c) for c in qmeta["AnswerCode"]]
    labels = raw.map(lambda code: lookup_meaning(qmeta, code)).dropna()
    if labels.empty:
        raise ValueError("no valid responses for this question")

    fig = charts.discrete_exclusive_chart(labels, ordered_labels)
    return {"type": "plot", "obj": fig}


def _discrete_inclusive(df: pd.DataFrame, question_code: str, qmeta: pd.DataFrame) -> dict:
    option_cols = [
        (code, f"{question_code}[{code}]")
        for code in qmeta["AnswerCode"]
        if f"{question_code}[{code}]" in df.columns
    ]
    if not option_cols:
        raise ValueError(f"no answer-option columns found for '{question_code}'")

    cols = [col for _, col in option_cols]
    answered = df[cols].notna().any(axis=1)
    n = int(answered.sum())
    if n == 0:
        raise ValueError("no valid responses for this question")

    props = {
        lookup_meaning(qmeta, code): (df.loc[answered, col] == "Y").sum() / n
        for code, col in option_cols
    }
    fig = charts.discrete_inclusive_chart(pd.Series(props))
    return {"type": "plot", "obj": fig}


def _continuous(df: pd.DataFrame, question_code: str, qmeta: pd.DataFrame) -> dict:
    col = _primary_column(df.columns, question_code)
    if col is None:
        raise ValueError(f"expected column '{question_code}' not found in the uploaded file")

    values = pd.to_numeric(df[col], errors="coerce").dropna()
    if values.empty:
        raise ValueError("no valid numeric responses for this question")

    fig = charts.continuous_chart(values)
    return {"type": "plot", "obj": fig}


def _ranking(df: pd.DataFrame, question_code: str, qmeta: pd.DataFrame) -> dict:
    n_ranks = len(qmeta)
    rank_cols = [
        (i, f"{question_code}[{i}]")
        for i in range(1, n_ranks + 1)
        if f"{question_code}[{i}]" in df.columns
    ]
    if not rank_cols:
        raise ValueError(f"no rank columns found for '{question_code}'")

    answered = df[[col for _, col in rank_cols]].notna().any(axis=1)
    n = int(answered.sum())
    if n == 0:
        raise ValueError("no valid responses for this question")

    option_labels = [lookup_meaning(qmeta, code) for code in qmeta["AnswerCode"]]
    rank_labels = [f"Ranked as #{i}" for i, _ in rank_cols]
    pivot = pd.DataFrame(0.0, index=rank_labels, columns=option_labels)
    for (_, col), rank_label in zip(rank_cols, rank_labels):
        counts = df.loc[answered, col].map(lambda code: lookup_meaning(qmeta, code)).value_counts()
        for label, count in counts.items():
            if label in pivot.columns:
                pivot.loc[rank_label, label] = count / n

    fig = charts.ranking_chart(pivot)
    return {"type": "plot", "obj": fig}


def _country(df: pd.DataFrame, question_code: str, countries: pd.DataFrame) -> dict:
    col = _primary_column(df.columns, question_code)
    if col is None:
        raise ValueError(f"expected column '{question_code}' not found in the uploaded file")

    codes = df[col].dropna().astype(str).str.strip()
    if codes.empty:
        raise ValueError("no valid responses for this question")

    counts = codes.value_counts()
    name_by_code = dict(zip(countries["Code"], countries["Country"]))
    table = pd.DataFrame({
        "Country": [name_by_code.get(code, code) for code in counts.index],
        "Number of responses": counts.values,
    }).sort_values("Number of responses", ascending=False).reset_index(drop=True)

    return {"type": "table", "obj": table}


_BUILDERS = {
    "DiscreteExclusive": _discrete_exclusive,
    "DiscreteInclusive": _discrete_inclusive,
    "Continuous": _continuous,
    "Ranking": _ranking,
}


def build_report(
    df: pd.DataFrame,
    metadata: pd.DataFrame,
    countries: pd.DataFrame,
    only_full_responses: bool,
) -> list[dict]:
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

    items: list[dict] = []
    for question_code in metadata["QuestionCode"].dropna().unique():
        qmeta = metadata[metadata["QuestionCode"] == question_code]
        if qmeta.empty:
            continue

        qtype = qmeta["Type"].iloc[0]
        title = qmeta["Question"].iloc[0]

        if pd.isna(qtype) or qtype == "NotRelevant":
            continue
        if qtype == "Country":
            builder = lambda d, q, m: _country(d, q, countries)  # noqa: E731
        elif qtype in _BUILDERS:
            builder = _BUILDERS[qtype]
        else:
            continue

        # A question whose columns simply aren't in this export (e.g. the
        # codes file has since grown a question this file predates) is
        # skipped quietly rather than shown as an error.
        if not _has_columns(df.columns, question_code):
            continue

        try:
            item = builder(df, question_code, qmeta)
            item["title"] = title
            items.append(item)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, isolates one bad question
            items.append({
                "type": "error",
                "title": f"{question_code} ({qtype})",
                "message": str(exc),
            })

    return items
