"""Turns one survey question into tidy, per-respondent values.

This is the single place that knows how each question type is laid out in a
LimeSurvey export. Both the full report and the Explore tab build on it, so
a question is always read the same way wherever it is shown.

Every Variable's `values` is indexed like the response DataFrame (one entry
per respondent row), with missing/unanswered entries as NaN, so two
variables can be lined up respondent-by-respondent for grouping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from metadata import lookup_meaning

# Question types the app knows how to read. "NotRelevant" and blank types
# in the codes file are skipped everywhere.
SUPPORTED_TYPES = ("DiscreteExclusive", "DiscreteInclusive", "Continuous", "Ranking", "Country")

# Label used for a question's free-text "Other" option. It's added as an
# answer only when someone actually chose it.
OTHER_ANSWER = "Other"

# Upper limits for numeric questions: answers above the limit are counted as
# the limit itself (e.g. a typo of 525 years of experience counts as 70).
VALUE_CAPS = {
    "G01Q02": 70,  # years of experience
}


@dataclass
class Variable:
    code: str
    title: str
    kind: str
    # DiscreteExclusive / Country: Series of text labels.
    # Continuous: Series of floats.
    # DiscreteInclusive: DataFrame, one bool column per option label; rows of
    #   respondents who ticked nothing are all-NaN.
    # Ranking: DataFrame, one column per option label holding the rank
    #   (1 = top) that respondent gave it; non-respondents are all-NaN.
    values: pd.Series | pd.DataFrame
    # Answer labels in codes-file order (empty for Continuous / Country),
    # plus OTHER_ANSWER last if anyone chose "Other".
    options: list[str]
    # What respondents typed in the "Other" box, one entry per respondent.
    other_texts: list[str] = field(default_factory=list)
    # Short remarks about how the data was cleaned, shown under the chart.
    notes: list[str] = field(default_factory=list)

    @property
    def answered(self) -> pd.Series:
        """Boolean mask: which respondent rows answered this question."""
        if isinstance(self.values, pd.DataFrame):
            return self.values.notna().any(axis=1)
        return self.values.notna()


def _matches_question(column: str, question_code: str) -> bool:
    """True if `column` belongs to `question_code`: either the exact code
    itself, or a bracketed sub-answer/rank column, e.g. "G05Q01[A1]" or
    "G07Q04[1]" for question "G05Q01"/"G07Q04". "...Time" bookkeeping
    columns never match either form."""
    return column == question_code or column.startswith(f"{question_code}[")


def has_columns(columns, question_code: str) -> bool:
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


def _option_labels(qmeta: pd.DataFrame) -> list[str]:
    """Answer labels in codes-file order, skipping codes with no Meaning."""
    labels = [lookup_meaning(qmeta, code) for code in qmeta["AnswerCode"]]
    return [lbl for lbl in labels if isinstance(lbl, str)]


def _other_texts(df: pd.DataFrame, question_code: str) -> pd.Series:
    """The free text typed in a question's "Other" box (blank -> missing)."""
    col = f"{question_code}[other]"
    if col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    text = df[col].astype("string").str.strip()
    return text.where(text != "")


def _require_column(df: pd.DataFrame, question_code: str) -> str:
    col = _primary_column(df.columns, question_code)
    if col is None:
        raise ValueError(f"expected column '{question_code}' not found in the uploaded file")
    return col


def _discrete_exclusive(df, question_code, qmeta, countries):
    raw = df[_require_column(df, question_code)].astype("string").str.strip()
    # LimeSurvey records a literal "-oth-" sentinel when the respondent chose
    # the free-text "Other" option instead of a coded answer; what they typed
    # is in the "[other]" column.
    chose_other = (raw == "-oth-").fillna(False)
    labels = raw.map(lambda code: lookup_meaning(qmeta, code) if pd.notna(code) else None)
    labels = labels.astype(object).where(labels.notna())
    labels[chose_other] = OTHER_ANSWER
    options = _option_labels(qmeta) + ([OTHER_ANSWER] if chose_other.any() else [])
    texts = _other_texts(df, question_code)[chose_other].dropna().tolist()
    return labels, options, texts, []


def _continuous(df, question_code, qmeta, countries):
    values = pd.to_numeric(df[_require_column(df, question_code)], errors="coerce")
    notes = []
    cap = VALUE_CAPS.get(question_code)
    if cap is not None:
        n_over = int((values > cap).sum())
        if n_over:
            values = values.clip(upper=cap)
            notes.append(
                f"{n_over} answer{'s' if n_over > 1 else ''} above {cap:g} "
                f"counted as {cap:g}."
            )
    return values, [], [], notes


def _country(df, question_code, qmeta, countries):
    codes = df[_require_column(df, question_code)].astype("string").str.strip()
    name_by_code = dict(zip(countries["Code"], countries["Country"]))
    # Unknown codes are kept as the raw code rather than dropped.
    names = codes.map(lambda c: name_by_code.get(c, c) if pd.notna(c) else None)
    return names.astype(object).where(names.notna()), [], [], []


def _discrete_inclusive(df, question_code, qmeta, countries):
    option_cols = [
        (lookup_meaning(qmeta, code), f"{question_code}[{code}]")
        for code in qmeta["AnswerCode"]
        if f"{question_code}[{code}]" in df.columns
    ]
    option_cols = [(lbl, col) for lbl, col in option_cols if isinstance(lbl, str)]
    if not option_cols:
        raise ValueError(f"no answer-option columns found for '{question_code}'")

    ticked = pd.DataFrame({lbl: df[col] == "Y" for lbl, col in option_cols}, index=df.index)
    options = [lbl for lbl, _ in option_cols]
    # Filling in the "Other" box counts as ticking "Other".
    other = _other_texts(df, question_code)
    if other.notna().any():
        ticked[OTHER_ANSWER] = other.notna()
        options.append(OTHER_ANSWER)
    ticked = ticked.astype("boolean")
    # Respondents who ticked nothing didn't answer the question at all - mark
    # them missing rather than "ticked no option", so they're left out of
    # every denominator.
    answered = ticked.any(axis=1)
    ticked.loc[~answered, :] = pd.NA
    return ticked, options, other.dropna().tolist(), []


def _ranking(df, question_code, qmeta, countries):
    options = _option_labels(qmeta)
    rank_cols = [
        (i, f"{question_code}[{i}]")
        for i in range(1, len(qmeta) + 1)
        if f"{question_code}[{i}]" in df.columns
    ]
    if not rank_cols:
        raise ValueError(f"no rank columns found for '{question_code}'")

    ranks = pd.DataFrame(float("nan"), index=df.index, columns=options)
    for rank, col in rank_cols:
        labels = df[col].map(lambda code: lookup_meaning(qmeta, code) if pd.notna(code) else None)
        for row, label in labels.dropna().items():
            if label in ranks.columns:
                ranks.at[row, label] = rank
    return ranks, options, [], []


_READERS = {
    "DiscreteExclusive": _discrete_exclusive,
    "DiscreteInclusive": _discrete_inclusive,
    "Continuous": _continuous,
    "Ranking": _ranking,
    "Country": _country,
}


def list_questions(metadata: pd.DataFrame, columns) -> list[dict]:
    """The questions that can be shown for this export, in survey order:
    dicts of code, title, kind and their metadata rows."""
    out = []
    for code in metadata["QuestionCode"].dropna().unique():
        qmeta = metadata[metadata["QuestionCode"] == code]
        kind = qmeta["Type"].iloc[0]
        if kind not in SUPPORTED_TYPES:
            continue
        # A question whose columns simply aren't in this export (e.g. the
        # codes file has since grown a question this file predates) is
        # skipped quietly rather than shown as an error.
        if not has_columns(columns, code):
            continue
        out.append({
            "code": code,
            "title": qmeta["Question"].iloc[0],
            "category": qmeta["Category"].iloc[0],
            "kind": kind,
            "qmeta": qmeta,
        })
    return out


def read_variable(df: pd.DataFrame, question: dict, countries: pd.DataFrame) -> Variable:
    values, options, other_texts, notes = _READERS[question["kind"]](
        df, question["code"], question["qmeta"], countries
    )
    return Variable(
        question["code"], question["title"], question["kind"], values, options,
        other_texts=other_texts, notes=notes,
    )
