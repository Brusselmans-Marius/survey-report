"""Country-wise figure builder: one question, one 100% bar per country.

Which questions can be shown:
    DiscreteExclusive    each respondent's answer
    Ranking              each respondent's highest-ranked (#1) option

Each bar is split into the question's answers in the codes-file order, the
same order for every bar so countries can be compared directly.

"Units" are what get a bar: a single country, a user-defined country group
(its countries' respondents pooled together), or - with a minimum n set -
one pooled bar for every unit with fewer respondents than that. Bars run:
single countries, the pooled bar, then country groups at the bottom.

The sidebar filters are applied before this module sees the data. Group
definitions never change with the filters: a group simply counts whichever
of its countries' respondents are left.
"""
from __future__ import annotations

import pandas as pd

import charts
from explore import MAX_GROUPS, OTHER, group_colors
from variables import read_variable

SUPPORTED_KINDS = ("DiscreteExclusive", "Ranking")
RANKING_NOTE = "Ranking question: each respondent counted by their highest-ranked (#1) option."
SORT_BY_SIZE = "Most respondents first"
SORT_ALPHABETICAL = "Alphabetical"


def country_options(questions: list[dict]) -> list[dict]:
    return [q for q in questions if q["kind"] in SUPPORTED_KINDS]


def _answers(var) -> pd.Series:
    """One answer label per respondent row (missing if not answered)."""
    if var.kind == "Ranking":
        is_first = var.values == 1
        return is_first.idxmax(axis=1).where(is_first.any(axis=1))
    return var.values


def small_pool_label(min_n: int) -> str:
    return f"Other countries (each n < {min_n})"


def build_country_item(
    df: pd.DataFrame,
    question: dict,
    country_q: dict,
    countries: pd.DataFrame,
    groups: list[dict],
    min_n: int = 1,
    sort: str = SORT_BY_SIZE,
    height_scale: float = 1.0,
) -> dict:
    """The country-wise chart for `question`. `groups` is a list of
    {"name": str, "countries": [str, ...]}; a country is in at most one."""
    try:
        return _build(df, question, country_q, countries, groups, min_n, sort, height_scale)
    except Exception as exc:  # noqa: BLE001 - show the problem inline, keep the app alive
        return {"type": "error", "message": str(exc)}


def _build(df, question, country_q, countries, groups, min_n, sort, height_scale) -> dict:
    var = read_variable(df, question, countries)
    answers = _answers(var)
    country = read_variable(df, country_q, countries).values

    answered = answers.notna()
    no_country = int((answered & country.isna()).sum())
    keep = answered & country.notna()
    if not keep.any():
        raise ValueError("no respondents answered this question and the country question")
    answers, country = answers[keep], country[keep]

    # Country -> unit: its group's name, or itself.
    group_of = {c: g["name"] for g in groups for c in g["countries"]}
    unit = country.map(lambda c: group_of.get(c, c))
    sizes = unit.value_counts()

    group_names = [g["name"] for g in groups if g["name"] in sizes.index]
    singles = [u for u in sizes.index if u not in group_of.values()]
    if sort == SORT_ALPHABETICAL:
        singles.sort()
    else:  # largest first, ties alphabetical
        singles.sort(key=lambda u: (-sizes[u], u))

    # Units below the minimum n are pooled into one bar, below the single
    # countries. (A lone unit below it is left as is - "pooling" it would
    # only rename it.)
    pooled = [u for u in singles + group_names if sizes[u] < min_n]
    if len(pooled) < 2:
        pooled = []
    pool = small_pool_label(min_n)
    if pooled:
        unit = unit.where(~unit.isin(pooled), pool)
        sizes = unit.value_counts()
    singles = [u for u in singles if u not in pooled]
    group_names = [u for u in group_names if u not in pooled]
    # Country groups go at the bottom.
    order = singles + ([pool] if pooled else []) + group_names

    options = [o for o in var.options if (answers == o).any()]
    coded = [o for o in options if o != OTHER]
    if len(coded) > MAX_GROUPS:
        raise ValueError(f"this question has more than {MAX_GROUPS} answers - too many to colour")
    shares = pd.crosstab(unit, answers, normalize="index").reindex(index=order, columns=options).fillna(0.0)

    colors = group_colors(var.options)
    kinds = ["group" if u in group_names else "pooled" if u == pool else "country" for u in order]

    fig = charts.country_stacked_chart(
        shares, [colors[o] for o in options], kinds, height_scale=height_scale,
        xlabel=("Share of respondents by their #1-ranked option" if var.kind == "Ranking"
                else "Share of respondents"),
    )

    notes = []
    if var.kind == "Ranking":
        notes.append(RANKING_NOTE)
    notes.append(
        f"Based on {int(keep.sum())} respondents who answered this question and "
        "the country question. Each bar adds up to 100%."
    )
    if no_country:
        notes.append(f"{no_country} respondent{'s' if no_country > 1 else ''} who didn't give a country "
                     "left out.")
    if group_names:
        notes.append("Country groups are shown in bold at the bottom.")
    if pooled:
        notes.append(f"Pooled into one bar (n < {min_n} each): {', '.join(pooled)}.")
    notes += var.notes

    data = (shares * 100).round(1).map(lambda v: f"{v:.1f}%")
    data.insert(0, "n", [sizes[u] for u in order])
    data.index = order  # n is listed in the data table, not on the plot
    data.columns.name = None
    return {
        "type": "plot", "obj": fig, "caption": " ".join(notes),
        "data": data.rename_axis("Country / group").reset_index(),
        "other_texts": var.other_texts,
    }
