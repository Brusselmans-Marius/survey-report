"""Explore tab: one question at a time, optionally split by a second one.

The chart type is chosen from the pair of question types:

    plotted question     grouped by any groupable question
    -----------------    -----------------------------------------------
    DiscreteExclusive    split bars, % of each group giving each answer
    DiscreteInclusive    split bars, % of each group ticking each option
    Continuous           box + dots per group (scatter if the group-by
                         question is Continuous too)
    Ranking              average-rank dot plot, one dot per group
    Country              table of countries x groups (counts)

Grouping by Country is the exception: there are far too many countries for
a split chart, so it always gives a table with one row per country.

How each question type becomes groups:
    DiscreteExclusive    its answers
    Continuous           3-4 ranges of roughly equal size
    Ranking              each respondent's #1 choice
    Country              one table row per country (see above)
    DiscreteInclusive    not offered - one respondent can be in several groups
"""
from __future__ import annotations

import pandas as pd

import charts
from variables import Variable, read_variable

GROUPABLE_TYPES = ("DiscreteExclusive", "Continuous", "Ranking", "Country")
MAX_GROUPS = len(charts.GROUP_PALETTE)
OTHER = "Other"


def group_by_options(questions: list[dict], target_code: str) -> list[dict]:
    return [q for q in questions if q["kind"] in GROUPABLE_TYPES and q["code"] != target_code]


def _fmt(x: float) -> str:
    return f"{x:g}"


def _bin_continuous(values: pd.Series) -> tuple[pd.Series, list[str]]:
    """Splits numbers into 3-4 ranges of roughly equal size. Tied values are
    never split across ranges, so ranges can't overlap; each is labelled
    with the real lowest-highest value it contains."""
    v = values.dropna()
    n_bins = 4 if len(v) >= 40 else 3
    bins = pd.qcut(v, n_bins, labels=False, duplicates="drop")
    if bins.nunique() < 2:
        raise ValueError("too few distinct values to split this question into groups")
    levels, labels = [], pd.Series(index=values.index, dtype=object)
    for b in sorted(bins.unique()):
        members = v[bins == b]
        lo, hi = members.min(), members.max()
        label = _fmt(lo) if lo == hi else f"{_fmt(lo)}–{_fmt(hi)}"
        levels.append(label)
        labels[members.index] = label
    return labels, levels


def _fold_to_other(labels: pd.Series, keep: int) -> tuple[pd.Series, list[str]]:
    """Keeps the `keep` most common labels and folds the rest into "Other"."""
    counts = labels.value_counts()
    top = list(counts.index[:keep])
    if len(counts) <= keep:
        return labels, top
    folded = labels.where(labels.isin(top) | labels.isna(), OTHER)
    return folded, top + [OTHER]


def to_groups(var: Variable) -> tuple[pd.Series, list[str]]:
    """(group label per respondent row, group levels in display order).
    Levels are listed in the question's own order, so a level's colour is
    stable across charts even if some groups are empty after filtering."""
    if var.kind == "DiscreteExclusive":
        labels, levels = var.values, list(var.options)
    elif var.kind == "Continuous":
        labels, levels = _bin_continuous(var.values)
    elif var.kind == "Ranking":
        is_first = var.values == 1
        has_first = is_first.any(axis=1)
        labels = is_first.idxmax(axis=1).where(has_first)
        levels = list(var.options)
    else:
        raise ValueError(f"questions of type {var.kind} can't be used to group")

    if len(levels) > MAX_GROUPS:
        labels, levels = _fold_to_other(labels.dropna(), MAX_GROUPS - 1)
        labels = labels.reindex(var.values.index)
    return labels, levels


def group_colors(levels: list[str]) -> dict[str, str]:
    colors, slot = {}, 0
    for level in levels:
        if level == OTHER:
            colors[level] = charts.OTHER_GRAY
        else:
            colors[level] = charts.GROUP_PALETTE[slot]
            slot += 1
    return colors


def supports_scale_to_100(target_kind: str, group_kind: str) -> bool:
    """Whether the "Scale to 100%" view exists for this pair: single-choice
    answers split by groups (not countries, which are a table)."""
    return target_kind == "DiscreteExclusive" and group_kind in GROUPABLE_TYPES and group_kind != "Country"


def build_grouped_item(
    df: pd.DataFrame,
    target_q: dict,
    group_q: dict,
    countries: pd.DataFrame,
    scale_to_100: bool = False,
) -> dict:
    """The grouped chart for `target_q` split by `group_q`. Returns a dict with
    type ("plot" / "table" / "error"), obj, caption, other_texts and - for
    charts - a `data` DataFrame for the table view."""
    try:
        target = read_variable(df, target_q, countries)
        grouper = read_variable(df, group_q, countries)
        item = _build_grouped(target, grouper, scale_to_100)
    except Exception as exc:  # noqa: BLE001 - show the problem inline, keep the app alive
        return {"type": "error", "message": str(exc)}
    item["other_texts"] = target.other_texts
    notes = target.notes + grouper.notes
    if notes:
        item["caption"] = (item.get("caption", "") + " " + " ".join(notes)).strip()
    return item


def _build_grouped(target: Variable, grouper: Variable, scale_to_100: bool) -> dict:

    if target.kind == "Continuous" and grouper.kind == "Continuous":
        both = target.values.notna() & grouper.values.notna()
        if not both.any():
            raise ValueError("no respondents answered both questions")
        x, y = grouper.values[both], target.values[both]
        fig = charts.scatter_chart(x, y, xlabel=grouper.title, ylabel=target.title)
        data = pd.DataFrame({grouper.title: x, target.title: y}).reset_index(drop=True)
        return {"type": "plot", "obj": fig, "data": data,
                "caption": f"Based on {int(both.sum())} respondents who answered both questions."}

    if grouper.kind == "Country":
        return _by_country_table(target, grouper)

    group_labels, levels = to_groups(grouper)
    keep = target.answered & group_labels.notna()
    if not keep.any():
        raise ValueError("no respondents answered both questions")
    g = group_labels[keep]

    # Only groups that actually have respondents are drawn, but colours come
    # from the full level list so they never shift.
    colors = group_colors(levels)
    present = [lvl for lvl in levels if (g == lvl).any()]
    sizes = {lvl: int((g == lvl).sum()) for lvl in present}
    legend = {lvl: f"{lvl} (n={sizes[lvl]})" for lvl in present}
    color_list = [colors[lvl] for lvl in present]
    grouped_by = grouper.title + (" (#1 choice)" if grouper.kind == "Ranking" else "")
    caption = (
        f"Based on {int(keep.sum())} respondents who answered both questions. "
        f"Grouped by: {grouped_by}"
        + ("" if grouped_by.endswith(("?", ".", "!", ")")) else ".")
    )

    if target.kind == "DiscreteExclusive":
        answers = target.values[keep]
        shares = pd.crosstab(answers, g, normalize="columns")
        options = [o for o in target.options if o in shares.index]
        pct = shares.reindex(index=options, columns=present).fillna(0.0)
        pct.columns = [legend[lvl] for lvl in present]
        if scale_to_100:
            # One bar per group, split into that group's answers.
            answer_colors = group_colors(target.options)
            fig = charts.stacked_100_chart(pct.T, [answer_colors[o] for o in options])
            caption += " Each bar is one group's answers, adding up to 100%."
            return {"type": "plot", "obj": fig, "caption": caption, "data": _pct_table(pct.T)}
        return _answer_bars(pct, target, color_list, caption)

    if target.kind == "DiscreteInclusive":
        ticked = target.values[keep].astype(float)
        pct = ticked.groupby(g).mean().reindex(present).T
        pct.columns = [legend[lvl] for lvl in present]
        caption += " Respondents could tick several options, so a group's bars don't add up to 100%."
        return _answer_bars(pct, target, color_list, caption)

    if target.kind == "Continuous":
        vals = target.values[keep]
        groups = [(legend[lvl], vals[g == lvl]) for lvl in present]
        fig = charts.grouped_box_chart(groups, color_list)
        data = pd.DataFrame({
            "Group": [legend[lvl] for lvl in present],
            "n": [sizes[lvl] for lvl in present],
            "Median": [vals[g == lvl].median() for lvl in present],
            "Q1": [vals[g == lvl].quantile(0.25) for lvl in present],
            "Q3": [vals[g == lvl].quantile(0.75) for lvl in present],
            "Min": [vals[g == lvl].min() for lvl in present],
            "Max": [vals[g == lvl].max() for lvl in present],
        })
        return {"type": "plot", "obj": fig, "caption": caption, "data": data}

    if target.kind == "Ranking":
        means = target.values[keep].groupby(g).mean().reindex(present)
        means.index = [legend[lvl] for lvl in present]
        # One row per group, one dot per option (coloured by option).
        option_colors = group_colors(target.options)
        fig = charts.ranking_by_group_chart(
            means.T, [option_colors[o] for o in means.columns], n_ranks=len(target.options)
        )
        caption += " Dots show each option's average rank within the group; further left = more preferred."
        data = means.round(2).rename_axis("Group").reset_index()
        return {"type": "plot", "obj": fig, "caption": caption, "data": data}

    if target.kind == "Country":
        counts = pd.crosstab(target.values[keep], g).reindex(columns=present).fillna(0).astype(int)
        counts.columns = [legend[lvl] for lvl in present]
        counts["Total"] = counts.sum(axis=1)
        counts = counts.sort_values("Total", ascending=False).rename_axis("Country").reset_index()
        return {"type": "table", "obj": counts, "caption": caption}

    raise ValueError(f"unsupported question type {target.kind}")


def _by_country_table(target: Variable, grouper: Variable) -> dict:
    """One row per country summarising `target` for that country's
    respondents, most responses first."""
    keep = target.answered & grouper.values.notna()
    if not keep.any():
        raise ValueError("no respondents answered both questions")
    country = grouper.values[keep]
    n = country.value_counts().rename("n")

    if target.kind == "DiscreteExclusive":
        shares = pd.crosstab(country, target.values[keep], normalize="index") * 100
        cols = [o for o in target.options if o in shares.columns]
        body = shares[cols].round(1)
        body.columns = [f"{c} (%)" for c in cols]
        note = "Percentages are within each country (each row adds up to 100%)."
    elif target.kind == "DiscreteInclusive":
        body = (target.values[keep].astype(float).groupby(country).mean() * 100).round(1)
        body.columns = [f"{c} (%)" for c in body.columns]
        note = ("% of each country's respondents who ticked each option; "
                "several options could be ticked, so rows don't add up to 100%.")
    elif target.kind == "Continuous":
        vals = target.values[keep].groupby(country)
        body = pd.DataFrame({
            "Median": vals.median(), "Q1": vals.quantile(0.25), "Q3": vals.quantile(0.75),
            "Min": vals.min(), "Max": vals.max(),
        }).round(1)
        note = ""
    elif target.kind == "Ranking":
        body = target.values[keep].groupby(country).mean().round(2)
        body.columns = [f"{c} (avg rank)" for c in body.columns]
        note = "Average rank per option (1 = most preferred)."
    else:
        raise ValueError(f"unsupported question type {target.kind}")

    table = (
        pd.concat([n, body], axis=1)
        .sort_values("n", ascending=False, kind="stable")
        .rename_axis("Country")
        .reset_index()
    )
    caption = (
        f"Based on {int(keep.sum())} respondents who answered both questions, "
        f"from {len(table)} countries. {note}"
    ).strip()
    return {"type": "table", "obj": table, "caption": caption}


def _answer_bars(pct: pd.DataFrame, target: Variable, group_color_list: list[str], caption: str) -> dict:
    """Split bars with one row per group and one bar per answer, coloured by
    answer. pct: index = answers, columns = group legend labels.

    A question with more answers than there are colours (only the
    preoxygenation question) keeps one row per answer with bars coloured by
    group instead - colours are never repeated or generated."""
    coded = [o for o in target.options if o != OTHER]
    if len(coded) > MAX_GROUPS:
        fig = charts.grouped_bar_chart(pct, group_color_list)
        caption += (
            f" This question has {len(target.options)} answers - too many to give each "
            "its own colour - so here rows show answers and colours show groups."
        )
        return {"type": "plot", "obj": fig, "caption": caption, "data": _pct_table(pct, "Answer")}
    answer_colors = group_colors(target.options)
    fig = charts.grouped_bar_chart(pct.T, [answer_colors[o] for o in pct.index])
    return {"type": "plot", "obj": fig, "caption": caption, "data": _pct_table(pct.T)}


def _pct_table(pct: pd.DataFrame, first_column: str = "Group") -> pd.DataFrame:
    """Percent table for the data view; one row per entry of pct's index."""
    return (pct * 100).round(1).map(lambda v: f"{v:.1f}%").rename_axis(first_column).reset_index()
