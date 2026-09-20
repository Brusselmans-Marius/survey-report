"""Matplotlib figure builders, one per survey question type.

Each function takes already-translated, already-shaped data (report.py does
the code lookup and reshaping) and returns a ready-to-render Figure.

Color choices follow the project's data-viz palette (validated categorical
order + neutral chart ink):
- Charts where color carries no identity (the category label already does
  that job) use a single neutral blue, matching series slot 1.
- Ranking is the one chart where color distinguishes several answer options
  at once, so it uses the fixed categorical hue order (first 3 slots, which
  validate for both colorblind- and normal-vision separation together -
  every ranking question in this survey has exactly 3 options).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES_1 = "#2a78d6"       # single-series bars / histogram fill
SERIES_1_DARK = "#184f95"  # boxplot outline - darker step, same hue family
CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0, colors=INK_MUTED)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)


def discrete_exclusive_chart(labels: pd.Series, ordered_labels: list[str]) -> plt.Figure:
    """labels: one translated answer per respondent (single-choice question)."""
    counts = labels.value_counts()
    present = [lbl for lbl in ordered_labels if lbl in counts.index]
    counts = counts.reindex(present)[::-1]  # reversed so option 1 plots on top

    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(counts) + 1.2))
    ax.barh(counts.index, counts.values, color=SERIES_1, edgecolor=INK_PRIMARY, linewidth=0.8)
    pad = max(counts.values) * 0.02
    for y, v in enumerate(counts.values):
        ax.text(v + pad, y, str(int(v)), va="center", ha="left", color=INK_PRIMARY, fontsize=9)
    ax.set_xlim(0, max(counts.values) * 1.12)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def discrete_inclusive_chart(props: pd.Series) -> plt.Figure:
    """props: index = translated option labels (in question order), values =
    proportion of respondents who selected that option (0-1)."""
    props = props[::-1]  # reversed so option 1 plots on top

    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(props) + 1.2))
    ax.barh(props.index, props.values, color=SERIES_1, edgecolor=INK_PRIMARY, linewidth=0.8)
    for y, v in enumerate(props.values):
        ax.text(v + 0.015, y, f"{v * 100:.0f}%", va="center", ha="left", color=INK_PRIMARY, fontsize=9)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    _style_axes(ax)
    fig.tight_layout()
    return fig


def continuous_chart(values: pd.Series) -> plt.Figure:
    """values: numeric responses to a free-number question."""
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1, sharex=True, figsize=(6.5, 3.4), layout="constrained",
        gridspec_kw={"height_ratios": [1, 4], "hspace": 0.06},
    )

    ax_box.boxplot(
        values, vert=False, widths=0.6,
        boxprops=dict(color=SERIES_1_DARK, linewidth=1.5),
        medianprops=dict(color=SERIES_1_DARK, linewidth=1.5),
        whiskerprops=dict(color=SERIES_1_DARK, linewidth=1.5),
        capprops=dict(color=SERIES_1_DARK, linewidth=1.5),
        flierprops=dict(markeredgecolor=SERIES_1_DARK, markersize=4),
    )
    ax_box.set_yticks([])
    ax_box.set_facecolor(SURFACE)
    for spine in ax_box.spines.values():
        spine.set_visible(False)

    ax_hist.hist(values, bins=15, color=SERIES_1, edgecolor=INK_PRIMARY, linewidth=0.8, alpha=0.85)
    quartiles = np.round(np.percentile(values, [0, 25, 50, 75, 100]))
    ax_hist.set_xticks(sorted(set(quartiles)))
    ax_hist.set_yticks([])
    _style_axes(ax_hist)

    return fig


def ranking_chart(pivot: pd.DataFrame) -> plt.Figure:
    """pivot: index = rank-position labels ("Ranked as #1" first), columns =
    translated option labels (fixed order), values = proportion of
    respondents (0-1) who placed that option at that rank."""
    options = list(pivot.columns)
    colors = dict(zip(options, CATEGORICAL))
    ranks = list(pivot.index)[::-1]  # reversed so "#1" plots on top

    fig, ax = plt.subplots(figsize=(6.5, 0.7 * len(ranks) + 1.6))
    left = np.zeros(len(ranks))
    for opt in options:
        vals = pivot.loc[ranks, opt].to_numpy()
        ax.barh(ranks, vals, left=left, color=colors[opt], edgecolor=SURFACE, linewidth=1, label=opt)
        left = left + vals
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    _style_axes(ax)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=min(len(options), 4), frameon=False, fontsize=9,
    )
    fig.tight_layout()
    return fig
