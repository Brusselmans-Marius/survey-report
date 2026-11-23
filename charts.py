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

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES_1 = "#2a78d6"       # single-series bars / histogram fill
SERIES_1_DARK = "#184f95"  # boxplot outline - darker step, same hue family
CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

# Softer categorical order for the Explore tab's "group by" charts. A group's
# colour is fixed by its position in the grouping question's answer list, so
# e.g. "Fellow" is the same colour in every grouped chart. Validated with the
# data-viz palette checker on the light chart surface (all charts render on
# SURFACE): lightness band, chroma floor, adjacent CVD separation (worst 9.9)
# and normal-vision separation (worst 15.7) all pass. Contrast vs the surface
# is below 3:1, so every grouped chart also gets a data-table view.
GROUP_PALETTE = ["#5b9bd5", "#ee8a5c", "#4fb99f", "#d6a140", "#8f80d2", "#e0849c"]
OTHER_GRAY = "#b5b3ab"     # an "Other" fold-in group is never a series hue

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


# Long answer/question texts (the "extended" view) are wrapped onto several
# lines rather than squeezing the plot. Short labels are unaffected.
LABEL_WRAP = 42    # characters per line for axis labels
LEGEND_WRAP = 34   # characters per line for legend entries
LINE_HEIGHT = 0.16  # inches per wrapped text line at fontsize ~9-10

# Explore-tab charts are capped at this height (inches; shown at ~100 px per
# inch) so a chart plus its controls fits on a laptop screen. Charts whose
# wrapped labels genuinely need more room grow past it.
MAX_GROUPED_HEIGHT = 4.2


def wrap(text, width: int = LABEL_WRAP) -> str:
    return textwrap.fill(str(text), width)


def _max_lines(labels) -> int:
    return max((str(lbl).count("\n") + 1 for lbl in labels), default=1)


def _row_height(labels, base: float = 0.5) -> float:
    """Height per bar row: `base`, or more if wrapped labels need it."""
    return max(base, _max_lines(labels) * LINE_HEIGHT + 0.15)


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
    names = [wrap(lbl) for lbl in counts.index]

    fig, ax = plt.subplots(figsize=(6.5, _row_height(names) * len(counts) + 1.2))
    ax.barh(names, counts.values, color=SERIES_1, edgecolor=INK_PRIMARY, linewidth=0.8)
    total = len(labels)  # everyone who answered: the percentages add up to 100
    pad = max(counts.values) * 0.02
    for y, v in enumerate(counts.values):
        ax.text(v + pad, y, f"{int(v)} ({v / total * 100:.0f}%)", va="center", ha="left",
                color=INK_PRIMARY, fontsize=9)
    ax.set_xlim(0, max(counts.values) * 1.22)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def discrete_inclusive_chart(props: pd.Series) -> plt.Figure:
    """props: index = translated option labels (in question order), values =
    proportion of respondents who selected that option (0-1)."""
    props = props[::-1]  # reversed so option 1 plots on top
    names = [wrap(lbl) for lbl in props.index]

    fig, ax = plt.subplots(figsize=(6.5, _row_height(names) * len(props) + 1.2))
    ax.barh(names, props.values, color=SERIES_1, edgecolor=INK_PRIMARY, linewidth=0.8)
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
        ax.barh(ranks, vals, left=left, color=colors[opt], edgecolor=SURFACE, linewidth=1,
                label=wrap(opt, LEGEND_WRAP))
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


# ---- Grouped charts (Explore tab) ------------------------------------------
# Each takes the group labels (legend text, e.g. "Fellow (n=12)") in display
# order and a matching list of colours, so colour assignment stays in one
# place (explore.py).


def _legend_columns(labels: list[str], max_chars: int = 80) -> int:
    """Most legend columns (up to 4) whose widest row still fits the figure
    width, so long answer labels wrap onto extra rows instead of overflowing."""
    for ncol in (4, 3, 2):
        if ncol > len(labels):
            continue
        rows = [labels[i:i + ncol] for i in range(0, len(labels), ncol)]
        if all(sum(_longest_line(lbl) + 6 for lbl in row) <= max_chars for row in rows):
            return ncol
    return 1


def _longest_line(label: str) -> int:
    return max(len(line) for line in label.split("\n"))


def _group_legend(fig, ax, labels: list[str], colors: list[str], marker: str = "s") -> None:
    if max(len(lbl) for lbl in labels) > 2 * LEGEND_WRAP:
        # Long (extended-view) labels: one entry per row across the full
        # width, which takes far fewer lines than narrow side-by-side columns.
        labels = [wrap(lbl, 95) for lbl in labels]
        ncol = 1
    else:
        labels = [wrap(lbl, LEGEND_WRAP) for lbl in labels]
        ncol = _legend_columns(labels)
    handles = [
        plt.Line2D([], [], linestyle="", marker=marker, markersize=9,
                   markerfacecolor=c, markeredgecolor=c)
        for c in colors
    ]
    return fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
        ncol=ncol, frameon=False, fontsize=9,
        handletextpad=0.3, columnspacing=1.2, labelcolor=INK_PRIMARY,
    )


def grouped_bar_chart(pct: pd.DataFrame, colors: list[str]) -> plt.Figure:
    """Split bars: one row per index entry, one coloured bar per column.
    In the Explore tab rows are groups and bars are answers (or, for a
    question with too many answers to colour, the other way round). Values
    are proportions (0-1); `colors` matches the columns."""
    options = list(pct.index)
    groups = list(pct.columns)
    k = len(groups)
    band = 0.8
    h = band / k

    names = [wrap(o) for o in options]
    n_opt = len(options)
    natural = n_opt * (0.22 * k + 0.25) + 0.9
    text_need = n_opt * (_max_lines(names) * LINE_HEIGHT + 0.1) + 0.9
    fig, ax = plt.subplots(figsize=(7.0, max(min(natural, MAX_GROUPED_HEIGHT), text_need)))
    for j, (group, color) in enumerate(zip(groups, colors)):
        # Option 1 on top; within each option, group 1 on top.
        ys = [(n_opt - 1 - i) + ((k - 1) / 2 - j) * h for i in range(n_opt)]
        ax.barh(ys, pct[group].to_numpy(), height=h, color=color,
                edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(range(n_opt))
    ax.set_yticklabels(names[::-1], color=INK_PRIMARY)
    ax.set_ylim(-0.5, n_opt - 0.5)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    _style_axes(ax)
    fig.tight_layout()
    _group_legend(fig, ax, groups, colors)
    return fig


def grouped_box_chart(groups: list[tuple[str, pd.Series]], colors: list[str]) -> plt.Figure:
    """groups: (group label, numeric values) in display order (top first).
    Each group gets a box plus every individual response as a dot."""
    k = len(groups)
    names = [wrap(label) for label, _ in groups]
    fig, ax = plt.subplots(figsize=(7.0, min(_row_height(names, 0.6) * k + 1.0, MAX_GROUPED_HEIGHT)))
    rng = np.random.default_rng(0)  # fixed seed: same jitter on every rerun
    for j, ((label, vals), color) in enumerate(zip(groups, colors)):
        y = k - 1 - j
        ax.boxplot(
            vals, positions=[y], vert=False, widths=0.5, patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color + "40", edgecolor=color, linewidth=1.5),
            medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
            whiskerprops=dict(color=color, linewidth=1.5),
            capprops=dict(color=color, linewidth=1.5),
        )
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(vals, y + jitter, s=30, color=color, edgecolor=SURFACE,
                   linewidth=1, zorder=3)
    ax.set_yticks(range(k))
    ax.set_yticklabels(names[::-1], color=INK_PRIMARY)
    ax.set_ylim(-0.6, k - 0.4)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def ranking_by_group_chart(mean_ranks: pd.DataFrame, colors: list[str], n_ranks: int) -> plt.Figure:
    """Dot plot of average ranks: one row per column of `mean_ranks`, one
    coloured dot per index entry (`colors` matches the index). The Explore
    tab passes rows = groups, dots = ranked options. Values: average rank,
    1 = top."""
    options = list(mean_ranks.columns)
    groups = list(mean_ranks.index)
    k = len(groups)
    n_opt = len(options)
    names = [wrap(o) for o in options]
    fig, ax = plt.subplots(figsize=(7.0, min(_row_height(names, 0.75) * n_opt + 1.0, MAX_GROUPED_HEIGHT)))
    for i, opt in enumerate(options):
        y = n_opt - 1 - i
        vals = mean_ranks[opt].dropna()
        if len(vals):
            ax.plot([vals.min(), vals.max()], [y, y], color=BASELINE, linewidth=2,
                    solid_capstyle="round", zorder=1)
        for j, (group, color) in enumerate(zip(groups, colors)):
            v = mean_ranks.at[group, opt]
            if pd.notna(v):
                # Small vertical offset per group so equal averages don't
                # hide each other.
                ax.scatter(v, y + ((k - 1) / 2 - j) * 0.12, s=80, color=color,
                           edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(range(n_opt))
    ax.set_yticklabels(names[::-1], color=INK_PRIMARY)
    ax.set_ylim(-0.6, n_opt - 0.4)
    ax.set_xlim(0.7, n_ranks + 0.3)
    ax.set_xticks(range(1, n_ranks + 1))
    ax.set_xlabel("Average rank (1 = most preferred)", color=INK_MUTED, fontsize=9)
    _style_axes(ax)
    fig.tight_layout()
    _group_legend(fig, ax, groups, colors, marker="o")
    return fig


def scatter_chart(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str) -> plt.Figure:
    """Two numeric questions against each other, one dot per respondent."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.scatter(x, y, s=40, color=SERIES_1, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xlabel(wrap(xlabel, 80), color=INK_MUTED, fontsize=9)
    ax.set_ylabel(wrap(ylabel, 50), color=INK_MUTED, fontsize=9)
    _style_axes(ax)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    fig.tight_layout()
    return fig


def stacked_100_chart(shares: pd.DataFrame, colors: list[str]) -> plt.Figure:
    """One 100% bar per group. shares: index = group labels (display order,
    top first), columns = answer labels (question order), values = share
    (0-1) of that group giving that answer; each row sums to 1. Segments
    wide enough to hold it are labelled with their percentage."""
    groups = list(shares.index)
    answers = list(shares.columns)
    names = [wrap(g) for g in groups]
    k = len(groups)
    fig, ax = plt.subplots(figsize=(7.0, min(_row_height(names, 0.55) * k + 1.0, MAX_GROUPED_HEIGHT)))
    ys = [k - 1 - j for j in range(k)]  # first group on top
    left = np.zeros(k)
    for answer, color in zip(answers, colors):
        vals = shares[answer].to_numpy(dtype=float)
        ax.barh(ys, vals, left=left, height=0.62, color=color,
                edgecolor=SURFACE, linewidth=2)
        for y, x0, v in zip(ys, left, vals):
            if v >= 0.08:
                ax.text(x0 + v / 2, y, f"{v * 100:.0f}%", ha="center", va="center",
                        color=INK_PRIMARY, fontsize=8)
        left = left + vals
    ax.set_yticks(range(k))
    ax.set_yticklabels(names[::-1], color=INK_PRIMARY)
    ax.set_ylim(-0.6, k - 0.4)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    _style_axes(ax)
    fig.tight_layout()
    _group_legend(fig, ax, answers, colors)
    return fig


# ---- Country-wise figure builder -------------------------------------------

COUNTRY_ROW_HEIGHT = 0.36  # inches per bar at height scale 100%
TITLE_WRAP = 95            # most characters per line of a figure title


def _country_row_height(names: list[str], height_scale: float, base: float = COUNTRY_ROW_HEIGHT) -> float:
    return max(base * height_scale, _max_lines(names) * LINE_HEIGHT + 0.06)


def _style_country_labels(ax, kinds: list[str]) -> None:
    """Country groups in bold, the pooled "Other countries" bar muted italic."""
    for tick, kind in zip(ax.get_yticklabels(), kinds):
        if kind == "group":
            tick.set_fontweight("bold")
        if kind == "pooled":
            tick.set_fontstyle("italic")
            tick.set_color(INK_MUTED)
        else:
            tick.set_color(INK_PRIMARY)


def add_figure_title(fig, title: str | None, above=None) -> None:
    """Puts the question title on the figure itself (so downloads carry it),
    left-aligned with the country labels and above everything else - the
    legend, if `above` is given."""
    if not title:
        return
    fig.draw_without_rendering()  # settle the layout first (constrained layout moves axes)
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    if above is not None:
        boxes.append(above.get_window_extent(renderer))
    left_px, right_px = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    to_fig = fig.transFigure.inverted()
    x0, _ = to_fig.transform((left_px, 0))
    _, y1 = to_fig.transform((0, max(b.y1 for b in boxes)))
    gap = 0.12 / fig.get_figheight()  # inches -> figure fraction
    # Wrap to the width the chart takes up (~0.62 em per bold character).
    width_pt = (right_px - left_px) / fig.dpi * 72
    chars = max(40, min(TITLE_WRAP, int(width_pt / (11 * 0.62))))
    fig.text(x0, y1 + gap, wrap(title, chars), ha="left", va="bottom",
             fontsize=11, fontweight="bold", color=INK_PRIMARY)


def country_stacked_chart(
    shares: pd.DataFrame,
    colors: list[str],
    kinds: list[str],
    height_scale: float = 1.0,
    xlabel: str = "Share of respondents",
    title: str | None = None,
) -> plt.Figure:
    """One 100% bar per country or country group, first row on top.
    shares: index = row labels, columns = answers (same order for every
    row), values = shares (0-1). kinds: "country", "group" (drawn with a
    bold label) or "pooled" (muted italic label), one per row. The figure
    grows with the number of rows, times `height_scale`."""
    rows = list(shares.index)
    answers = list(shares.columns)
    k = len(rows)
    names = [wrap(r) for r in rows]
    row_h = _country_row_height(names, height_scale)
    fig, ax = plt.subplots(figsize=(7.0, row_h * k + 0.9))
    ys = [k - 1 - j for j in range(k)]
    left = np.zeros(k)
    for answer, color in zip(answers, colors):
        vals = shares[answer].to_numpy(dtype=float)
        ax.barh(ys, vals, left=left, height=0.7, color=color, edgecolor=SURFACE, linewidth=2)
        # Percent labels only where the segment and the row are big enough.
        if row_h >= 0.22:
            for y, x0, v in zip(ys, left, vals):
                if v >= 0.08:
                    ax.text(x0 + v / 2, y, f"{v * 100:.0f}%", ha="center", va="center",
                            color=INK_PRIMARY, fontsize=8)
        left = left + vals
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, k - 0.4)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    _style_axes(ax)
    _style_country_labels(ax, kinds)
    fig.tight_layout()
    legend = _group_legend(fig, ax, answers, colors)
    add_figure_title(fig, title, above=legend)
    return fig


MEDIAN_RED = "#d03b32"  # median line of the country box plots


def country_box_chart(
    groups: list[tuple[str, pd.Series]],
    kinds: list[str],
    height_scale: float = 1.0,
    title: str | None = None,
) -> plt.Figure:
    """One box plot per country or country group (first on top), with every
    respondent's answer as a dot. One neutral colour: the row label already
    says which country it is. kinds as in country_stacked_chart."""
    k = len(groups)
    names = [wrap(label) for label, _ in groups]
    row_h = _country_row_height(names, height_scale, base=COUNTRY_ROW_HEIGHT * 1.2)
    fig, ax = plt.subplots(figsize=(7.0, row_h * k + 0.8))
    rng = np.random.default_rng(0)  # fixed seed: same jitter on every rerun
    ys = [k - 1 - j for j in range(k)]
    for y, (_, vals) in zip(ys, groups):
        ax.boxplot(
            vals, positions=[y], vert=False, widths=0.55, patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=SERIES_1 + "33", edgecolor=SERIES_1_DARK, linewidth=1.2),
            medianprops=dict(color=MEDIAN_RED, linewidth=2),
            whiskerprops=dict(color=SERIES_1_DARK, linewidth=1.2),
            capprops=dict(color=SERIES_1_DARK, linewidth=1.2),
        )
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(vals, y + jitter, s=22, color=SERIES_1, alpha=0.45, edgecolor="none",
                   zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, k - 0.4)
    _style_axes(ax)
    _style_country_labels(ax, kinds)
    fig.tight_layout()
    add_figure_title(fig, title)
    return fig


# Diverging scale for the country heatmap: blue (few respondents ticked the
# option) -> neutral grey at 50% -> red (most ticked it). Two hues and a
# neutral midpoint, never a rainbow; red/blue stays distinguishable with
# the common colour-vision deficiencies.
HEAT_LOW = "#2f6db5"
HEAT_MID = "#ecebe6"
HEAT_HIGH = "#c8423b"
HEAT_LABEL_WRAP = 26   # characters per line of a column heading
HEAT_LABEL_MAX = 80    # longer headings are shortened (full text in the data table)


def _short(text: str, limit: int = HEAT_LABEL_MAX) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def country_heatmap_chart(
    pct: pd.DataFrame,
    kinds: list[str],
    height_scale: float = 1.0,
    title: str | None = None,
) -> plt.Figure:
    """Heatmap for tick-all-that-apply questions: one row per country or
    group (first on top), one column per option; each cell is the share
    (0-1) of that row's respondents who ticked the option, printed in the
    cell and coloured on a blue-grey-red scale centred on 50%."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("heat", [HEAT_LOW, HEAT_MID, HEAT_HIGH])
    rows, cols = list(pct.index), list(pct.columns)
    k, n_opt = len(rows), len(cols)
    names = [wrap(r) for r in rows]
    row_h = _country_row_height(names, height_scale)
    many = n_opt > 5  # many columns: steeper, wider headings and a wider figure
    fig, ax = plt.subplots(figsize=(max(7.0, 2.2 + 0.68 * n_opt), row_h * k + 2.0))
    values = pct.to_numpy(dtype=float)
    ax.pcolormesh(values, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    if row_h >= 0.22:
        for i in range(k):
            for j in range(n_opt):
                v = values[i, j]
                ax.text(j + 0.5, i + 0.5, f"{v * 100:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if abs(v - 0.5) > 0.3 else INK_PRIMARY)
    ax.invert_yaxis()  # first row on top
    ax.set_yticks([i + 0.5 for i in range(k)])
    ax.set_yticklabels(names)
    ax.xaxis.tick_top()
    ax.set_xticks([j + 0.5 for j in range(n_opt)])
    ax.set_xticklabels([wrap(_short(c), HEAT_LABEL_WRAP + (12 if many else 0)) for c in cols],
                       rotation=60 if many else 35, ha="left", rotation_mode="anchor",
                       fontsize=8.5, color=INK_PRIMARY)
    ax.tick_params(length=0, colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _style_country_labels(ax, kinds)
    for tick in ax.get_xticklabels():
        tick.set_color(INK_PRIMARY)

    bar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), ax=ax,
        orientation="horizontal", location="bottom", fraction=0.04, pad=0.03, aspect=40,
    )
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, colors=INK_MUTED, labelsize=8)
    bar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    bar.set_label("Share of respondents who ticked the option", color=INK_MUTED, fontsize=9)
    add_figure_title(fig, title)
    return fig
