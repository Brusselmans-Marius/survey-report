"""Streamlit app: visualises coded survey results as simple charts.

Everything runs client-side / in-session - the uploaded response file is
held only in memory for this browser session and is never written to disk
or sent anywhere else. The question/answer translation tables
(lisasurvey_codes.xlsx and lisasurvey_codes_extended.xlsx) ship with the app
itself; see metadata.py for how to update them.

Three tabs:
- Full report: one chart/table per question.
- Explore: one question at a time, optionally grouped by a second question.
- Country-wise figure builder: one question, one 100% bar per country, with
  user-defined country groups.

Sidebar filters ("Full responses only", "European respondents only") apply
immediately to both tabs. A sidebar toggle switches every label between the
compact (short) texts and the extended (full question / answer) texts.
"""
import io
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from countrywise import (
    RANKING_NOTE, SORT_ALPHABETICAL, SORT_BY_SIZE, build_country_item, country_options,
)
from explore import build_grouped_item, group_by_options, supports_scale_to_100
from metadata import load_metadata
from report import build_report_items, prepare_responses, respondent_countries
from variables import list_questions

st.set_page_config(page_title="Survey results", layout="wide")
st.title("Attitudes towards anesthesia for pregnant women: a survey [results]")

if "raw_responses" not in st.session_state:
    # The uploaded file as read; the sidebar filters are applied on top of
    # it on every run, so changing a filter needs no new upload.
    st.session_state.raw_responses = None
    # Full report per (text view, filters), built on first use and reused
    # until a new file is loaded.
    st.session_state.reports = {}
if "cw_groups" not in st.session_state:
    # Country groups of the Country-wise tab: [{"name", "countries"}, ...].
    # Kept for the whole browser session, whatever the filters or file.
    st.session_state.cw_groups = []

with st.sidebar:
    uploaded_file = st.file_uploader("Drag the survey responses .xlsx here", type=["xlsx"])
    run_clicked = st.button("Get results", type="primary")
    status = st.empty()
    st.divider()
    full_only = st.checkbox("Full responses only", value=False)
    europe_only = st.toggle(
        "European respondents only",
        value=False,
        help="Only respondents practising in Europe, including Turkey, Russia, "
             "Cyprus, the Caucasus and Kazakhstan. Respondents who didn't "
             "answer the country question are left out.",
    )
    filter_status = st.empty()
    st.divider()
    extended = st.toggle(
        "Full question & answer text",
        value=True,
        key="extended_view",
        help="Off: short titles and answer labels. On: the full question and "
             "answer texts as respondents saw them.",
    )

view = "extended" if extended else "compact"
metadata, countries = load_metadata(view)

if run_clicked:
    if uploaded_file is None:
        status.warning("Please upload a survey responses file first.")
    else:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=0)
            prepare_responses(df, only_full_responses=False)  # is it a survey export?
        except Exception as exc:
            st.session_state.raw_responses = None
            status.error(
                f"Error: {exc}" if isinstance(exc, ValueError)
                else "Couldn't read that file - please upload a valid .xlsx survey export."
            )
        else:
            st.session_state.raw_responses = df
        st.session_state.reports = {}

# Apply the sidebar filters to the loaded file.
responses = None
raw = st.session_state.raw_responses
if raw is not None:
    try:
        responses = prepare_responses(raw, full_only, europe_only, metadata, countries)
    except ValueError as exc:
        filter_status.warning(str(exc))
    else:
        filter_status.caption(f"Showing {len(responses)} of {len(raw)} responses.")

def figure_png(fig) -> bytes:
    """Renders a figure the way st.pyplot does (tight crop, 200 dpi)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def heading(title: str) -> None:
    # Full question texts can run to a few lines - use a smaller heading.
    if extended:
        st.markdown(f"##### {title}")
    else:
        st.subheader(title)


report_key = (view, full_only, europe_only)
if responses is not None and report_key not in st.session_state.reports:
    with st.spinner("Building report..."):
        items = build_report_items(responses, metadata, countries)
        # Render each chart to an image once, here, instead of on every
        # rerun of the page (re-drawing 34 charts made every click slow).
        for item in items:
            if item["type"] == "plot":
                item["png"] = figure_png(item.pop("obj"))
        st.session_state.reports[report_key] = items


def show_figure_fixed(fig) -> None:
    """Shows a figure at its natural size (~100 px per inch) instead of
    stretching it to the page width, which made tall charts overflow a
    laptop screen."""
    png = figure_png(fig)
    width_px = plt.imread(io.BytesIO(png)).shape[1]
    st.image(png, width=width_px // 2)


def show_other_texts(texts: list[str]) -> None:
    """Scrollable list of what respondents typed in the "Other" box."""
    st.markdown(f"**\"Other\" answers** ({len(texts)})")
    with st.container(height=260, border=True):
        for text in texts:
            st.markdown(f"- {_escape_markdown(text)}")


def _escape_markdown(text: str) -> str:
    """Shows respondents' text literally (no accidental bold, links, etc.)."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$])", r"\\\1", text)


def show_item(item: dict, fixed_size: bool = False) -> None:
    """Draws one chart/table/error item (shared by both tabs). If anyone
    typed an "Other" answer, the texts are listed in a box beside it."""
    if item["type"] == "error":
        st.error(f"Couldn't render this: {item['message']}")
        return
    if item.get("other_texts"):
        col_main, col_side = st.columns([3, 1])
        with col_side:
            show_other_texts(item["other_texts"])
        with col_main:
            _show_item_body(item, fixed_size)
    else:
        _show_item_body(item, fixed_size)


def _show_item_body(item: dict, fixed_size: bool) -> None:
    if item["type"] == "table":
        st.dataframe(item["obj"], hide_index=True, width="stretch")
    elif "png" in item:
        st.image(item["png"], width="stretch")
    elif fixed_size:
        show_figure_fixed(item["obj"])
    else:
        st.pyplot(item["obj"])
        plt.close(item["obj"])
    if item.get("caption"):
        st.caption(item["caption"])
    if item.get("data") is not None:
        with st.expander("Data table"):
            st.dataframe(item["data"], hide_index=True, width="stretch")


def show_no_data_message() -> None:
    if st.session_state.raw_responses is None:
        st.info("Upload a survey responses file and click **Get results**.")
    else:
        st.info("No responses match the current filters in the sidebar.")


tab_report, tab_explore, tab_country = st.tabs(
    ["Full report", "Explore", "Country-wise figure builder"]
)

with tab_report:
    items = st.session_state.reports.get(report_key) if responses is not None else None
    if items:
        for item in items:
            heading(item["title"])
            show_item(item)
    else:
        show_no_data_message()


# ---- Explore tab ---------------------------------------------------------
# Runs as a fragment: using its dropdowns only re-runs this function, not
# the whole page (so the Full report isn't redrawn on every click). Nothing
# is plotted until both a question and a grouping question are chosen.

def _reset(*keys: str):
    def callback():
        for key in keys:
            st.session_state[key] = None
    return callback


def _clear_grouping_if_same():
    # Plotting the question you were grouping by: clear the grouping.
    if st.session_state.get("ex_question") == st.session_state.get("ex_group_question"):
        st.session_state["ex_group_question"] = None


def _question_picker(label: str, topics: list[str], codes_in, title_of,
                     topic_key: str, question_key: str, on_question_change=None):
    col_topic, col_question = st.columns([1, 2])
    topic = col_topic.selectbox(
        f"{label}: topic", topics, index=None, placeholder="Choose a topic",
        key=topic_key, on_change=_reset(question_key),
    )
    codes = codes_in(topic) if topic else []
    code = col_question.selectbox(
        f"{label}: question", codes, index=None, format_func=title_of,
        placeholder="Choose a question" if topic else "Choose a topic first",
        disabled=not codes, key=question_key, on_change=on_question_change,
    )
    return code


@st.fragment
def explore_tab(responses, metadata, countries) -> None:
    questions = list_questions(metadata, responses.columns)
    by_code = {q["code"]: q for q in questions}
    groupable = group_by_options(questions, target_code=None)

    def topics_of(qs):
        return list(dict.fromkeys(q["category"] for q in qs))  # survey order

    def title_of(code):
        return by_code[code]["title"]

    target_code = _question_picker(
        "Plot", topics_of(questions),
        lambda t: [q["code"] for q in questions if q["category"] == t],
        title_of, "ex_topic", "ex_question", on_question_change=_clear_grouping_if_same,
    )
    group_code = _question_picker(
        "Group by", topics_of(groupable),
        # The plotted question can't also be the grouping question.
        lambda t: [q["code"] for q in groupable if q["category"] == t and q["code"] != target_code],
        title_of, "ex_group_topic", "ex_group_question",
    )

    supported = (
        target_code is not None and group_code is not None
        and supports_scale_to_100(by_code[target_code]["kind"], by_code[group_code]["kind"])
    )
    scale_on = st.toggle(
        "Scale to 100%", key="ex_scale_100", disabled=not supported,
        help="One bar per group, split into that group's answers and adding "
             "up to 100% - easier for comparing proportions. Available for "
             "single-choice questions.",
    )

    if target_code is None or group_code is None:
        st.info("Choose a question to plot and a question to group it by.")
        return
    if scale_on and not supported:
        st.caption(
            "Scale to 100% applies to single-choice questions only (not "
            "numbers, tick-all, ranking or country questions, or grouping by "
            "country). It switches back on automatically."
        )

    st.divider()
    heading(by_code[target_code]["title"])
    with st.spinner("Drawing chart..."):
        item = build_grouped_item(
            responses, by_code[target_code], by_code[group_code], countries,
            scale_to_100=scale_on and supported,
        )
    show_item(item, fixed_size=True)


with tab_explore:
    if responses is None:
        show_no_data_message()
    else:
        explore_tab(responses, metadata, countries)


# ---- Country-wise figure builder tab ---------------------------------------
# Also a fragment. Country groups live in st.session_state.cw_groups and are
# never changed by the sidebar filters: the filtered responses are simply
# counted into whichever groups exist.

def _add_group(all_countries: list[str]):
    ss = st.session_state
    name = (ss.get("cw_new_name") or "").strip()
    members = list(ss.get("cw_new_members") or [])
    taken = {g["name"].casefold() for g in ss.cw_groups} | {c.casefold() for c in all_countries}
    if not name:
        ss.cw_error = "Give the group a name."
    elif name.casefold() in taken:
        ss.cw_error = f"\"{name}\" is already used by a group or a country - pick another name."
    elif len(members) < 2:
        ss.cw_error = "Pick at least two countries."
    else:
        ss.cw_groups = ss.cw_groups + [{"name": name, "countries": sorted(members)}]
        ss.cw_new_name, ss.cw_new_members, ss.cw_making = "", [], False
        ss.cw_error = None


def _remove_group(name: str):
    def callback():
        st.session_state.cw_groups = [g for g in st.session_state.cw_groups if g["name"] != name]
    return callback


def _group_manager(all_countries: list[str], present: set[str]) -> None:
    """The "country groups" section below the plot: active groups with a
    remove button each, and the form to make a new one."""
    groups = st.session_state.cw_groups
    st.markdown("**Country groups**")
    if not groups:
        st.caption("No groups yet - every country is its own bar.")
    for group in groups:
        members = [
            c if c in present else f"{c} *(filtered out)*" for c in group["countries"]
        ]
        with st.container(border=True):
            col_text, col_x = st.columns([12, 1], vertical_alignment="center")
            note = "" if any(c in present for c in group["countries"]) else (
                "  \n:gray[No respondents under the current filters - not shown in the plot.]"
            )
            col_text.markdown(f"**{_escape_markdown(group['name'])}**: {', '.join(members)}{note}")
            col_x.button("✕", key=f"cw_remove_{group['name']}", help="Remove this group",
                         on_click=_remove_group(group["name"]))

    if st.toggle("Make new grouping", key="cw_making"):
        grouped = {c for g in groups for c in g["countries"]}
        free = [c for c in all_countries if c not in grouped]
        with st.container(border=True):
            st.multiselect(
                "Countries in the group", free, key="cw_new_members",
                placeholder="Choose countries",
                help="Countries already in a group aren't listed; remove that group first.",
            )
            st.text_input("Group name", key="cw_new_name", placeholder="e.g. Nordic countries")
            st.button("Confirm group", type="primary", on_click=_add_group, args=(all_countries,))
            if st.session_state.get("cw_error"):
                st.warning(st.session_state.cw_error)


def _download_buttons(fig, name: str) -> None:
    """Download the chart as a vector PDF or a 300 dpi PNG (the on-screen
    image is lower resolution to keep the page fast)."""
    pdf, png = io.BytesIO(), io.BytesIO()
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    col_pdf, col_png, _ = st.columns([1, 1, 4])
    col_pdf.download_button("Download PDF (vector)", pdf.getvalue(), f"{name}.pdf",
                            mime="application/pdf", on_click="ignore")
    col_png.download_button("Download PNG (300 dpi)", png.getvalue(), f"{name}.png",
                            mime="image/png", on_click="ignore")


@st.fragment
def country_tab(responses, raw, metadata, countries) -> None:
    questions = list_questions(metadata, responses.columns)
    country_qs = [q for q in questions if q["kind"] == "Country"]
    if not country_qs:
        st.error("No country question found in this file.")
        return
    options = country_options(questions)
    by_code = {q["code"]: q for q in options}

    code = _question_picker(
        "Plot", list(dict.fromkeys(q["category"] for q in options)),
        lambda t: [q["code"] for q in options if q["category"] == t],
        lambda c: by_code[c]["title"], "cw_topic", "cw_question",
    )
    st.caption("Single-choice and ranking questions. Ranking questions are shown by "
               "each respondent's highest-ranked (#1) option.")

    col_sort, col_min, col_height = st.columns(3)
    sort = col_sort.radio("Order of countries", [SORT_BY_SIZE, SORT_ALPHABETICAL],
                          key="cw_sort", horizontal=True,
                          help="Country groups are always listed first.")
    min_n = col_min.slider(
        "Minimum n per bar", 1, 30, 10, key="cw_min_n",
        help="Countries and groups with fewer respondents than this are pooled "
             "into one \"Other countries\" bar.",
    )
    height_pct = col_height.slider("Plot height", 50, 200, 100, step=10, format="%d%%",
                                   key="cw_height")

    # Every country in the uploaded file (not just the filtered responses),
    # so the group builder's list doesn't change with the sidebar filters.
    all_countries = sorted(respondent_countries(raw, metadata, countries).dropna().unique())
    present = set(respondent_countries(responses, metadata, countries).dropna())

    if code is None:
        st.info("Choose a question to plot.")
    else:
        question = by_code[code]
        st.divider()
        title = question["title"]
        heading(title)
        if question["kind"] == "Ranking":
            st.markdown(f"**{RANKING_NOTE}**")
        with st.spinner("Drawing chart..."):
            item = build_country_item(
                responses, question, country_qs[0], countries, st.session_state.cw_groups,
                min_n=min_n, sort=sort, height_scale=height_pct / 100,
            )
        if item["type"] == "plot":
            _download_buttons(item["obj"], f"{code}_by_country")
        show_item(item, fixed_size=True)

    st.divider()
    _group_manager(all_countries, present)


with tab_country:
    if responses is None:
        show_no_data_message()
    else:
        country_tab(responses, st.session_state.raw_responses, metadata, countries)
