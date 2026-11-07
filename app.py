"""Streamlit app: visualises coded survey results as simple charts.

Everything runs client-side / in-session - the uploaded response file is
held only in memory for this browser session and is never written to disk
or sent anywhere else. The question/answer translation tables
(lisasurvey_codes.xlsx and lisasurvey_codes_extended.xlsx) ship with the app
itself; see metadata.py for how to update them.

Two tabs:
- Full report: one chart/table per question.
- Explore: one question at a time, optionally grouped by a second question.

A sidebar toggle switches every label between the compact (short) texts and
the extended (full question / answer) texts.
"""
import io
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from explore import build_grouped_item, group_by_options, supports_scale_to_100
from metadata import load_metadata
from report import build_report_items, prepare_responses
from variables import list_questions

st.set_page_config(page_title="Survey results", layout="wide")
st.title("Attitudes towards anesthesia for pregnant women: a survey [results]")

if "responses" not in st.session_state:
    st.session_state.responses = None
    # Full report per text view ("compact"/"extended"), built on first use
    # and reused until a new file is loaded.
    st.session_state.reports = {}

with st.sidebar:
    uploaded_file = st.file_uploader("Drag the survey responses .xlsx here", type=["xlsx"])
    full_only = st.checkbox("Full responses only", value=False)
    run_clicked = st.button("Get results", type="primary")
    status = st.empty()
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
        with st.spinner("Building report..."):
            try:
                df = pd.read_excel(uploaded_file, sheet_name=0)
            except Exception:
                st.session_state.responses = None
                st.session_state.reports = {}
                status.error("Couldn't read that file - please upload a valid .xlsx survey export.")
            else:
                try:
                    st.session_state.responses = prepare_responses(df, full_only)
                    st.session_state.reports = {}
                    status.empty()
                except Exception as exc:
                    st.session_state.responses = None
                    st.session_state.reports = {}
                    status.error(f"Error: {exc}")

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


responses = st.session_state.responses
if responses is not None and view not in st.session_state.reports:
    with st.spinner("Building report..."):
        items = build_report_items(responses, metadata, countries)
        # Render each chart to an image once, here, instead of on every
        # rerun of the page (re-drawing 34 charts made every click slow).
        for item in items:
            if item["type"] == "plot":
                item["png"] = figure_png(item.pop("obj"))
        st.session_state.reports[view] = items


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


tab_report, tab_explore = st.tabs(["Full report", "Explore"])

with tab_report:
    items = st.session_state.reports.get(view) if responses is not None else None
    if items:
        for item in items:
            heading(item["title"])
            show_item(item)
    else:
        st.info("Upload a survey responses file and click **Get results**.")


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
        st.info("Upload a survey responses file and click **Get results**.")
    else:
        explore_tab(responses, metadata, countries)
