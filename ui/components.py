"""Small HTML helpers for things Streamlit widgets can't do."""

from html import escape
from pathlib import Path

import altair as alt
import streamlit as st

UI_DIR = Path(__file__).parent
# dark-safe chart colours: brass, green, blue, red, grey, teal
CHART = ["#C8A464", "#8DBA83", "#7FA7C9", "#E07A66", "#B9B3A6", "#6FB0A8"]


def load_css() -> None:
    # styles.css first, then any page-specific sheets (lab.css, codex.css...)
    sheets = [UI_DIR / "styles.css"] + sorted(p for p in UI_DIR.glob("*.css") if p.name != "styles.css")
    st.markdown("<style>" + "\n".join(p.read_text() for p in sheets) + "</style>", unsafe_allow_html=True)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def title_card(kicker: str, title: str, story: str = "") -> None:
    text = f'<p class="gl-story">{escape(story)}</p>' if story else ""
    _html(f'<div class="gl-titlecard"><div class="chapter">{escape(kicker)}</div>'
          f'<div class="title">{escape(title)}</div><div class="rule"></div>{text}</div>')


def scene_header(kicker: str, title: str, story: str) -> None:
    title_card(kicker, title, story)


def objective(text: str) -> None:
    _html(f'<div class="gl-objective"><span>OBJECTIVE</span>{text}</div>')


def feedback(text: str, tone: str = "") -> None:
    _html(f'<div class="gl-feedback {tone}">{text}</div>')


def stage(key: str, ratio=(2.3, 1)):
    """Scene on the left, the handheld scanner on the right."""
    left, right = st.columns(list(ratio), gap="medium")
    return left.container(key=f"stage_{key}"), right.container(key=f"scanner_{key}")


def scanner_head(title: str, status: str = "") -> None:
    extra = f"<span>{escape(status)}</span>" if status else ""
    _html(f'<div class="gl-scanner-head"><span>{escape(title)}</span>{extra}</div>')


def evidence(img_rgb, caption: str) -> None:
    st.image(img_rgb, width="stretch")
    _html(f'<div class="gl-caption">{escape(caption)}</div>')


def chart(c: alt.Chart) -> None:
    dim, line, text = "#A7A398", "#2F322D", "#E7E4DA"
    c = (c.configure(background="transparent")
         .configure_view(stroke=None)
         .configure_axis(labelColor=dim, titleColor=dim, gridColor=line, domainColor=line, tickColor=line)
         .configure_legend(labelColor=text, titleColor=dim)
         .configure_title(color=text))
    st.altair_chart(c, width="stretch")


def stamp(status: str) -> str:
    return f'<span class="gl-stamp {status}">{status}</span>'


def tag(kind: str) -> str:
    # kind: measured / published / gameplay
    return f'<span class="gl-tag {kind}">{kind.upper()}</span>'


def caption(text: str) -> None:
    _html(f'<div class="gl-caption">{escape(text)}</div>')


def readouts(items: list[tuple[str, str, str]]) -> None:
    # items: (label, value, tone) with tone in "", "ok", "warn", "bad"
    cells = "".join(
        f'<div class="gl-readout {tone}"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div></div>'
        for label, value, tone in items
    )
    _html(f'<div class="gl-readouts">{cells}</div>')


def message(text: str, tone: str = "") -> None:
    _html(f'<div class="gl-msg {tone}">{text}</div>')


def lesson(points: list[str], title: str = "WHAT YOU LEARNED") -> None:
    items = "".join(f"<li>{p}</li>" for p in points)
    _html(f'<div class="gl-lesson"><div class="title">{title}</div><ul>{items}</ul></div>')


def kv_table(rows: list[tuple[str, str]], cls: str = "gl-ledger") -> str:
    # two-column table for ledgers and the case summary; values may hold html
    body = "".join(f"<tr><td>{escape(a)}</td><td>{b}</td></tr>" for a, b in rows)
    return f'<table class="{cls}">{body}</table>'
