"""Small HTML helpers for things Streamlit widgets can't do."""

from html import escape
from pathlib import Path

import streamlit as st

UI_DIR = Path(__file__).parent


def load_css() -> None:
    # styles.css first, then any page-specific sheets (lab.css, codex.css...)
    sheets = [UI_DIR / "styles.css"] + sorted(p for p in UI_DIR.glob("*.css") if p.name != "styles.css")
    st.markdown("<style>" + "\n".join(p.read_text() for p in sheets) + "</style>", unsafe_allow_html=True)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def scene_header(kicker: str, title: str, story: str) -> None:
    _html(f'<div class="gl-kicker">{escape(kicker)}</div>')
    st.header(title, anchor=False)
    _html(f'<p class="gl-story">{escape(story)}</p>')


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
