"""Shared helpers for playing chapters headlessly with Streamlit's AppTest, on a pinned case."""

import pytest
from streamlit.testing.v1 import AppTest

from cv.edge import BENCHMARK_FILE
from game import case, device

needs_models = pytest.mark.skipif(not BENCHMARK_FILE.exists(), reason="run setup_models.py")
SEED = 3                    # room 209, pocket watch, full inventory brief, wall 457
CASE = case.build_case(SEED)
START = device.BATTERY_START

SCRIPT = """
import streamlit as st
from game import device, state, level1, level2, level3, level4
state.init_state(st.session_state)
device.init_device(st.session_state)
st.session_state.setdefault("case_seed", {seed})
st.session_state.setdefault("chapter", {level})
{{1: level1, 2: level2, 3: level3, 4: level4}}[st.session_state.chapter].render()
"""


def page(level: int, seed: int = SEED) -> AppTest:
    return AppTest.from_string(SCRIPT.format(seed=seed, level=level), default_timeout=180).run()


def go(at: AppTest, chapter: int) -> AppTest:
    at.session_state.chapter = chapter
    return at.run()


def click(at: AppTest, label: str) -> AppTest:
    next(b for b in at.button if b.label.startswith(label)).click()
    return at.run()


def click_key(at: AppTest, key: str) -> AppTest:
    at.button(key=key).click()
    return at.run()


def texts(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)


def seed_where(**want) -> int:
    return next(n for n in range(1000) if all(getattr(case.build_case(n), k) == v for k, v in want.items()))


def open_page(at: AppTest, url_path: str) -> AppTest:
    # AppTest can't follow st.switch_page to a function page from one run to the next, so point it there directly
    at._page_hash = next(h for h, p in at._registered_pages.items() if p["url_pathname"] == url_path)
    return at.run()


def assert_play_clean(at: AppTest) -> None:
    """A play screen: no expanders, tables, charts, lesson boxes or tags, and at most one feedback line."""
    assert not at.expander, "expander on a play screen"
    assert not at.dataframe, "dataframe on a play screen"
    assert not at.get("vega_lite_chart"), "chart on a play screen"
    marks = [m.value for m in at.markdown]
    assert not any("gl-lesson" in m or "gl-tag" in m for m in marks), "lesson box or tag on a play screen"
    assert sum(m.count("gl-feedback") for m in marks) <= 1, "more than one feedback line"


def is_cleared(at: AppTest, n: int) -> bool:
    keys = {b.key for b in at.button}
    return f"CHAPTER {n} · CLEARED" in texts(at) and f"l{n}_continue" in keys
