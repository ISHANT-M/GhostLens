"""Scripted runs of chapters 1 and 2 for the tests."""

from streamlit.testing.v1 import AppTest

from game import case
from tests.play import CASE, click, click_key


def solve_ch1(at: AppTest, answer: str) -> AppTest:
    """Enhance, set the reference gamma and contrast, type the answer and analyze."""
    at = click_key(at, "l1_mode_Enhance")
    at.slider(key="l1_gamma").set_value(case.REFERENCE["gamma"])
    at.slider(key="l1_contrast").set_value(case.REFERENCE["contrast"])
    at.run()
    at.text_input[0].input(answer)
    return click(at, "Analyze clue")


def start_ch2(at: AppTest, tier: str = "light") -> AppTest:
    at = click_key(at, "l2_mode_Classify")
    return click(at, f"Load {tier}")


def tag_anchor(at: AppTest, c: case.Case = CASE) -> AppTest:
    at = click_key(at, f"scan_{c.anchor}")
    return click_key(at, f"l2_tag_{c.anchor}")


def aim(at: AppTest, win: str) -> AppTest:
    """Point the room reticle at a window id: 'full' or 'half-{col}-{row}'."""
    from game import level2
    if win == "full":
        at.radio(key="l2_win_size").set_value(level2.WIN_SIZES[0])
        return at.run()
    col, row = (int(v) for v in win.split("-")[1:])
    at.radio(key="l2_win_size").set_value(level2.WIN_SIZES[1]).run()
    at.select_slider(key="l2_pan").set_value(level2.PAN[col])
    at.select_slider(key="l2_tilt").set_value(level2.TILT[row])
    return at.run()


def scan_window(at: AppTest, win: str) -> AppTest:
    return click_key(aim(at, win), "l2_scan_window")


def solve_ch2(at: AppTest, c: case.Case = CASE, tier: str = "light") -> AppTest:
    from game import level2
    at = tag_anchor(start_ch2(at, tier), c)
    for win in level2.IDEAL_WINDOWS:
        if 2 in at.session_state.completed_levels:
            break
        at = scan_window(at, win)
    return at
