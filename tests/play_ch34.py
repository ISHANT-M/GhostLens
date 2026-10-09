"""Scripted runs of chapters 3 and 4 for the tests."""

from streamlit.testing.v1 import AppTest

from tests.play import click, click_key


def solve_ch3(at: AppTest, threshold: float = 0.2) -> AppTest:
    """Detect, load the balanced detector, run it, set the threshold and submit."""
    at = click_key(at, "l3_mode_Detect")
    at = click(at, "Load balanced")
    at = click(at, "Run YOLO26s")
    at.slider(key="l3_threshold").set_value(threshold).run()
    return click_key(at, "l3_submit")


def solve_ch4(at: AppTest) -> AppTest:
    """Segment, quantize, load U-Net Standard INT8, run it and purify at the default threshold."""
    at = click_key(at, "l4_mode_Segment")
    at.toggle(key="l4_int8").set_value(True).run()
    at = click(at, "Load balanced")
    at = click(at, "Run U-Net Standard INT8")
    return click_key(at, "l4_purify")
