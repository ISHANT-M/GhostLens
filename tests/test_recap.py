"""The HUD chips, the loading screen, wrong-mode lines and the case recap."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from cv.edge import BENCHMARK_FILE
from game import flow, levels

needs_models = pytest.mark.skipif(not BENCHMARK_FILE.exists(), reason="run setup_models.py")


def test_chip_names_the_mode_only_once_chosen():
    assert flow.chip_label(3, {}) == "CH 3"
    assert flow.chip_label(3, {"l3_mode": "Detect"}) == "CH 3 · DETECT"


def test_cell_bar_rounds_to_the_nearest_cell():
    def lit(units):
        return flow.cellbar_html(units).count('class="on"')
    assert [lit(u) for u in (200, 151, 138, 68, 5, 0)] == [4, 3, 3, 1, 1, 0]
    assert lit(1000) == flow.CELLS


def test_loading_screen_and_warmup_never_name_a_task():
    words = ("enhanc", "classif", "detect", "segment", "u-net")
    for n, info in levels.LEVELS.items():
        text = (flow.loading_html(n, info["title"]) + " ".join(label for label, _ in levels.warmup(n))).lower()
        assert not any(w in text for w in words), n


def test_asked_line():
    assert flow.asked_line("Detect", "a name for each single object") == (
        "You asked for a box and a label for each object. This question needs a name for each single object.")


def test_wrong_mode_lines_list_each_wrong_try():
    script = """
import streamlit as st
from game import flow
st.session_state.l2_tried = ["Detect", "Segment", "Classify"]
st.session_state.l2_mode = "Classify"
options = {"Classify": {"verdict": "Right."}, "Detect": {"verdict": "Boxes you didn't need."},
           "Segment": {"verdict": "Every pixel, for one name."}}
for line in flow.wrong_mode_lines(2, options):
    st.markdown(line)
"""
    at = AppTest.from_string(script).run()
    assert [m.value for m in at.markdown] == ["You tried Detect first: Boxes you didn't need.",
                                              "You tried Segment first: Every pixel, for one name."]


SOLVED = {
    "completed_levels": {1, 2, 3, 4},
    "l1_final": {"quality": 0.847, "clip": 0.0, "ms": 2.1},
    "l2_model": "n-cls", "l2_anchor_label": "stopwatch",
    "l3_model": "s", "best_scores": {3: 7 / 9},
    "l4_deployed": {"id": "std-int8", "name": "U-Net Standard INT8", "iou": 0.968},
}
BENCH = {
    "classifiers": [{"id": "n-cls", "name": "YOLO26n-cls", "latency_ms": 7.0}],
    "detectors": [{"id": "s", "name": "YOLO26s", "latency_ms": 49.2}],
    "unets": [{"id": "standard-fp32", "latency_ms": 64.1, "size_mb": 0.5, "iou": 0.96},
              {"id": "standard-int8", "latency_ms": 22.1, "size_mb": 0.13, "iou": 0.959},
              {"id": "std-int8", "latency_ms": 22.1, "size_mb": 0.13, "iou": 0.959}],
}


def test_recap_rows():
    rows = levels.recap_rows(SOLVED, BENCH)
    assert [r["task"] for r in rows] == ["Enhance", "Classify", "Detect", "Segment"]
    assert rows[0]["number"] == "evidence quality 0.85 in 2.1 ms"
    assert rows[1]["model"] == "YOLO26n-cls" and "stopwatch" in rows[1]["number"]
    assert rows[2]["number"] == "F1 0.78 at 49.2 ms"
    assert rows[3]["number"] == "IoU 0.968 at 22.1 ms"
    assert all(r["module"].startswith("Module") for r in rows)
    assert levels.recap_rows({"completed_levels": {1}}, BENCH)[0]["number"] == "–"


def test_talking_points_use_your_numbers():
    points = levels.talking_points(SOLVED, BENCH)
    assert len(points) == 4 and "0.85" in points[0] and "0.78" in points[2] and "3.8×" in points[3]
    assert not any(p.rstrip().endswith("?") for p in points)        # statements, not questions


def test_recap_markdown_has_no_forecasts():
    points = levels.talking_points(SOLVED, BENCH)
    md = levels.recap_markdown("CASE 0003", levels.recap_rows(SOLVED, BENCH), points)
    assert "Forecasts" not in md and "Say it out loud" not in md
    assert "## Talking points" in md and md.count("| ") > 8 and points[3] in md


@needs_models
def test_int8_ratios_come_from_the_benchmark():
    bench = json.loads(BENCHMARK_FILE.read_text())
    smaller, faster, delta = levels.int8_ratios(bench)
    fp = levels.bench_row(bench, "unets", "standard-fp32")
    q = levels.bench_row(bench, "unets", "standard-int8")
    assert smaller == fp["size_mb"] / q["size_mb"] and faster == fp["latency_ms"] / q["latency_ms"]
    assert smaller > 1 and faster > 1 and abs(delta) < 0.05
