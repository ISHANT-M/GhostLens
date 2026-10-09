"""Chapter 4 played headlessly: segmentation, cleanup, INT8 and the cleared screen."""

import re

import numpy as np
import pytest

from cv import segmentation as seg
from cv.edge import load_benchmark
from cv.models import MODELS_DIR
from game import device, level4
from tests.play import (CASE, START, assert_play_clean, click, click_key, is_cleared, needs_models, page, texts)
from tests.play_ch34 import solve_ch4


def rows(at) -> str:
    return " ".join(m.value for m in at.markdown)


def to_play(at, int8: bool = True):
    at = click_key(at, "l4_mode_Segment")
    if int8:
        at.toggle(key="l4_int8").set_value(True).run()
    at = click(at, "Load balanced")
    return click(at, f"Run U-Net Standard {'INT8' if int8 else 'FP32'}")


@needs_models
def test_only_standard_int8_passes():
    at = page(4)
    assert f"WALL #{CASE.wall_seed}" in texts(at)
    at = to_play(at, int8=False)
    at = click_key(at, "l4_purify")
    assert 4 not in at.session_state.completed_levels               # FP32 is too slow
    assert re.search(r"Latency ≤ 40 ms</td><td class='bad'>✗", rows(at))
    assert re.search(r"Model IoU ≥ 0.95 \(150 walls\)</td><td class='ok'>✓", rows(at))
    at.toggle(key="l4_int8").set_value(True).run()
    at = click(at, "Load balanced")
    at = click(at, "Run U-Net Standard INT8")
    at.slider(key="l4_thr").set_value(0.9).run()
    at = click_key(at, "l4_purify")
    assert 4 not in at.session_state.completed_levels               # a good model, but a bad threshold on this wall
    assert "On this wall the mask" in rows(at)
    at.slider(key="l4_thr").set_value(0.5).run()
    at = click_key(at, "l4_purify")
    assert not at.exception
    assert 4 in at.session_state.completed_levels
    assert at.session_state.grades[4] != "A"                        # two failed purifications


@needs_models
@pytest.mark.skipif(not (MODELS_DIR / "stain_unet_lite.pt").exists(), reason="run setup_models.py")
def test_lite_model_leaks():
    at = click_key(page(4), "l4_mode_Segment")
    at = click(at, "Load light")
    at = click(at, "Run U-Net Lite")
    at = click_key(at, "l4_purify")
    assert "masks leak" in rows(at)
    assert 4 not in at.session_state.completed_levels


@needs_models
def test_recovery_from_an_empty_battery():
    at = page(4)
    at.session_state.battery = 0
    at.session_state.battery_low_mark = 0
    at.session_state.xp = 25
    at = click_key(at.run(), "l4_mode_Segment")
    at.toggle(key="l4_int8").set_value(True).run()
    assert at.button(key="l4_model_standard-int8").disabled
    at.button(key="lobby_l4_picker").click().run()
    at = click(at, "Load balanced")
    at = click(at, "Run U-Net Standard INT8")
    at = click_key(at, "l4_purify")
    s = at.session_state
    assert not at.exception
    assert 4 in s.completed_levels and s.grades[4] != "A"
    assert s.battery == device.CHARGER_UNITS - device.used_in_level(s, 4)


@needs_models
def test_cleanup_off_leaves_iou_alone_and_on_changes_it():
    seed = CASE.wall_seed
    _, truth = level4.scene(seed)
    raw = level4.wall_probs("standard", True, seed) > 0.5
    assert seg.mask_iou(seg.clean_mask(raw, 0, 0), truth) == seg.mask_iou(raw, truth)
    assert seg.mask_iou(seg.clean_mask(raw, 9, 0), truth) != seg.mask_iou(raw, truth)

    at = to_play(page(4))
    base = f"{seg.mask_iou(raw, truth):.3f} → {seg.mask_iou(raw, truth):.3f}"
    assert base in rows(at)
    at.select_slider(key="l4_open").set_value(9).run()
    cleaned = seg.mask_iou(seg.clean_mask(raw, 9, 0), truth)
    assert f"{seg.mask_iou(raw, truth):.3f} → {cleaned:.3f}" in rows(at)
    at.select_slider(key="l4_open").set_value(0).run()
    at.select_slider(key="l4_close").set_value(3).run()   # still passes on every pool wall
    at = click_key(at, "l4_purify")
    d = at.session_state.l4_deployed
    assert d["close_k"] == 3 and d["open_k"] == 0
    assert d["iou"] == pytest.approx(seg.mask_iou(seg.clean_mask(raw, 0, 3), truth))
    assert d["raw_iou"] == pytest.approx(seg.mask_iou(raw, truth))


@needs_models
def test_probability_view_and_int8_diff_render():
    at = to_play(page(4), int8=False)
    assert at.radio(key="l4_view").options == ["Mask", "Probability", "Box"]   # no diff before both have run
    at.radio(key="l4_view").set_value("Probability").run()
    assert "PROBABILITY" in texts(at) and not at.exception
    at.radio(key="l4_view").set_value("Box").run()
    assert "HEALTHY WALL" in texts(at)
    at.toggle(key="l4_int8").set_value(True).run()
    at = click(at, "Load balanced")
    at = click(at, "Run U-Net Standard INT8")
    assert "INT8 diff" in at.radio(key="l4_view").options
    at.radio(key="l4_view").set_value("INT8 diff").run()
    seed = CASE.wall_seed
    a = level4.wall_probs("standard", True, seed) > 0.5
    b = level4.wall_probs("standard", False, seed) > 0.5
    n = int((a ^ b).sum())
    assert f"{n} PX DIFFER ({n / a.size:.1%})" in texts(at) and not at.exception


@needs_models
def test_play_screen_is_clean():
    at = click_key(page(4), "l4_mode_Detect")
    assert_play_clean(at)
    at = to_play(at, int8=False)
    assert_play_clean(at)
    at = click_key(at, "l4_purify")                                 # fails on latency: one line plus the checks
    assert_play_clean(at)
    assert 4 not in at.session_state.completed_levels


@needs_models
def test_cleared_screen():
    at = solve_ch4(page(4))
    s = at.session_state
    assert not at.exception and s.grades[4] == "A" and is_cleared(at, 4)
    assert s.battery == START - level4.par() + device.A_GRADE_UNITS
    text = texts(at)
    assert "Corruption purified: U-Net Standard INT8, IoU" in text
    assert at.get("vega_lite_chart") and at.dataframe
    assert at.button(key="l4_continue").label == "Close the case"
    assert at.button(key="l4_lab_pruning")
    assert "No cleanup" in text


@needs_models
def test_frontier_rows_only_standard_int8_is_inside_the_box():
    assert [r["name"] for r in level4.frontier_rows(load_benchmark()) if r["meets"]] == ["U-Net Standard INT8"]


def test_frontier_rows_mark_the_models_inside_the_limits():
    fake = {"unets": [
        {"name": "U-Net A", "variant": "a", "precision": "FP32", "latency_ms": 64.0, "iou": 0.96, "size_mb": 0.5},
        {"name": "U-Net A", "variant": "a", "precision": "INT8", "latency_ms": 22.0, "iou": 0.959, "size_mb": 0.13},
        {"name": "U-Net B", "variant": "b", "precision": "INT8", "latency_ms": 5.0, "iou": 0.92, "size_mb": 0.02},
    ]}
    rows = level4.frontier_rows(fake)
    assert [r["meets"] for r in rows] == [False, True, False]
    assert rows[1]["name"] == "U-Net A INT8"


def test_pixel_accuracy_flatters_an_empty_mask():
    truth = np.zeros((10, 10), bool)
    truth[:1] = True
    assert level4.pixel_accuracy(np.zeros_like(truth), truth) == 0.9


def test_diff_view_counts_disagreeing_pixels():
    img = np.full((4, 4, 3), 200, np.uint8)
    a, b = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    a[0, :2] = True
    b[0, 1:3] = True
    out, n = level4.diff_view(img, a, b)
    assert n == 2 and tuple(out[0, 0]) == level4.DIFF_COLOR and tuple(out[0, 1]) != level4.DIFF_COLOR


def test_probability_lut_is_monotonic_in_brightness():
    grey = level4.LUT[:, 0].astype(int).sum(axis=1)
    assert all(a <= b for a, b in zip(grey, grey[1:]))
