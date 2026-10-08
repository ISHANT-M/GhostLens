"""Play each chapter headlessly with Streamlit's AppTest."""

import pytest
from streamlit.testing.v1 import AppTest

from cv.edge import BENCHMARK_FILE
from cv.models import MODELS_DIR

needs_models = pytest.mark.skipif(not BENCHMARK_FILE.exists(), reason="run setup_models.py")


def page(level: int) -> AppTest:
    at = AppTest.from_string(f"""
import streamlit as st
from game import device, state, level{level}
state.init_state(st.session_state)
device.init_device(st.session_state)
level{level}.render()
""", default_timeout=180)
    return at.run()


def click(at: AppTest, label: str) -> AppTest:
    next(b for b in at.button if b.label.startswith(label)).click()
    return at.run()


def test_level1_wrong_mode_costs_battery_and_explains():
    at = page(1)
    at = click(at, "Use Retrain")
    assert any("augmentation" in m.value for m in at.markdown)
    assert at.session_state.battery == 100          # retraining isn't run, so it's free
    at = click(at, "Use Enhance")
    assert not at.exception
    assert at.slider(key="l1_gamma")


def test_level1_solves_with_sensible_settings():
    at = click(page(1), "Use Enhance")
    at.slider(key="l1_gamma").set_value(2.4)
    at.slider(key="l1_contrast").set_value(2.5)
    at.run()
    at.text_input[0].input("217")
    at = click(at, "Analyze clue")
    assert not at.exception
    assert 1 in at.session_state.completed_levels
    assert at.session_state.grades[1] == "A"
    assert at.session_state.battery < 100


def test_level1_warns_about_clipping():
    at = click(page(1), "Use Enhance")
    at.slider(key="l1_gamma").set_value(2.4)
    at.slider(key="l1_contrast").set_value(6.0)
    at.run()
    assert any("pure black or pure white" in m.value for m in at.markdown)


@needs_models
def test_level2_efficient_run_gets_an_a():
    at = click(page(2), "Use Classify")
    at = click(at, "Load light")
    at.button(key="scan_pocket watch").click().run()
    at.radio[0].set_value("pocket watch").run()
    at = click(at, "Tag as anchor")
    at.button(key="scan_room").click().run()
    quiz = at.radio[-1]
    quiz.set_value(quiz.options[2]).run()
    at = click(at, "Log answer")
    assert not at.exception
    assert 2 in at.session_state.completed_levels
    assert at.session_state.grades[2] == "A"


@needs_models
def test_level2_segmenting_a_single_object_wastes_battery():
    at = click(page(2), "Use Segment")
    assert at.session_state.battery < 90
    assert any("most expensive way" in m.value for m in at.markdown)


@needs_models
def test_level3_heavy_detector_does_not_fit_and_nano_falls_short():
    at = click(page(3), "Use Detect")
    heavy = next(b for b in at.button if b.key == "l3_model_yolo26m.pt")
    assert heavy.disabled                                   # 44 MB doesn't fit in 32 MB
    at = click(at, "Load light")
    at = click(at, "Run YOLO26n")
    at.slider(key="l3_threshold").set_value(0.2).run()
    at = click(at, "Submit report")
    assert 3 not in at.session_state.completed_levels
    assert any("can't do this at any threshold" in m.value for m in at.markdown)
    at = click(at, "Load balanced")
    at = click(at, "Run YOLO26s")
    at.slider(key="l3_threshold").set_value(0.2).run()
    at = click(at, "Submit report")
    assert not at.exception
    assert 3 in at.session_state.completed_levels
    assert at.session_state.grades[3] in "BC"               # it worked, but not efficiently


@needs_models
def test_level4_only_quantized_balanced_model_passes():
    at = click(page(4), "Use Segment")
    at = click(at, "Load balanced")
    at = click(at, "Run U-Net Standard FP32")
    at = click(at, "Purify")
    assert 4 not in at.session_state.completed_levels       # FP32 is too slow
    at.toggle(key="l4_int8").set_value(True).run()
    at = click(at, "Load balanced")
    at = click(at, "Run U-Net Standard INT8")
    at = click(at, "Purify")
    assert not at.exception
    assert 4 in at.session_state.completed_levels


@needs_models
@pytest.mark.skipif(not (MODELS_DIR / "stain_unet_lite.pt").exists(), reason="run setup_models.py")
def test_level4_lite_model_leaks():
    at = click(page(4), "Use Segment")
    at = click(at, "Load light")
    at = click(at, "Run U-Net Lite")
    at = click(at, "Purify")
    assert any("masks leak" in m.value for m in at.markdown)


@needs_models
def test_empty_battery_leaves_only_light_models():
    at = click(page(2), "Use Classify")
    at.session_state.battery = 0
    at.run()
    locked = {b.key: b.disabled for b in at.button if b.key.startswith("l2_model_")}
    assert locked == {"l2_model_yolo26n-cls.pt": False, "l2_model_yolo26s-cls.pt": True, "l2_model_yolo26m-cls.pt": True}
    assert any("EMERGENCY RESERVE" in m.value for m in at.markdown)
