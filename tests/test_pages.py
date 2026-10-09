"""Play each chapter headlessly with Streamlit's AppTest, on a pinned case."""

from html import escape

import pytest
from streamlit.testing.v1 import AppTest

from cv.edge import BENCHMARK_FILE
from cv.models import MODELS_DIR
from game import achievements, case, device

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


def texts(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)


def seed_where(**want) -> int:
    return next(n for n in range(1000) if all(getattr(case.build_case(n), k) == v for k, v in want.items()))


# chapter 1

def solve_ch1(at: AppTest, answer: str) -> AppTest:
    at = click(at, "Use Enhance")
    at.slider(key="l1_gamma").set_value(case.REFERENCE["gamma"])
    at.slider(key="l1_contrast").set_value(case.REFERENCE["contrast"])
    at.run()
    at.text_input[0].input(answer)
    return click(at, "Analyze clue")


def test_level1_wrong_mode_costs_battery_and_explains():
    at = page(1)
    at = click(at, "Use Retrain")
    assert any("augmentation" in m.value for m in at.markdown)
    assert at.session_state.battery == START          # retraining isn't run, so it's free
    at = click(at, "Use Enhance")
    assert not at.exception
    assert at.slider(key="l1_gamma")


@pytest.mark.parametrize("clue", case.CLUES, ids=lambda c: c["id"])
def test_level1_every_pool_photo_solves_with_the_reference_settings(clue):
    seed = next(n for n in range(1000) if case.build_case(n).clue["id"] == clue["id"])
    at = page(1, seed)
    assert escape(clue["place"]) in texts(at)
    at = solve_ch1(at, f" {clue['answer'].upper()} ")    # spaces and case don't matter
    assert not at.exception
    assert 1 in at.session_state.completed_levels
    assert at.session_state.grades[1] == "A"
    assert at.session_state.l1_xp["charge"] == device.A_GRADE_UNITS
    assert f"The plate reads {clue['answer']}" in texts(at)


def test_level1_wrong_answer_is_rejected():
    at = solve_ch1(page(1), "999")
    assert 1 not in at.session_state.completed_levels
    assert any("isn't what the plate says" in m.value for m in at.markdown)
    assert at.session_state.battery < START


def test_level1_warns_about_clipping():
    at = click(page(1), "Use Enhance")
    at.slider(key="l1_gamma").set_value(2.4)
    at.slider(key="l1_contrast").set_value(6.0)
    at.run()
    assert any("pure black or pure white" in m.value for m in at.markdown)


def test_level1_warns_about_non_local_means():
    at = click(page(1), "Use Enhance")
    at.selectbox(key="l1_denoise").set_value("Non-local means").run()
    assert any("Non-local means costs about" in m.value for m in at.markdown)


# chapter 2

def solve_ch2(at: AppTest, c: case.Case = CASE, tier: str = "light") -> AppTest:
    at = click(at, "Use Classify")
    at = click(at, f"Load {tier}")
    at.button(key=f"scan_{c.anchor}").click().run()
    at.radio(key="l2_anchor_choice").set_value(c.anchor).run()
    at = click(at, "Tag as anchor")
    at.button(key="scan_room").click().run()
    quiz = at.radio(key="l2_quiz")
    quiz.set_value(quiz.options[2]).run()
    return click(at, "Log answer")


@needs_models
def test_level2_efficient_run_gets_an_a():
    at = page(2)
    assert f"CHAPTER 2 · {CASE.clue['label'].upper()}" in texts(at)
    at = solve_ch2(at)
    assert not at.exception
    assert 2 in at.session_state.completed_levels
    assert at.session_state.grades[2] == "A"
    assert case.ANCHORS[CASE.anchor]["entity"] in texts(at)


@needs_models
def test_level2_wrong_anchor_gets_the_riddle_miss():
    at = click(page(2), "Use Classify")
    at = click(at, "Load light")
    wrong = next(n for n in CASE.evidence_order if n != CASE.anchor)
    at.button(key=f"scan_{wrong}").click().run()
    at.radio(key="l2_anchor_choice").set_value(wrong).run()
    at = click(at, "Tag as anchor")
    assert case.ANCHORS[CASE.anchor]["miss"] in texts(at)


@needs_models
def test_level2_quiz_uses_the_label_the_model_really_gave():
    from game import level2
    at = click(page(2), "Use Classify")
    at = click(at, "Load balanced")
    at.button(key=f"scan_{CASE.anchor}").click().run()
    at.radio(key="l2_anchor_choice").set_value(CASE.anchor).run()
    at = click(at, "Tag as anchor")
    at.button(key="scan_room").click().run()
    label = level2.scan(level2.ROOM_PHOTO, "yolo26s-cls.pt")["top"][0][0]
    assert f"'{label}'" in at.radio(key="l2_quiz").options[2]


@needs_models
def test_level2_segmenting_a_single_object_wastes_battery():
    at = click(page(2), "Use Segment")
    assert at.session_state.battery < START - 50
    assert any("most expensive way" in m.value for m in at.markdown)


@needs_models
def test_empty_battery_leaves_only_light_models():
    at = click(page(2), "Use Classify")
    at.session_state.battery = 0
    at.run()
    locked = {b.key: b.disabled for b in at.button if b.key.startswith("l2_model_")}
    assert locked == {"l2_model_yolo26n-cls.pt": False, "l2_model_yolo26s-cls.pt": True, "l2_model_yolo26m-cls.pt": True}
    assert any("EMERGENCY RESERVE" in m.value for m in at.markdown)


@needs_models
def test_unaffordable_wrong_mode_is_not_run():
    at = page(2)
    at.session_state.battery = 30
    at.run()
    at = click(at, "Use Detect")
    assert at.session_state.battery == 30
    assert any("not enough charge to run it" in m.value for m in at.markdown)


# chapter 3

def solve_ch3(at: AppTest, threshold: float = 0.2) -> AppTest:
    at = click(at, "Use Detect")
    at = click(at, "Load balanced")
    at = click(at, "Run YOLO26s")
    at.slider(key="l3_threshold").set_value(threshold).run()
    return click(at, "Submit report")


@needs_models
def test_level3_heavy_detector_does_not_fit_and_nano_falls_short():
    at = click(page(3), "Use Detect")
    assert CASE.brief == "inventory" and "Full inventory" in texts(at)
    heavy = next(b for b in at.button if b.key == "l3_model_yolo26m.pt")
    assert heavy.disabled                                   # 44 MB doesn't fit in 32 MB
    at = click(at, "Load light")
    at = click(at, "Run YOLO26n")
    assert at.slider(key="l3_threshold").value == case.BRIEFS["inventory"]["start"]
    at.slider(key="l3_threshold").set_value(0.2).run()
    at = click(at, "Submit report")
    assert 3 not in at.session_state.completed_levels
    assert any("can't meet this brief at any threshold" in m.value for m in at.markdown)
    at = click(at, "Load balanced")
    at = click(at, "Run YOLO26s")
    at.slider(key="l3_threshold").set_value(0.2).run()
    at = click(at, "Submit report")
    assert not at.exception
    assert 3 in at.session_state.completed_levels
    assert at.session_state.grades[3] in "BC"               # it worked, but not efficiently
    assert case.MOVABLE[CASE.moved] in texts(at)


@needs_models
def test_level3_miss_nothing_brief():
    seed = seed_where(brief="miss_nothing")
    at = click(page(3, seed), "Use Detect")
    assert "Miss nothing" in texts(at)
    at = click(at, "Load balanced")
    at = click(at, "Run YOLO26s")
    assert at.slider(key="l3_threshold").value == case.BRIEFS["miss_nothing"]["start"]
    at = click(at, "Submit report")                         # 0.05: everything found, too many false alarms
    assert any("Nothing missed" in m.value for m in at.markdown)
    at.slider(key="l3_threshold").set_value(0.3).run()
    at = click(at, "Submit report")                         # 0.30: one cup missed
    assert any("The brief says miss nothing" in m.value for m in at.markdown)
    at.slider(key="l3_threshold").set_value(0.25).run()
    at = click(at, "Submit report")
    assert 3 in at.session_state.completed_levels


# chapter 4

def solve_ch4(at: AppTest) -> AppTest:
    at = click(at, "Use Segment")
    at.toggle(key="l4_int8").set_value(True).run()
    at = click(at, "Load balanced")
    at = click(at, "Run U-Net Standard INT8")
    return click(at, "Purify")


@needs_models
def test_level4_only_quantized_balanced_model_passes():
    at = page(4)
    assert f"WALL #{CASE.wall_seed}" in texts(at)
    at = click(at, "Use Segment")
    at = click(at, "Load balanced")
    at = click(at, "Run U-Net Standard FP32")
    at = click(at, "Purify")
    assert 4 not in at.session_state.completed_levels       # FP32 is too slow
    at.toggle(key="l4_int8").set_value(True).run()
    at = click(at, "Load balanced")
    at = click(at, "Run U-Net Standard INT8")
    at.slider(key="l4_thr").set_value(0.9).run()
    at = click(at, "Purify")
    assert 4 not in at.session_state.completed_levels       # a good model, but a bad threshold on this wall
    assert any("On this wall the mask" in m.value for m in at.markdown)
    at.slider(key="l4_thr").set_value(0.5).run()
    at = click(at, "Purify")
    assert not at.exception
    assert 4 in at.session_state.completed_levels
    assert at.session_state.grades[4] != "A"                # two failed purifications


@needs_models
@pytest.mark.skipif(not (MODELS_DIR / "stain_unet_lite.pt").exists(), reason="run setup_models.py")
def test_level4_lite_model_leaks():
    at = click(page(4), "Use Segment")
    at = click(at, "Load light")
    at = click(at, "Run U-Net Lite")
    at = click(at, "Purify")
    assert any("masks leak" in m.value for m in at.markdown)
    assert 4 not in at.session_state.completed_levels


# recovery from an empty battery: one lobby trip, the chapter still solves, but not with an A

def from_empty(level: int, seed: int = SEED) -> AppTest:
    at = page(level, seed)
    at.session_state.battery = 0
    at.session_state.battery_low_mark = 0
    at.session_state.xp = 25
    return at.run()


@needs_models
def test_level3_recovery_from_an_empty_battery():
    at = click(from_empty(3, seed_where(brief="miss_nothing")), "Use Detect")
    assert next(b for b in at.button if b.key == "l3_model_yolo26s.pt").disabled
    at.button(key="lobby_l3_picker").click().run()
    assert at.session_state.battery == device.CHARGER_UNITS and at.session_state.xp == 0
    at = click(at, "Load balanced")
    at = click(at, "Run YOLO26s")
    at.slider(key="l3_threshold").set_value(0.2).run()
    at = click(at, "Submit report")
    s = at.session_state
    assert not at.exception
    assert 3 in s.completed_levels and s.grades[3] != "A"
    assert s.l3_report["checks"]["Stayed in the field"][0] is False
    assert device.lobby_trips(s, 3) == 1


@needs_models
def test_level4_recovery_from_an_empty_battery():
    at = click(from_empty(4), "Use Segment")
    at.toggle(key="l4_int8").set_value(True).run()
    assert next(b for b in at.button if b.key == "l4_model_standard-int8").disabled
    at.button(key="lobby_l4_picker").click().run()
    at = click(at, "Load balanced")
    at = click(at, "Run U-Net Standard INT8")
    at = click(at, "Purify")
    s = at.session_state
    assert not at.exception
    assert 4 in s.completed_levels and s.grades[4] != "A"
    assert s.battery == device.CHARGER_UNITS - device.used_in_level(s, 4)


# the whole case

@needs_models
def test_ideal_play_through_then_every_side_scan():
    at = solve_ch1(page(1), CASE.number)
    at = solve_ch2(go(at, 2))
    at = solve_ch3(go(at, 3))
    at = solve_ch4(go(at, 4))
    s = at.session_state
    assert not at.exception
    assert s.grades == {1: "A", 2: "A", 3: "A", 4: "A"}
    d = device.summary(s)
    assert d["spare_cells"] == 4 and d["lobby_trips"] == 0 and d["spent_side"] == 0
    assert s.battery == START - d["spent_main"] + 4 * device.A_GRADE_UNITS
    assert d["spent_main"] <= START // 2                      # about 9% of a full pack
    assert {"frugal", "straight_a", "right_tool", "quantizer"} <= achievements.earned_ids(s)
    assert f"{case.ANCHORS[CASE.anchor]['entity']} is gone." in texts(at)

    for chapter, key in ((1, "cam04"), (2, "scout"), (3, "tray"), (4, "wall2")):
        at = go(at, chapter)
        at.button(key=f"side_{key}").click().run()
        assert not at.exception
        assert at.session_state.side_scans[key] is True, key
    s = at.session_state
    assert len(s.side_clues) == 4
    assert "thorough" in achievements.earned_ids(s)
    assert "Four side clues, one timeline" in texts(at)
    assert at.get("vega_lite_chart")                         # the battery timeline on the case summary
    assert s.intel["l3_light"].startswith("Doorway scout")
    assert s.battery == START - d["spent_main"] - device.summary(s)["spent_side"] + 4 * device.A_GRADE_UNITS
    assert device.summary(s)["spent_main"] == d["spent_main"]     # side scans go on their own line
    assert s.grades == {1: "A", 2: "A", 3: "A", 4: "A"}       # side scans never touch a grade


def open_page(at: AppTest, url_path: str) -> AppTest:
    # AppTest can't follow st.switch_page to a function page from one run to the next, so point it there directly
    at._page_hash = next(h for h, p in at._registered_pages.items() if p["url_pathname"] == url_path)
    return at.run()


@needs_models
def test_full_app_play_through_and_restart():
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.session_state.case_seed = SEED
    at.run()
    assert "CASE 0003" in texts(at) and "Battery rules" in texts(at)
    at = solve_ch1(open_page(at, "level1"), CASE.number)
    at = solve_ch2(open_page(at, "level2"))
    at = solve_ch3(open_page(at, "level3"))
    at = solve_ch4(open_page(at, "level4"))
    s = at.session_state
    assert not at.exception
    assert s.grades == {1: "A", 2: "A", 3: "A", 4: "A"}
    assert "SOLVED 4/4" in texts(at)
    for path in ("lab", "guide", "about", ""):
        at = open_page(at, path)
        assert not at.exception, path
    assert f"{case.ANCHORS[CASE.anchor]['entity']} is gone." in texts(at)     # home shows the case summary
    s.guide_xp = 15
    at = click(at, "Restart the case")
    s = at.session_state
    assert not at.exception
    assert s.completed_levels == set() and s.battery == START and s.grades == {}
    assert s.case_seed != SEED and s.guide_xp == 15
    assert not [k for k in s if k.startswith("l4_")]


def test_about_page_explains_the_battery():
    at = AppTest.from_string("from game import levels\nlevels.about()").run()
    assert not at.exception
    assert "0.1% of a full pack" in texts(at)
    assert "Battery rules" in texts(at)
