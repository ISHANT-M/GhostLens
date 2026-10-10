"""The app shell, the cleared screen, home and the case summary, played headlessly on a pinned case."""

from streamlit.testing.v1 import AppTest

from cv import edge
from game import achievements, case, device
from tests.play import CASE, SEED, START, click, click_key, go, is_cleared, needs_models, open_page, texts

SCRIPT = """
import streamlit as st
from game import device, levels, state, level1, level2, level3, level4
state.init_state(st.session_state)
device.init_device(st.session_state)
st.session_state.setdefault("case_seed", {seed})
st.session_state.setdefault("chapter", 1)
pages = {{0: levels.home, 1: level1.render, 2: level2.render, 3: level3.render, 4: level4.render}}
pages[st.session_state.chapter]()
"""


def app() -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.session_state.case_seed = SEED
    return at.run()


def test_app_renders_home_about_and_guide():
    at = app()
    assert not at.exception
    assert "CASE 0003" in texts(at) and "GHOSTLENS" in texts(at)
    assert "Battery rules" not in texts(at)                  # the full rules live on About
    assert "SOLVED 0/4" in texts(at)
    assert not at.sidebar.children
    for path in ("about", "guide"):
        at = open_page(at, path)
        assert not at.exception, path
    assert "Battery rules" not in texts(open_page(at, "guide"))


def test_about_page_explains_the_battery():
    at = AppTest.from_string("from game import levels\nlevels.about()").run()
    assert not at.exception
    assert "0.1% of a full pack" in texts(at)
    assert "Battery rules" in texts(at)


def test_hud_and_home_do_not_name_the_task_before_it_is_chosen():
    script = """
import streamlit as st
from game import device, flow, levels, state
state.init_state(st.session_state)
device.init_device(st.session_state)
st.session_state.setdefault("case_seed", 3)
st.session_state.completed_levels = {1, 2}
st.session_state.l2_mode = "Classify"
st.markdown(flow.hud_html(3), unsafe_allow_html=True)
levels.home()
"""
    at = AppTest.from_string(script).run()
    assert not at.exception
    hud = at.markdown[0].value
    assert "CH 3<" in hud and "CH 2 · CLASSIFY" in hud and "SOLVED 2/4" in hud
    assert not any(m in hud for m in ("ENHANCE", "DETECT", "SEGMENT"))
    home = texts(at)
    assert "Classify · " in home and "Detect · " not in home and "Segment · " not in home
    at.session_state.l3_mode = "Detect"
    at.run()
    assert "CH 3 · DETECT" in at.markdown[0].value


CLEARED = """
import streamlit as st
from game import device, levels, state
state.init_state(st.session_state)
device.init_device(st.session_state)
s = st.session_state
s.setdefault("case_seed", 3)
if "setup" not in s:
    s.setup = True
    s.completed_levels = {1}
    s.l1_report = {"checks": {"Clipping under 1%": (True, "0.0%")}, "grade": "A"}
    s.l1_xp = {"base": 50, "quality_bonus": 20, "edge_bonus": 30, "attempt_penalty": 0, "total": 100, "charge": 10}
    s.just_cleared = 1
c = levels.Cleared(headline="Plate recovered in 2.1 ms.", happened=["You tried Detect first: no number."],
                   why=["Gamma lifted the shadows."], concept="Image enhancement",
                   numbers=lambda: st.markdown("NUMBERS TABLE"), stats=[("Clue", "209")], par=5)
if levels.show_cleared(1):
    levels.level_cleared(1, c)
else:
    levels.review_banner(1)
    st.markdown("THE SCENE")
"""


def test_level_cleared_screen_and_back_to_the_scene():
    at = AppTest.from_string(CLEARED).run()
    assert not at.exception
    text = texts(at)
    assert "CHAPTER 1 · CLEARED" in text and "Plate recovered in 2.1 ms." in text
    assert 'gl-grade gA animate' in text and "SPARE CELL" in text
    assert "The plate reads" in text and "NEXT QUESTION" in text
    assert "measured" in text and "published" in text and "game rule" in text
    assert "You tried Detect first" in text and "NUMBERS TABLE" in text and "Image enhancement" in text
    assert [t.label for t in at.tabs] == ["Debrief", "Numbers", "Bonus scan"]
    assert at.button(key="l1_continue").label == "Continue · Chapter 2: Identify the Entity"
    assert is_cleared(at, 1)

    at.run()
    assert "animate" not in texts(at)                       # the stamp only lands once
    at = click_key(at, "l1_back")
    assert at.session_state.l1_review is True
    assert "THE SCENE" in texts(at) and not is_cleared(at, 1)
    at = click_key(at, "l1_report_btn")
    assert is_cleared(at, 1)


def test_finish_level_shows_the_cleared_screen():
    script = """
import streamlit as st
from game import device, levels, state
state.init_state(st.session_state)
device.init_device(st.session_state)
st.session_state.setdefault("case_seed", 3)
st.session_state.setdefault("l4_review", True)
if levels.show_cleared(4):
    levels.level_cleared(4, levels.Cleared("Corruption purified.", [], [], "Segmentation"))
elif st.button("Purify"):
    levels.finish_level(4, 0.97, 1, {"IoU": (True, "0.968")})
"""
    at = click(AppTest.from_string(script).run(), "Purify")
    s = at.session_state
    assert not at.exception
    assert s.l4_review is False and 4 in s.completed_levels
    assert is_cleared(at, 4) and at.button(key="l4_continue").label == "Close the case"


@needs_models
def test_case_closed_renders_from_a_solved_state():
    script = """
import streamlit as st
from game import device, levels, state
state.init_state(st.session_state)
device.init_device(st.session_state)
s = st.session_state
s.setdefault("case_seed", 3)
s.completed_levels = {1, 2, 3, 4}
s.grades = {1: "A", 2: "A", 3: "B", 4: "A"}
s.best_scores = {1: 0.85, 3: 7 / 9}
s.l1_final = {"settings": {}, "quality": 0.85, "clip": 0.0, "ms": 2.0}
s.l2_model, s.l2_anchor_label, s.l3_model = "yolo26n-cls.pt", "stopwatch", "yolo26s.pt"
s.l4_deployed = {"id": "standard-int8", "name": "U-Net Standard INT8", "variant": "standard", "int8": True,
                 "iou": 0.968}
s.lab_seen = {"Convolution"}
levels.case_closed()
"""
    at = AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception
    text = texts(at)
    assert "is gone." in text and "CASE CLOSED" in text
    assert "Forecasts" not in text and "Say it out loud" not in text
    ms = next(r["latency_ms"] for r in edge.load_benchmark()["unets"] if r["id"] == "standard-int8")
    assert "U-Net Standard INT8" in text and f"IoU 0.968 at {ms:.1f} ms" in text
    assert at.get("download_button")
    labels = [b.label for b in at.button]
    assert "Lab · Pruning" in labels and "Lab · Convolution" not in labels


@needs_models
def test_ideal_play_through_then_every_side_scan():
    from tests.play_ch12 import solve_ch1, solve_ch2
    from tests.play_ch34 import solve_ch3, solve_ch4
    at = AppTest.from_string(SCRIPT.format(seed=SEED), default_timeout=300).run()
    at = solve_ch1(at, CASE.number)
    at = solve_ch2(go(at, 2))
    at = solve_ch3(go(at, 3))
    at = solve_ch4(go(at, 4))
    s = at.session_state
    assert not at.exception
    assert s.grades == {1: "A", 2: "A", 3: "A", 4: "A"}
    assert is_cleared(at, 4)
    d = device.summary(s)
    assert d["spare_cells"] == 4 and d["lobby_trips"] == 0 and d["spent_side"] == 0
    assert s.battery == START - d["spent_main"] + 4 * device.A_GRADE_UNITS
    assert d["spent_main"] <= START - 50                      # about 12% of a full pack
    assert {"frugal", "straight_a", "right_tool", "quantizer"} <= achievements.earned_ids(s)

    for chapter, key in ((1, "cam04"), (2, "scout"), (3, "tray"), (4, "wall2")):
        at = go(at, chapter)
        assert is_cleared(at, chapter)
        at = click_key(at, f"side_{key}")
        assert not at.exception
        assert at.session_state.side_scans[key] is True, key
    s = at.session_state
    assert len(s.side_clues) == 4
    assert "thorough" in achievements.earned_ids(s)
    assert s.intel["l3_light"].startswith("Doorway scout")
    assert s.battery == START - d["spent_main"] - device.summary(s)["spent_side"] + 4 * device.A_GRADE_UNITS
    assert device.summary(s)["spent_main"] == d["spent_main"]     # side scans go on their own line
    assert s.grades == {1: "A", 2: "A", 3: "A", 4: "A"}       # side scans never touch a grade

    at = go(at, 0)
    text = texts(at)
    assert not at.exception
    assert f"{case.ANCHORS[CASE.anchor]['entity']} is gone." in text
    assert "Four side clues, one timeline" in text
    assert at.get("vega_lite_chart")                         # the battery timeline
    assert "Forecasts" not in text


@needs_models
def test_full_app_play_through_and_restart():
    from tests.play_ch12 import solve_ch1, solve_ch2
    from tests.play_ch34 import solve_ch3, solve_ch4
    at = app()
    at = solve_ch1(open_page(at, "level1"), CASE.number)
    at = solve_ch2(open_page(at, "level2"))
    at = solve_ch3(open_page(at, "level3"))
    at = solve_ch4(open_page(at, "level4"))
    s = at.session_state
    assert not at.exception
    assert s.grades == {1: "A", 2: "A", 3: "A", 4: "A"}
    assert "SOLVED 4/4" in texts(at)
    assert "guide_xp" not in s
    for path in ("lab", "guide", "about", ""):
        at = open_page(at, path)
        assert not at.exception, path
        assert ("Battery rules" in texts(at)) == (path == "about"), path
    assert f"{case.ANCHORS[CASE.anchor]['entity']} is gone." in texts(at)     # home shows the case summary
    s.seen_tips = {1, 2, 3}
    at = click(at, "Restart the case")
    s = at.session_state
    assert not at.exception
    assert s.completed_levels == set() and s.battery == START and s.grades == {}
    assert s.case_seed != SEED and s.seen_tips >= {1, 2, 3}
    assert not [k for k in s if k.startswith("l4_")]


def test_project_page_reads_as_a_course_project():
    at = AppTest.from_string("from game import levels\nlevels.project()").run()
    assert not at.exception
    text = texts(at)
    for part in ("Edge AI in this project", "On-device, offline", "Memory budget", "Latency budget",
                 "1 ms = 1 unit = 0.1%", "INT8 quantization", "Model selection", "Battery rules",
                 "UCS668 Edge AI and Robotics", levels_team(), "github.com/ISHANT-M/GhostLens", "MIT licence"):
        assert part in text, part
    for n in (1, 2, 3, 4):
        assert f"CH {n}<b>" in text
    assert "precision, recall, F1 at IoU 0.5" in text and "TinyUNet" in text
    bench = edge.load_benchmark()
    if bench is None:
        assert "setup_models.py" in text
    else:
        table = next(m.value for m in at.markdown if "gl-bench" in m.value)
        assert table.count("<tr>") == 1 + sum(len(bench[g]) for g in ("classifiers", "detectors", "segmenters", "unets"))
        assert "Machine:" in table and "YOLO26n-cls" in table and "U-Net Standard INT8" in table


def levels_team() -> str:
    from game import levels
    return levels.TEAM


def test_project_page_is_in_the_menu_and_on_home():
    at = app()
    assert any(getattr(p, "label", "") == "Project" for p in at.get("page_link"))
    assert any(p.label.startswith("Project · UCS668") for p in at.get("page_link"))
    at = open_page(at, "about")
    assert not at.exception
    assert "Edge AI in this project" in texts(at)


def test_menu_hides_the_lobby_charger_after_the_case_is_closed():
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.session_state.case_seed = SEED
    at.session_state.completed_levels = {1, 2, 3}
    at.session_state.ledger = []
    at.session_state.battery = 30
    at.run()
    assert not at.exception
    assert at.button(key="lobby_menu")
    at.session_state.completed_levels = {1, 2, 3, 4}
    at.run()
    assert not [b for b in at.button if b.key == "lobby_menu"]


def test_sound_stays_off_once_switched_off():
    at = app()
    assert at.session_state.sound_on and at.get("iframe")
    at.toggle(key="sound").set_value(False).run()
    assert at.session_state.sound_on is False
    at.run()
    assert at.session_state.sound_on is False and not at.get("iframe")
    assert at.toggle(key="sound").value is False
