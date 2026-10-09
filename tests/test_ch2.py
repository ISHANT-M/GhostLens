"""Chapter 2, Identify the Entity: tag the anchor, then sweep the guest's room with a sliding window."""

import pytest

from cv import classification as clf
from cv.edge import energy_units
from game import case, level2, runtime
from tests.play import CASE, START, assert_play_clean, click_key, is_cleared, needs_models, page, texts
from tests.play_ch12 import aim, scan_window, solve_ch2, start_ch2, tag_anchor

W, H = level2.ROOM_SIZE
ANCHOR_LABELS = set().union(*(a["labels"] for a in case.ANCHORS.values()))


def feedback_lines(at) -> list[str]:
    return [m.value for m in at.markdown if "gl-feedback" in m.value]


def light_label(win: str) -> str:
    return level2.scan_window(win, level2.LIGHT)["top"][0][0]


# the window grid (no models needed)

def test_room_windows_cover_the_photo():
    wins = clf.room_windows(W, H)
    assert len(wins) == 10 and wins["full"] == (0, 0, W, H)
    grid = [b for k, b in wins.items() if k != "full"]
    assert {(x1 - x0, y1 - y0) for x0, y0, x1, y1 in grid} == {(560, 374)}
    assert all(0 <= x0 and 0 <= y0 and x1 <= W and y1 <= H for x0, y0, x1, y1 in grid)
    assert clf.coverage(grid, W, H) == 1.0
    assert wins["half-1-1"] == (360, 240, 920, 614)
    assert level2.window_id("Close-up (560×374)", "right", "top") == "half-2-0"
    assert level2.window_id("Whole photo", "right", "top") == "full"
    assert level2.window_name("half-1-0") == "Top-middle window"
    assert level2.window_name("half-0-2") == "Bottom-left window"


def test_coverage_counts_overlaps_once():
    assert clf.coverage([], 10, 10) == 0.0
    assert clf.coverage([(0, 0, 5, 10)], 10, 10) == 0.5
    assert clf.coverage([(0, 0, 5, 10), (0, 0, 5, 10)], 10, 10) == 0.5
    assert clf.coverage([(0, 0, 6, 10), (4, 0, 10, 10)], 10, 10) == 1.0
    assert clf.coverage([(-5, -5, 20, 20)], 10, 10) == 1.0     # clipped to the photo
    wins = clf.room_windows(W, H)
    one = clf.coverage([wins["half-0-0"]], W, H)
    assert one == pytest.approx(560 * 374 / (W * H))
    assert clf.coverage([wins["half-0-0"], wins["half-1-0"]], W, H) == pytest.approx((920 * 374) / (W * H))


def test_room_names_keep_order_and_drop_repeats():
    scans = [{"label": "a"}, {"label": "b"}, {"label": "a"}, {"label": "c"}]
    assert level2.room_names(scans) == ["a", "b", "c"]


# data: the sweep has to be winnable on every classifier and never name an anchor

@needs_models
@pytest.mark.parametrize("weights", ["yolo26n-cls.pt", "yolo26s-cls.pt", "yolo26m-cls.pt"])
def test_ideal_windows_name_three_things_and_never_an_anchor(weights):
    labels = {win: level2.scan_window(win, weights)["top"][0][0] for win in level2.WINDOWS}
    assert len({labels[w] for w in level2.IDEAL_WINDOWS}) >= 3, labels
    assert not set(labels.values()) & ANCHOR_LABELS, labels


@needs_models
def test_par_is_six_light_scans():
    assert level2.PAR_SCANS == 6
    assert level2.par() == 6 * energy_units(runtime.profile("classifiers", level2.LIGHT)["latency_ms"])


# played headlessly

@needs_models
def test_efficient_run_gets_an_a():
    at = page(2)
    assert f"CHAPTER 2 · {CASE.clue['label'].upper()}" in texts(at)
    at = solve_ch2(at)
    assert not at.exception
    s = at.session_state
    assert 2 in s.completed_levels and s.grades[2] == "A"
    assert len(s.l2_room_scans) == 3 and s.l2_room_scans[0]["win"] == level2.IDEAL_WINDOWS[0]
    assert s.l2_anchor_label in case.ANCHORS[CASE.anchor]["labels"]
    assert is_cleared(at, 2)
    assert f"Entity: {case.ANCHORS[CASE.anchor]['entity']}, anchored to the {CASE.anchor}." in texts(at)


@needs_models
def test_wrong_tag_gives_the_miss_line():
    at = start_ch2(page(2))
    wrong = next(n for n in CASE.evidence_order if n != CASE.anchor)
    assert not any(b.key == f"l2_tag_{wrong}" for b in at.button)       # scan before you can tag
    at = click_key(at, f"scan_{wrong}")
    at = click_key(at, f"l2_tag_{wrong}")
    label, p = level2.scan(level2.OBJECTS[wrong], level2.LIGHT)["top"][0]
    line = feedback_lines(at)
    assert len(line) == 1 and f"GhostLens says {label} ({p:.0%})" in line[0]
    assert case.ANCHORS[CASE.anchor]["miss"] in line[0]
    assert not at.session_state.get("l2_anchor_found")
    assert not any(r.key == "l2_anchor_choice" for r in at.radio)


@needs_models
def test_segmenting_a_single_object_wastes_battery():
    at = click_key(page(2), "l2_mode_Segment")
    assert at.session_state.battery < START - 50
    assert any("most expensive way" in m for m in feedback_lines(at))
    assert_play_clean(at)


@needs_models
def test_empty_battery_leaves_only_light_models():
    at = click_key(page(2), "l2_mode_Classify")
    at.session_state.battery = 0
    at.run()
    locked = {b.key: b.disabled for b in at.button if b.key.startswith("l2_model_")}
    assert locked == {"l2_model_yolo26n-cls.pt": False, "l2_model_yolo26s-cls.pt": True,
                      "l2_model_yolo26m-cls.pt": True}


@needs_models
def test_unaffordable_wrong_mode_is_not_run():
    at = page(2)
    at.session_state.battery = 30
    at.run()
    at = click_key(at, "l2_mode_Detect")
    assert at.session_state.battery == 30
    assert any("not enough charge to run it" in m for m in feedback_lines(at))


@needs_models
def test_each_window_costs_one_scan_and_a_rescan_is_disabled():
    at = tag_anchor(start_ch2(page(2)))
    unit = energy_units(runtime.profile("classifiers", level2.LIGHT)["latency_ms"])
    before = at.session_state.battery
    at = scan_window(at, "half-1-0")
    assert at.session_state.battery == before - unit
    button = at.button(key="l2_scan_window")
    assert button.disabled and button.label == "Already scanned"
    at = aim(at, "half-1-1")                                   # moving the reticle is free
    assert at.session_state.battery == before - unit
    assert not at.button(key="l2_scan_window").disabled
    at = aim(at, "full")
    assert at.select_slider(key="l2_pan").disabled
    line = feedback_lines(scan_window(at, "half-1-1"))
    assert len(line) == 1 and "One label per window, no position inside it." in line[0]


@needs_models
def test_wins_at_three_distinct_labels():
    first, second, third = level2.IDEAL_WINDOWS
    repeat = next((w for w in level2.WINDOWS if w not in level2.IDEAL_WINDOWS
                   and light_label(w) in {light_label(first), light_label(second)}), None)
    at = tag_anchor(start_ch2(page(2)))
    for win in (first, second) + ((repeat,) if repeat else ()):
        at = scan_window(at, win)
        assert 2 not in at.session_state.completed_levels
    assert "2/3" in texts(at)
    at = scan_window(at, third)
    s = at.session_state
    assert 2 in s.completed_levels
    expected = 3 / len(s.l2_room_scans)
    assert s.best_scores[2] == pytest.approx(min(1.0, expected))


@needs_models
def test_play_screens_are_clean():
    at = page(2)
    assert_play_clean(at)
    at = start_ch2(at)
    assert_play_clean(at)
    at = tag_anchor(at)
    assert "Name three different things in the guest's room." in texts(at)
    assert_play_clean(at)
    at = scan_window(at, "full")
    assert_play_clean(at)
    assert "ROOM COVERED" in texts(at).upper()


@needs_models
def test_cleared_shows_both_tables():
    at = solve_ch2(page(2))
    assert is_cleared(at, 2)
    assert len(at.dataframe) == 2
    sweep = at.dataframe[1].value
    assert len(sweep) == 10 and set(sweep.columns) == {"window", "top-1", "p", "ms"}
    text = texts(at)
    assert "A full sweep = 10 passes" in text and "still no boxes" in text
    assert "sliding window" in text and "detection" not in text.lower()
    assert any(b.key == "side_scout" for b in at.button)


@needs_models
def test_back_to_the_room_after_clearing():
    at = solve_ch2(page(2))
    at = click_key(at, "l2_back")
    assert any(b.key == "l2_report_btn" for b in at.button)
    assert any(b.key == "l2_scan_window" for b in at.button)
    assert not at.exception
