"""Chapter 3 played headlessly: detection, the threshold, the box inspector and the cleared screen."""

import numpy as np

from cv import detection as det
from game import case, device, level3
from tests.play import (CASE, START, assert_play_clean, click, click_key, is_cleared, needs_models, page,
                        seed_where, texts)
from tests.play_ch34 import solve_ch3


def to_play(at, tier: str = "balanced", run: str = "Run YOLO26s"):
    at = click_key(at, "l3_mode_Detect")
    at = click(at, f"Load {tier}")
    return click(at, run)


@needs_models
def test_heavy_detector_does_not_fit_and_nano_falls_short():
    at = click_key(page(3), "l3_mode_Detect")
    assert CASE.brief == "inventory" and "Full inventory" in texts(at)
    assert at.button(key="l3_model_yolo26m.pt").disabled             # 44 MB doesn't fit in 32 MB
    at = click(at, "Load light")
    at = click(at, "Run YOLO26n")
    assert at.slider(key="l3_threshold").value == case.BRIEFS["inventory"]["start"]
    at.slider(key="l3_threshold").set_value(0.2).run()
    at = click_key(at, "l3_submit")
    assert 3 not in at.session_state.completed_levels
    assert "can't meet this brief at any threshold" in texts(at)
    at = click(at, "Load balanced")
    at = click(at, "Run YOLO26s")
    at.slider(key="l3_threshold").set_value(0.2).run()
    at = click_key(at, "l3_submit")
    assert not at.exception
    assert 3 in at.session_state.completed_levels
    assert at.session_state.grades[3] in "BC"                       # it worked, but not efficiently
    assert is_cleared(at, 3) and case.MOVABLE[CASE.moved] in texts(at)


@needs_models
def test_miss_nothing_lines():
    at = to_play(page(3, seed_where(brief="miss_nothing")))
    assert "Miss nothing" in texts(at)
    assert at.slider(key="l3_threshold").value == case.BRIEFS["miss_nothing"]["start"]
    at = click_key(at, "l3_submit")                                 # 0.05: everything found, too many false alarms
    assert "Nothing missed" in texts(at)
    at.slider(key="l3_threshold").set_value(0.3).run()
    at = click_key(at, "l3_submit")                                 # 0.30: one cup missed
    assert "The brief says miss nothing" in texts(at)
    at.slider(key="l3_threshold").set_value(0.25).run()
    at = click_key(at, "l3_submit")
    assert 3 in at.session_state.completed_levels and len(at.session_state.l3_reports) == 3


@needs_models
def test_recovery_from_an_empty_battery():
    at = page(3, seed_where(brief="miss_nothing"))
    at.session_state.battery = 0
    at.session_state.battery_low_mark = 0
    at.session_state.xp = 25
    at = click_key(at.run(), "l3_mode_Detect")
    assert at.button(key="l3_model_yolo26s.pt").disabled
    at.button(key="lobby_l3_picker").click().run()
    assert at.session_state.battery == device.CHARGER_UNITS and at.session_state.xp == 0
    at = click(at, "Load balanced")
    at = click(at, "Run YOLO26s")
    at.slider(key="l3_threshold").set_value(0.2).run()
    at = click_key(at, "l3_submit")
    s = at.session_state
    assert not at.exception
    assert 3 in s.completed_levels and s.grades[3] != "A"
    assert s.l3_report["checks"]["Stayed in the field"][0] is False
    assert device.lobby_trips(s, 3) == 1


@needs_models
def test_inspector_shows_input_size_then_the_reason_after_a_report():
    at = to_play(page(3))
    at.slider(key="l3_threshold").set_value(0.2).run()
    radio = at.radio(key="l3_inspect")
    assert radio.value is None
    dets = level3.scene_detections("yolo26s.pt")
    cup = next(i for i, d in enumerate(dets) if d["conf"] >= 0.2 and d["label"] == "cup")
    radio.set_value(cup).run()
    text = texts(at)
    box = dets[cup]["box"]
    w, h = round(box[2] - box[0]), round(box[3] - box[1])
    iw, ih = det.input_size(box, level3.load_scene().shape)
    assert f"{w}×{h} px here → {iw}×{ih} px at the model's 640-px input" in text
    assert "correct · IoU" not in text and "false alarm" not in text      # no verdict before a report
    assert cup in at.session_state.l3_inspected

    at.slider(key="l3_threshold").set_value(0.05).run()
    at = click_key(at, "l3_submit")                                 # 0.05: lots of false alarms, nothing solved
    m = det.evaluate(dets, level3.TRUTH["objects"], 0.05)[0]
    alarm = det.explain_false_alarms(m, level3.TRUTH["objects"])[0]
    i = next(k for k, d in enumerate(dets) if d["box"] == alarm["det"]["box"] and d["conf"] == alarm["det"]["conf"])
    at.radio(key="l3_inspect").set_value(i).run()
    assert f"false alarm · {det.REASONS[alarm['reason']]}" in texts(at)
    tp = m["tp"][0]
    j = next(k for k, d in enumerate(dets) if d["box"] == tp["box"] and d["conf"] == tp["conf"])
    at.radio(key="l3_inspect").set_value(j).run()
    assert f"correct · IoU {tp['iou']:.2f}" in texts(at)
    assert not at.exception


@needs_models
def test_threshold_moves_never_charge():
    at = to_play(page(3))
    battery = at.session_state.battery
    for t in (0.1, 0.5, 0.9, 0.3):
        at.slider(key="l3_threshold").set_value(t).run()
    assert at.session_state.battery == battery and not at.exception


@needs_models
def test_play_screen_is_clean():
    at = click_key(page(3), "l3_mode_Classify")
    assert_play_clean(at)
    at = to_play(at)
    assert_play_clean(at)
    at.slider(key="l3_threshold").set_value(0.05).run()
    at = click_key(at, "l3_submit")
    assert_play_clean(at)
    assert 3 not in at.session_state.completed_levels


@needs_models
def test_cleared_shows_the_pr_chart_and_label_lines():
    at = solve_ch3(page(3))
    s = at.session_state
    assert not at.exception and s.grades[3] == "A" and is_cleared(at, 3)
    assert s.battery == START - level3.par() + device.A_GRADE_UNITS
    text = texts(at)
    assert "Report accepted: F1" in text and "at threshold 0.20" in text
    assert at.get("vega_lite_chart")
    assert len(at.code[0].value.splitlines()) == len(level3.TRUTH["objects"])
    assert at.select_slider(key="l3_match_iou")
    at.button(key="l3_back").click().run()
    assert at.session_state.l3_review and not is_cleared(at, 3) and not at.exception
    assert at.button(key="l3_report_btn")


# the maths behind the chapter

def random_case(rng):
    labels = ["cup", "chair"]
    truth, dets = [], []
    for _ in range(rng.integers(1, 6)):
        x, y = rng.uniform(0, 200, 2)
        truth.append({"label": str(rng.choice(labels)), "box": [x, y, x + rng.uniform(10, 60), y + rng.uniform(10, 60)]})
    for _ in range(rng.integers(0, 10)):
        t = truth[rng.integers(len(truth))]["box"]
        j = rng.normal(0, 8, 4)
        dets.append({"label": str(rng.choice(labels)), "conf": float(rng.uniform(0.05, 1)),
                     "box": [t[0] + j[0], t[1] + j[1], t[2] + abs(j[2]), t[3] + abs(j[3])]})
    return dets, truth


def test_lowering_the_threshold_never_lowers_recall():
    rng = np.random.default_rng(0)
    for _ in range(200):
        dets, truth = random_case(rng)
        recalls = [det.evaluate(dets, truth, t)[1]["recall"] for t in level3.THRESHOLDS]
        assert all(a >= b for a, b in zip(recalls, recalls[1:]))     # high threshold, low recall


def test_stricter_matching_iou_never_adds_matches():
    rng = np.random.default_rng(1)
    for _ in range(200):
        dets, truth = random_case(rng)
        tps = [r["tp"] for r in level3.match_sweep(dets, truth, 0.2, level3.MATCH_IOUS)]
        assert all(a >= b for a, b in zip(tps, tps[1:]))


@needs_models
def test_review_is_read_only():
    at = solve_ch3(page(3))
    at = click_key(at, "l3_back")
    loads = [b for b in at.button if b.key and b.key.startswith("l3_model_")]
    assert loads and all(b.disabled for b in loads)
    assert not [b for b in at.button if b.key and b.key.startswith("lobby_")]
