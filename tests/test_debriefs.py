"""The small resolvers behind predictions, debriefs and the case recap."""

import json

import numpy as np
import pytest

from cv import detection as det
from cv.edge import BENCHMARK_FILE
from game import flow, level1, level2, level3, level4, levels

needs_models = pytest.mark.skipif(not BENCHMARK_FILE.exists(), reason="run setup_models.py")


def bench() -> dict:
    return json.loads(BENCHMARK_FILE.read_text())


# chapter 1

def test_best_tools_is_the_argmax_and_keeps_ties():
    assert level1.best_tools({"a": 0.3, "b": 0.6, "c": 0.1}) == {1}
    assert level1.best_tools({"a": 0.5, "b": 0.2, "c": 0.5}) == {0, 2}


@pytest.mark.parametrize("clue", level1.case.CLUES, ids=lambda c: c["id"])
def test_single_tool_answer_comes_from_the_measurement(clue):
    q = level1.single_tool_quality(clue["id"])
    best, why = level1.resolve_tool(clue)
    assert best == level1.best_tools(q)
    assert all(f"{v:.2f}" in why for v in q.values())


def test_debrief_frame_puts_your_pass_first():
    runs = [{"setup": "ref", "quality": 0.85, "clip": 0.0, "ms": 2.0}]
    df = level1.debrief_frame({"quality": 0.9, "clip": 0.002, "ms": 3.0}, runs)
    assert list(df.setup) == ["Yours", "ref"] and df.battery[1] == "0.2%"
    assert list(level1.debrief_frame(None, runs).setup) == ["ref"]


def test_clip_tone_uses_one_limit():
    assert level1.clip_tone(0.009) == ""
    assert level1.clip_tone(0.015) == "warn"
    assert level1.clip_tone(0.031) == "bad"


# chapter 2

def test_bigger_model_resolver():
    same = {"n": {"padlock": "padlock", "room": "window shade"}, "s": {"padlock": "padlock", "room": "window shade"}}
    assert level2.changed_items(same, "n") == set()
    assert level2.bigger_model_answer(set()) == 0
    room = {**same, "s": {"padlock": "padlock", "room": "studio couch"}}
    assert level2.changed_items(room, "n") == {"room"}
    assert level2.bigger_model_answer({"room"}) == 2
    obj = {**same, "m": {"padlock": "combination lock", "room": "studio couch"}}
    assert level2.changed_items(obj, "n") == {"padlock", "room"}
    assert level2.bigger_model_answer({"padlock", "room"}) == 1


# chapter 3

def test_which_drops_and_answer():
    before = {"precision": 0.67, "recall": 0.29, "f1": 0.4}
    after = {"precision": 0.64, "recall": 1.0, "f1": 0.78}
    assert level3.which_drops(before, after) == {"precision"}
    assert level3.drops_answer({"precision"}) == 0
    assert level3.drops_answer({"recall"}) == 1
    assert level3.drops_answer({"precision", "recall", "f1"}) == 2
    assert level3.drops_answer(set()) == 3


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


TRUTH = [{"label": "cup", "box": [0, 0, 10, 10]}, {"label": "chair", "box": [50, 50, 90, 90]},
         {"label": "cup", "box": [200, 200, 220, 220]}]


def test_false_alarm_reasons():
    dets = [
        {"label": "cup", "conf": 0.9, "box": [0, 0, 10, 10]},          # correct
        {"label": "cup", "conf": 0.8, "box": [0, 0, 10, 9]},           # second box on the same cup
        {"label": "cup", "conf": 0.7, "box": [200, 200, 240, 240]},    # IoU 0.25 with the other cup
        {"label": "cup", "conf": 0.6, "box": [52, 52, 88, 88]},        # on the chair
        {"label": "cup", "conf": 0.5, "box": [400, 400, 420, 420]},    # nothing there
    ]
    m = det.match(dets, TRUTH)
    assert m["tp"][0]["truth"] == 0
    reasons = {a["det"]["conf"]: (a["reason"], a["truth"]) for a in det.explain_false_alarms(m, TRUTH)}
    assert reasons == {0.8: ("duplicate", 0), 0.7: ("loose", 2), 0.6: ("other_class", 1), 0.5: ("unlabelled", None)}
    assert set(det.REASONS) == {"duplicate", "loose", "other_class", "unlabelled"}


def test_yolo_lines_are_normalised_centres():
    lines = det.yolo_lines([{"label": "cup", "box": [100, 50, 300, 150]}], 400, 200, {"cup": 41})
    assert lines == ["41 0.5000 0.5000 0.5000 0.5000"]


# chapter 4

def test_int8_bins():
    assert level4.int8_bin(-0.06) == 0
    assert level4.int8_bin(-0.05) == 1 and level4.int8_bin(-0.01) == 1
    assert level4.int8_bin(-0.001) == 2 and level4.int8_bin(0.01) == 2
    assert level4.int8_bin(0.02) == 3
    assert len(level4.INT8_OPTIONS) == 4


@needs_models
def test_int8_answer_comes_from_the_benchmark():
    smaller, faster, delta = levels.int8_ratios(bench())
    assert level4.INT8_OPTIONS[level4.int8_bin(delta)] == "Changes by less than 0.01"
    assert smaller > 3 and faster > 2


def test_frontier_rows_mark_the_models_inside_the_limits():
    fake = {"unets": [
        {"name": "U-Net A", "variant": "a", "precision": "FP32", "latency_ms": 64.0, "iou": 0.96, "size_mb": 0.5},
        {"name": "U-Net A", "variant": "a", "precision": "INT8", "latency_ms": 22.0, "iou": 0.959, "size_mb": 0.13},
        {"name": "U-Net B", "variant": "b", "precision": "INT8", "latency_ms": 5.0, "iou": 0.92, "size_mb": 0.02},
    ]}
    rows = level4.frontier_rows(fake)
    assert [r["meets"] for r in rows] == [False, True, False]
    assert rows[1]["name"] == "U-Net A INT8"


@needs_models
def test_only_standard_int8_is_inside_the_box():
    assert [r["name"] for r in level4.frontier_rows(bench()) if r["meets"]] == ["U-Net Standard INT8"]


def test_pixel_accuracy_flatters_an_empty_mask():
    truth = np.zeros((10, 10), bool)
    truth[:1] = True
    assert level4.pixel_accuracy(np.zeros_like(truth), truth) == 0.9


# hud and the recap

def test_chip_names_the_mode_only_once_chosen():
    assert flow.chip_label(3, {}) == "CH 3"
    assert flow.chip_label(3, {"l3_mode": "Detect"}) == "CH 3 · DETECT"


def test_loading_screen_and_warmup_never_name_a_task():
    words = ("enhanc", "classif", "detect", "segment", "u-net")
    for n, info in levels.LEVELS.items():
        text = (flow.loading_html(n, info["title"]) + " ".join(label for label, _ in levels.warmup(n))).lower()
        assert not any(w in text for w in words), n


def test_wrong_mode_line():
    assert flow.asked_line("Detect", "a name for each single object") == (
        "You asked for a box and a label for each object. This question needs a name for each single object.")


SOLVED = {
    "completed_levels": {1, 2, 3, 4},
    "l1_final": {"quality": 0.847, "clip": 0.0, "ms": 2.1},
    "l2_model": "n-cls", "l2_anchor_label": "stopwatch",
    "l3_model": "s", "best_scores": {3: 7 / 9},
    "l4_deployed": {"id": "std-int8", "name": "U-Net Standard INT8", "iou": 0.968},
    "l1_pred_tool": {"q": "Which?", "options": ["a", "b"], "choice": 1, "correct": [1], "why": "b.", "right": True},
    "l1_pred_tool_choice": "b",
    "lab_pred_x": {"q": "Lab?", "options": ["a"], "choice": 0, "correct": [0], "why": "", "right": True},
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


def test_forecasts_and_viva_prompts_use_your_numbers():
    forecasts = levels.chapter_forecasts(SOLVED)
    assert len(forecasts) == 1 and forecasts[0]["right"]          # the lab prediction isn't a chapter one
    prompts = levels.viva_prompts(SOLVED, BENCH)
    assert len(prompts) == 4 and "0.85" in prompts[0] and "0.78" in prompts[2] and "3.8×" in prompts[3]
    md = levels.recap_markdown("CASE 0003", levels.recap_rows(SOLVED, BENCH), forecasts, prompts)
    assert "Forecasts: 1/1" in md and md.count("| ") > 8
