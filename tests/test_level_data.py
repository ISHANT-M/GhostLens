"""Every option in the case pool, checked with the real models: right choices work, wrong ones fail."""

import numpy as np
import pytest

from cv import detection as det
from cv import segmentation as seg
from cv.edge import BENCHMARK_FILE, load_benchmark
from game import case

pytestmark = pytest.mark.skipif(not BENCHMARK_FILE.exists(), reason="run setup_models.py")

CLASSIFIERS = ["yolo26n-cls.pt", "yolo26s-cls.pt", "yolo26m-cls.pt"]
MASK_THRESHOLDS = [round(float(t), 2) for t in np.arange(0.1, 0.91, 0.05)]   # the chapter 4 slider
LIMIT = 0.95


@pytest.mark.parametrize("weights", CLASSIFIERS)
def test_each_anchor_is_named_inside_its_own_label_set(weights):
    from game import level2
    for name, anchor in case.ANCHORS.items():
        top = level2.scan(level2.OBJECTS[name], weights)["top"][0][0]
        assert top in anchor["labels"], (weights, name, top)
        others = set().union(*(a["labels"] for n, a in case.ANCHORS.items() if n != name))
        assert top not in others


def scores(weights: str) -> dict[float, dict]:
    from game import level3
    dets = level3.scene_detections(weights)
    return {t: det.evaluate(dets, level3.TRUTH["objects"], t) for t in level3.THRESHOLDS}


@pytest.mark.parametrize("brief", sorted(case.BRIEFS))
def test_balanced_detector_meets_each_brief_and_light_never_does(brief):
    small = [t for t, (_, s) in scores("yolo26s.pt").items() if case.brief_passes(brief, s)]
    assert 0.2 in small
    start = case.BRIEFS[brief]["start"]
    assert start not in small                       # the slider starts outside the passing window
    assert not any(case.brief_passes(brief, s) for _, s in scores("yolo26n.pt").values())


def test_moved_cups_are_found_wherever_a_brief_passes():
    from game import level3
    truth = level3.TRUTH["objects"]
    for t, (m, s) in scores("yolo26s.pt").items():
        if not any(case.brief_passes(b, s) for b in case.BRIEFS):
            continue
        for i in case.MOVABLE:
            assert truth[i]["label"] == "cup"
            assert any(d["label"] == "cup" and det.iou(d["box"], truth[i]["box"]) >= det.IOU_MATCH for d in m["tp"]), (t, i)


def test_scout_sees_the_furniture_but_no_cups():
    from game import level2
    labels = sorted(d["label"] for d in level2.scout_matches()["tp"])
    assert labels == ["chair", "chair", "dining table"]
    assert level2.scout_text(level2.scout_matches()).endswith("none of the 4 cups")


def test_tray_crop_finds_the_cups():
    from game import level3
    assert len(level3.tray_matches()["tp"]) >= 3
    full, _ = det.evaluate(level3.scene_detections("yolo26n.pt"), level3.cups(), level3.SIDE_CONF)
    assert not full["tp"]                           # the same model on the full frame finds none


@pytest.mark.parametrize("wall", case.WALL_SEEDS)
def test_wall_pool(wall):
    from game import level4
    _, truth = level4.scene(wall)

    def iou(variant: str, int8: bool, t: float) -> float:
        return seg.mask_iou(level4.wall_probs(variant, int8, wall) > t, truth)

    for int8 in (False, True):
        assert max(iou("lite", int8, t) for t in MASK_THRESHOLDS) < LIMIT          # the light model always leaks
        assert iou("standard", int8, 0.5) >= 0.96
        assert iou("pro", int8, 0.5) >= 0.96                                       # pro fails on latency only


def test_benchmark_agrees_with_the_purify_check():
    unets = {r["id"]: r for r in load_benchmark()["unets"]}
    assert unets["lite-fp32"]["iou"] < LIMIT and unets["lite-int8"]["iou"] < LIMIT
    assert unets["standard-int8"]["iou"] >= LIMIT and unets["standard-int8"]["latency_ms"] <= 40
    assert unets["standard-fp32"]["latency_ms"] > 40 and unets["pro-int8"]["latency_ms"] > 40
