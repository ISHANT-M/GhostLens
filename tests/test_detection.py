import pytest

from cv.detection import evaluate, iou, match, scores


def test_iou_basic_cases():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


TRUTH = [{"label": "cup", "box": [0, 0, 10, 10]}, {"label": "chair", "box": [50, 50, 90, 90]}]


def test_match_counts_tp_fp_and_missed():
    dets = [
        {"label": "cup", "conf": 0.9, "box": [1, 1, 10, 10]},      # right
        {"label": "cup", "conf": 0.8, "box": [0, 0, 9, 9]},        # duplicate of the same cup
        {"label": "chair", "conf": 0.7, "box": [0, 0, 10, 10]},    # wrong label on the cup
    ]
    m = match(dets, TRUTH)
    assert len(m["tp"]) == 1 and len(m["fp"]) == 2
    assert [t["label"] for t in m["missed"]] == ["chair"]


def test_scores():
    s = scores({"tp": [1, 1, 1], "fp": [1], "missed": [1, 1]})
    assert s["precision"] == 0.75
    assert s["recall"] == 0.6
    assert s["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def test_empty_prediction_has_full_precision_zero_recall():
    s = scores(match([], TRUTH))
    assert s["precision"] == 1.0 and s["recall"] == 0.0 and s["f1"] == 0.0


def test_threshold_trades_precision_for_recall():
    dets = [
        {"label": "cup", "conf": 0.9, "box": [0, 0, 10, 10]},
        {"label": "chair", "conf": 0.3, "box": [50, 50, 90, 90]},
        {"label": "cup", "conf": 0.2, "box": [100, 100, 120, 120]},
    ]
    _, low = evaluate(dets, TRUTH, 0.1)
    _, high = evaluate(dets, TRUTH, 0.5)
    assert low["recall"] > high["recall"]
    assert low["precision"] < high["precision"]
