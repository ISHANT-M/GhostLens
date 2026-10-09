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


# why a false alarm didn't count, and the label format

from cv import detection as det  # noqa: E402

TRUTH3 = [{"label": "cup", "box": [0, 0, 10, 10]}, {"label": "chair", "box": [50, 50, 90, 90]},
          {"label": "cup", "box": [200, 200, 220, 220]}]


def test_false_alarm_reasons():
    dets = [
        {"label": "cup", "conf": 0.9, "box": [0, 0, 10, 10]},          # correct
        {"label": "cup", "conf": 0.8, "box": [0, 0, 10, 9]},           # second box on the same cup
        {"label": "cup", "conf": 0.7, "box": [200, 200, 240, 240]},    # IoU 0.25 with the other cup
        {"label": "cup", "conf": 0.6, "box": [52, 52, 88, 88]},        # on the chair
        {"label": "cup", "conf": 0.5, "box": [400, 400, 420, 420]},    # nothing there
    ]
    m = det.match(dets, TRUTH3)
    assert m["tp"][0]["truth"] == 0
    reasons = {a["det"]["conf"]: (a["reason"], a["truth"]) for a in det.explain_false_alarms(m, TRUTH3)}
    assert reasons == {0.8: ("duplicate", 0), 0.7: ("loose", 2), 0.6: ("other_class", 1), 0.5: ("unlabelled", None)}
    assert set(det.REASONS) == {"duplicate", "loose", "other_class", "unlabelled"}


def test_yolo_lines_are_normalised_centres():
    lines = det.yolo_lines([{"label": "cup", "box": [100, 50, 300, 150]}], 400, 200, {"cup": 41})
    assert lines == ["41 0.5000 0.5000 0.5000 0.5000"]


def test_model_input_size_scales_to_640():
    assert det.input_size([10, 20, 46, 50], (720, 1280)) == (18, 15)
    assert det.input_scale((720, 1280)) == 0.5
