from game import achievements
from game.codex import QUIZ


def ids(store):
    return achievements.earned_ids(store)


def test_empty_store_has_no_badges():
    assert achievements.earned({}) == []


def test_perfect_run():
    store = {
        "completed_levels": {1, 2, 3, 4}, "battery": 70,
        "l1_tried": ["Enhance"], "l2_tried": ["Classify"], "l3_tried": ["Detect"], "l4_tried": ["Segment"],
        "grades": {1: "A", 2: "A", 3: "A", 4: "A"}, "l4_model": "unet-int8",
        "best_scores": {1: 0.9, 3: 0.8},
    }
    assert ids(store) == {"frugal", "right_tool", "straight_a", "quantizer", "balanced_eye", "clean_frame"}


def test_wrong_mode_and_low_battery():
    store = {"completed_levels": {1, 2, 3, 4}, "battery": 0, "l2_tried": ["Detect", "Classify"],
             "grades": {1: "A", 2: "B", 3: "A", 4: "A"}, "l4_model": "unet-fp32", "best_scores": {3: 0.7}}
    assert ids(store) == {"fumes"}


def test_tips_and_quiz():
    store = {"seen_tips": set(range(15)), "quiz_correct": {"q1", "q2", "q3", "q4"}}
    assert ids(store) == {"scholar"}
    store["quiz_correct"].add("q5")
    assert "student" in ids(store)


def test_all_badges_have_text():
    for b in achievements.all_badges():
        assert b["name"] and b["description"] and b["hint"] and "check" not in b


def test_quiz_bank_valid():
    assert len({q[0] for q in QUIZ}) == len(QUIZ)
    for _, text, options, answer, why in QUIZ:
        assert text and why and 0 <= answer < len(options)
