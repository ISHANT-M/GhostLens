from game import achievements
from game.codex import QUIZ


def ids(store):
    return achievements.earned_ids(store)


def test_empty_store_has_no_badges():
    assert achievements.earned({}) == []


def test_perfect_run():
    store = {
        "completed_levels": {1, 2, 3, 4}, "battery": 139, "battery_low_mark": 139, "ledger": [],
        "l1_tried": ["Enhance"], "l2_tried": ["Classify"], "l3_tried": ["Detect"], "l4_tried": ["Segment"],
        "grades": {1: "A", 2: "A", 3: "A", 4: "A"}, "l4_model": "standard-int8",
        "best_scores": {1: 0.896, 3: 7 / 9},
    }
    assert ids(store) == {"frugal", "right_tool", "straight_a", "quantizer", "balanced_eye", "clean_frame"}


def test_reference_settings_do_not_earn_clean_frame():
    assert "clean_frame" not in ids({"best_scores": {1: 0.847}})


def test_frugal_needs_no_lobby_trips():
    store = {"completed_levels": {1, 2, 3, 4}, "battery": 120, "ledger": []}
    assert "frugal" in ids(store)
    store["ledger"] = [{"level": 3, "kind": "lobby", "units": 100}]
    assert "frugal" not in ids(store)
    assert "frugal" not in ids({"completed_levels": {1, 2, 3, 4}, "battery": 99, "ledger": []})


def test_fumes_sticks_after_recharging():
    store = {"battery": 100, "battery_low_mark": 0}
    assert "fumes" in ids(store)
    assert "fumes" not in ids({"battery": 5, "battery_low_mark": 5})


def test_thorough_needs_all_side_clues():
    store = {"side_clues": ["a", "b", "c"]}
    assert "thorough" not in ids(store)
    store["side_clues"].append("d")
    assert "thorough" in ids(store)


def test_wrong_mode_and_low_battery():
    store = {"completed_levels": {1, 2, 3, 4}, "battery": 0, "battery_low_mark": 0, "ledger": [],
             "l2_tried": ["Detect", "Classify"], "grades": {1: "A", 2: "B", 3: "A", 4: "A"},
             "l4_model": "standard-fp32", "best_scores": {3: 0.7}}
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
