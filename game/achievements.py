"""Badges, checked against st.session_state (or a plain dict in tests)."""

from collections.abc import Mapping

CORRECT_MODES = {1: "Enhance", 2: "Classify", 3: "Detect", 4: "Segment"}
QUIZ_TARGET = 5
TIPS_TARGET = 15


def _solved(store: Mapping) -> set:
    return set(store.get("completed_levels", set()))


def _closed(store: Mapping) -> bool:
    return _solved(store) >= {1, 2, 3, 4}


def _best(store: Mapping, level: int) -> float:
    return store.get("best_scores", {}).get(level, 0.0) or 0.0


def _frugal(s):
    return _closed(s) and s.get("battery", 0) >= 60


def _right_tool(s):
    return _closed(s) and all(set(s.get(f"l{n}_tried", [])) <= {m} for n, m in CORRECT_MODES.items())


def _straight_a(s):
    grades = s.get("grades", {})
    return all(grades.get(n) == "A" for n in CORRECT_MODES)


def _quantizer(s):
    return 4 in _solved(s) and str(s.get("l4_model") or "").endswith("-int8")


BADGES = [
    {"id": "frugal", "name": "Frugal", "description": "Closed the case with at least 60 battery left.",
     "hint": "Finish all four chapters without wasting battery.", "check": _frugal},
    {"id": "right_tool", "name": "Right tool every time",
     "description": "Solved every chapter without trying a wrong mode.",
     "hint": "Pick the correct task on the first try in all four chapters.", "check": _right_tool},
    {"id": "straight_a", "name": "Straight A", "description": "Got an A in all four chapters.",
     "hint": "Meet every check in each mission report.", "check": _straight_a},
    {"id": "quantizer", "name": "Quantizer", "description": "Solved chapter 4 with an INT8 model.",
     "hint": "Try the quantized model in chapter 4.", "check": _quantizer},
    {"id": "balanced_eye", "name": "Balanced eye", "description": "Reached F1 of 0.78 or more in chapter 3.",
     "hint": "Tune the threshold so precision and recall are both good.",
     "check": lambda s: _best(s, 3) >= 0.78},
    {"id": "clean_frame", "name": "Clean frame", "description": "Scored 85% or more on the chapter 1 frame.",
     "hint": "Brighten the dark frame without clipping it.", "check": lambda s: _best(s, 1) >= 0.85},
    {"id": "scholar", "name": "Scholar", "description": f"Found {TIPS_TARGET} or more loading screen tips.",
     "hint": "Read the tips on loading screens.",
     "check": lambda s: len(s.get("seen_tips", ())) >= TIPS_TARGET},
    {"id": "student", "name": "Student of the guide",
     "description": f"Answered {QUIZ_TARGET} field guide quiz questions right.",
     "hint": "Try the quiz in the field guide.",
     "check": lambda s: len(s.get("quiz_correct", ())) >= QUIZ_TARGET},
    {"id": "fumes", "name": "Running on fumes", "description": "Let the battery hit zero. Oops.",
     "hint": "Not one to aim for.", "check": lambda s: s.get("battery") == 0},
]


def all_badges() -> list[dict]:
    return [{k: v for k, v in b.items() if k != "check"} for b in BADGES]


def earned(store: Mapping) -> list[dict]:
    return [{k: v for k, v in b.items() if k != "check"} for b in BADGES if b["check"](store)]


def earned_ids(store: Mapping) -> set[str]:
    return {b["id"] for b in earned(store)}
