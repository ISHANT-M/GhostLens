"""Badges, checked against st.session_state (or a plain dict in tests)."""

import re
from collections.abc import Mapping

from game import device

CORRECT_MODES = {1: "Enhance", 2: "Classify", 3: "Detect", 4: "Segment"}
QUIZ_TARGET = 5
TIPS_TARGET = 15
CLEAN_FRAME = 0.88       # reference settings give 0.847, careful cheap tuning reaches about 0.896
SIDE_CLUES = 4
FORECASTS = 3            # chapter predictions called right


def _solved(store: Mapping) -> set:
    return set(store.get("completed_levels", set()))


def _closed(store: Mapping) -> bool:
    return _solved(store) >= {1, 2, 3, 4}


def _best(store: Mapping, level: int) -> float:
    return store.get("best_scores", {}).get(level, 0.0) or 0.0


def _lobby_trips(s) -> int:
    return sum(1 for r in s.get("ledger", []) if r["kind"] == "lobby")


def _frugal(s):
    return _closed(s) and s.get("battery", 0) >= device.LOW_POWER_BELOW and _lobby_trips(s) == 0


def _fumes(s):
    return s.get("battery_low_mark", s.get("battery")) == 0


def _right_tool(s):
    return _closed(s) and all(set(s.get(f"l{n}_tried", [])) <= {m} for n, m in CORRECT_MODES.items())


def _straight_a(s):
    grades = s.get("grades", {})
    return all(grades.get(n) == "A" for n in CORRECT_MODES)


def _forecasts_right(s) -> int:
    return sum(1 for k, v in s.items() if re.match(r"^l\d_pred_", str(k)) and isinstance(v, dict) and v.get("right"))


def _quantizer(s):
    return 4 in _solved(s) and str(s.get("l4_model") or "").endswith("-int8")


BADGES = [
    {"id": "frugal", "name": "Frugal",
     "description": f"Closed the case with at least {device.pct(device.LOW_POWER_BELOW)} battery and no lobby trips.",
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
     "check": lambda s: round(_best(s, 3), 2) >= 0.78},
    {"id": "clean_frame", "name": "Clean frame",
     "description": f"Scored {CLEAN_FRAME:.0%} or more on the chapter 1 frame.",
     "hint": "Tune past the reference settings: brighter plate, no clipping.",
     "check": lambda s: round(_best(s, 1), 2) >= CLEAN_FRAME},
    {"id": "thorough", "name": "Thorough", "description": f"Logged all {SIDE_CLUES} side clues.",
     "hint": "Run the optional side scan after each chapter.",
     "check": lambda s: len(s.get("side_clues", ())) >= SIDE_CLUES},
    {"id": "scholar", "name": "Scholar", "description": f"Found {TIPS_TARGET} or more loading screen tips.",
     "hint": "Read the tips on loading screens.",
     "check": lambda s: len(s.get("seen_tips", ())) >= TIPS_TARGET},
    {"id": "student", "name": "Student of the guide",
     "description": f"Answered {QUIZ_TARGET} field guide quiz questions right.",
     "hint": "Try the quiz in the field guide.",
     "check": lambda s: len(s.get("quiz_correct", ())) >= QUIZ_TARGET},
    {"id": "forecaster", "name": "Forecaster",
     "description": f"Called {FORECASTS} chapter predictions right before seeing the measurement.",
     "hint": "Answer the 'Predict first' questions in the chapters.",
     "check": lambda s: _forecasts_right(s) >= FORECASTS},
    {"id": "fumes", "name": "Running on fumes", "description": "Let the battery hit zero. Oops.",
     "hint": "Not one to aim for.", "check": _fumes},
]


def all_badges() -> list[dict]:
    return [{k: v for k, v in b.items() if k != "check"} for b in BADGES]


def earned(store: Mapping) -> list[dict]:
    return [{k: v for k, v in b.items() if k != "check"} for b in BADGES if b["check"](store)]


def earned_ids(store: Mapping) -> set[str]:
    return {b["id"] for b in earned(store)}
