"""Game progress. `store` is st.session_state in the app and a plain dict in tests."""

from collections.abc import MutableMapping

LEVEL_COUNT = 4

DEFAULTS = {
    "current_level": 1,
    "completed_levels": set(),
    "xp": 0,
    "clues_found": [],
    "demo_mode": False,
    "best_scores": {},
    "grades": {},
}


def init_state(store: MutableMapping) -> None:
    # runs on every rerun, so never overwrite what is already there
    for key, value in DEFAULTS.items():
        if key not in store:
            store[key] = value.copy() if hasattr(value, "copy") else value


def reset_progress(store: MutableMapping) -> None:
    # demo_mode belongs to a sidebar widget, so it is left alone
    for key, value in DEFAULTS.items():
        if key != "demo_mode":
            store[key] = value.copy() if hasattr(value, "copy") else value


def is_unlocked(store: MutableMapping, level: int) -> bool:
    if not 1 <= level <= LEVEL_COUNT:
        return False
    if store["demo_mode"] or level == 1:
        return True
    return (level - 1) in store["completed_levels"]


def status(store: MutableMapping, level: int) -> str:
    if level in store["completed_levels"]:
        return "SOLVED"
    return "OPEN" if is_unlocked(store, level) else "LOCKED"


def complete_level(store: MutableMapping, level: int, xp: int, clue: str | None = None,
                   score: float | None = None) -> bool:
    # returns False on replays so XP is only given once
    if score is not None and score > store["best_scores"].get(level, float("-inf")):
        store["best_scores"][level] = score
    if level in store["completed_levels"]:
        return False
    store["completed_levels"].add(level)
    store["xp"] += xp
    if clue and clue not in store["clues_found"]:
        store["clues_found"].append(clue)
    store["current_level"] = min(level + 1, LEVEL_COUNT)
    return True


def all_solved(store: MutableMapping) -> bool:
    return len(store["completed_levels"]) == LEVEL_COUNT
