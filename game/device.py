"""The GhostLens device: battery, model memory and a ledger of everything that used or added charge."""

from collections.abc import MutableMapping

# 1 unit = 1 ms of measured compute = 0.1% of a full pack
CAPACITY = 1000
UNITS_PER_PERCENT = 10
BATTERY_START = 200
LOW_POWER_BELOW = 100
CHARGER_UNITS = 100
CHARGER_XP = 40
A_GRADE_UNITS = 10
MEMORY_MB = 32.0
LEVELS = (1, 2, 3, 4)


def pct(units: float) -> str:
    return f"{units / UNITS_PER_PERCENT:.1f}%"


def init_device(store: MutableMapping) -> None:
    if "ledger" not in store:  # new session, or one from before 0.6 with the old battery scale
        reset_device(store)
    store.setdefault("resident", {})


def reset_device(store: MutableMapping) -> None:
    store["battery"] = BATTERY_START
    store["battery_low_mark"] = BATTERY_START
    store["resident"] = {}         # models kept loaded between chapters: {slot: (name, size_mb)}
    store["ledger"] = []           # every change to the battery, in order
    store["side_scans"] = {}       # {scan key: True once logged, False after a failed try}
    store["side_clues"] = []
    store["intel"] = {}
    store["low_power_warned"] = False
    store.pop("runs", None)        # the old log, replaced by the ledger


def _log(store: MutableMapping, level: int, what: str, ms: float, units: int, kind: str) -> None:
    store["ledger"].append({"level": level, "what": what, "ms": ms, "units": units, "kind": kind,
                            "battery": store["battery"]})


def _add(store: MutableMapping, units: int) -> None:
    store["battery"] = min(CAPACITY, store["battery"] + units)


def spend(store: MutableMapping, level: int, what: str, latency_ms: float, units: int, kind: str = "main") -> None:
    store["battery"] = max(0, store["battery"] - units)
    store["battery_low_mark"] = min(store.get("battery_low_mark", BATTERY_START), store["battery"])
    _log(store, level, what, latency_ms, units, kind)


def used_in_level(store: MutableMapping, level: int, kind: str = "main") -> int:
    return sum(r["units"] for r in store["ledger"] if r["level"] == level and r["kind"] == kind)


def can_afford(store: MutableMapping, units: int) -> bool:
    return store["battery"] >= units


def can_run(store: MutableMapping, units: int, tier: str) -> bool:
    # light models still run on the emergency reserve
    return can_afford(store, units) or tier == "Light"


def low_power(store: MutableMapping) -> bool:
    return store["battery"] < LOW_POWER_BELOW


def charger_available(store: MutableMapping) -> bool:
    return low_power(store)


def active_level(store: MutableMapping, level: int) -> int:
    """The chapter a lobby trip counts against: this one if unsolved, else the first unsolved one."""
    solved = store.get("completed_levels", set())
    if level not in solved:
        return level
    return next((n for n in LEVELS if n not in solved), level)


def lobby_charge(store: MutableMapping, level: int) -> int:
    if not charger_available(store):
        return 0
    _add(store, CHARGER_UNITS)
    store["xp"] = max(0, store.get("xp", 0) - CHARGER_XP)
    _log(store, active_level(store, level), "Lobby charger", 0.0, CHARGER_UNITS, "lobby")
    return CHARGER_UNITS


def lobby_trips(store: MutableMapping, level: int | None = None) -> int:
    return sum(1 for r in store["ledger"] if r["kind"] == "lobby" and (level is None or r["level"] == level))


def lobby_check(store: MutableMapping, level: int) -> tuple[str, tuple[bool, str]]:
    trips = lobby_trips(store, level)
    detail = "no trips to the lobby charger" if trips == 0 else f"walked to the lobby charger {trips}×"
    return "Stayed in the field", (trips == 0, detail)


def grade_reward(store: MutableMapping, level: int, grade: str) -> int:
    got = any(r["kind"] == "cell" and r["level"] == level for r in store["ledger"])
    if grade != "A" or got:
        return 0
    _add(store, A_GRADE_UNITS)
    _log(store, level, "Spare cell", 0.0, A_GRADE_UNITS, "cell")
    return A_GRADE_UNITS


def summary(store: MutableMapping) -> dict:
    ledger = store["ledger"]
    return {
        "start": BATTERY_START,
        "left": store["battery"],
        "lowest": store.get("battery_low_mark", store["battery"]),
        "spent_main": sum(r["units"] for r in ledger if r["kind"] == "main"),
        "spent_side": sum(r["units"] for r in ledger if r["kind"] == "side"),
        "spare_cells": sum(1 for r in ledger if r["kind"] == "cell"),
        "lobby_trips": lobby_trips(store),
        "side_scans": sum(1 for ok in store.get("side_scans", {}).values() if ok),
    }


def memory_used(store: MutableMapping) -> float:
    return sum(size for _, size in store["resident"].values())


def free_memory(store: MutableMapping) -> float:
    return MEMORY_MB - memory_used(store)


def fits(store: MutableMapping, size_mb: float, replacing: str | None = None) -> bool:
    freed = store["resident"].get(replacing, (None, 0.0))[1] if replacing else 0.0
    return size_mb <= free_memory(store) + freed


def load_model(store: MutableMapping, slot: str, name: str, size_mb: float) -> None:
    store["resident"][slot] = (name, size_mb)


def unload(store: MutableMapping, slot: str) -> None:
    store["resident"].pop(slot, None)
