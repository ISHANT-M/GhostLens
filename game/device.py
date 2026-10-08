"""The GhostLens device: battery, model memory and a log of what ran."""

from collections.abc import MutableMapping

BATTERY_START = 100
MEMORY_MB = 32.0
LOW_POWER_BELOW = 20


def init_device(store: MutableMapping) -> None:
    store.setdefault("battery", BATTERY_START)
    store.setdefault("resident", {})      # models kept loaded between chapters: {slot: (name, size_mb)}
    store.setdefault("runs", [])          # every inference: level, what, measured ms, units


def reset_device(store: MutableMapping) -> None:
    store["battery"] = BATTERY_START
    store["resident"] = {}
    store["runs"] = []


def spend(store: MutableMapping, level: int, what: str, latency_ms: float, units: int) -> None:
    store["battery"] = max(0, store["battery"] - units)
    store["runs"].append({"level": level, "what": what, "ms": latency_ms, "units": units})


def used_in_level(store: MutableMapping, level: int) -> int:
    return sum(r["units"] for r in store["runs"] if r["level"] == level)


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


def low_power(store: MutableMapping) -> bool:
    return store["battery"] < LOW_POWER_BELOW
