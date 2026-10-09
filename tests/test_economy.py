"""The battery economy: guarantees from the measured benchmark, and the shared paid widgets."""

import pytest
from streamlit.testing.v1 import AppTest

from cv.edge import energy_units, load_benchmark
from game import device

BENCH = load_benchmark()
needs_bench = pytest.mark.skipif(BENCH is None, reason="run setup_models.py")
CH1_PASS = 5   # a reference enhancement pass is measured live at about 2 ms; 5 leaves room for a slow machine


def cost(group: str, model_id: str) -> int:
    return energy_units(next(r for r in BENCH[group] if r["id"] == model_id)["latency_ms"])


def test_energy_units():
    assert energy_units(49.2) == 49
    assert energy_units(0.2) == 1


@needs_bench
def test_ideal_run_leaves_a_margin():
    ideal = CH1_PASS + 4 * cost("classifiers", "yolo26n-cls.pt") + cost("detectors", "yolo26s.pt") \
        + cost("unets", "standard-int8")
    assert ideal <= device.BATTERY_START - 50


@needs_bench
def test_one_lobby_trip_always_pays_for_a_mandatory_run():
    # if you can't afford one of these you are in low power, so the charger is open and one trip is enough
    for units in (cost("detectors", "yolo26s.pt"), cost("unets", "standard-int8")):
        assert units < device.LOW_POWER_BELOW <= device.CHARGER_UNITS


@needs_bench
def test_chapter_two_light_model_runs_on_the_reserve():
    store = {}
    device.init_device(store)
    store["battery"] = 0
    assert device.can_run(store, cost("classifiers", "yolo26n-cls.pt"), "Light")


RUN_BUTTON = """
import streamlit as st
from game import device, flow, state
state.init_state(st.session_state)
device.init_device(st.session_state)
if flow.run_button(3, "Run YOLO26s", "YOLO26s on the parlour", 49.2, "Balanced", key="run_s"):
    st.session_state.ran = True
flow.run_button(3, "Run YOLO26n", "YOLO26n on the parlour", 24.9, "Light", key="run_n")
"""


def app(script: str) -> AppTest:
    return AppTest.from_string(script, default_timeout=60).run()


def texts(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)


def test_run_button_charges_and_forecasts():
    at = app(RUN_BUTTON)
    assert at.button(key="run_s").label == "Run YOLO26s · 4.9%"
    assert "→ 15.1% left" in texts(at)
    at.button(key="run_s").click().run()
    assert at.session_state.ran and at.session_state.battery == 151
    assert device.used_in_level(at.session_state, 3) == 49


def test_run_button_blocked_until_lobby_trip():
    at = app(RUN_BUTTON)
    at.session_state.battery = 30
    at.session_state.xp = 25
    at.run()
    assert at.button(key="run_s").disabled
    assert not at.button(key="run_n").disabled
    assert "needs 4.9% · 3.0% left" in texts(at)
    assert "runs on the emergency reserve" not in texts(at)
    at.session_state.battery = 10
    at.run()
    assert "runs on the emergency reserve" in texts(at)
    at.button(key="lobby_run_s").click().run()
    assert at.session_state.battery == 110 and at.session_state.xp == 0
    assert not at.button(key="run_s").disabled
    assert device.lobby_check(at.session_state, 3)[1][0] is False


SIDE_SCAN = """
import streamlit as st
from game import device, flow, state
state.init_state(st.session_state)
device.init_device(st.session_state)

def run():
    st.session_state.tries = st.session_state.get("tries", 0) + 1
    return st.session_state.tries > 1

flow.side_scan(3, flow.SideScan(
    key="tray", title="Zoom on the tea tray", blurb="Crop the tray and look again.", what="YOLO26n on the tray",
    latency_ms=24.9, reward_xp=25, clue="Tray: 4 of 4 cups", run=run, show=lambda: st.write("tray shown"),
    intel=("l3_light", "found the cups up close"), tip_topic="Input resolution"))
"""


def test_side_scan_retries_then_rewards_once():
    at = app(SIDE_SCAN)
    assert at.button(key="side_tray").label == "Run side scan · 2.5%"
    at.button(key="side_tray").click().run()
    assert at.session_state.side_scans == {"tray": False} and at.session_state.xp == 0
    assert "You can try again" in texts(at)
    at.button(key="side_tray").click().run()
    s = at.session_state
    assert s.side_scans == {"tray": True}
    assert s.xp == 25 and s.side_clues == ["Tray: 4 of 4 cups"]
    assert s.intel == {"l3_light": "found the cups up close"}
    assert len(s.seen_tips) == 1
    assert s.battery == device.BATTERY_START - 50
    assert device.used_in_level(s, 3) == 0 and device.used_in_level(s, 3, kind="side") == 50
    at.run()
    assert at.session_state.xp == 25
    assert not [b for b in at.button if b.key == "side_tray"]
    assert "LOGGED" in texts(at)


def test_side_scan_never_uses_the_reserve():
    at = app(SIDE_SCAN)
    at.session_state.battery = 10
    at.run()
    assert at.button(key="side_tray").disabled


PICKER = """
import streamlit as st
from game import device, flow, state
state.init_state(st.session_state)
device.init_device(st.session_state)
card = {"accuracy": "x", "accuracy_tag": "published"}
flow.model_picker(3, [
    {**card, "id": "n", "name": "Nano", "tier": "Light", "size_mb": 5.5, "latency_ms": 24.9,
     "intel": "Doorway scout found no cups."},
    {**card, "id": "s", "name": "Small", "tier": "Balanced", "size_mb": 20.4, "latency_ms": 49.2},
    {**card, "id": "m", "name": "Medium", "tier": "Heavy", "size_mb": 10.0, "latency_ms": 110.5},
    {**card, "id": "x", "name": "Huge", "tier": "Heavy", "size_mb": 44.3, "latency_ms": 110.5},
], {"latency_ms": 60}, slot="watchdog")
"""


def picker_at(battery: int) -> AppTest:
    at = app(PICKER)
    at.session_state.battery = battery
    return at.run()


def card_html(at: AppTest, name: str) -> str:
    return next(m.value for m in at.markdown if f'<div class="name">{name}</div>' in m.value)


def locked(at: AppTest) -> dict:
    return {b.key: b.disabled for b in at.button if b.key.startswith("l3_model_")}


def test_picker_on_an_empty_battery():
    at = picker_at(0)
    assert locked(at) == {"l3_model_n": False, "l3_model_s": True, "l3_model_m": True, "l3_model_x": True}
    assert "EMERGENCY RESERVE" in card_html(at, "Nano")
    assert "FIELD INTEL" in card_html(at, "Nano")
    assert "NEEDS 4.9%" in card_html(at, "Small")
    assert "LOW POWER" in card_html(at, "Medium")
    assert "DOES NOT FIT" in card_html(at, "Huge")
    assert at.button(key="lobby_l3_picker")


def test_picker_in_low_power_with_some_charge():
    at = picker_at(50)
    assert locked(at) == {"l3_model_n": False, "l3_model_s": False, "l3_model_m": True, "l3_model_x": True}
    assert "gl-badge" not in card_html(at, "Small")
    assert "LOW POWER" in card_html(at, "Medium")


def test_picker_with_a_healthy_battery():
    at = picker_at(150)
    assert locked(at)["l3_model_m"] is False
    assert "over target" in card_html(at, "Medium")
    assert not [b for b in at.button if b.key == "lobby_l3_picker"]
