from game import device


def fresh(**extra):
    store = {"xp": 0, "completed_levels": set()}
    device.init_device(store)
    store.update(extra)
    return store


def test_starts_at_twenty_percent():
    s = fresh()
    assert s["battery"] == device.BATTERY_START == 200
    assert device.pct(s["battery"]) == "20.0%"
    assert device.pct(49) == "4.9%"


def test_spend_clamps_at_zero_and_keeps_low_mark():
    s = fresh()
    device.spend(s, 1, "big model", 900, 900)
    assert s["battery"] == 0 and s["battery_low_mark"] == 0
    assert device.used_in_level(s, 1) == 900
    assert device.used_in_level(s, 2) == 0


def test_side_spend_not_counted_as_main():
    s = fresh()
    device.spend(s, 2, "scout", 25, 25, kind="side")
    device.spend(s, 2, "scan", 7, 7)
    assert device.used_in_level(s, 2) == 7
    assert device.used_in_level(s, 2, kind="side") == 25


def test_light_models_run_on_the_reserve():
    s = fresh(battery=0)
    assert device.can_run(s, 25, "Light")
    assert not device.can_run(s, 49, "Balanced")
    s["battery"] = 60
    assert device.can_run(s, 49, "Balanced") and device.can_afford(s, 49)


def test_low_power_below_ten_percent():
    s = fresh(battery=100)
    assert not device.low_power(s)
    s["battery"] = 99
    assert device.low_power(s) and device.charger_available(s)


def test_lobby_only_in_low_power():
    s = fresh(battery=120, xp=100)
    assert device.lobby_charge(s, 1) == 0
    assert s["battery"] == 120 and s["xp"] == 100


def test_lobby_charge_costs_xp_and_fails_check():
    s = fresh(battery=30, xp=25)
    assert device.lobby_charge(s, 3) == device.CHARGER_UNITS
    assert s["battery"] == 130 and s["xp"] == 0           # xp floors at 0
    assert device.lobby_trips(s, 3) == 1 and device.lobby_trips(s) == 1
    assert device.lobby_check(s, 3) == ("Stayed in the field", (False, "walked to the lobby charger 1×"))
    assert device.lobby_check(s, 2)[1][0]


def test_lobby_trip_counts_against_first_unsolved_chapter():
    s = fresh(battery=10, completed_levels={1, 2})
    device.lobby_charge(s, 2)
    assert device.lobby_trips(s, 3) == 1 and device.lobby_trips(s, 2) == 0


def test_grade_reward_once_per_chapter():
    s = fresh()
    assert device.grade_reward(s, 1, "B") == 0
    assert device.grade_reward(s, 1, "A") == device.A_GRADE_UNITS
    assert device.grade_reward(s, 1, "A") == 0
    assert s["battery"] == 210


def test_summary():
    s = fresh()
    device.spend(s, 1, "pass", 2, 2)
    device.spend(s, 1, "cam 04", 2, 2, kind="side")
    device.grade_reward(s, 1, "A")
    s["side_scans"]["cam04"] = True
    got = device.summary(s)
    assert got == {"start": 200, "left": 206, "lowest": 196, "spent_main": 2, "spent_side": 2,
                   "spare_cells": 1, "lobby_trips": 0, "side_scans": 1}


def test_reset_and_old_session_migration():
    s = fresh()
    device.spend(s, 1, "x", 0, 50)
    device.reset_device(s)
    assert s["battery"] == device.BATTERY_START and s["ledger"] == []
    old = {"battery": 37, "runs": [{"level": 1}], "resident": {}}
    device.init_device(old)
    assert old["battery"] == device.BATTERY_START and "runs" not in old and old["ledger"] == []


def test_memory_and_fit():
    s = fresh()
    device.load_model(s, "watchdog", "YOLO26s", 20.4)
    assert device.free_memory(s) == device.MEMORY_MB - 20.4
    assert not device.fits(s, 15)
    assert device.fits(s, 15, replacing="watchdog")
    device.unload(s, "watchdog")
    assert device.memory_used(s) == 0
