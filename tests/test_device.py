from game import device
from game.scoring import edge_grade, efficient, level_xp


def fresh():
    store = {}
    device.init_device(store)
    return store


def test_spend_never_goes_below_zero_and_is_logged():
    s = fresh()
    device.spend(s, 1, "big model", 900, 120)
    assert s["battery"] == 0
    assert device.used_in_level(s, 1) == 120
    assert device.used_in_level(s, 2) == 0


def test_memory_and_fit():
    s = fresh()
    device.load_model(s, "watchdog", "YOLO26s", 20.4)
    assert device.free_memory(s) == device.MEMORY_MB - 20.4
    assert not device.fits(s, 15)
    assert device.fits(s, 15, replacing="watchdog")
    device.unload(s, "watchdog")
    assert device.memory_used(s) == 0


def test_low_power():
    s = fresh()
    device.spend(s, 1, "x", 0, 85)
    assert device.low_power(s)


def test_reset():
    s = fresh()
    device.spend(s, 1, "x", 0, 50)
    device.reset_device(s)
    assert s["battery"] == device.BATTERY_START and s["runs"] == []


def test_grades():
    assert edge_grade({"a": True, "b": True}) == "A"
    assert edge_grade({"a": False, "b": True}) == "B"
    assert edge_grade({"a": False, "b": False, "c": False, "d": False}) == "D"


def test_efficiency_has_some_slack():
    assert efficient(6, 4)
    assert not efficient(20, 4)


def test_grade_bonus_in_xp():
    assert level_xp(1.0, 1, "A")["total"] == 200
    assert level_xp(1.0, 1, "C")["total"] == 150
