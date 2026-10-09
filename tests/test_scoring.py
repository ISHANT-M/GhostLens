from game.scoring import edge_grade, efficient, level_xp


def test_perfect_first_attempt():
    assert level_xp(1.0, 1)["total"] == 150


def test_extra_attempts_cost_xp():
    result = level_xp(0.8, 3)
    assert result["quality_bonus"] == 40
    assert result["attempt_penalty"] == -20
    assert result["total"] == 120


def test_never_below_minimum():
    assert level_xp(0.0, 20)["total"] == 50


def test_quality_is_clamped():
    assert level_xp(5.0, 1)["total"] == 150
    assert level_xp(-1.0, 1)["total"] == 100


def test_grade_bonus_in_xp():
    assert level_xp(1.0, 1, "A")["total"] == 200
    assert level_xp(1.0, 1, "C")["total"] == 150


def test_grades():
    assert edge_grade({"a": True, "b": True}) == "A"
    assert edge_grade({"a": False, "b": True}) == "B"
    assert edge_grade({"a": False, "b": False, "c": False, "d": False}) == "D"


def test_trying_nano_then_small_detector_is_wasteful():
    assert not efficient(25 + 49, 49)


def test_lite_then_standard_int8_is_fine():
    assert efficient(5 + 22, 22)


def test_three_cheap_passes_are_fine():
    assert efficient(3 * 2, 2)


def test_nl_means_pass_is_wasteful():
    assert not efficient(170, 2)
