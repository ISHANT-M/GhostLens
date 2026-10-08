from game.scoring import level_xp


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
