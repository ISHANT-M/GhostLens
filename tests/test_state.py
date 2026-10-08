from game import state


def fresh():
    store = {}
    state.init_state(store)
    return store


def test_init_does_not_overwrite_progress():
    store = fresh()
    store["xp"] = 120
    state.init_state(store)
    assert store["xp"] == 120


def test_sessions_do_not_share_mutable_defaults():
    a, b = fresh(), fresh()
    a["completed_levels"].add(1)
    assert b["completed_levels"] == set()
    assert state.DEFAULTS["completed_levels"] == set()


def test_only_level_one_open_at_start():
    store = fresh()
    assert state.status(store, 1) == "OPEN"
    assert [state.is_unlocked(store, n) for n in (2, 3, 4)] == [False, False, False]


def test_completing_a_level_unlocks_the_next_only():
    store = fresh()
    state.complete_level(store, 1, xp=100, clue="217")
    assert state.status(store, 1) == "SOLVED"
    assert state.is_unlocked(store, 2)
    assert not state.is_unlocked(store, 3)
    assert store["clues_found"] == ["217"]
    assert store["current_level"] == 2


def test_xp_is_not_awarded_twice_but_best_score_improves():
    store = fresh()
    assert state.complete_level(store, 1, xp=100, score=0.6)
    assert not state.complete_level(store, 1, xp=100, score=0.9)
    assert store["xp"] == 100
    assert store["best_scores"][1] == 0.9


def test_demo_mode_unlocks_everything():
    store = fresh()
    store["demo_mode"] = True
    assert all(state.is_unlocked(store, n) for n in range(1, 5))


def test_out_of_range_levels_are_locked():
    store = fresh()
    store["demo_mode"] = True
    assert not state.is_unlocked(store, 0)
    assert not state.is_unlocked(store, 5)


def test_reset_keeps_demo_mode():
    store = fresh()
    store["demo_mode"] = True
    state.complete_level(store, 1, xp=100)
    state.reset_progress(store)
    assert store["completed_levels"] == set()
    assert store["xp"] == 0
    assert store["demo_mode"] is True


def test_all_solved():
    store = fresh()
    for n in range(1, 5):
        state.complete_level(store, n, xp=50)
    assert state.all_solved(store)
