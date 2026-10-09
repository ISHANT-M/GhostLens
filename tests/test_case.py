"""The case file: one seed decides everything, and every option in the pool gets used."""

from game import case

SEEDS = range(300)


def test_same_seed_same_case():
    for seed in (0, 7, 123_456):
        assert case.build_case(seed) == case.build_case.__wrapped__(seed)


def test_every_option_shows_up():
    cases = [case.build_case(n) for n in SEEDS]
    assert {c.clue["id"] for c in cases} == {c["id"] for c in case.CLUES}
    assert {c.cam04["id"] for c in cases} == {c["id"] for c in case.CLUES}
    assert {c.anchor for c in cases} == set(case.ANCHORS)
    assert {c.brief for c in cases} == set(case.BRIEFS)
    assert {c.moved for c in cases} == set(case.MOVABLE)
    assert {c.wall_seed for c in cases} == set(case.WALL_SEEDS)
    assert {c.wall2_seed for c in cases} == set(case.WALL_SEEDS)


def test_each_case_is_consistent():
    for n in SEEDS:
        c = case.build_case(n)
        assert c.cam04["id"] != c.clue["id"]
        assert c.wall2_seed != c.wall_seed
        assert sorted(c.evidence_order) == sorted(case.ANCHORS)
        assert c.number == c.clue["answer"]


def test_wall_217_is_not_in_the_pool():
    # U-Net Lite gets 0.967 IoU on that wall, which would make the light model look good enough
    assert 217 not in case.WALL_SEEDS


def test_get_case_draws_a_seed_once_and_reset_drops_it():
    store = {}
    first = case.get_case(store)
    assert store["case_seed"] == first.seed
    assert case.get_case(store) is first
    case.reset_case(store)
    assert "case_seed" not in store
    seeds = set()
    for _ in range(5):
        case.reset_case(store)
        seeds.add(case.get_case(store).seed)
    assert len(seeds) > 1


def test_case_label():
    assert case.case_label({"case_seed": 30217}) == "CASE 0217"
    assert case.case_label({"case_seed": 7}) == "CASE 0007"


def test_anchor_label_sets_do_not_overlap():
    names = list(case.ANCHORS)
    for a in names:
        for b in names:
            if a != b:
                assert not case.ANCHORS[a]["labels"] & case.ANCHORS[b]["labels"]


def test_briefs():
    seven_of_nine = {"tp": 7, "fp": 4, "fn": 0, "precision": 7 / 11, "recall": 1.0, "f1": 7 / 9}
    exactly = {"tp": 6, "fp": 3, "fn": 1, "precision": 6 / 9, "recall": 6 / 7, "f1": 2 * 6 / (2 * 6 + 3 + 1)}
    loose = {"tp": 7, "fp": 9, "fn": 0, "precision": 7 / 16, "recall": 1.0, "f1": 14 / 23}
    assert case.brief_passes("inventory", seven_of_nine) and case.brief_passes("miss_nothing", seven_of_nine)
    assert case.brief_passes("inventory", exactly)                 # F1 is exactly 0.75
    assert not case.brief_passes("miss_nothing", exactly)          # one cup missed
    assert not case.brief_passes("inventory", loose) and not case.brief_passes("miss_nothing", loose)
    for b in case.BRIEFS.values():
        assert b["title"] and b["rule"] and 0.05 <= b["start"] <= 0.95
