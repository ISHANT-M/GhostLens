from streamlit.testing.v1 import AppTest

from game import codex


def test_forget_the_field_guide():
    store = {"seen_tips": {1}, "quiz_correct": {"q1"}, "guide_xp": 5, "quiz_q3": 1, "xp": 10}
    codex.forget_guide(store)
    assert store == {"xp": 10}


def test_glossary_has_no_brand_names():
    assert not any("Jetson" in text for _, text in codex.GLOSSARY)


def test_no_quiz_left_in_the_guide():
    assert not any(hasattr(codex, name) for name in ("QUIZ", "QUIZ_BY_ID", "CHAPTER_QUIZ", "inline_quiz",
                                                       "check_answer", "quiz_tab"))


def test_field_guide_has_three_tabs_and_no_quiz():
    at = AppTest.from_string("from game import codex\ncodex.render()").run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Tips found", "Glossary", "Badges"]
    assert not at.radio and not any(b.label in ("Check", "Skip") for b in at.button)


def test_forget_button_clears_the_tips():
    at = AppTest.from_string("from game import codex\ncodex.render()")
    at.session_state["seen_tips"] = {0, 1}
    at.run()
    at.button(key="forget_guide").click().run()
    assert "seen_tips" not in at.session_state and not at.exception


# loading screen tips

from game import tips  # noqa: E402


def test_chapter_topics_exist():
    topics = {t for t, _ in tips.TIPS}
    assert all(t in topics for ts in tips.CHAPTER_TOPICS.values() for t in ts)


def test_pick_tip_never_shows_this_or_a_later_chapter():
    for chapter in (1, 2, 3, 4):
        banned = {t for n, ts in tips.CHAPTER_TOPICS.items() if n >= chapter for t in ts}
        for _ in range(50):
            assert tips.pick_tip(chapter, set())[1] not in banned


def test_pick_tip_prefers_the_previous_chapter():
    for _ in range(30):
        assert tips.pick_tip(3, set())[1] in tips.CHAPTER_TOPICS[2]


def test_pick_tip_moves_on_when_previous_chapter_is_seen():
    seen = {i for i, (t, _) in enumerate(tips.TIPS) if t in tips.CHAPTER_TOPICS[2]}
    i, topic, _ = tips.pick_tip(3, seen)
    assert i not in seen and topic not in tips.CHAPTER_TOPICS[3] + tips.CHAPTER_TOPICS[4]


def test_power_tip_says_it_is_a_game_rule():
    text = next(t for topic, t in tips.TIPS if topic == "Power")
    assert "game rule" in text and "twice the battery" not in text
