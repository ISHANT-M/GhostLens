from game import codex


def test_right_answer_goes_to_guide_xp_once():
    store = {"xp": 30}
    qid, _, _, answer, _ = codex.QUIZ[0]
    codex.check_answer(store, qid, answer)
    codex.check_answer(store, qid, answer)
    assert store["guide_xp"] == codex.QUIZ_XP
    assert store["xp"] == 30
    assert store["quiz_correct"] == {qid}


def test_wrong_answer_gives_nothing():
    store = {}
    qid, _, options, answer, _ = codex.QUIZ[0]
    codex.check_answer(store, qid, (answer + 1) % len(options))
    assert store.get("guide_xp", 0) == 0 and store["quiz_feedback"][0] == "bad"


def test_forget_the_field_guide():
    store = {"seen_tips": {1}, "quiz_correct": {"q1"}, "guide_xp": 5, "xp": 10}
    codex.forget_guide(store)
    assert store == {"xp": 10}


def test_glossary_has_no_brand_names():
    assert not any("Jetson" in text for _, text in codex.GLOSSARY)


def test_chapter_quiz_ids_exist():
    for ids in codex.CHAPTER_QUIZ.values():
        assert ids and all(q in codex.QUIZ_BY_ID for q in ids)


def test_inline_ids_prefer_unanswered_and_stick():
    store = {"quiz_correct": {codex.CHAPTER_QUIZ[3][0]}}
    ids = codex.inline_ids(store, 3, 2)
    assert ids == codex.CHAPTER_QUIZ[3][1:3]
    store["quiz_correct"].add(ids[0])
    assert codex.inline_ids(store, 3, 2) == ids


def test_inline_answer_uses_its_own_feedback_and_guide_xp():
    store = {"xp": 10}
    qid = codex.CHAPTER_QUIZ[1][0]
    assert codex.check_answer(store, qid, codex.QUIZ_BY_ID[qid][3], feedback_key="l1_iq_x_feedback")
    assert store["guide_xp"] == codex.QUIZ_XP and store["xp"] == 10
    assert store["l1_iq_x_feedback"][0] == "ok" and "quiz_feedback" not in store


def test_inline_quiz_renders_and_pays():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("from game import codex\ncodex.inline_quiz(2, n=2)").run()
    assert len(at.radio) == 2
    qid = at.session_state["l2_iq_ids"][0]
    at.radio(key=f"l2_iq_{qid}").set_value(codex.QUIZ_BY_ID[qid][3]).run()
    next(b for b in at.button if b.key == f"l2_iq_{qid}_check").click().run()
    assert at.session_state["guide_xp"] == codex.QUIZ_XP and not at.exception


def test_field_guide_page_renders():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_string("from game import codex\ncodex.render()").run()
    assert not at.exception


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
