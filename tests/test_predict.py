from streamlit.testing.v1 import AppTest

SCRIPT = """
import streamlit as st
from game import flow
st.session_state.setdefault("xp", 0)
flow.predict("l9_pred_demo", "Which is bigger?", ["two", "three"],
             resolve=lambda: (1, "three is bigger."), reveal=lambda: st.write("revealed"))
"""


def run(choice: str) -> AppTest:
    at = AppTest.from_string(SCRIPT).run()
    at.radio[0].set_value(choice).run()
    next(b for b in at.button if b.label == "Lock in").click()
    return at.run()


def test_right_guess_pays_once():
    at = run("three")
    assert at.session_state.xp == 5
    at.run()
    assert at.session_state.xp == 5
    assert any("revealed" in m.value for m in at.markdown)


def test_wrong_guess_is_free_and_explained():
    at = run("two")
    assert at.session_state.xp == 0
    record = at.session_state["l9_pred_demo"]
    assert record["right"] is False and record["correct"] == [1]
    assert any("three is bigger" in m.value for m in at.markdown)


def test_predictions_lists_only_records():
    from game.flow import predictions
    store = {"l1_pred_a": {"choice": 0}, "l1_pred_a_choice": "x", "xp": 3}
    assert predictions(store) == [{"choice": 0}]
