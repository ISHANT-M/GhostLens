"""Chapter 1, The Dark Frame: played headlessly, plus the pieces the cleared screen uses."""

from html import escape

import pytest

from cv.edge import energy_units
from game import case, device, level1
from tests.play import CASE, START, assert_play_clean, click, click_key, is_cleared, page, texts
from tests.play_ch12 import solve_ch1


def feedback_lines(at) -> list[str]:
    return [m.value for m in at.markdown if "gl-feedback" in m.value]


def enhance() -> object:
    return click_key(page(1), "l1_mode_Enhance")


@pytest.mark.parametrize("clue", case.CLUES, ids=lambda c: c["id"])
def test_every_pool_photo_solves_with_an_a(clue):
    seed = next(n for n in range(1000) if case.build_case(n).clue["id"] == clue["id"])
    at = page(1, seed)
    assert escape(clue["place"]) in texts(at)              # the story is on the title card
    at = solve_ch1(at, f" {clue['answer'].upper()} ")      # spaces and case don't matter
    assert not at.exception
    s = at.session_state
    assert 1 in s.completed_levels and s.grades[1] == "A"
    assert s.l1_xp["charge"] == device.A_GRADE_UNITS
    assert is_cleared(at, 1)
    assert f"Plate {clue['answer']} recovered at" in texts(at)
    assert s.l1_final["pipeline"]["gamma"] == case.REFERENCE["gamma"]


def test_clipping_over_the_limit_gives_a_b():
    at = enhance()
    for key, value in (("l1_gamma", 3.0), ("l1_contrast", 2.5), ("l1_brightness", 30)):
        at.slider(key=key).set_value(value)
    at.run()
    assert any("over the 1.0% limit" in m for m in feedback_lines(at))
    at.text_input[0].input(CASE.number)
    at = click(at, "Analyze clue")
    s = at.session_state
    assert 1 in s.completed_levels and 0.01 < s.l1_final["clip"] < 0.02
    assert s.l1_report["checks"]["Clipping under 1%"][0] is False
    # the 30 fps check is timed, so a busy machine can cost one more grade
    fast = s.l1_report["checks"]["Fast enough for live video"][0]
    assert s.grades[1] == ("B" if fast else "C")


def test_wrong_answer_is_rejected():
    at = solve_ch1(page(1), "999")
    assert 1 not in at.session_state.completed_levels
    assert any("isn't what the plate says" in m for m in feedback_lines(at))
    assert at.session_state.battery < START
    assert_play_clean(at)


def test_clipping_and_non_local_means_lines():
    at = enhance()
    at.slider(key="l1_gamma").set_value(2.4)
    at.slider(key="l1_contrast").set_value(6.0)
    at.run()
    assert any("pure black or pure white" in m for m in feedback_lines(at))

    at = enhance()
    at.radio(key="l1_section").set_value("Detail").run()
    at.selectbox(key="l1_denoise").set_value("Non-local means").run()
    assert any("Non-local means costs about" in m for m in feedback_lines(at))
    at.radio(key="l1_section").set_value("Tone").run()      # hidden widgets keep their values
    assert at.session_state.l1_lab["l1_denoise"] == "Non-local means"


def test_retrain_is_free_and_detect_costs():
    at = click_key(page(1), "l1_mode_Retrain")
    assert at.session_state.battery == START
    assert any("big machine before deployment" in m for m in feedback_lines(at))
    at = click_key(at, "l1_mode_Detect")
    assert at.session_state.battery < START
    assert any("Detector found" in m and "no number" in m for m in feedback_lines(at))
    assert any("YOLO26S" in m.value for m in at.markdown)
    at = click_key(at, "l1_mode_Enhance")
    assert not at.exception and at.slider(key="l1_gamma")


def test_presets_set_the_sliders():
    at = enhance()
    at.slider(key="l1_contrast").set_value(3.0).run()
    at = click_key(at, "l1_preset_gamma")
    assert at.slider(key="l1_gamma").value == 2.4 and at.slider(key="l1_contrast").value == 1.0
    at = click_key(at, "l1_preset_bright")
    assert at.slider(key="l1_brightness").value == 100 and at.slider(key="l1_gamma").value == 1.0
    assert any("Lighter, not clearer" in m for m in feedback_lines(at))
    at = click_key(at, "l1_preset_contrast")
    assert at.slider(key="l1_contrast").value == 4.0 and at.slider(key="l1_brightness").value == 0
    at = click_key(at, "l1_preset_reset")
    assert at.session_state.l1_lab == level1.DEFAULTS
    assert at.session_state.battery == START                # presets are tuning, not a forensic pass


def test_wipe_slider_is_free():
    at = enhance()
    before = len(at.session_state.ledger)
    at.slider(key="l1_wipe").set_value(0.1).run()
    at.slider(key="l1_wipe").set_value(0.9).run()
    assert at.session_state.battery == START and len(at.session_state.ledger) == before


def test_play_screens_are_clean():
    at = page(1)
    assert_play_clean(at)
    at = click_key(at, "l1_mode_Detect")
    assert_play_clean(at)
    at = click_key(at, "l1_mode_Enhance")
    assert_play_clean(at)
    assert "OBJECTIVE" in texts(at) and "Read the number on the plate." in texts(at)
    for section in level1.SECTIONS:
        at.radio(key="l1_section").set_value(section).run()
        assert_play_clean(at)


def test_cleared_shows_the_single_tool_numbers_and_the_debrief_table():
    at = solve_ch1(page(1), CASE.number)
    assert is_cleared(at, 1)
    text = texts(at)
    assert "One tool on its own" in text and " · best" in text
    q = level1.single_tool_quality(CASE.clue["id"])
    assert all(name in text for name in q)
    df = at.dataframe[0].value
    assert list(df.setup)[:2] == ["Yours", "Reference (gamma 2.4, contrast 2.5)"]
    assert "Image enhancement" in text and "measured" in text
    assert any(e.label == "The maths" for e in at.expander)
    assert any(b.key == "side_cam04" for b in at.button)     # the side scan lives on the Bonus scan tab


def test_back_to_the_scene_and_report():
    at = solve_ch1(page(1), CASE.number)
    at = click_key(at, "l1_back")
    assert not is_cleared(at, 1) and at.session_state.l1_review
    assert any(b.key == "l1_report_btn" for b in at.button)
    assert not any(b.key == "l1_submit" for b in at.button)
    at = click_key(at, "l1_report_btn")
    assert is_cleared(at, 1)


def test_side_scan_uses_the_final_settings():
    at = solve_ch1(page(1), CASE.number)
    at = click_key(at, "side_cam04")
    assert not at.exception
    assert at.session_state.side_scans["cam04"] is True
    assert at.session_state.l1_cam04["settings"].gamma == case.REFERENCE["gamma"]


# units

def test_best_tools():
    assert level1.best_tools({"a": 0.3, "b": 0.6, "c": 0.1}) == {1}
    assert level1.best_tools({"a": 0.5, "b": 0.2, "c": 0.5}) == {0, 2}


@pytest.mark.parametrize("clue", case.CLUES, ids=lambda c: c["id"])
def test_gamma_is_the_best_single_tool(clue):
    q = level1.single_tool_quality(clue["id"])
    assert list(q)[min(level1.best_tools(q))] == "Gamma 2.4"
    assert q["Brightness +100"] < 0.2


def test_clip_tone():
    assert level1.clip_tone(0.009) == ""
    assert level1.clip_tone(0.015) == "warn"
    assert level1.clip_tone(0.031) == "bad"


def test_debrief_frame():
    runs = [{"setup": "ref", "quality": 0.85, "clip": 0.0, "ms": 2.0}]
    df = level1.debrief_frame({"quality": 0.9, "clip": 0.002, "ms": 3.0}, runs)
    assert list(df.setup) == ["Yours", "ref"]
    assert list(df.clipped) == ["0.2%", "0.0%"]
    assert list(df.battery) == [device.pct(energy_units(3.0)), device.pct(energy_units(2.0))]
    assert list(level1.debrief_frame(None, runs).setup) == ["ref"]


def test_describe_and_settings_from():
    s = level1.settings_from({"l1_gamma": 2.4, "l1_contrast": 2.5, "l1_equalizer": "CLAHE"})
    assert level1.describe(s) == "gamma 2.4, contrast ×2.5, CLAHE clip 2"
    assert level1.describe(level1.settings_from({})) == "nothing switched on"


def test_par_is_timed_once_and_stored():
    at = solve_ch1(page(1), CASE.number)
    s = at.session_state
    par = s.l1_par
    report = s.l1_report["checks"]["Battery"][1]
    assert report.endswith(f"{device.pct(par)} would have done it")
    for _ in range(3):                                     # re-rendering the cleared screen doesn't re-time it
        at.run()
        assert at.session_state.l1_par == par
    assert f"<tr><td>Par</td><td>{device.pct(par)}</td></tr>" in texts(at)
