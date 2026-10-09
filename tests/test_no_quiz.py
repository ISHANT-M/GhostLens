"""No quiz is left anywhere in the game: no locked-in guesses, no quiz bank, no guide XP."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BANNED = ["Lock in", "QUIZ", "inline_quiz", "flow.predict", "guide_xp", "Quick check", "index=None"]
# the chapter 3 box inspector starts with nothing picked; the field guide clears old pre-0.8 session keys
ALLOWED = {("index=None", "game/level3.py"), ("guide_xp", "game/codex.py")}


def sources() -> list[Path]:
    return sorted(p for folder in ("game", "ui") for p in (ROOT / folder).rglob("*")
                  if p.suffix in (".py", ".css"))


@pytest.mark.parametrize("word", BANNED)
def test_no_quiz_words_in_the_game(word):
    hits = [str(p.relative_to(ROOT)) for p in sources() if word in p.read_text()]
    assert [h for h in hits if (word, h) not in ALLOWED] == []


def test_app_has_no_quiz_or_sidebar():
    text = (ROOT / "app.py").read_text()
    assert "guide_xp" not in text and "GUIDE XP" not in text and "st.sidebar" not in text
