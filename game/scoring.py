"""XP and the Edge Engineering grade."""

BASE_XP = 100
QUALITY_BONUS = 50
ATTEMPT_PENALTY = 10
MIN_XP = 50
GRADE_BONUS = {"A": 50, "B": 25, "C": 0, "D": 0}


def level_xp(quality: float, attempts: int, grade: str | None = None) -> dict:
    # quality is 0..1 (evidence quality, F1, IoU...), attempts includes the successful one
    quality = min(max(quality, 0.0), 1.0)
    attempts = max(attempts, 1)
    bonus = round(QUALITY_BONUS * quality)
    penalty = ATTEMPT_PENALTY * (attempts - 1)
    edge = GRADE_BONUS.get(grade, 0)
    total = max(BASE_XP + bonus + edge - penalty, MIN_XP)
    return {"base": BASE_XP, "quality_bonus": bonus, "edge_bonus": edge, "attempt_penalty": -penalty, "total": total}


def edge_grade(checks: dict[str, bool]) -> str:
    """A = every check passed, then one letter down per failed check."""
    failed = sum(not ok for ok in checks.values())
    return "ABCD"[min(failed, 3)]


def efficient(used: int, optimal: int, slack: float = 1.5) -> bool:
    return used <= max(optimal * slack, optimal + 2)
