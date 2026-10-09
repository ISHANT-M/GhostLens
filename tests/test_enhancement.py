import itertools
from pathlib import Path

import numpy as np
import pytest

from cv import enhancement as enh
from game import case
from game.achievements import CLEAN_FRAME
from game.level1 import CAM03, CAM04

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level1"
POOL = [(clue, cam) for clue in case.CLUES for cam in (CAM03, CAM04)]
IDS = [f"{clue['id']}-{cam[1][:6].replace(' ', '')}" for clue, cam in POOL]


def load(clue: dict, cam: tuple[int, str] = CAM03) -> tuple[np.ndarray, np.ndarray]:
    ref = enh.load_image(ASSETS / clue["source"], crop=clue["crop"])
    return ref, enh.make_dark_frame(ref, seed=cam[0], stamp=cam[1])


def quality(ref: np.ndarray, dark: np.ndarray, box: list[float], settings: enh.Settings) -> tuple[float, float]:
    """(evidence quality, newly clipped share), measured the same way as chapter 1."""
    out, _ = enh.run_pipeline(dark, settings)
    clip = max(0.0, sum(enh.clipped(out)) - sum(enh.clipped(dark)))
    return enh.evidence_quality(enh.legibility(out, ref, box), clip), clip


@pytest.fixture(scope="module", params=POOL, ids=IDS)
def frames(request):
    clue, cam = request.param
    ref, dark = load(clue, cam)
    return clue, cam, ref, dark


def test_missing_image_raises():
    with pytest.raises(FileNotFoundError):
        enh.load_image("does/not/exist.jpg")


def test_pool_data():
    answers = [c["answer"].lower() for c in case.CLUES]
    assert len(set(answers)) == len(answers) >= 3
    assert len({c["id"] for c in case.CLUES}) == len(case.CLUES)
    for c in case.CLUES:
        x0, y0, x1, y1 = c["clue_box"]
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 0.9       # the bottom 10% is the camera stamp
        assert 0.005 <= (x1 - x0) * (y1 - y0) <= 0.03
        assert c["label"] and c["place"] and c["floor"] and (ASSETS / c["source"]).exists()


def test_crop_is_applied_before_resizing():
    clue = next(c for c in case.CLUES if c["crop"])
    full, cropped = enh.load_image(ASSETS / clue["source"]), enh.load_image(ASSETS / clue["source"], crop=clue["crop"])
    assert full.shape[1] == cropped.shape[1] == 960
    assert cropped.shape[0] < full.shape[0]


def test_dark_frame_is_dark_and_deterministic(frames):
    clue, cam, ref, dark = frames
    assert enh.gray(dark).mean() < 15
    assert np.array_equal(dark, enh.make_dark_frame(ref, seed=cam[0], stamp=cam[1]))


def test_reference_settings_pass_without_clipping(frames):
    clue, _, ref, dark = frames
    good, clip = quality(ref, dark, clue["clue_box"], enh.Settings(**case.REFERENCE))
    assert good >= case.PASS_QUALITY
    assert clip <= 0.01


def test_bad_settings_score_lower(frames):
    clue, _, ref, dark = frames
    box = clue["clue_box"]
    good, _ = quality(ref, dark, box, enh.Settings(**case.REFERENCE))
    assert quality(ref, dark, box, enh.Settings())[0] < 0.2                  # untouched dark frame
    assert quality(ref, dark, box, enh.Settings(brightness=120))[0] < 0.2    # brightness alone does nothing
    assert quality(ref, dark, box, enh.Settings(gamma=2.4, contrast=4))[0] < good      # clipping is punished
    assert quality(ref, dark, box, enh.Settings(gamma=2.4, contrast=2.5, sharpen=3))[0] < good


def test_cheap_tuning_reaches_a_clean_frame(frames):
    clue, _, ref, dark = frames
    gammas = np.arange(1.6, 3.41, 0.2)
    contrasts = np.arange(1.5, 4.01, 0.5)
    best = max(quality(ref, dark, clue["clue_box"], enh.Settings(gamma=g, contrast=k, equalizer=e))[0]
               for g, k, e in itertools.product(gammas, contrasts, ["None", "CLAHE"]))
    assert best >= CLEAN_FRAME


def test_gamma_one_is_identity():
    _, dark = load(case.CLUES[0])
    assert np.array_equal(enh.gamma_correct(dark, 1.0), dark)


def test_brightness_contrast_clips():
    img = np.full((4, 4, 3), 200, np.uint8)
    out = enh.brightness_contrast(img, brightness=100)
    assert out.max() == 255 and out.dtype == np.uint8
    assert enh.clipped(out) == (0.0, 1.0)


def test_histogram_sums_to_one():
    assert enh.histogram(load(case.CLUES[0])[1]).sum() == pytest.approx(1.0)


def test_pipeline_reports_timings():
    _, dark = load(case.CLUES[0])
    out, timings = enh.run_pipeline(dark, enh.Settings(gamma=2.0, equalizer="CLAHE"))
    assert out.shape == dark.shape
    assert [name for name, _ in timings] == ["Gamma", "CLAHE"]
    assert all(ms >= 0 for _, ms in timings)


def test_random_augment_keeps_shape():
    ref, _ = load(case.CLUES[0])
    rng = np.random.default_rng(0)
    for _ in range(10):
        out, applied = enh.random_augment(ref, rng)
        assert out.shape == ref.shape and applied


def test_wipe_splits_before_and_after():
    before = np.zeros((10, 20, 3), np.uint8)
    after = np.full((10, 20, 3), 200, np.uint8)
    out = enh.wipe(before, after, 0.5)
    assert out.shape == before.shape
    assert (out[:, :9] == 0).all() and (out[:, 12:] == 200).all()
    assert tuple(out[5, 10]) == enh.BRASS                         # the divider
    assert np.array_equal(enh.wipe(before, after, 0.0), after)    # no divider at the edges
    assert np.array_equal(enh.wipe(before, after, 1.0), before)


def test_histogram_strip_marks_clipping():
    _, dark = load(case.CLUES[0])
    strip = enh.histogram_strip(dark, enh.gamma_correct(dark, 2.4), width=400, height=100)
    assert strip.shape == (100, 400, 3) and strip.dtype == np.uint8
    tall = enh.histogram_strip(dark, enh.brightness_contrast(dark, contrast=20), width=400, height=100)

    def white_tick(img):
        return np.all(img[:, 395:] == enh.CLIP_RED, axis=2).sum()
    assert white_tick(tall) > white_tick(strip) > 0               # more pixels at 255, taller tick
