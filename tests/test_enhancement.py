import json
from pathlib import Path

import numpy as np
import pytest

from cv import enhancement as enh

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level1"
CLUE = json.loads((ASSETS / "clue.json").read_text())


@pytest.fixture(scope="module")
def frames():
    ref = enh.load_image(ASSETS / CLUE["source"])
    return ref, enh.make_dark_frame(ref)


def test_missing_image_raises():
    with pytest.raises(FileNotFoundError):
        enh.load_image("does/not/exist.jpg")


def test_dark_frame_is_dark_and_deterministic(frames):
    ref, dark = frames
    assert enh.gray(dark).mean() < 15
    assert np.array_equal(dark, enh.make_dark_frame(ref))


def test_gamma_one_is_identity(frames):
    _, dark = frames
    assert np.array_equal(enh.gamma_correct(dark, 1.0), dark)


def test_brightness_contrast_clips():
    img = np.full((4, 4, 3), 200, np.uint8)
    out = enh.brightness_contrast(img, brightness=100)
    assert out.max() == 255 and out.dtype == np.uint8
    assert enh.clipped(out) == (0.0, 1.0)


def test_histogram_sums_to_one(frames):
    assert enh.histogram(frames[1]).sum() == pytest.approx(1.0)


def test_good_settings_beat_bad_ones(frames):
    ref, dark = frames
    box = CLUE["clue_box"]
    base = sum(enh.clipped(dark))

    def quality(settings):
        out, _ = enh.run_pipeline(dark, settings)
        return enh.evidence_quality(enh.legibility(out, ref, box), max(0, sum(enh.clipped(out)) - base))

    good = quality(enh.Settings(gamma=2.4, contrast=2.5))
    assert good >= CLUE["pass_quality"]
    assert quality(enh.Settings()) < 0.2                          # untouched dark frame
    assert quality(enh.Settings(brightness=120)) < 0.2            # brightness alone does nothing
    assert quality(enh.Settings(gamma=2.4, contrast=4)) < good    # clipping is punished
    assert quality(enh.Settings(gamma=2.4, contrast=2.5, sharpen=3)) < good


def test_pipeline_reports_timings(frames):
    _, dark = frames
    out, timings = enh.run_pipeline(dark, enh.Settings(gamma=2.0, equalizer="CLAHE"))
    assert out.shape == dark.shape
    assert [name for name, _ in timings] == ["Gamma", "CLAHE"]
    assert all(ms >= 0 for _, ms in timings)


def test_random_augment_keeps_shape(frames):
    ref, _ = frames
    rng = np.random.default_rng(0)
    for _ in range(10):
        out, applied = enh.random_augment(ref, rng)
        assert out.shape == ref.shape and applied
