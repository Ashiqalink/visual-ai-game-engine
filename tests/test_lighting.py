"""Tests for imaging.normalize_lighting / clipped_fraction.

Synthetic inputs with a known defect — a colour cast, a shoulder of highlights
piling up near white, a flat side-lit face — so "did it help" is measurable
rather than a matter of taste.
"""

import numpy as np
import pytest

from visual_ai.imaging import clipped_fraction, normalize_lighting


def gradient(height=64, width=64, low=40, high=255):
    """A left-to-right ramp, as three identical channels."""
    ramp = np.linspace(low, high, width, dtype=np.float32)
    return np.repeat(np.tile(ramp, (height, 1))[..., None], 3, axis=2).astype(np.uint8)


def test_clipped_fraction_counts_only_all_channel_white():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:5] = 255            # blown
    image[5:, :, 0] = 255      # red channel only — not blown
    assert clipped_fraction(image) == pytest.approx(0.5)


def test_clipped_fraction_respects_a_mask():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:5] = 255
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5:] = 255  # look only at the half that is not blown

    assert clipped_fraction(image, mask=mask) == pytest.approx(0.0)


def test_clipped_fraction_of_an_empty_mask_is_zero():
    image = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert clipped_fraction(image, mask=np.zeros((10, 10), np.uint8)) == 0.0


def test_highlights_stop_piling_up_at_white():
    blown = np.clip(gradient().astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
    before = clipped_fraction(blown)
    after = clipped_fraction(normalize_lighting(blown)[..., :3])

    assert before > 0.2          # the input really is blown out
    assert after < before / 2    # and the correction pulls most of it back


def test_the_rolloff_leaves_midtones_alone():
    image = gradient(low=0, high=255)
    fixed = normalize_lighting(image, white_balance=False, clip_limit=0.0)

    dark = image[..., 0] < 0.7 * 255
    assert np.array_equal(fixed[..., :3][dark], image[dark])


def test_a_colour_cast_is_pulled_back_toward_neutral():
    grey = np.full((64, 64, 3), 120, dtype=np.uint8)
    tinted = grey.copy()
    tinted[..., 0] = 170  # a red cast over an otherwise neutral scene

    fixed = normalize_lighting(tinted, clip_limit=0.0, highlight_knee=1.0)
    spread_before = float(np.ptp(tinted.reshape(-1, 3).mean(axis=0)))
    spread_after = float(np.ptp(fixed[..., :3].reshape(-1, 3).mean(axis=0)))

    assert spread_after < spread_before


def test_a_flat_white_patch_stops_being_white():
    # The complaint this function exists for: a bright region that reads as a
    # hole punched in the sprite rather than as a lit surface.
    flat = np.full((64, 64, 3), 252, dtype=np.uint8)
    flat[20:44, 20:44] = 255

    fixed = normalize_lighting(flat)[..., :3]

    assert clipped_fraction(flat) == pytest.approx(1.0)  # every pixel was white
    assert clipped_fraction(fixed) == 0.0                # none of them still is
    assert fixed.max() < 250


def test_bright_shading_stays_ordered_after_the_pull_down():
    # A flat patch has no shading to keep, so structure is checked on a ramp:
    # brighter must stay brighter, or the fix has flattened the subject.
    ramp = gradient(low=170, high=255)[..., :3]

    fixed = normalize_lighting(ramp)[..., :3].astype(np.float32).mean(axis=2)
    row = fixed[32]

    assert row[-1] > row[len(row) // 2] > row[0]


def test_alpha_survives_untouched():
    rgba = np.dstack([gradient(), np.linspace(0, 255, 64, dtype=np.uint8)
                      .repeat(64).reshape(64, 64)])
    fixed = normalize_lighting(rgba)

    assert fixed.shape == rgba.shape
    assert np.array_equal(fixed[..., 3], rgba[..., 3])


def test_zero_strength_is_the_identity():
    image = gradient()
    assert np.array_equal(normalize_lighting(image, strength=0.0)[..., :3], image)


def test_a_mask_confines_the_correction():
    image = np.clip(gradient().astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[:32] = 255

    fixed = normalize_lighting(image, mask=mask)[..., :3]

    assert np.array_equal(fixed[32:], image[32:])   # outside the mask: untouched
    assert not np.array_equal(fixed[:32], image[:32])


def test_the_mask_also_decides_what_is_measured():
    # Left half neutral grey, right half a heavy blue cast. Measuring the whole
    # frame would chase the cast; masked to the left, there is nothing to fix.
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    image[:, 32:, 2] = 250
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[:, :32] = 255

    fixed = normalize_lighting(image, mask=mask, clip_limit=0.0, highlight_knee=1.0)

    assert np.array_equal(fixed[..., :3][:, :32], image[:, :32])


def test_a_mismatched_mask_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        normalize_lighting(gradient(), mask=np.zeros((8, 8), np.uint8))
