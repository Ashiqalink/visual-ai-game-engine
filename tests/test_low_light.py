"""LowLightBoost: it must lift dark frames and stay out of the way otherwise."""

import numpy as np
import pytest

from visual_ai.low_light import (
    DARK_LUMA,
    MAX_GAIN,
    LowLightBoost,
    measure_luma,
)


def flat_frame(level, size=(120, 160)):
    """A uniform grey BGR frame at the given 0-255 level."""
    return np.full((size[0], size[1], 3), level, dtype=np.uint8)


def textured_frame(level, size=(120, 160), seed=0):
    """Grey frame with a brighter blob -- something CLAHE can act on."""
    rng = np.random.default_rng(seed)
    frame = np.full((size[0], size[1], 3), level, dtype=np.uint8)
    frame[30:90, 40:120] = min(255, int(level * 1.6))
    noise = rng.integers(-3, 4, frame.shape, dtype=np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def settle(boost, frame, frames=60):
    """Run the same frame through until the slewed gain converges."""
    out = frame
    for _ in range(frames):
        out = boost.apply(frame)
    return out


class TestMeasureLuma:
    def test_matches_full_frame_mean_on_flat_input(self):
        assert measure_luma(flat_frame(64)) == pytest.approx(64.0, abs=0.5)

    def test_subsampling_is_close_to_the_full_measure(self):
        frame = textured_frame(70, seed=1)
        assert measure_luma(frame, stride=8) == pytest.approx(
            measure_luma(frame, stride=1), abs=2.0)


class TestIdleOnWellLitFrames:
    def test_bright_frame_is_returned_untouched(self):
        boost = LowLightBoost()
        frame = textured_frame(150)
        out = settle(boost, frame)
        # Identity, not just equality: a well-lit stream must not pay a copy.
        assert out is frame
        assert boost.gain == 1.0
        assert boost.active is False

    def test_luma_is_reported_even_when_idle(self):
        boost = LowLightBoost()
        boost.apply(textured_frame(150))
        assert boost.luma == pytest.approx(measure_luma(textured_frame(150)),
                                           abs=1.0)

    def test_frame_just_above_the_threshold_is_left_alone(self):
        boost = LowLightBoost()
        out = settle(boost, flat_frame(int(DARK_LUMA) + 5))
        assert boost.gain == 1.0
        assert out is not None


class TestBoostsDarkFrames:
    def test_dark_frame_gets_brighter(self):
        boost = LowLightBoost()
        frame = textured_frame(25)
        out = settle(boost, frame)
        assert boost.active is True
        assert boost.gain > 1.5
        assert measure_luma(out) > measure_luma(frame) * 1.5

    def test_gain_is_capped(self):
        boost = LowLightBoost()
        settle(boost, flat_frame(2))
        assert boost.gain <= MAX_GAIN + 1e-6

    def test_black_frame_does_not_divide_by_zero(self):
        boost = LowLightBoost()
        out = settle(boost, flat_frame(0))
        assert out.shape == (120, 160, 3)
        assert np.isfinite(boost.gain)

    def test_output_keeps_shape_and_dtype(self):
        boost = LowLightBoost()
        frame = textured_frame(20)
        out = settle(boost, frame)
        assert out.shape == frame.shape
        assert out.dtype == np.uint8

    def test_highlights_saturate_rather_than_wrap(self):
        # A wrapped highlight turns the brightest part of a hand black, which
        # is worse for the detector than the underexposure being fixed.
        boost = LowLightBoost()
        frame = textured_frame(20)
        frame[0:10, 0:10] = 250
        out = settle(boost, frame)
        assert out[0:10, 0:10].mean() > 200

    def test_local_contrast_improves(self):
        # The blob against its background is what the detector keys on, so
        # the gap between them is the number that has to grow.
        boost = LowLightBoost()
        frame = textured_frame(25)
        out = settle(boost, frame)
        before = float(frame[30:90, 40:120].mean() - frame[0:20, 0:20].mean())
        after = float(out[30:90, 40:120].mean() - out[0:20, 0:20].mean())
        assert after > before


class TestSlew:
    def test_gain_ramps_rather_than_snapping(self):
        boost = LowLightBoost()
        dark = textured_frame(20)
        boost.apply(dark)
        first = boost.gain
        settled = settle(boost, dark).mean()
        assert 1.0 < first < boost.gain, "gain should still be climbing"
        assert settled > 0

    def test_returns_to_unity_when_the_lights_come_back(self):
        boost = LowLightBoost()
        settle(boost, textured_frame(20))
        assert boost.gain > 1.0
        out = settle(boost, textured_frame(160))
        assert boost.gain == 1.0
        assert boost.active is False
        assert out is not None

    def test_reset_clears_state(self):
        boost = LowLightBoost()
        settle(boost, textured_frame(20))
        boost.reset()
        assert boost.gain == 1.0
        assert boost.active is False


class TestColour:
    def test_chroma_is_untouched(self):
        # Gain and CLAHE both act on Y alone, so U and V come back unchanged.
        # That is the invariant worth testing: a skin tone whose chroma drifts
        # is a skin tone the hand model was not trained on. (The BGR *ratios*
        # do compress -- lifting Y adds the same amount to every channel --
        # which is what raising exposure on a real camera does too.)
        import cv2
        boost = LowLightBoost()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :] = (30, 45, 70)          # dim BGR, warm
        out = settle(boost, frame)
        before = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        after = cv2.cvtColor(out, cv2.COLOR_BGR2YUV)
        assert float(after[:, :, 1].mean()) == pytest.approx(
            float(before[:, :, 1].mean()), abs=2.0)
        assert float(after[:, :, 2].mean()) == pytest.approx(
            float(before[:, :, 2].mean()), abs=2.0)
        assert float(after[:, :, 0].mean()) > float(before[:, :, 0].mean())

    def test_channel_ordering_is_preserved(self):
        boost = LowLightBoost()
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :] = (30, 45, 70)
        out = settle(boost, frame)
        b, g, r = [float(out[:, :, i].mean()) for i in range(3)]
        assert r > g > b
