"""
Bit-identity guard for `blit_sprite`'s cached alpha composite.

`blit_sprite` memoises the frame-independent half of the blend — `alpha * rgb`
and `1 - alpha` — for any sprite that came out of the resize cache. That is a
speed change and must never be a pixel change, so every test here compares the
cached path against the arithmetic it replaced, byte for byte, rather than
within a tolerance.
"""

import unittest

import numpy as np

from visual_ai.imaging import blit_sprite, invalidate_sprite_cache, resize, to_rgba


def reference(frame, sprite, cx, cy, size=None):
    """
    The pre-cache composite, transcribed.

    Scaling goes through the engine's own `resize` — the point here is to pin
    the blend, and reimplementing the premultiplied resample would only test
    that reimplementation. Operates on a copy.
    """
    out = frame.copy()
    sprite = to_rgba(sprite)
    if size is not None and (size != sprite.shape[0] or size != sprite.shape[1]):
        sprite = resize(sprite, size, size)

    height, width = sprite.shape[:2]
    y0, x0 = int(cy) - height // 2, int(cx) - width // 2
    fy0, fy1 = max(0, y0), min(out.shape[0], y0 + height)
    fx0, fx1 = max(0, x0), min(out.shape[1], x0 + width)
    if fy0 >= fy1 or fx0 >= fx1:
        return out

    patch = sprite[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    target = out[fy0:fy1, fx0:fx1]
    alpha = (patch[..., 3:4].astype(np.float32)) / 255.0
    out[fy0:fy1, fx0:fx1] = (
        alpha * patch[..., :3] + (1.0 - alpha) * target).astype(out.dtype)
    return out


class TestCachedBlitIsBitIdentical(unittest.TestCase):

    def setUp(self):
        invalidate_sprite_cache()
        rng = np.random.default_rng(20260820)
        self.frame = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        self.sprite = rng.integers(0, 256, (48, 48, 4), dtype=np.uint8)

    def _check(self, cx, cy, size):
        want = reference(self.frame, self.sprite, cx, cy, size)
        got = self.frame.copy()
        blit_sprite(got, self.sprite, cx, cy, size=size)
        np.testing.assert_array_equal(got, want)
        return got

    def test_matches_the_uncached_arithmetic(self):
        self._check(80, 60, 32)

    def test_second_blit_takes_the_cache_and_still_matches(self):
        # The first call populates the blend cache; the second is the one the
        # optimisation is for, and is the one that could drift.
        first = self._check(80, 60, 32)
        second = self._check(80, 60, 32)
        np.testing.assert_array_equal(first, second)

    def test_clipping_at_every_edge(self):
        # A clipped sprite slices the cached layers rather than the RGBA
        # buffer, which is where an off-by-one would hide.
        for cx, cy in ((0, 60), (159, 60), (80, 0), (80, 119),
                       (2, 2), (158, 118)):
            with self.subTest(cx=cx, cy=cy):
                invalidate_sprite_cache()
                self._check(cx, cy, 32)          # cold
                self._check(cx, cy, 32)          # warm

    def test_several_sizes_of_one_sprite(self):
        for size in (16, 24, 32, 48, 64):
            with self.subTest(size=size):
                self._check(80, 60, size)
                self._check(80, 60, size)

    def test_rotated_blits_do_not_use_the_cache(self):
        # A rotation allocates a fresh array every call, so caching it would
        # both miss and grow without bound. Drawing rotated then unrotated at
        # the same size must still give the unrotated answer.
        rotated = self.frame.copy()
        blit_sprite(rotated, self.sprite, 80, 60, size=32, angle=37.0)
        self.assertFalse(np.array_equal(rotated, self.frame))
        self._check(80, 60, 32)

    def test_invalidate_clears_the_blend_layers_too(self):
        self._check(80, 60, 32)
        invalidate_sprite_cache(self.sprite)
        self._check(80, 60, 32)

    def test_fully_transparent_sprite_leaves_the_frame_alone(self):
        clear = self.sprite.copy()
        clear[..., 3] = 0
        got = self.frame.copy()
        blit_sprite(got, clear, 80, 60, size=32)
        blit_sprite(got, clear, 80, 60, size=32)
        np.testing.assert_array_equal(got, self.frame)


if __name__ == "__main__":
    unittest.main()
