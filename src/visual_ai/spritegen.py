"""
spritegen.py — creature sprites as data, not code.

A game should not carry six hundred lines of hard-coded polygon coordinates per
character. Here a creature is a :class:`CreatureSpec` — a small bundle of
colours, proportions and feature toggles — and :func:`render_creature` turns it
into an RGBA sprite. Front and side views come from the same spec, so the two
can never drift out of sync, and adding a character is a dict rather than a new
``_draw_whatever`` function.

Rendering is signed-distance-field rasterisation in pure NumPy. That buys three
things worth having:

* **No new dependency.** NumPy is already required by the pipeline. Pillow and
  OpenCV are not needed to produce artwork.
* **Real antialiasing at native size.** Coverage is computed from the distance
  field itself, so edges are exact — no 4x supersample-and-shrink, which is
  what the previous approach paid for much softer results.
* **Resolution independence.** Every measurement is in sprite-local units on
  ``[-1, 1]``, so the same spec renders correctly at 64px or 4096px.

Colours are RGB. The rest of this repo is largely BGR because it grew out of
OpenCV; :func:`visual_ai.imaging.bgr_to_rgb` converts at the boundary.

    >>> from visual_ai import CreatureSpec, render_creature
    >>> sprite = render_creature(CreatureSpec(name="ruby"), view="side", size=192)
    >>> sprite.shape
    (192, 192, 4)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np

__all__ = [
    "CreatureSpec",
    "render_creature",
    "render_views",
    "DEFAULT_CAST",
    "spec_from_dict",
    "spec_to_dict",
    "BODY_SHAPES",
    "VIEWS",
]

RGB = tuple[int, int, int]
View = Literal["front", "side"]

BODY_SHAPES = ("round", "oval", "tall", "wedge")
VIEWS = ("front", "side")


# ── Signed distance fields ────────────────────────────────────────────────────
#
# Every function returns distance in sprite-local units: negative inside the
# shape, positive outside, zero on the boundary. Distances for the ellipse and
# the smooth-union operator are approximations, which is fine — they are only
# ever consumed by a coverage threshold a fraction of a pixel wide.

def _grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates spanning [-1, 1], y increasing downward."""
    axis = (np.arange(size, dtype=np.float32) + 0.5) / size * 2.0 - 1.0
    return np.meshgrid(axis, axis)


def _sd_circle(x, y, cx: float, cy: float, r: float) -> np.ndarray:
    return np.hypot(x - cx, y - cy) - r


def _sd_ellipse(x, y, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    dx = (x - cx) / rx
    dy = (y - cy) / ry
    # Scaling the unit-circle distance back by the smaller radius keeps the
    # gradient close enough to 1 that a one-pixel coverage ramp stays even.
    return (np.hypot(dx, dy) - 1.0) * min(rx, ry)


def _sd_segment(x, y, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    pax, pay = x - ax, y - ay
    bax, bay = bx - ax, by - ay
    denom = bax * bax + bay * bay
    h = np.clip((pax * bax + pay * bay) / max(denom, 1e-9), 0.0, 1.0)
    return np.hypot(pax - bax * h, pay - bay * h)


def _sd_triangle(x, y, p0, p1, p2) -> np.ndarray:
    """Exact triangle field: nearest edge distance, signed by winding."""
    d = np.minimum(
        np.minimum(_sd_segment(x, y, *p0, *p1), _sd_segment(x, y, *p1, *p2)),
        _sd_segment(x, y, *p2, *p0),
    )

    def side(a, b):
        return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])

    s0, s1, s2 = side(p0, p1), side(p1, p2), side(p2, p0)
    inside = ((s0 >= 0) & (s1 >= 0) & (s2 >= 0)) | ((s0 <= 0) & (s1 <= 0) & (s2 <= 0))
    return np.where(inside, -d, d)


def _op_round(sdf: np.ndarray, radius: float) -> np.ndarray:
    """Round off corners by radius — a wedge body needs this to look drawn."""
    return sdf - radius


def _op_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)


def _op_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, -b)


# ── Compositing ───────────────────────────────────────────────────────────────

class _Canvas:
    """A straight-alpha RGBA accumulator with SDF-driven coverage."""

    def __init__(self, size: int):
        self.size = size
        # Units span [-1, 1] across `size` pixels, so one unit is size/2 pixels.
        # Coverage ramps over exactly one pixel, which is what makes the edges
        # look drawn rather than aliased or blurred.
        self.px_per_unit = size / 2.0
        self.rgb = np.zeros((size, size, 3), dtype=np.float32)
        self.alpha = np.zeros((size, size), dtype=np.float32)

    def coverage(self, sdf: np.ndarray) -> np.ndarray:
        return np.clip(0.5 - sdf * self.px_per_unit, 0.0, 1.0)

    def fill(self, sdf: np.ndarray, colour: RGB, opacity: float = 1.0) -> None:
        self._over(self.coverage(sdf) * float(opacity), colour)

    def stroke(self, sdf: np.ndarray, colour: RGB, width: float,
               opacity: float = 1.0) -> None:
        self._over(self.coverage(np.abs(sdf) - width * 0.5) * float(opacity), colour)

    def _over(self, cov: np.ndarray, colour: RGB) -> None:
        """Source-over compositing in straight (non-premultiplied) alpha."""
        if not cov.any():
            return
        src = np.asarray(colour, dtype=np.float32) / 255.0
        src_a = cov[..., None]
        dst_a = self.alpha[..., None]
        out_a = src_a + dst_a * (1.0 - src_a)
        # Where nothing has been drawn yet out_a is 0 and the colour is
        # undefined; np.where keeps the divide from producing NaNs that would
        # later show up as black speckle in the transparent margin.
        self.rgb = np.where(
            out_a > 1e-6,
            (src * src_a + self.rgb * dst_a * (1.0 - src_a)) / np.maximum(out_a, 1e-6),
            self.rgb,
        )
        self.alpha = out_a[..., 0]

    def to_rgba(self) -> np.ndarray:
        out = np.empty((self.size, self.size, 4), dtype=np.uint8)
        out[..., :3] = np.clip(self.rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        out[..., 3] = np.clip(self.alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return out


# ── The spec ──────────────────────────────────────────────────────────────────

@dataclass
class CreatureSpec:
    """
    One character, as data.

    Proportions are fractions of the sprite half-extent, so they are stable at
    any ``size``. The defaults describe a plain round bird: two dot eyes, a
    small beak, no brows and no crest. That plainness is deliberate — expressive
    brows and head crests are exactly the traits that make a cartoon bird
    recognisable as somebody else's cartoon bird.
    """

    name: str = "bird"
    body: str = "round"                     # one of BODY_SHAPES
    colour: RGB = (226, 74, 58)
    belly: RGB | None = (245, 214, 190)     # lighter underside; None to disable
    outline: RGB = (28, 28, 34)
    outline_width: float = 0.055
    beak: RGB = (247, 181, 56)
    beak_size: float = 0.19                 # 0 disables the beak
    eye_size: float = 0.095
    eye_colour: RGB = (255, 255, 255)
    pupil: RGB = (28, 28, 34)
    gloss: float = 0.22                     # top highlight strength; 0 disables
    tail: float = 0.0                       # side-view tail length; 0 disables
    scale: float = 1.0                      # overall size within the sprite box

    def __post_init__(self):
        if self.body not in BODY_SHAPES:
            raise ValueError(
                f"unknown body {self.body!r}; expected one of {BODY_SHAPES}")
        # JSON round-trips turn tuples into lists, and a list colour would break
        # the float conversion in _over. Normalise on the way in instead of
        # defending against it at every use site.
        for field_name in ("colour", "belly", "outline", "beak", "eye_colour", "pupil"):
            value = getattr(self, field_name)
            if isinstance(value, list):
                object.__setattr__(self, field_name, tuple(value))

    def variant(self, **changes) -> CreatureSpec:
        """A copy with fields overridden — handy for palette swaps."""
        return replace(self, **changes)


def spec_to_dict(spec: CreatureSpec) -> dict:
    return asdict(spec)


def spec_from_dict(data: dict) -> CreatureSpec:
    known = {f for f in CreatureSpec.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown spec fields: {sorted(unknown)}")
    return CreatureSpec(**data)


# ── Body shapes ───────────────────────────────────────────────────────────────

def _body_field(spec: CreatureSpec, x, y, cx: float, cy: float) -> np.ndarray:
    """The silhouette, before any features are drawn on top of it."""
    s = spec.scale
    if spec.body == "round":
        return _sd_circle(x, y, cx, cy, 0.70 * s)
    if spec.body == "oval":
        return _sd_ellipse(x, y, cx, cy, 0.62 * s, 0.76 * s)
    if spec.body == "tall":
        return _sd_ellipse(x, y, cx, cy, 0.54 * s, 0.84 * s)
    if spec.body == "wedge":
        # A rounded triangle. Rounding grows the shape outward by its radius,
        # so the raw triangle has to be inset by that much or the corners push
        # past the canvas edge and get clipped.
        r = 0.14 * s
        w, h = 0.64 * s, 0.70 * s
        base = cy + h * 0.84
        return _op_round(
            _sd_triangle(x, y, (cx, cy - h), (cx - w, base), (cx + w, base)),
            r,
        )
    raise ValueError(spec.body)  # pragma: no cover — guarded in __post_init__


def _draw_eye(canvas: _Canvas, x, y, spec: CreatureSpec,
              ex: float, ey: float, look: float = 0.0) -> None:
    r = spec.eye_size * spec.scale
    canvas.fill(_sd_circle(x, y, ex, ey, r), spec.eye_colour)
    canvas.stroke(_sd_circle(x, y, ex, ey, r), spec.outline,
                  spec.outline_width * 0.6)
    canvas.fill(_sd_circle(x, y, ex + look * r * 0.35, ey + r * 0.1, r * 0.52),
                spec.pupil)


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_creature(spec: CreatureSpec, view: View = "front",
                    size: int = 192) -> np.ndarray:
    """
    Render ``spec`` to an ``(size, size, 4)`` uint8 RGBA array.

    ``view="side"`` is the slingshot pose: the creature faces right, so it
    points where it is about to be launched.
    """
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    if size < 8:
        raise ValueError("size must be at least 8 pixels")

    x, y = _grid(size)
    canvas = _Canvas(size)
    s = spec.scale
    stroke_w = spec.outline_width * s

    # The side view sits right of centre: the tail needs more room than the
    # beak does, and both have to stay inside the canvas at every scale.
    cx = 0.0 if view == "front" else 0.04
    cy = 0.02

    body = _body_field(spec, x, y, cx, cy)

    # Tail first, so the body silhouette overlaps and hides its root.
    if view == "side" and spec.tail > 0:
        # The tip is measured from the body edge outward rather than from the
        # centre, so `tail` means "how far it sticks out" regardless of body.
        root_x = cx - 0.40 * s
        tip_x = cx - (0.70 + spec.tail) * s
        for offset, droop in ((-0.13, -0.09), (0.11, 0.15)):
            oy = cy + offset * s
            tail = _sd_triangle(
                x, y,
                (root_x, oy - 0.15 * s),
                (tip_x, oy + droop * s),
                (root_x, oy + 0.17 * s),
            )
            canvas.fill(tail, spec.colour)
            canvas.stroke(tail, spec.outline, stroke_w * 0.8)

    # Body fill.
    canvas.fill(body, spec.colour)

    # Belly: an ellipse clipped to the lower part of the body.
    if spec.belly is not None:
        belly = _sd_ellipse(x, y, cx + (0.06 * s if view == "side" else 0.0),
                            cy + 0.34 * s, 0.44 * s, 0.36 * s)
        canvas.fill(_op_intersect(belly, body), spec.belly)

    # Gloss: a highlight on the upper body, clipped to the silhouette. A single
    # fill reads as a hard-edged oval sticker, so it is built from a few
    # concentric passes instead — cheap, and the stacked coverage gives a
    # falloff that looks like light rather than paint.
    if spec.gloss > 0:
        for step in range(3):
            spread = 1.0 + step * 0.45
            gloss = _sd_ellipse(x, y, cx - 0.19 * s, cy - 0.38 * s,
                                0.20 * s * spread, 0.13 * s * spread)
            canvas.fill(_op_intersect(gloss, body + stroke_w),
                        (255, 255, 255), opacity=spec.gloss * 0.42)

    canvas.stroke(body, spec.outline, stroke_w)

    # Beak.
    if spec.beak_size > 0:
        b = spec.beak_size * s
        if view == "front":
            # Pointing down, between and just below the eyes.
            beak = _sd_triangle(x, y,
                                (cx - b * 0.95, cy + 0.06 * s),
                                (cx + b * 0.95, cy + 0.06 * s),
                                (cx, cy + 0.06 * s + b * 1.30))
        else:
            # Pointing right. The root is set well inside the silhouette so the
            # beak reads as attached to the head rather than stuck onto it.
            nose = cx + 0.48 * s
            beak = _sd_triangle(x, y,
                                (nose, cy - b * 1.00),
                                (nose + b * 1.95, cy + b * 0.05),
                                (nose, cy + b * 0.95))
        canvas.fill(beak, spec.beak)
        canvas.stroke(beak, spec.outline, stroke_w * 0.85)

    # Eyes last, so they sit on top of everything.
    if view == "front":
        eye_dx = 0.26 * s
        _draw_eye(canvas, x, y, spec, cx - eye_dx, cy - 0.16 * s, look=+0.3)
        _draw_eye(canvas, x, y, spec, cx + eye_dx, cy - 0.16 * s, look=-0.3)
    else:
        _draw_eye(canvas, x, y, spec, cx + 0.26 * s, cy - 0.22 * s, look=+0.6)

    return canvas.to_rgba()


def render_views(spec: CreatureSpec, size: int = 192) -> dict[str, np.ndarray]:
    """Both views of one spec, keyed by view name."""
    return {view: render_creature(spec, view=view, size=size) for view in VIEWS}


# ── A default cast ────────────────────────────────────────────────────────────
#
# Five plainly-drawn birds distinguished only by colour, silhouette and size.
# No brows, no crests, no species markings — nothing that reads as a particular
# existing character.

DEFAULT_CAST: tuple[CreatureSpec, ...] = (
    CreatureSpec(
        name="ruby", body="round", colour=(226, 74, 58),
        belly=(246, 214, 196), beak=(247, 181, 56), scale=1.0, tail=0.20,
    ),
    CreatureSpec(
        name="amber", body="wedge", colour=(240, 190, 52),
        belly=(250, 228, 160), beak=(232, 128, 44), scale=0.94,
        eye_size=0.085, tail=0.16,
    ),
    CreatureSpec(
        name="slate", body="round", colour=(74, 82, 96),
        belly=(120, 130, 148), beak=(196, 168, 92), scale=1.06,
        gloss=0.28, tail=0.18,
    ),
    CreatureSpec(
        name="azure", body="round", colour=(72, 148, 214),
        belly=(198, 226, 246), beak=(240, 176, 72), scale=0.80,
        eye_size=0.105, tail=0.24,
    ),
    CreatureSpec(
        name="ivory", body="oval", colour=(238, 238, 232),
        belly=(255, 255, 252), outline=(96, 96, 104),
        beak=(244, 168, 60), scale=0.96, gloss=0.15, tail=0.20,
    ),
)


def cast_by_name(cast: Iterable[CreatureSpec] | None = None) -> dict[str, CreatureSpec]:
    return {spec.name: spec for spec in (cast if cast is not None else DEFAULT_CAST)}
