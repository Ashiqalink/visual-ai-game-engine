"""
imaging.py — the sprite pipeline, so games never import an image library.

Everything a game needs to turn a picture into a usable sprite lives here:
loading, cutting the background out, trimming, and writing a PNG back. A game
that wants artwork imports :mod:`visual_ai` and nothing else.

The interesting function is :func:`clean_sprite`, which takes whatever you feed
it — a render, a photo, an image off a generator — and returns a tightly
cropped RGBA sprite with a real alpha channel.

Two ways to cut a background, and the difference matters:

* :func:`chroma_key` — for flat artwork on a solid backdrop. It measures the
  backdrop from the border, keys on colour distance, restricts the cut to
  regions actually connected to the border, and un-mixes the backdrop out of
  the semi-transparent edge pixels. On flat art this is effectively lossless
  and keeps hard lines hard.
* :func:`remove_background` — a neural matte via ``rembg``, for images whose
  background you could not control. It handles anything, but it was trained on
  photographs, so it renders crisp cartoon edges noticeably softer.

Prefer the first when you can choose the backdrop. Generate art on a solid
magenta field and this module cuts it out perfectly.

Arrays are ``uint8``. Three-channel input is assumed RGB; use
:func:`bgr_to_rgb` at the boundary with OpenCV-flavoured code.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

__all__ = [
    "load_image",
    "save_png",
    "to_rgba",
    "bgr_to_rgb",
    "rgb_to_bgr",
    "chroma_key",
    "remove_background",
    "clean_sprite",
    "autocrop",
    "pad_to",
    "resize",
    "bleed_edges",
    "background_uniformity",
    "composite_over",
    "blit_sprite",
    "REMBG_AVAILABLE",
]

BackgroundMode = Literal["auto", "chroma", "rembg", "none"]


def _cv2():
    """OpenCV, imported lazily so importing this module never costs a load."""
    import cv2

    return cv2


def _probe_rembg() -> bool:
    """
    Is rembg importable *and* usable?

    A rembg installed without an ONNX backend does not raise on import — it
    prints an install hint and calls ``sys.exit()``. That is a ``SystemExit``,
    which does not inherit from ``Exception``, so the obvious ``except
    Exception`` guard lets it straight through and kills the importing process.
    Catching ``BaseException`` here is deliberate, and the stream redirect keeps
    rembg's hint off the console of anyone who never asked for it.
    """
    import contextlib
    import io

    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            import rembg  # noqa: F401
        return True
    except BaseException:
        return False


REMBG_AVAILABLE = _probe_rembg()


# ── Channel helpers ───────────────────────────────────────────────────────────

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Flip channel order. Works for 3- and 4-channel images; alpha is kept."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected an HxWx3 or HxWx4 image, got {image.shape}")
    out = image.copy()
    out[..., :3] = image[..., 2::-1]
    return out


rgb_to_bgr = bgr_to_rgb  # the swap is its own inverse


def to_rgba(image: np.ndarray) -> np.ndarray:
    """Promote greyscale or RGB to RGBA with an opaque alpha channel."""
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"cannot interpret shape {image.shape} as an image")
    if image.shape[2] == 4:
        return image
    if image.shape[2] != 3:
        raise ValueError(f"expected 3 or 4 channels, got {image.shape[2]}")
    alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
    return np.concatenate([image, alpha], axis=2)


# ── File I/O ──────────────────────────────────────────────────────────────────

def load_image(path: str | os.PathLike) -> np.ndarray:
    """Read any image file as RGBA."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    cv2 = _cv2()
    # imread mangles non-ASCII paths on Windows; reading the bytes ourselves
    # and decoding from memory sidesteps that entirely.
    with open(path, "rb") as handle:
        buffer = np.frombuffer(handle.read(), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"could not decode an image from {path}")
    if image.ndim == 3 and image.shape[2] in (3, 4):
        image = bgr_to_rgb(image)
    return to_rgba(image)


def save_png(path: str | os.PathLike, image: np.ndarray) -> str:
    """Write RGBA (or RGB) out as a PNG, creating parent folders as needed."""
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    cv2 = _cv2()
    ok, buffer = cv2.imencode(".png", rgb_to_bgr(to_rgba(image)))
    if not ok:
        raise RuntimeError(f"PNG encoding failed for {path}")
    with open(path, "wb") as handle:
        handle.write(buffer.tobytes())
    return path


# ── Background removal ────────────────────────────────────────────────────────

def _border_pixels(rgb: np.ndarray, width: int = 2) -> np.ndarray:
    """Every pixel within `width` of an edge, as a flat Nx3 array."""
    w = max(1, int(width))
    edges = [rgb[:w], rgb[-w:], rgb[:, :w].reshape(-1, 3), rgb[:, -w:].reshape(-1, 3)]
    return np.concatenate([e.reshape(-1, 3) for e in edges], axis=0)


def background_uniformity(image: np.ndarray, width: int = 2) -> float:
    """
    How uniform the border is, from 0 (noisy) to 1 (a single flat colour).

    :func:`clean_sprite` uses this to decide whether chroma keying will work,
    so it does not have to be told which kind of image it was handed.
    """
    rgb = to_rgba(image)[..., :3].astype(np.float32)
    border = _border_pixels(rgb, width)
    spread = np.linalg.norm(border - np.median(border, axis=0), axis=1)
    # 32 units of RGB distance is roughly where a gradient stops reading as
    # "one colour" to the eye; past that, treat the border as non-uniform.
    return float(np.clip(1.0 - np.mean(spread) / 32.0, 0.0, 1.0))


def chroma_key(image: np.ndarray, key: tuple[int, int, int] | None = None,
               tolerance: float = 42.0, softness: float = 26.0,
               connected: bool = True, unmix: bool = True) -> np.ndarray:
    """
    Cut a solid backdrop out of flat artwork.

    ``key`` defaults to the median border colour. ``tolerance`` is the RGB
    distance treated as fully background; pixels fade to opaque over the next
    ``softness`` units, which is what gives a clean antialiased edge rather
    than a jagged one.

    ``connected`` restricts the cut to background regions that actually touch
    the border. Without it, a sprite the same colour as the backdrop would get
    holes punched through it — a white bird on a white field would lose its
    belly. ``unmix`` removes the backdrop's colour contribution from
    semi-transparent edge pixels, which is what kills the coloured halo.
    """
    rgba = to_rgba(image)
    rgb = rgba[..., :3].astype(np.float32)

    key_rgb = (np.median(_border_pixels(rgb), axis=0) if key is None
               else np.asarray(key, dtype=np.float32))

    distance = np.linalg.norm(rgb - key_rgb, axis=2)
    softness = max(float(softness), 1e-3)
    tolerance = float(tolerance)
    ramp = np.clip((distance - tolerance) / softness, 0.0, 1.0)

    # Difference matting.
    #
    # A distance ramp alone cannot recover partial coverage, and the failure is
    # not subtle. Take a pixel that is half dark outline, half magenta backdrop:
    # it sits ~159 units from the key, far past any sane threshold, so the ramp
    # calls it fully opaque and it keeps its blended purple. Every antialiased
    # edge pixel ends up doing this, which is precisely the halo.
    #
    # The blend is linear — observed = a*F + (1-a)*key — so given the true
    # foreground colour F the coverage falls straight out as
    # |observed - key| / |F - key|. F is not known at the edge, but it is known
    # a few pixels inside, where the sprite is unambiguously opaque. Flooding
    # that colour outward supplies F for exactly the band that needs it.
    core = ramp >= 1.0
    alpha = ramp
    if core.any():
        foreground, reached = _propagate_colour(rgb, core.astype(np.float32),
                                                iterations=8)
        spread = np.linalg.norm(foreground - key_rgb, axis=2)
        # A foreground indistinguishable from the key carries no information
        # about coverage; fall back to the ramp rather than dividing by ~0.
        usable = reached & (spread > max(tolerance, 8.0))
        alpha = np.where(usable, np.clip(distance / np.maximum(spread, 1e-6), 0.0, 1.0),
                         ramp)
        # Anything the ramp was already certain about stays certain: interiors
        # opaque, and the flat backdrop transparent.
        alpha = np.where(core, 1.0, alpha)
        alpha = np.where(distance <= tolerance, 0.0, alpha)

    if connected:
        cv2 = _cv2()
        background = (alpha < 1.0).astype(np.uint8)
        count, labels = cv2.connectedComponents(background, connectivity=4)
        if count > 1:
            border_labels = np.unique(
                np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
            border_labels = border_labels[border_labels != 0]
            reachable = np.isin(labels, border_labels)
            alpha = np.where(reachable, alpha, 1.0)

    out = rgba.copy()
    if unmix:
        # Observed = alpha*true + (1-alpha)*key. Solving for true is only
        # stable where alpha is meaningfully above zero; fully transparent
        # pixels have no colour worth recovering.
        safe = alpha > 0.02
        recovered = np.where(
            safe[..., None],
            (rgb - (1.0 - alpha[..., None]) * key_rgb) / np.maximum(alpha[..., None], 0.02),
            rgb,
        )
        out[..., :3] = np.clip(recovered, 0, 255).astype(np.uint8)

    out[..., 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    # The margin is still full of backdrop colour at this point. Left there it
    # would reappear as a rim the moment anything filtered the sprite.
    return bleed_edges(out)


def remove_background(image: np.ndarray, model: str = "isnet-general-use") -> np.ndarray:
    """
    Neural background removal via ``rembg``.

    ``isnet-general-use`` holds hard edges better than the default ``u2net``,
    which matters for line art. The first call downloads the model.
    """
    if not REMBG_AVAILABLE:
        raise RuntimeError(
            "rembg is not installed. Install it with:\n"
            '    pip install "rembg[cpu]"\n'
            "Or generate artwork on a solid backdrop and use chroma_key(), "
            "which needs no extra dependency and gives crisper edges on flat art."
        )
    try:
        from rembg import new_session, remove
    except BaseException as exc:  # SystemExit — see _probe_rembg
        raise RuntimeError(f"rembg failed to import: {exc}") from exc

    try:
        session = new_session(model)
        cut = remove(to_rgba(image)[..., :3], session=session)
    except Exception as exc:
        # The overwhelmingly common cause is a rembg install without an ONNX
        # backend, which otherwise surfaces as an opaque error deep in rembg.
        raise RuntimeError(
            f"rembg could not run ({exc}).\n"
            'If this mentions a missing backend, install one with:\n'
            '    pip install "rembg[cpu]"'
        ) from exc
    return to_rgba(np.asarray(cut))


# ── Geometry ──────────────────────────────────────────────────────────────────

def autocrop(image: np.ndarray, threshold: int = 8, margin: int = 0) -> np.ndarray:
    """Trim fully transparent borders. Returns the input if it is all empty."""
    rgba = to_rgba(image)
    solid = rgba[..., 3] > threshold
    if not solid.any():
        return rgba
    rows = np.flatnonzero(solid.any(axis=1))
    cols = np.flatnonzero(solid.any(axis=0))
    y0 = max(int(rows[0]) - margin, 0)
    y1 = min(int(rows[-1]) + 1 + margin, rgba.shape[0])
    x0 = max(int(cols[0]) - margin, 0)
    x1 = min(int(cols[-1]) + 1 + margin, rgba.shape[1])
    return rgba[y0:y1, x0:x1]


def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Resample without the halo.

    Resizing straight-alpha RGBA directly is the classic way to get a fringe:
    the filter averages colour channels with no regard for alpha, so whatever
    colour sits in the transparent pixels bleeds into the edge. Premultiplying
    first weights every contribution by its own opacity, which is what the
    compositing maths actually calls for; dividing back out afterwards
    restores straight alpha for the caller.
    """
    cv2 = _cv2()
    rgba = to_rgba(image)
    shrinking = width < rgba.shape[1] or height < rgba.shape[0]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC

    source = rgba.astype(np.float32)
    alpha = source[..., 3:4] / 255.0
    premultiplied = np.concatenate([source[..., :3] * alpha, source[..., 3:4]], axis=2)

    scaled = cv2.resize(premultiplied, (int(width), int(height)), interpolation=interp)
    out_alpha = np.clip(scaled[..., 3:4], 0.0, 255.0)
    # Below roughly one part in 255 the colour is unrecoverable and the divide
    # only amplifies noise, so leave those pixels at zero — they are invisible.
    unpremultiplied = np.where(
        out_alpha > 0.5,
        scaled[..., :3] / np.maximum(out_alpha / 255.0, 1e-6),
        0.0,
    )

    out = np.empty(scaled.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = np.clip(unpremultiplied, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(out_alpha[..., 0], 0, 255).astype(np.uint8)
    return out


def _propagate_colour(colour: np.ndarray, known: np.ndarray,
                      iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Flood known colours outward one ring at a time.

    Each pass averages neighbouring known values, weighted by how many
    neighbours are known, then promotes the newly filled ring to known. Returns
    the filled colour field and the mask of everything now known — callers need
    the mask because pixels further than ``iterations`` rings away were never
    reached and still hold their original value.
    """
    cv2 = _cv2()
    colour = colour.astype(np.float32, copy=True)
    known = known.astype(np.float32, copy=True)

    for _ in range(max(0, int(iterations))):
        if known.min() > 0:
            break
        weight = cv2.blur(known, (3, 3))
        total = cv2.blur(colour * known[..., None], (3, 3))
        fillable = (known == 0) & (weight > 1e-6)
        if not fillable.any():
            break
        averaged = total / np.maximum(weight, 1e-6)[..., None]
        colour = np.where(fillable[..., None], averaged, colour)
        known = np.where(fillable, 1.0, known)

    return colour, known > 0


def bleed_edges(image: np.ndarray, iterations: int = 6) -> np.ndarray:
    """
    Push opaque colour outward into the transparent margin.

    Transparent pixels still hold *some* RGB value, and after a background cut
    that value is the background. Anything that samples colour without
    weighting by alpha — a GPU texture filter, pygame's ``smoothscale`` — will
    drag it back into view as a coloured rim. Flooding the margin with the
    nearest real colour makes the sprite safe for those consumers.

    Each pass averages the neighbouring known colours, weighted by how many
    neighbours are known, then marks the filled ring as known for the next one.
    """
    cv2 = _cv2()
    rgba = to_rgba(image)
    colour, _ = _propagate_colour(rgba[..., :3], (rgba[..., 3] > 0).astype(np.float32),
                                  iterations)
    out = rgba.copy()
    out[..., :3] = np.clip(colour, 0, 255).astype(np.uint8)
    return out


def pad_to(image: np.ndarray, size: int, fit: float = 0.88) -> np.ndarray:
    """
    Centre the image on a transparent ``size``x``size`` canvas.

    ``fit`` is how much of the canvas the content should span, leaving a margin
    so rotation in-game does not clip the corners.
    """
    rgba = autocrop(image)
    h, w = rgba.shape[:2]
    longest = max(h, w)
    if longest == 0:
        raise ValueError("cannot pad an empty image")

    target = max(1, int(round(size * float(fit))))
    scale = target / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scaled = resize(rgba, new_w, new_h)

    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = scaled
    return canvas


def blit_sprite(frame: np.ndarray, sprite: np.ndarray, cx: int, cy: int,
                size: int | None = None, angle: float = 0.0) -> None:
    """
    Composite a 4-channel sprite centred on ``(cx, cy)``, in place.

    Channel order is whatever the two already share — an OpenCV caller passes a
    BGR frame and a BGRA sprite and gets the right answer without a conversion.

    ``size`` scales the sprite to a square of that many pixels, ``angle``
    rotates it in degrees. Both are applied before clipping, and the sprite is
    clipped rather than rejected when it hangs off an edge, so a creature can
    walk out of frame without popping.
    """
    cv2 = _cv2()
    sprite = to_rgba(sprite)

    if size is not None:
        size = int(size)
        if size <= 0:
            return
        if size != sprite.shape[0] or size != sprite.shape[1]:
            sprite = resize(sprite, size, size)

    if angle:
        dim = sprite.shape[0]
        matrix = cv2.getRotationMatrix2D((dim / 2.0, dim / 2.0), angle, 1.0)
        sprite = cv2.warpAffine(sprite, matrix, (dim, dim), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    height, width = sprite.shape[:2]
    y0, x0 = int(cy) - height // 2, int(cx) - width // 2

    fy0, fy1 = max(0, y0), min(frame.shape[0], y0 + height)
    fx0, fx1 = max(0, x0), min(frame.shape[1], x0 + width)
    if fy0 >= fy1 or fx0 >= fx1:
        return

    patch = sprite[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0]
    target = frame[fy0:fy1, fx0:fx1]
    alpha = (patch[..., 3:4].astype(np.float32)) / 255.0
    frame[fy0:fy1, fx0:fx1] = (
        alpha * patch[..., :3] + (1.0 - alpha) * target).astype(frame.dtype)


def composite_over(foreground: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Alpha-composite one RGBA image over another of the same shape."""
    fg = to_rgba(foreground).astype(np.float32)
    bg = to_rgba(background).astype(np.float32)
    if fg.shape != bg.shape:
        raise ValueError(f"shape mismatch: {fg.shape} over {bg.shape}")
    fa = fg[..., 3:4] / 255.0
    ba = bg[..., 3:4] / 255.0
    out_a = fa + ba * (1.0 - fa)
    rgb = np.where(
        out_a > 1e-6,
        (fg[..., :3] * fa + bg[..., :3] * ba * (1.0 - fa)) / np.maximum(out_a, 1e-6),
        0.0,
    )
    out = np.empty_like(fg)
    out[..., :3] = np.clip(rgb, 0, 255)
    out[..., 3] = np.clip(out_a[..., 0] * 255.0, 0, 255)
    return out.astype(np.uint8)


# ── The one-call pipeline ─────────────────────────────────────────────────────

def clean_sprite(image: np.ndarray, mode: BackgroundMode = "auto",
                 size: int | None = None, key: tuple[int, int, int] | None = None,
                 tolerance: float = 42.0, fit: float = 0.88) -> np.ndarray:
    """
    Any image in, a clean sprite out.

    ``mode="auto"`` measures the border and picks chroma keying for a uniform
    backdrop, ``rembg`` otherwise — falling back to chroma keying if rembg is
    not installed, since a mediocre cut beats a crash. Pass ``size`` to also
    centre the result on a square canvas.
    """
    rgba = to_rgba(image)

    if mode == "auto":
        if rgba[..., 3].min() < 250:
            mode = "none"          # already has alpha; trust it
        elif background_uniformity(rgba) > 0.75:
            mode = "chroma"
        else:
            mode = "rembg" if REMBG_AVAILABLE else "chroma"

    if mode == "chroma":
        rgba = chroma_key(rgba, key=key, tolerance=tolerance)
    elif mode == "rembg":
        rgba = remove_background(rgba)
    elif mode != "none":
        raise ValueError(f"unknown mode {mode!r}")

    rgba = autocrop(rgba)
    if size is not None:
        rgba = pad_to(rgba, int(size), fit=fit)
    return rgba
