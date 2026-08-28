"""
visual_ai.display - the window a game presents through.

Replaces `cv2.imshow` + `cv2.waitKey`. Those two are not a renderer; they are a
GDI blit and a Win32 message pump, and on Windows the pump is the expensive
half. Measured on the development machine, presenting a native 1920x1080 canvas
costs **24.0 ms** - the whole of a 60 fps budget, before a single pixel of the
game is drawn. The same frame through SDL costs 2.62 ms. That difference is the
reason this module exists.

    from visual_ai import display

    win = display.open_window("My Game", 1280, 720, fps=60)
    while True:
        key = win.present(canvas)          # show, pump, pace, return a key
        if key == ord('f'):
            win.toggle_fullscreen()        # same canvas, scaled to the screen
        if key in (27, ord('q')):
            break
    win.close()

Fullscreen is off unless asked for, by the game (``fullscreen=True``) or by the
player (``VISUAL_AI_FULLSCREEN=1``). Either way the logical frame size is
unchanged - the canvas is scaled up, never re-shaped - so no drawing code cares
which one it is running in.

`present` deliberately does all three jobs `imshow` and `waitKey` did jointly.
Splitting them would hand every game the pacing problem back, and the three
copies of that logic already in the games repo disagreed with each other.

Backends
--------
``d3d11``  Win32 + Direct3D 11, in the compiled engine. The tuned path.
``sdl``    pygame. Portable, and what CI runs.
``null``   No window at all. Validates and counts, never sleeps.

``auto`` (the default) picks ``d3d11`` on Windows when the compiled engine
offers it, and ``sdl`` otherwise. It never picks ``null``: a game that silently
renders nothing looks exactly like a game that works. Ask for ``null`` by name,
or set ``VISUAL_AI_DISPLAY=null``.

An explicitly named backend never degrades - if you asked for ``d3d11`` and it
is not there, that is an error, because the whole point of naming it was to get
it. Only ``auto`` falls back quietly. The probe itself follows the house style
from `capture.py`: ask the object whether it has the thing, take a safe default
if not.

Import cost
-----------
`visual_ai/__init__.py` does **not** import this module, and must not start:
resolving a backend initialises SDL, and a game that never opens a window
should not pay for one. Import it as ``from visual_ai import display``.

That spelling is also what makes the headless driver work. `headless.py`
replaces `open_window` on this module object, and the substitution is only seen
by callers that look the attribute up at call time. ``from visual_ai.display
import open_window`` binds the real function once at import and silently opts a
game out of every regression test.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from ._keys import NO_KEY
from ._pacing import Pacer

# Set before anything can import pygame. Importing it prints a community banner
# to stdout, and `run_lab_tests.py` parses the headless report as JSON off
# stdout - so an unsuppressed banner fails the regression gate with a message
# about malformed JSON that never mentions pygame. `sdl.py` sets this too; this
# is the copy that catches an import from somewhere else entirely.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

__all__ = [
    "BACKENDS", "NO_KEY", "Window",
    "available_backends", "resolve_backend", "resolve_fullscreen",
    "open_window", "show", "find_window", "close_all",
]

#: Every backend name, best first. `auto` resolves left to right.
BACKENDS = ("d3d11", "sdl", "null")

#: The environment override, for a player or a test that wants a specific path.
ENV_VAR = "VISUAL_AI_DISPLAY"

#: The fullscreen override, for a player who wants it without editing a game.
FULLSCREEN_ENV_VAR = "VISUAL_AI_FULLSCREEN"

#: Open windows, keyed by title.
#
# `cv2.namedWindow` is title-keyed and idempotent, and at least one game leans
# on that: Sling paints a loading splash from module scope and then opens "its"
# window again once the pipeline is up, expecting the same window back. A
# registry keeps that working, and keeps `close_all` honest.
_WINDOWS: dict[str, Window] = {}


# ── Backend selection ─────────────────────────────────────────────────────────

def _d3d11_available() -> bool:
    """True when the compiled engine was built with the D3D11 window in it."""
    if sys.platform != "win32":
        return False
    try:
        import engine_core  # noqa: PLC0415 - probing, not depending
    except ImportError:
        return False
    return getattr(engine_core, "open_window", None) is not None


def _sdl_available() -> bool:
    # `find_spec`, not `import`: asking whether pygame is installed must not
    # cost the ~0.2 s of importing it, and must not initialise SDL, in a
    # process that may well resolve to another backend.
    return importlib.util.find_spec("pygame") is not None


_PROBES = {
    "d3d11": _d3d11_available,
    "sdl": _sdl_available,
    "null": lambda: True,
}


def available_backends() -> tuple[str, ...]:
    """The backends this machine can actually open a window with."""
    return tuple(name for name in BACKENDS if _PROBES[name]())


def resolve_backend(name: str | None = None) -> str:
    """
    Turn a request - or the lack of one - into exactly one backend name.

    `name` wins, then ``$VISUAL_AI_DISPLAY``, then ``auto``. Raises rather than
    degrades whenever a specific backend was asked for and is not there.
    """
    requested = name or os.environ.get(ENV_VAR) or "auto"
    requested = requested.strip().lower()

    if requested == "auto":
        for candidate in ("d3d11", "sdl"):
            if _PROBES[candidate]():
                return candidate
        raise RuntimeError(
            "no display backend is available: d3d11 needs a Windows build of "
            "engine_core, sdl needs pygame. Install pygame, or set "
            f"{ENV_VAR}=null if this process is not meant to show anything."
        )

    if requested not in _PROBES:
        raise ValueError(
            f"unknown display backend {requested!r}; "
            f"expected one of {', '.join(BACKENDS)} or 'auto'"
        )

    if not _PROBES[requested]():
        raise RuntimeError(
            f"display backend {requested!r} was asked for but is not available "
            f"on this machine (via {ENV_VAR} or backend=). Use 'auto' to fall "
            f"back; available: {', '.join(available_backends())}"
        )
    return requested


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def resolve_fullscreen(fullscreen: bool | None = None) -> bool:
    """
    Turn a request - or the lack of one - into a fullscreen decision.

    `fullscreen` wins, then ``$VISUAL_AI_FULLSCREEN``, then windowed. The
    environment is what a player has: every game in the workspace hard-codes
    its `open_window` call, so without this the only way to play fullscreen is
    to edit the game. An unrecognised value raises rather than being read as
    False - a player who typed ``VISUAL_AI_FULLSCREEN=full`` and got a window
    would have no way to tell that from the flag not working.
    """
    if fullscreen is not None:
        return bool(fullscreen)
    raw = os.environ.get(FULLSCREEN_ENV_VAR)
    if raw is None or not raw.strip():
        return False
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(
        f"{FULLSCREEN_ENV_VAR}={raw!r} is not a yes/no value; "
        f"expected one of {', '.join(sorted(_TRUE | _FALSE))}"
    )


# ── The window ────────────────────────────────────────────────────────────────

class Window:
    """
    Base class holding everything that must not differ between backends.

    Frame validation, pacing, the present counter and the key queue live here
    on purpose: they are the observable contract, and a backend that
    reimplemented any of them would be free to drift. Subclasses supply only
    mechanism - `_show`, `_pump`, `_shutdown`.
    """

    #: The null backend opts out of pacing entirely; see `null.py`.
    _paces = True

    def __init__(self, title: str, width: int, height: int, *,
                 fps: float | None = 60.0,
                 fullscreen: bool = False) -> None:
        self.title = title
        self.width = int(width)
        self.height = int(height)
        self.backend = "base"
        self.closed = False
        # The *logical* size above never changes with this; fullscreen scales
        # the same canvas to the screen, so a game's frames stay the shape
        # `_check` was told to expect.
        self.fullscreen = bool(fullscreen)
        self.presents = 0
        self._pacer = Pacer(fps)
        # Bounded on purpose. Every dispatch chain in both repos reads one key
        # per frame, so an unbounded queue would only let a keyboard-mashing
        # player build a backlog the game replays long after they stopped.
        self._keys: list[int] = []

    # -- subclass hooks --------------------------------------------------------

    def _show(self, frame: np.ndarray) -> None:
        raise NotImplementedError

    def _pump(self) -> None:
        """Drain OS events, appending keys with `_push_key`."""

    def _shutdown(self) -> None:
        """Release the OS resources. Called at most once."""

    def _set_fullscreen(self, fullscreen: bool) -> None:
        """
        Put the window into - or take it out of - fullscreen.

        Deliberately not a no-op default: a backend that gained a fullscreen
        argument and never implemented it would otherwise report
        ``fullscreen=True`` from a window that is plainly still a window.
        """
        raise NotImplementedError(
            f"the {self.backend!r} display backend cannot go fullscreen")

    # -- shared machinery ------------------------------------------------------

    def _push_key(self, key: int) -> None:
        if len(self._keys) < 8:
            self._keys.append(key)

    def _take_key(self) -> int:
        return self._keys.pop(0) if self._keys else NO_KEY

    def _check(self, frame: np.ndarray) -> np.ndarray:
        """
        Validate the frame and hand back something safe to upload.

        The contiguity copy is here rather than in each caller because it was
        already the subtle part: a BGR->RGB flip produces a reversed-stride view
        that looks like an ndarray and is not accepted by any buffer upload.
        """
        if self.closed:
            raise RuntimeError(f"window {self.title!r} is closed")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy array, got {type(frame).__name__}")
        if frame.dtype != np.uint8:
            raise ValueError(f"frame must be uint8, got {frame.dtype}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must be HxWx3 BGR, got shape {frame.shape}")
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame is {frame.shape[1]}x{frame.shape[0]}, "
                f"window {self.title!r} is {self.width}x{self.height}")
        return frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame)

    # -- public API ------------------------------------------------------------

    def present(self, frame: np.ndarray) -> int:
        """
        Show `frame`, pump the OS, pace the loop, and return one key.

        The return is `cv2.waitKey`'s: a byte, or `NO_KEY` (255) if nothing was
        pressed. Call this exactly once per frame - it is what counts a frame.
        """
        frame = self._check(frame)
        self._show(frame)
        self.presents += 1
        self._pump()
        if self._paces:
            self._pacer.wait()
        return self._take_key()

    def paint(self, frame: np.ndarray) -> None:
        """
        Put `frame` on screen right now. No pacing, no key, not a frame.

        For the two places that need pixels up before something blocking: a
        loading splash before an import, and a "cutting you out" notice before
        a matte that takes seconds. Under `cv2` both spelled this `imshow` plus
        `waitKey(1)`, which made them indistinguishable from a real frame - so
        the headless driver counted them, timed them, and fed them a scripted
        keypress meant for the game loop.
        """
        frame = self._check(frame)
        self._show(frame)
        self._pump()

    def poll(self) -> int:
        """One key, without presenting or pacing."""
        if not self.closed:
            self._pump()
        return self._take_key()

    def set_fullscreen(self, fullscreen: bool) -> None:
        """
        Go fullscreen, or come back. Idempotent, and takes effect immediately.

        The frame size a game presents is unaffected - the canvas is scaled to
        the screen, not re-sized - so a game may call this from its key
        dispatch without touching a single drawing call.
        """
        if self.closed:
            raise RuntimeError(f"window {self.title!r} is closed")
        fullscreen = bool(fullscreen)
        if fullscreen == self.fullscreen:
            return
        self._set_fullscreen(fullscreen)
        self.fullscreen = fullscreen

    def toggle_fullscreen(self) -> bool:
        """Flip fullscreen and return the new state. What an F11 handler wants."""
        self.set_fullscreen(not self.fullscreen)
        return self.fullscreen

    def set_fps(self, fps: float | None) -> None:
        """Change the render budget. Takes effect from the next frame."""
        self._pacer.set_fps(fps)

    @property
    def fps(self) -> float | None:
        return self._pacer.fps

    def close(self) -> None:
        """Release the window. Idempotent."""
        if self.closed:
            return
        self.closed = True
        _WINDOWS.pop(self.title, None)
        self._shutdown()

    def __enter__(self) -> Window:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self.closed else (
            f"{self.width}x{self.height}" + (" fullscreen" if self.fullscreen else ""))
        return f"<{type(self).__name__} {self.title!r} {state} backend={self.backend}>"


# ── Factory ───────────────────────────────────────────────────────────────────

def open_window(title: str, width: int, height: int, *,
                backend: str | None = None,
                fps: float | None = 60.0,
                vsync: bool = True,
                resizable: bool = True,
                fullscreen: bool | None = None) -> Window:
    """
    Open - or return - the window named `title`.

    `fps` is the render budget; `None` means never wait, which is what an
    unpaced `waitKey(1)` loop does today. `width`/`height` are the *logical*
    frame size and stay fixed; a resizable window scales to fit, the way
    `cv2.WINDOW_NORMAL` did.

    `fullscreen` scales that same logical canvas to the whole screen. `None`,
    the default, means "whatever ``$VISUAL_AI_FULLSCREEN`` says", so a game
    that passes nothing still honours the player's choice; pass `True`/`False`
    to decide for them. Flip it later with `Window.set_fullscreen`.

    Re-opening an existing title returns the same window rather than a second
    one, matching `cv2.namedWindow`. It is returned as it is - the mode a
    caller asks for on a second open is ignored, because the first caller may
    since have toggled it and quietly undoing that would be worse.
    """
    existing = _WINDOWS.get(title)
    if existing is not None and not existing.closed:
        return existing

    name = resolve_backend(backend)
    want_fullscreen = resolve_fullscreen(fullscreen)
    if name == "null":
        from .null import NullWindow as impl  # noqa: PLC0415 - backends load lazily
    elif name == "sdl":
        from .sdl import SDLWindow as impl  # noqa: PLC0415
    else:
        from .d3d11 import D3D11Window as impl  # noqa: PLC0415

    window = impl(title, width, height, fps=fps, vsync=vsync, resizable=resizable,
                  fullscreen=want_fullscreen)
    _WINDOWS[title] = window
    return window


def show(title: str, frame: np.ndarray, *,
         backend: str | None = None,
         fps: float | None = 60.0) -> int:
    """
    Show `frame` in the window named `title`, sizing the window to fit it.

    The line-for-line replacement for ``cv2.imshow(title, frame)`` followed by
    ``cv2.waitKey(1) & 0xFF``, and it exists for the same reason `imshow` had no
    size argument: a script whose canvas is whatever the camera happened to hand
    it cannot state a size up front, and several of the bundled examples start
    on an 800x600 placeholder and switch to the camera's real resolution on the
    first payload. A resize reopens the window.

    **Games should use `open_window` instead.** Its fixed size is a guard, not a
    limitation - a canvas that quietly changes shape mid-game is a bug, and this
    function would paper over it. Reach for this in demos, tools and one-off
    scripts, where the shape really is whatever arrives.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError(f"frame must be HxWx3 BGR, got {getattr(frame, 'shape', frame)!r}")
    height, width = frame.shape[:2]

    window = _WINDOWS.get(title)
    if window is not None and not window.closed \
            and (window.width, window.height) != (width, height):
        window.close()
        window = None
    if window is None or window.closed:
        # Late-bound on purpose: `headless.py` replaces `open_window` on this
        # module, and a direct call to the global is what lets it see the swap.
        window = open_window(title, width, height, backend=backend, fps=fps)
    return window.present(frame)


def find_window(title: str) -> Window | None:
    """The open window with this title, if there is one."""
    window = _WINDOWS.get(title)
    return None if window is None or window.closed else window


def close_all() -> None:
    """Close every open window. The replacement for `cv2.destroyAllWindows`."""
    for window in list(_WINDOWS.values()):
        window.close()
    _WINDOWS.clear()
