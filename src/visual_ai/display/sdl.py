"""
The pygame/SDL backend. Portable, and what CI runs.

SDL picks a hardware presentation path per platform (D3D11 on Windows, Metal on
macOS, GL elsewhere), which is why it costs 2.62 ms at 1080p against OpenCV's
24.0 ms on the same machine.

Two things here are not obvious and both were paid for once already:

**The frame is uploaded with no copy at all.** ``frombuffer(frame.data, size,
"BGR")`` wraps the game's own buffer - SDL understands BGR, so the BGR->RGB
flip every OpenCV-shaped renderer reaches for is simply not needed. That flip
is not free: ``ascontiguousarray(frame[:, :, ::-1]).tobytes()`` measures
**6.17 ms** at 1080p, against **0.00075 ms** for the buffer wrap. Getting this
wrong costs more than the entire saving this module exists for, and looks
identical on screen. `tests/test_display.py` pins it.

**Only the video subsystem is initialised.** ``pygame.init()`` also starts the
mixer and prints a community banner to stdout - and `run_lab_tests.py` parses
the headless report as JSON off stdout, so a full init breaks the regression
gate with a message about invalid JSON that says nothing about pygame.
``PYGAME_HIDE_SUPPORT_PROMPT`` is belt and braces for anything that imports
pygame after us.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402 - the environment must be set before the import

from . import Window
from ._keys import CLOSE, ESCAPE, UNMAPPED, from_text


class SDLWindow(Window):
    """A window backed by an SDL surface."""

    def __init__(self, title: str, width: int, height: int, *,
                 fps: float | None = 60.0,
                 vsync: bool = True,
                 resizable: bool = True,
                 fullscreen: bool = False) -> None:
        super().__init__(title, width, height, fps=fps, fullscreen=fullscreen)
        self.backend = "sdl"
        self._resizable = bool(resizable)

        if not pygame.display.get_init():
            pygame.display.init()
        pygame.display.set_caption(title)

        self.vsync = bool(vsync)
        self._set_mode(self.fullscreen)

    def _set_mode(self, fullscreen: bool) -> None:
        """(Re)create the surface. The one place a display mode is chosen."""
        # SCALED keeps the drawing surface at the logical size and lets SDL
        # stretch it to whatever the window has been dragged to, which is what
        # cv2.WINDOW_NORMAL did. Paired with FULLSCREEN it is also what makes
        # fullscreen free of consequences for a game: SDL takes the desktop
        # resolution and scales the same logical canvas onto it, so no frame
        # a game hands over ever changes shape. Some drivers refuse a vsynced
        # mode outright, so each fallback drops one request rather than
        # giving up.
        flags = pygame.SCALED
        if self._resizable:
            flags |= pygame.RESIZABLE
        if fullscreen:
            flags |= pygame.FULLSCREEN
        size = (self.width, self.height)
        try:
            self._screen = pygame.display.set_mode(size, flags,
                                                   vsync=1 if self.vsync else 0)
        except pygame.error:
            try:
                self._screen = pygame.display.set_mode(size, flags)
                self.vsync = False
            except pygame.error:
                self._screen = pygame.display.set_mode(size, flags & pygame.FULLSCREEN)
                self.vsync = False

    def _set_fullscreen(self, fullscreen: bool) -> None:
        # `toggle_fullscreen`, not another `set_mode`: a SCALED window cannot
        # be re-`set_mode` at all. Measured on the development machine, on the
        # real `windows` driver as well as the dummy one, the second call
        # raises ``pygame.error: failed to create renderer`` and takes the
        # game's surface with it. `toggle_fullscreen` is supported on every
        # platform for a SCALED window and keeps the same surface object -
        # verified here at 320x180: window 1280x720 -> 1440x900 (the desktop),
        # logical size unchanged.
        try:
            pygame.display.toggle_fullscreen()
        except pygame.error:
            # The dummy driver CI runs has no toggle. Rebuilding the mode does
            # work there, because the renderer it fails to create is the
            # SCALED one and dropping SCALED is free with nothing on screen.
            self._set_mode(fullscreen)
        self._screen = pygame.display.get_surface()

        # The check that can fail: `toggle_fullscreen` returns 0 whether or not
        # it did anything, so without this a driver that ignored us would leave
        # the base class reporting a fullscreen window that is still a window.
        if bool(self._screen.get_flags() & pygame.FULLSCREEN) != fullscreen:
            raise RuntimeError(
                f"SDL ignored the request to "
                f"{'enter' if fullscreen else 'leave'} fullscreen "
                f"(driver {pygame.display.get_driver()!r})")

    def _show(self, frame: np.ndarray) -> None:
        surface = pygame.image.frombuffer(frame.data, (self.width, self.height), "BGR")
        self._screen.blit(surface, (0, 0))
        pygame.display.flip()

    def _pump(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                self._push_key(CLOSE)
            elif event.type == pygame.KEYDOWN:
                self._push_key(self._translate(event))

    @staticmethod
    def _translate(event: pygame.event.Event) -> int:
        # Escape before text: it has a text form on some layouts and not
        # others, so reading it from the keycode is the only stable spelling.
        if event.key == pygame.K_ESCAPE:
            return ESCAPE
        code = from_text(getattr(event, "unicode", ""))
        return UNMAPPED if code is None else code

    def _shutdown(self) -> None:
        # Quit the whole video subsystem only when this was the last window;
        # the registry in __init__ is what knows, so ask it rather than
        # guessing. A second window would otherwise lose its surface.
        from . import _WINDOWS  # noqa: PLC0415 - circular by design, resolved late

        if not any(isinstance(w, SDLWindow) and not w.closed
                   for w in _WINDOWS.values()):
            pygame.display.quit()
