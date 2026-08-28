"""
The backend that opens nothing.

Two callers, and the difference between them matters:

* A benchmark or a CI smoke run that wants the game's real code path without a
  display. It gets full frame validation - a game that starts handing over a
  wrong-shaped canvas must fail here, not sail through and fail on a machine
  with a monitor.

* `headless.py`'s regression driver, which subclasses this to add a scripted
  key feed and per-frame timestamps.

**This backend must never sleep.** The headless driver's entire timing report
is `[t1-t0, t2-t1, ...]` across consecutive `present` calls, and that is only
the cost of one update-and-render iteration because nothing in between waits.
Honouring `fps` here would silently turn every recorded frame time into
16.7 ms and no test would notice. `_paces = False` is the enforcement, and
`test_null_never_paces_however_it_is_asked` is the check.
"""

from __future__ import annotations

import numpy as np

from . import Window


class NullWindow(Window):
    """A window that validates and counts, and does nothing else."""

    _paces = False

    def __init__(self, title: str, width: int, height: int, *,
                 fps: float | None = 60.0,
                 vsync: bool = True,
                 resizable: bool = True,
                 fullscreen: bool = False) -> None:
        super().__init__(title, width, height, fps=fps, fullscreen=fullscreen)
        self.backend = "null"

    def _show(self, frame: np.ndarray) -> None:
        """Deliberately empty: the frame was validated, and there is no screen."""

    def _set_fullscreen(self, fullscreen: bool) -> None:
        """
        Deliberately empty: there is no screen to fill.

        Tracked all the same, because a headless run exists to exercise the
        game's real code path - one that toggles fullscreen must not raise
        here and pass on a machine with a monitor.
        """
