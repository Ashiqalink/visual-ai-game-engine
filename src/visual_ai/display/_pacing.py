"""
The frame pacer, moved into the SDK.

This is `gameloop.frame_pacer`'s algorithm, unchanged in behaviour and kept for
the same reason it was written: without a governor a render loop spins at a few
hundred iterations a second while the pipeline delivers thirty payloads, so the
same unchanged state is cleared, drawn and blitted eight times per camera frame.
That is where a laptop's heat goes.

Two properties are load-bearing and easy to lose in a rewrite:

* **Re-base, never accumulate.** A frame that overran its budget must not bank
  credit and let the next few run back-to-back; that converts one slow frame
  into a visible stutter. `max(now, deadline + dt)` drops the debt.

* **Waiting is separate from presenting.** Under OpenCV the two were the same
  call, and asking `waitKey` for exactly the remaining milliseconds rounded up
  over highgui's ~15.9 ms tick and cost a whole extra tick - which is what had
  punchy running at 32 fps against a 60 fps budget. Here the wait is a real
  sleep with no tick to round over, so the remaining time is simply correct.
  (Python 3.11 moved `time.sleep` onto a high-resolution waitable timer on
  Windows, so sub-tick sleeps are honoured; the floor was 15.6 ms before that.)

`vsync` and this pacer compose rather than fight: a vsynced present already
blocks until the refresh, so at 60 Hz there is nothing left to sleep and at
144 Hz there is about half a frame left. Neither waits twice.
"""

from __future__ import annotations

import time


class Pacer:
    """A deadline, advanced once per frame. `fps=None` disables the wait."""

    __slots__ = ("_dt", "_deadline")

    def __init__(self, fps: float | None) -> None:
        self._deadline: float | None = None
        self.set_fps(fps)

    def set_fps(self, fps: float | None) -> None:
        if fps is None:
            self._dt = None
            return
        fps = float(fps)
        if fps <= 0.0:
            raise ValueError(f"fps must be positive or None, got {fps!r}")
        self._dt = 1.0 / fps
        # Leave the deadline alone: changing the rate mid-run should take
        # effect from the next frame, not skip or double one.

    @property
    def fps(self) -> float | None:
        return None if self._dt is None else 1.0 / self._dt

    def wait(self) -> None:
        """Sleep out whatever is left of this frame's budget."""
        if self._dt is None:
            return
        if self._deadline is None:
            # First frame: there is no budget to wait out yet, only one to set.
            self._deadline = time.perf_counter() + self._dt
            return
        remaining = self._deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        self._deadline = max(time.perf_counter(), self._deadline + self._dt)
