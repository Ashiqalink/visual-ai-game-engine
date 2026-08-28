"""
The one key truth table, shared by every backend.

`cv2.waitKey` is what every game in this workspace was written against, so its
return contract is the contract here: a byte, `255` when nothing was pressed.
Reproducing it exactly is what lets a game swap renderers without touching its
key dispatch, and the three rows below are the ones that are invisible when
they are wrong:

* **Printable keys come from the text event, not the keycode.** Shift+q must
  give ``'Q'`` and a non-US layout must give the character the player actually
  typed. SDL's ``event.unicode`` and Win32's ``WM_CHAR`` both already did the
  layout work; deriving a character from a raw keycode would undo it.

* **Escape is explicit.** It has a text form (``'\\x1b'``) on some layouts and
  not others, so it is mapped from the keycode and never from text.

* **An unmapped key returns 0, not 255.** `cv2.waitKey` returns 2490368 for Up
  on Windows and the games mask it to a byte, so an arrow key arrives as 0 —
  and three loops read "not 255" as *any key dismisses this card*
  (`labkit.py`, `punchy/tof_punch.py`, `Sling/main.py`). Returning 255 for
  arrows would make those cards look broken while nothing raised.

`NO_KEY` is `cv2`'s idle value and `headless.py`'s `IDLE` — the same 255, now
named in one place.
"""

from __future__ import annotations

#: What a masked `waitKey` returns when no key arrived.
NO_KEY = 255

#: What a masked `waitKey` returns for a key with no byte representation.
UNMAPPED = 0

#: Escape, spelled once.
ESCAPE = 27

#: Closing the window reports itself as Escape.
#
# Nothing in either repo calls `getWindowProperty`, so no loop can currently
# notice its window being closed - the X button kills the process and takes the
# run log with it. Every `27` handler in both repos means "quit", so reporting a
# close as Escape makes every existing loop shut down cleanly with no new code.
# `Window.closed` is there for anyone who needs to tell the two apart.
CLOSE = ESCAPE


def from_text(text: str) -> int | None:
    """
    The byte a text event carries, or None if it carries no single byte.

    Anything outside Latin-1 has no byte form, so it is left to the keycode
    path rather than being truncated into a different key.
    """
    if not text or len(text) != 1:
        return None
    code = ord(text)
    if code > 0xFF:
        return None
    return code
