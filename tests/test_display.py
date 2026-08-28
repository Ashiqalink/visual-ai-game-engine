"""
The display contract, asserted against every backend this machine has.

`visual_ai.display` exists so a game can present a frame without knowing which
of three very different things is underneath it. That only holds if the three
behave identically where a game can see, so - like `test_engine_parity.py` does
for the two engines - every property here runs against every available backend.
A backend that cannot be reached is skipped, not failed; the ones present still
guard the contract.

The pixel readback test is the important one. Everything else fails loudly; a
channel swap, a wrong row stride or a stray colour conversion fails silently,
on someone else's machine, in a screenshot.
"""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

# Before `visual_ai.display` can import pygame. A developer running the suite
# should not have windows popping up in front of them, and CI has no display at
# all. Overridable, so this file can be pointed at a real driver by hand.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from visual_ai import display  # noqa: E402 - the environment must be set first

BACKENDS = display.available_backends()

#: SDL under the dummy driver has no accelerated renderer, so vsync is off for
#: every test here - the pacing assertions must measure the pacer and not a
#: monitor's refresh rate.
OPTS = {"fps": None, "vsync": False}


@pytest.fixture(autouse=True)
def _no_leaked_windows():
    """Every test starts and ends with no window open."""
    display.close_all()
    yield
    display.close_all()


def frame(width: int = 32, height: int = 16) -> np.ndarray:
    return np.zeros((height, width, 3), np.uint8)


# ── The contract, per backend ─────────────────────────────────────────────────

@pytest.mark.parametrize("backend", BACKENDS)
def test_reports_the_backend_it_actually_opened(backend):
    """`.backend` is the resolved name, never 'auto' and never a lie."""
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    assert win.backend == backend
    assert (win.width, win.height) == (32, 16)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reopening_a_title_returns_the_same_window(backend):
    """
    `cv2.namedWindow` is title-keyed and idempotent, and Sling depends on it:
    it paints a loading splash from module scope, then opens 'its' window again
    once the pipeline is up and expects to be talking to the same one.
    """
    first = display.open_window("t", 32, 16, backend=backend, **OPTS)
    assert display.open_window("t", 32, 16, backend=backend, **OPTS) is first
    assert display.find_window("t") is first

    first.close()
    assert display.find_window("t") is None
    assert display.open_window("t", 32, 16, backend=backend, **OPTS) is not first


@pytest.mark.parametrize("backend", BACKENDS)
def test_present_returns_no_key_when_nothing_was_pressed(backend):
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    assert win.present(frame()) == display.NO_KEY
    assert win.presents == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_paint_shows_a_frame_without_counting_it(backend):
    """
    `paint` is for the two places that need pixels up before something blocking
    - a loading splash, a 'cutting you out' notice. It must not look like a
    frame: the headless driver times and scripts frames, and counting these
    would corrupt both.
    """
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    assert win.paint(frame()) is None
    assert win.presents == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_non_contiguous_frame_is_accepted(backend):
    """
    A BGR<->RGB flip produces a reversed-stride view that is still an ndarray
    and is not something any buffer upload will take. Callers used to copy it
    themselves, one at a time, and the ones that forgot found out at runtime.
    """
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    flipped = frame()[:, :, ::-1]
    assert not flipped.flags["C_CONTIGUOUS"]
    assert win.present(flipped) == display.NO_KEY


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_wrong_frame_is_refused_the_same_way_everywhere(backend):
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)

    with pytest.raises(ValueError, match="32x16"):
        win.present(frame(64, 48))
    with pytest.raises(ValueError, match="uint8"):
        win.present(np.zeros((16, 32, 3), np.float32))
    with pytest.raises(ValueError, match="HxWx3"):
        win.present(np.zeros((16, 32), np.uint8))
    with pytest.raises(ValueError, match="HxWx3"):
        win.present(np.zeros((16, 32, 4), np.uint8))
    with pytest.raises(TypeError, match="numpy array"):
        win.present([[0, 0, 0]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_close_is_idempotent_and_presenting_after_it_raises(backend):
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    win.close()
    win.close()
    assert win.closed
    with pytest.raises(RuntimeError, match="closed"):
        win.present(frame())


@pytest.mark.parametrize("backend", BACKENDS)
def test_context_manager_closes(backend):
    with display.open_window("t", 32, 16, backend=backend, **OPTS) as win:
        win.present(frame())
    assert win.closed


# ── Pacing ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend", BACKENDS)
def test_fps_none_does_not_wait(backend):
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    canvas = frame()
    start = time.perf_counter()
    for _ in range(30):
        win.present(canvas)
    assert time.perf_counter() - start < 0.25


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "null"])
def test_fps_60_paces_the_loop(backend):
    """
    30 presents at 60 fps is 29 waited frames, ~0.48 s. The band is wide enough
    for a loaded machine and far too tight to pass unpaced (0.04 s) or
    double-paced (0.97 s).
    """
    win = display.open_window("t", 32, 16, backend=backend, fps=60.0, vsync=False)
    canvas = frame()
    start = time.perf_counter()
    for _ in range(30):
        win.present(canvas)
    assert 0.39 < time.perf_counter() - start < 0.62


def test_null_never_paces_however_it_is_asked():
    """
    The load-bearing property of the headless seam. `headless.drive` reports
    the gap between consecutive presents as the cost of one update-and-render
    iteration, which is only true because nothing in between sleeps. Honouring
    `fps` here would quietly rewrite every recorded frame time as 16.7 ms, and
    the regression gate does not assert on timings, so nothing would notice.
    """
    win = display.open_window("t", 32, 16, backend="null", fps=60.0)
    canvas = frame()
    start = time.perf_counter()
    for _ in range(30):
        win.present(canvas)
    assert time.perf_counter() - start < 0.05


def test_set_fps_changes_the_budget_without_skipping_a_frame():
    win = display.open_window("t", 32, 16, backend="null", fps=None)
    assert win.fps is None
    win.set_fps(30.0)
    assert win.fps == pytest.approx(30.0)
    with pytest.raises(ValueError, match="positive"):
        win.set_fps(0)


# ── Pixels ────────────────────────────────────────────────────────────────────

@pytest.mark.skipif("sdl" not in BACKENDS, reason="needs pygame")
def test_a_bgr_frame_reaches_the_surface_as_bgr():
    """
    The test that catches a silent renderer bug.

    Present a frame whose two halves are unambiguously blue and red *in BGR*,
    read the surface back, and check the colours survived. A channel swap, a
    wrong row stride or an accidental sRGB conversion all fail here and are
    invisible everywhere else.
    """
    import pygame

    canvas = np.zeros((16, 32, 3), np.uint8)
    canvas[:, :16] = (255, 0, 0)      # BGR blue
    canvas[:, 16:] = (0, 0, 255)      # BGR red

    win = display.open_window("t", 32, 16, backend="sdl", **OPTS)
    win.present(canvas)

    shown = pygame.surfarray.array3d(pygame.display.get_surface()).swapaxes(0, 1)
    assert tuple(shown[0, 0]) == (0, 0, 255)      # RGB blue
    assert tuple(shown[0, 31]) == (255, 0, 0)     # RGB red


@pytest.mark.skipif("sdl" not in BACKENDS, reason="needs pygame")
def test_the_upload_does_not_copy_the_frame():
    """
    Guards the whole point of the module.

    `frombuffer(frame.data, size, "BGR")` wraps the caller's buffer; the
    ``ascontiguousarray(frame[:, :, ::-1]).tobytes()`` form that an OpenCV
    habit reaches for measures 6.17 ms at 1080p against 0.00075 ms for the
    wrap. A regression to it would eat more than this module saves and would
    look identical on screen, so assert the surface really is a view: mutate
    the source array and watch the surface change.
    """
    import pygame

    canvas = np.zeros((16, 32, 3), np.uint8)
    surface = pygame.image.frombuffer(canvas.data, (32, 16), "BGR")
    canvas[0, 0] = (255, 0, 0)
    assert surface.get_at((0, 0))[:3] == (0, 0, 255)


# ── Keys ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif("sdl" not in BACKENDS, reason="needs pygame")
@pytest.mark.parametrize("key, unicode_, expected", [
    ("K_q", "q", ord("q")),
    ("K_q", "Q", ord("Q")),
    ("K_ESCAPE", "\x1b", 27),
    ("K_SPACE", " ", 32),
    ("K_UP", "", 0),
    ("K_F1", "", 0),
])
def test_keys_arrive_as_masked_bytes(key, unicode_, expected):
    """
    `cv2.waitKey`'s contract, reproduced exactly, because every dispatch chain
    in both repos was written against it.

    The arrow and F-key rows are the ones worth having: cv2 returns 2490368 for
    Up on Windows and the games mask it to a byte, so those loops see 0 - and
    three of them read "not 255" as *any key dismisses this card*. A backend
    returning 255 for an arrow would leave those cards looking broken.
    """
    import pygame

    win = display.open_window("t", 32, 16, backend="sdl", **OPTS)
    pygame.event.post(pygame.event.Event(
        pygame.KEYDOWN, key=getattr(pygame, key), unicode=unicode_, mod=0))
    assert win.present(frame()) == expected


@pytest.mark.skipif("sdl" not in BACKENDS, reason="needs pygame")
def test_closing_the_window_reads_as_escape():
    """
    New behaviour, and wanted: nothing in either repo calls `getWindowProperty`,
    so no loop can currently notice its window being closed - the X button
    kills the process mid-run. Every `27` handler in both repos means quit, so
    reporting a close as Escape shuts every existing loop down cleanly.
    """
    import pygame

    win = display.open_window("t", 32, 16, backend="sdl", **OPTS)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert win.present(frame()) == 27
    assert win.closed is True


@pytest.mark.skipif("sdl" not in BACKENDS, reason="needs pygame")
def test_one_key_per_frame_and_no_unbounded_backlog():
    """
    Every dispatch chain reads one key per frame, so the queue is FIFO and
    capped - a player holding a key down must not build a backlog the game
    replays after they stop.
    """
    import pygame

    win = display.open_window("t", 32, 16, backend="sdl", **OPTS)
    for ch in "abc":
        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_a, unicode=ch, mod=0))
    assert [win.present(frame()) for _ in range(4)] == [
        ord("a"), ord("b"), ord("c"), display.NO_KEY]

    for _ in range(50):
        pygame.event.post(pygame.event.Event(
            pygame.KEYDOWN, key=pygame.K_z, unicode="z", mod=0))
    win.present(frame())
    assert len(win._keys) <= 8


@pytest.mark.skipif("sdl" not in BACKENDS, reason="needs pygame")
def test_poll_returns_a_key_without_presenting():
    import pygame

    win = display.open_window("t", 32, 16, backend="sdl", **OPTS)
    pygame.event.post(pygame.event.Event(
        pygame.KEYDOWN, key=pygame.K_h, unicode="h", mod=0))
    assert win.poll() == ord("h")
    assert win.presents == 0


# ── Backend selection ─────────────────────────────────────────────────────────

def test_explicit_backend_beats_the_environment(monkeypatch):
    monkeypatch.setenv(display.ENV_VAR, "null")
    assert display.resolve_backend("sdl") == "sdl"


def test_the_environment_is_used_when_nothing_was_asked_for(monkeypatch):
    monkeypatch.setenv(display.ENV_VAR, "null")
    assert display.resolve_backend() == "null"


def test_auto_never_resolves_to_null(monkeypatch):
    """
    A game that silently renders nothing looks exactly like a game that works,
    so `null` has to be asked for by name.
    """
    monkeypatch.delenv(display.ENV_VAR, raising=False)
    assert display.resolve_backend() != "null"


def test_auto_falls_back_to_sdl_when_d3d11_is_not_built(monkeypatch):
    monkeypatch.delenv(display.ENV_VAR, raising=False)
    monkeypatch.setitem(display._PROBES, "d3d11", lambda: False)
    assert display.resolve_backend() == "sdl"


def test_a_named_backend_never_falls_back(monkeypatch):
    """
    Only `auto` degrades quietly. Asking for `d3d11` and silently getting `sdl`
    would mean the tuned path is never exercised and nobody finds out - which
    is exactly what a stale prebuilt .pyd would cause.
    """
    monkeypatch.setitem(display._PROBES, "d3d11", lambda: False)
    with pytest.raises(RuntimeError, match=display.ENV_VAR):
        display.resolve_backend("d3d11")


def test_an_unknown_backend_names_the_valid_ones():
    with pytest.raises(ValueError, match="sdl"):
        display.resolve_backend("opengl")


def test_no_backend_at_all_is_an_error_not_a_silent_null(monkeypatch):
    monkeypatch.delenv(display.ENV_VAR, raising=False)
    monkeypatch.setitem(display._PROBES, "d3d11", lambda: False)
    monkeypatch.setitem(display._PROBES, "sdl", lambda: False)
    with pytest.raises(RuntimeError, match="no display backend"):
        display.resolve_backend()


def test_close_all_closes_every_window():
    a = display.open_window("a", 32, 16, backend="null")
    b = display.open_window("b", 32, 16, backend="null")
    display.close_all()
    assert a.closed and b.closed
    assert display.find_window("a") is None


# ── Fullscreen ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend", BACKENDS)
def test_a_window_is_not_fullscreen_unless_asked(backend):
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    assert win.fullscreen is False


@pytest.mark.parametrize("backend", BACKENDS)
def test_fullscreen_does_not_change_the_frame_a_game_hands_over(backend):
    """
    The whole point of scaling rather than resizing: a canvas that was legal
    windowed stays legal fullscreen. If this ever stopped holding, every game
    would start raising from `_check` the moment a player pressed the key.
    """
    win = display.open_window("t", 32, 16, backend=backend, fullscreen=True, **OPTS)
    assert (win.width, win.height) == (32, 16)
    win.present(frame())
    win.toggle_fullscreen()
    assert (win.width, win.height) == (32, 16)
    win.present(frame())
    assert win.presents == 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_toggle_reports_and_remembers_the_new_state(backend):
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    assert win.toggle_fullscreen() is True
    assert win.fullscreen is True
    assert win.toggle_fullscreen() is False
    assert win.fullscreen is False


@pytest.mark.parametrize("backend", BACKENDS)
def test_setting_the_mode_it_is_already_in_touches_nothing(backend):
    """
    Idempotent because a game may well call this every frame from a held key,
    and on SDL every non-idempotent call would rebuild the surface.
    """
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    calls = []
    monkey = win._set_fullscreen
    win._set_fullscreen = lambda flag: (calls.append(flag), monkey(flag))[1]
    win.set_fullscreen(False)
    assert calls == []
    win.set_fullscreen(True)
    win.set_fullscreen(True)
    assert calls == [True]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_closed_window_cannot_change_mode(backend):
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    win.close()
    with pytest.raises(RuntimeError, match="closed"):
        win.set_fullscreen(True)


def test_a_backend_that_never_implemented_it_says_so():
    """
    The default hook raises rather than passing: a window still plainly a
    window while `.fullscreen` reads True is the failure nobody would see.
    """
    win = display.Window("t", 32, 16, fps=None)
    with pytest.raises(NotImplementedError, match="fullscreen"):
        win.set_fullscreen(True)
    assert win.fullscreen is False


@pytest.mark.skipif("sdl" not in BACKENDS, reason="pygame is not installed")
def test_sdl_asks_for_a_scaled_fullscreen_mode(monkeypatch):
    """
    The state flag is cheap to set and proves nothing on its own - under the
    dummy driver a window that never went fullscreen looks identical. So
    assert on the flags SDL was actually handed when the window was made.

    SCALED is the half that matters: without it SDL changes the video mode to
    the logical size instead of scaling the canvas onto the desktop, which on
    a real screen is a 32x16 monitor and on this test a passing assertion.
    """
    import pygame

    from visual_ai.display import sdl as sdl_backend

    seen = []
    real = sdl_backend.pygame.display.set_mode

    def spy(size, flags=0, *args, **kwargs):
        seen.append(flags)
        return real(size, flags, *args, **kwargs)

    monkeypatch.setattr(sdl_backend.pygame.display, "set_mode", spy)
    display.open_window("t", 32, 16, backend="sdl", fullscreen=True, **OPTS)

    assert seen, "no mode was set at all"
    assert seen[0] & pygame.FULLSCREEN
    assert seen[0] & pygame.SCALED


@pytest.mark.skipif("sdl" not in BACKENDS, reason="pygame is not installed")
def test_sdl_refuses_to_report_a_fullscreen_it_did_not_get(monkeypatch):
    """
    `toggle_fullscreen` returns 0 whether or not it did anything, so the
    backend re-reads the surface. Simulate a driver that shrugs.
    """
    import pygame

    from visual_ai.display import sdl as sdl_backend

    win = display.open_window("t", 32, 16, backend="sdl", **OPTS)
    monkeypatch.setattr(sdl_backend.pygame.display, "toggle_fullscreen", lambda: 0)
    monkeypatch.setattr(sdl_backend.pygame.display, "set_mode",
                        lambda *a, **k: pygame.display.get_surface())

    with pytest.raises(RuntimeError, match="ignored the request"):
        win.set_fullscreen(True)
    assert win.fullscreen is False


# ── The player's fullscreen switch ────────────────────────────────────────────

def test_fullscreen_defaults_to_windowed(monkeypatch):
    monkeypatch.delenv(display.FULLSCREEN_ENV_VAR, raising=False)
    assert display.resolve_fullscreen() is False


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("no", False), ("off", False),
    ("", False), ("  ", False),
])
def test_the_environment_switch_reads_yes_and_no(monkeypatch, raw, expected):
    monkeypatch.setenv(display.FULLSCREEN_ENV_VAR, raw)
    assert display.resolve_fullscreen() is expected


def test_an_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv(display.FULLSCREEN_ENV_VAR, "1")
    assert display.resolve_fullscreen(False) is False
    monkeypatch.setenv(display.FULLSCREEN_ENV_VAR, "0")
    assert display.resolve_fullscreen(True) is True


def test_a_value_that_is_neither_is_an_error_not_a_no(monkeypatch):
    """A player who typed something wrong must be told, not quietly windowed."""
    monkeypatch.setenv(display.FULLSCREEN_ENV_VAR, "full")
    with pytest.raises(ValueError, match=display.FULLSCREEN_ENV_VAR):
        display.resolve_fullscreen()


@pytest.mark.parametrize("backend", BACKENDS)
def test_open_window_honours_the_environment_switch(monkeypatch, backend):
    """
    The point of the variable: every game hard-codes its `open_window` call,
    so a player with no way in through the argument still gets fullscreen.
    """
    monkeypatch.setenv(display.FULLSCREEN_ENV_VAR, "1")
    win = display.open_window("t", 32, 16, backend=backend, **OPTS)
    assert win.fullscreen is True


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_game_that_decides_for_itself_ignores_the_environment(monkeypatch, backend):
    monkeypatch.setenv(display.FULLSCREEN_ENV_VAR, "1")
    win = display.open_window("t", 32, 16, backend=backend, fullscreen=False, **OPTS)
    assert win.fullscreen is False


# ── Import cost ───────────────────────────────────────────────────────────────

def test_importing_visual_ai_does_not_import_pygame():
    """
    `visual_ai/__init__.py` must not import this module, and probing for a
    backend must not import pygame. Both would put SDL's startup cost on every
    game that never opens a window - the same argument the package already
    makes for deferring rembg.
    """
    probe = (
        "import sys; import visual_ai; "
        "sys.exit(1 if 'pygame' in sys.modules else 0)"
    )
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    env = dict(os.environ, PYTHONPATH=src)
    assert subprocess.run([sys.executable, "-c", probe], env=env).returncode == 0

    probe = (
        "import sys; from visual_ai import display; display.available_backends(); "
        "sys.exit(1 if 'pygame' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", probe], env=env).returncode == 0


# ── show() ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend", BACKENDS)
def test_show_opens_a_window_sized_to_the_frame(backend):
    assert display.show("t", frame(64, 48), backend=backend, fps=None) == display.NO_KEY
    win = display.find_window("t")
    assert (win.width, win.height) == (64, 48)
    assert win.presents == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_show_reopens_when_the_frame_changes_size(backend):
    """
    The case the bundled demos actually hit: they start on an 800x600
    placeholder and switch to the camera's real resolution on the first
    payload. `open_window` would raise here, correctly - `show` is the spelling
    that is allowed to follow the frame.
    """
    display.show("t", frame(64, 48), backend=backend, fps=None)
    first = display.find_window("t")
    display.show("t", frame(32, 16), backend=backend, fps=None)
    second = display.find_window("t")

    assert first.closed
    assert second is not first
    assert (second.width, second.height) == (32, 16)


@pytest.mark.parametrize("backend", BACKENDS)
def test_show_keeps_the_same_window_at_a_steady_size(backend):
    """A resize is a reopen; a steady stream of frames must not be."""
    display.show("t", frame(), backend=backend, fps=None)
    win = display.find_window("t")
    for _ in range(4):
        display.show("t", frame(), backend=backend, fps=None)
    assert display.find_window("t") is win
    assert win.presents == 5


def test_show_goes_through_the_module_level_open_window(monkeypatch):
    """
    The seam `headless.py` depends on. It replaces `open_window` on this module
    object, so `show` has to reach it by a late global lookup - resolving it at
    def time would opt every `show` caller out of the regression harness
    silently, which is the failure mode that motivated the same rule for games.
    """
    calls = []
    real = display.open_window

    def spy(*a, **kw):
        calls.append(a)
        return real(*a, **kw)

    monkeypatch.setattr(display, "open_window", spy)
    display.show("t", frame(), backend="null")
    assert calls == [("t", 32, 16)]


def test_show_refuses_a_frame_that_is_not_an_image():
    with pytest.raises(ValueError, match="HxWx3"):
        display.show("t", np.zeros((16, 32), np.uint8), backend="null")
