"""Which OpenCV capture backend to ask for, per platform.

`cv2.CAP_DSHOW` is a Windows-only constant that names a Windows-only API. Code
that hardcodes it does not merely lose a fast path on macOS and Linux -- it
asks for a backend that is not there, and the capture fails or silently falls
back depending on the OpenCV build. Anything opening a camera outside
`VisionPipeline` goes through here instead.

The pipeline itself deliberately calls `cv2.VideoCapture(index)` with no
backend at all and is left alone: letting OpenCV choose has worked on every
machine this has run on, and changing it would alter behaviour on Windows for
no gain.
"""

import sys

try:
    import cv2
except ImportError:                                    # pragma: no cover
    cv2 = None


def default_backend() -> int:
    """The capture API to request on this platform.

    DirectShow on Windows (more predictable than MSMF for webcams and the only
    one that reliably honours FOURCC requests), AVFoundation on macOS, V4L2 on
    Linux. Falls back to `CAP_ANY` -- OpenCV's own choice -- if a constant is
    missing from this build.
    """
    if cv2 is None:                                    # pragma: no cover
        return 0
    if sys.platform == "win32":
        return getattr(cv2, "CAP_DSHOW", 0)
    if sys.platform == "darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", 0)
    return getattr(cv2, "CAP_V4L2", 0)


def open_camera(index: int = 0, width: int | None = None, height: int | None = None,
                backend: int | None = None) -> "cv2.VideoCapture | None":
    """Open a camera with the platform's backend. Returns an open cap or None.

    Never raises and never hands back a capture that opened but cannot
    produce a frame -- on macOS in particular, a camera the user has not
    granted permission to opens cleanly and then reads nothing at all, which
    is otherwise diagnosed as "the camera is broken" rather than "the OS
    denied us".
    """
    if cv2 is None:                                    # pragma: no cover
        return None
    cap = cv2.VideoCapture(index, default_backend() if backend is None else backend)
    if not cap.isOpened():
        cap.release()
        return None
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    return cap
