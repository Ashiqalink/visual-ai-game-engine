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


