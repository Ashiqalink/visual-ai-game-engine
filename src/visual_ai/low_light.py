"""Adaptive low-light boost for the RGB tracking path.

MediaPipe's hand graph fails long before a human would call a room "dark": the
frame goes low-contrast and noisy, the palm stops separating from the
background, and detection drops out entirely. What it hands back is not a bad
landmark but no hand at all, so the game sees a dropout and the player sees a
bird drifting to the middle of the screen.

`LowLightBoost` sits between capture and detection. It measures the frame's
luminance, and when that falls below a threshold it lifts the image with a
gain and a CLAHE pass on the luma channel only:

    gain     recovers exposure -- a linear multiply, so hue is untouched
    CLAHE    recovers local contrast, which is what the detector actually
             needs; a global stretch just moves the whole histogram and leaves
             the palm as flat as it was

Three things keep it from doing harm:

* It is adaptive and one-sided. At normal exposure the gain solves to 1.0, and
  unless the frame also has a genuinely dark region the frame is returned
  untouched, with no allocation and no CLAHE.
* The gain is slewed, not snapped. A hand moving across a lamp swings frame
  luminance hard, and a per-frame gain would pump the whole image in step with
  it -- which reads as flicker and moves the landmarks.
* Gain is capped. Past roughly 4x, amplified sensor noise costs the detector
  more than the exposure gains it, so the cap is a real limit rather than a
  guard rail.

Luminance is measured on a subsampled grayscale view, not the full frame: at
stride 8 it is ~60x less work and the mean is within a fraction of a level.
"""

import cv2
import numpy as np

# Below this mean luma (0-255) the frame is treated as underexposed. Chosen
# from the detector's behaviour rather than perception: MediaPipe holds up
# well down to about here, and falls off a cliff below it.
DARK_LUMA = 90.0

# What the boost aims for. Not 128 -- pushing a genuinely dark frame that far
# means gains near the cap on every frame, and the noise comes up with it.
TARGET_LUMA = 110.0

# Amplified sensor noise beats the exposure win past this.
MAX_GAIN = 4.0

# Fraction of the way to the newly measured gain each frame. At 30 fps this
# settles a lighting change in ~0.3 s: fast enough not to feel sticky, slow
# enough that a hand waving over a lamp does not pump the image.
GAIN_SLEW = 0.25

# CLAHE parameters. A small clip limit: the point is to lift local contrast on
# the hand, not to sharpen noise in the flat dark regions around it.
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# Every Nth pixel in each axis when measuring luminance.
MEASURE_STRIDE = 8

# A frame can be bright on average and still hide the player. A lamp, a window
# or the monitor behind them holds the mean above DARK_LUMA while they sit in
# shadow, so the gain solves to 1.0 and nothing happens -- which is exactly the
# "it only works if I come close" report: walking in fills the frame with the
# dark subject and finally drags the mean down.
#
# Gain is the wrong tool there. Lifting a frame whose mean already meets the
# target only blows out the background. CLAHE is the right tool, because it
# works per tile and can lift the shadow the hand is in without touching the
# lit half. So the dark *region* gets its own gate, and when only that one
# trips the frame is equalised at gain 1.0.
#
# The threshold is deliberately deep shadow, not merely dim: at 30 fps this
# pass is not free, and an ordinary room with a dark jumper in it should not
# pay for it every frame.
DARK_REGION_LUMA = 45.0

#: Which percentile of the subsampled luma counts as "the dark part". Low,
#: because a player standing back is a small share of the frame -- but not so
#: low that a single dark speck (hair, a cable, a monitor bezel) trips it.
DARK_PERCENTILE = 10.0


def _luma_view(bgr_frame: np.ndarray, stride: int = MEASURE_STRIDE) -> np.ndarray:
    """Subsampled Rec. 601 luma plane -- what CLAHE will work on below."""
    view = bgr_frame[::stride, ::stride]
    return (view[:, :, 0] * 0.114 + view[:, :, 1] * 0.587 + view[:, :, 2] * 0.299)


def measure_luma(bgr_frame: np.ndarray, stride: int = MEASURE_STRIDE) -> float:
    """Mean luma (0-255) of a subsampled view of a BGR frame."""
    return float(_luma_view(bgr_frame, stride).mean())


class LowLightBoost:
    """Stateful adaptive boost. One instance per capture stream.

    Attributes
    ----------
    luma : float
        Mean luma of the last frame seen, before boosting.
    dark_region : float
        Luma at `DARK_PERCENTILE` of the last frame. Low with a bright mean
        means a shadowed subject in front of a lit background.
    gain : float
        The exposure gain currently applied. 1.0 means no exposure change,
        which no longer implies the frame was untouched -- CLAHE can still
        have run on a dark region. `active` is the idle test.
    active : bool
        Whether the last frame was actually boosted, by gain or by CLAHE.
    """

    def __init__(self, dark_luma: float = DARK_LUMA, target_luma: float = TARGET_LUMA,
                 max_gain: float = MAX_GAIN, slew: float = GAIN_SLEW,
                 clahe_clip: float = CLAHE_CLIP,
                 clahe_grid: tuple[int, int] = CLAHE_GRID,
                 dark_region_luma: float = DARK_REGION_LUMA) -> None:
        self.dark_luma = float(dark_luma)
        self.target_luma = float(target_luma)
        self.max_gain = float(max_gain)
        self.slew = float(slew)
        self.dark_region_luma = float(dark_region_luma)
        self.luma = 0.0
        self.dark_region = 0.0
        self.gain = 1.0
        self.active = False
        # CLAHE objects hold their own tile state; build once, not per frame.
        self._clahe = cv2.createCLAHE(clipLimit=float(clahe_clip),
                                      tileGridSize=tuple(clahe_grid))

    def reset(self) -> None:
        self.gain = 1.0
        self.active = False

    def apply(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return a boosted copy, or the frame itself when it needs nothing.

        The caller may hand the result straight to a detector and to
        consumers: when no boost is applied the same object comes back, so a
        well-lit stream pays one subsampled mean and nothing else.
        """
        luma_view = _luma_view(bgr_frame)
        self.luma = float(luma_view.mean())
        self.dark_region = float(np.percentile(luma_view, DARK_PERCENTILE))

        if self.luma >= self.dark_luma:
            target_gain = 1.0
        elif self.luma <= 1.0:
            # A black frame has no exposure to recover. Dividing by it would
            # ask for the cap and amplify pure sensor noise.
            target_gain = self.max_gain
        else:
            target_gain = min(self.max_gain, self.target_luma / self.luma)

        self.gain += (target_gain - self.gain) * self.slew
        needs_gain = self.gain >= 1.02
        if not needs_gain:
            # Close enough to unity that the multiply would be invisible, and
            # snapping to exactly 1.0 is what lets `gain == 1.0` mean "no
            # exposure change" -- which is still true when only CLAHE runs.
            self.gain = 1.0
            if self.dark_region >= self.dark_region_luma:
                self.active = False
                return bgr_frame

        # Gain in YUV so only brightness moves: a BGR multiply clips the
        # channels unevenly and shifts skin tone away from what the detector's
        # training data looks like.
        yuv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV)
        y = yuv[:, :, 0]
        if needs_gain:
            # convertScaleAbs saturates rather than wrapping -- a wrapped
            # highlight would turn the brightest part of the hand black.
            y = cv2.convertScaleAbs(y, alpha=self.gain, beta=0.0)
        yuv[:, :, 0] = self._clahe.apply(y)
        self.active = True
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
