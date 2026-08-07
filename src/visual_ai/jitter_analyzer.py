"""
jitter_analyzer.py — Real-time motion spatial jitter calculation tool for Visual AI Game Engine.

Calculates:
- Point-to-point step variance (pixel delta between consecutive frames)
- Mean spatial jitter over a rolling window (STD deviation / RMSE)
- Maximum peak movement jump (spikes)
- Jitter spectrum (high-frequency noise metric)
"""

import math
from collections import deque
import numpy as np


class JitterAnalyzer:
    """
    Computes position jitter metrics for 2D/3D input landmark positions over a rolling frame window.
    
    Parameters
    ----------
    window_size : int
        Number of frames to keep in rolling history for metric calculation (default 30 ~ 1 second).
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.raw_positions = deque(maxlen=window_size)
        self.smoothed_positions = deque(maxlen=window_size)
        self.deltas_raw = deque(maxlen=window_size - 1)
        self.deltas_smoothed = deque(maxlen=window_size - 1)
        
        # Cumulative stats
        self.total_frames = 0

    def reset(self):
        """Clear history buffers."""
        self.raw_positions.clear()
        self.smoothed_positions.clear()
        self.deltas_raw.clear()
        self.deltas_smoothed.clear()
        self.total_frames = 0

    def update(self, raw_pos: tuple[float, float], smoothed_pos: tuple[float, float] | None = None) -> dict:
        """
        Record a new coordinate sample and update jitter metrics.

        Parameters
        ----------
        raw_pos : (x, y)
            Unfiltered input landmark coordinate.
        smoothed_pos : (x, y), optional
            Filtered / EMA smoothed coordinate for side-by-side jitter reduction assessment.

        Returns
        -------
        dict containing current frame jitter statistics.
        """
        self.total_frames += 1
        
        if smoothed_pos is None:
            smoothed_pos = raw_pos

        if len(self.raw_positions) > 0:
            prev_raw = self.raw_positions[-1]
            prev_smooth = self.smoothed_positions[-1]
            
            d_raw = math.hypot(raw_pos[0] - prev_raw[0], raw_pos[1] - prev_raw[1])
            d_smooth = math.hypot(smoothed_pos[0] - prev_smooth[0], smoothed_pos[1] - prev_smooth[1])
            
            self.deltas_raw.append(d_raw)
            self.deltas_smoothed.append(d_smooth)

        self.raw_positions.append(raw_pos)
        self.smoothed_positions.append(smoothed_pos)

        return self.get_stats()

    def get_stats(self) -> dict:
        """Returns statistical jitter calculations over the current rolling window."""
        if len(self.deltas_raw) < 2:
            return {
                "raw_jitter_std": 0.0,
                "raw_jitter_mean_px": 0.0,
                "raw_max_jump_px": 0.0,
                "smoothed_jitter_std": 0.0,
                "smoothed_jitter_mean_px": 0.0,
                "smoothed_max_jump_px": 0.0,
                "jitter_reduction_pct": 0.0,
                "sample_count": len(self.raw_positions),
            }

        raw_arr = np.array(self.deltas_raw)
        smooth_arr = np.array(self.deltas_smoothed)

        raw_std = float(np.std(raw_arr))
        raw_mean = float(np.mean(raw_arr))
        raw_max = float(np.max(raw_arr))

        smooth_std = float(np.std(smooth_arr))
        smooth_mean = float(np.mean(smooth_arr))
        smooth_max = float(np.max(smooth_arr))

        # Reduction percentage based on standard deviation (jitter amplitude)
        reduction_pct = (1.0 - (smooth_std / raw_std)) * 100.0 if raw_std > 1e-5 else 0.0

        return {
            "raw_jitter_std": round(raw_std, 2),
            "raw_jitter_mean_px": round(raw_mean, 2),
            "raw_max_jump_px": round(raw_max, 2),
            "smoothed_jitter_std": round(smooth_std, 2),
            "smoothed_jitter_mean_px": round(smooth_mean, 2),
            "smoothed_max_jump_px": round(smooth_max, 2),
            "jitter_reduction_pct": round(reduction_pct, 1),
            "sample_count": len(self.raw_positions),
        }
