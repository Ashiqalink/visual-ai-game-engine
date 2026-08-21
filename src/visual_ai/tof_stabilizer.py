"""
tof_stabilizer.py — deprecated import path for :mod:`visual_ai.depth_stabilizer`.

"ToF" named a time-of-flight sensor this engine never had. The class was
renamed to ``DepthStabilizer`` first; the module has now followed. This path
keeps working so nothing breaks mid-release, but it warns.

Note for benchmarks and tests: patching a module attribute here (the clock, in
particular) patches the shim, not the module the pipeline actually runs on.
Reach for ``visual_ai.depth_stabilizer`` when you mean the real thing.
"""

import warnings

from visual_ai.depth_stabilizer import DepthStabilizer, ToFStabilizer

warnings.warn(
    "visual_ai.tof_stabilizer is deprecated; import visual_ai.depth_stabilizer instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DepthStabilizer", "ToFStabilizer"]
