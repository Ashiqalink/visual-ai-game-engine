"""
gesture_math.py — Engine-level gesture and landmark geometry utilities.

Exposes geometric calculation helper functions on landmark points (MediaPipe Hand/Face format)
so game code can evaluate gesture properties without manual coordinate math.
"""

import math
from typing import List, Tuple, Dict, Any, Union


Point2D = Union[Tuple[float, float], Dict[str, float], Any]


def _extract_xy(pt: Point2D) -> Tuple[float, float]:
    """Helper to extract (x, y) coordinates from tuple, dict, or MediaPipe landmark object."""
    if isinstance(pt, (tuple, list)):
        return (float(pt[0]), float(pt[1]))
    if isinstance(pt, dict):
        return (float(pt.get("x", 0.0)), float(pt.get("y", 0.0)))
    if hasattr(pt, "x") and hasattr(pt, "y"):
        return (float(pt.x), float(pt.y))
    # ndarrays and other indexable sequences (the package re-exports numpy,
    # so arrays are a completely ordinary way for a game to hold a point).
    try:
        return (float(pt[0]), float(pt[1]))
    except (TypeError, IndexError, KeyError):
        raise ValueError(f"Unsupported point format: {pt}")


def get_landmark_distance(pt1: Point2D, pt2: Point2D) -> float:
    """Calculate Euclidean distance between two landmark points."""
    return math.dist(_extract_xy(pt1), _extract_xy(pt2))


def get_finger_angle(joint_a: Point2D, joint_b: Point2D, joint_c: Point2D) -> float:
    """
    Calculate joint flexion angle in degrees at joint_b (vertex) formed by joint_a -> joint_b -> joint_c.
    Returns angle in degrees [0, 180].
    """
    ax, ay = _extract_xy(joint_a)
    bx, by = _extract_xy(joint_b)
    cx, cy = _extract_xy(joint_c)

    ba_x, ba_y = ax - bx, ay - by
    bc_x, bc_y = cx - bx, cy - by

    len_ba = math.sqrt(ba_x * ba_x + ba_y * ba_y)
    len_bc = math.sqrt(bc_x * bc_x + bc_y * bc_y)

    if len_ba < 1e-9 or len_bc < 1e-9:
        return 0.0

    dot = ba_x * bc_x + ba_y * bc_y
    cos_theta = max(-1.0, min(1.0, dot / (len_ba * len_bc)))
    return math.degrees(math.acos(cos_theta))


def get_hand_center_and_radius(landmarks: List[Point2D]) -> Tuple[Tuple[float, float], float]:
    """
    Calculate hand palm center (average position) and bounding radius.
    Returns ((cx, cy), radius).
    """
    if not landmarks:
        return ((0.0, 0.0), 0.0)

    sum_x = 0.0
    sum_y = 0.0
    pts = []
    for lm in landmarks:
        x, y = _extract_xy(lm)
        sum_x += x
        sum_y += y
        pts.append((x, y))

    cx = sum_x / len(landmarks)
    cy = sum_y / len(landmarks)

    max_dist_sq = 0.0
    for px, py in pts:
        dx = px - cx
        dy = py - cy
        dist_sq = dx * dx + dy * dy
        if dist_sq > max_dist_sq:
            max_dist_sq = dist_sq

    return ((cx, cy), math.sqrt(max_dist_sq))


def get_landmark_velocity(
    prev_pt: Point2D, curr_pt: Point2D, dt: float
) -> Tuple[float, float]:
    """Calculate instantaneous velocity vector (vx, vy) for a landmark point over dt seconds."""
    if dt < 1e-9:
        return (0.0, 0.0)
    x1, y1 = _extract_xy(prev_pt)
    x2, y2 = _extract_xy(curr_pt)
    return ((x2 - x1) / dt, (y2 - y1) / dt)
