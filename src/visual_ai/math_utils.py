"""
math_utils.py — Engine core math utilities for Visual AI Game Engine.

Contains pure game-agnostic mathematical operations:
  • Transform2D & Transform3D composition & projection
  • Viewport & camera frame coordinate conversion
  • Vector2 & Vector3 operations (dot, cross, distance, angle_between, reflect)
  • Interpolation & Easing curves (lerp, slerp, ease_in_quad, spring, etc.)
  • Kinematics & Physics primitives (Euler integration, drag, AABB & Circle
    collisions, trajectory prediction)
  • Fixed-timestep update accumulator & Tween progress manager
  • Seeded RNG, weighted random selection, and noise utilities
"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


# ── Vector Primitives ─────────────────────────────────────────────────────────

@dataclass
class Vector2:
    x: float = 0.0
    y: float = 0.0

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self) -> "Vector2":
        length = self.length()
        if length < 1e-9:
            return Vector2(0.0, 0.0)
        return Vector2(self.x / length, self.y / length)

    def dot(self, other: "Vector2") -> float:
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: "Vector2") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def angle_between(self, other: "Vector2") -> float:
        """Return angle between vectors in radians [0, pi]."""
        l1, l2 = self.length(), other.length()
        if l1 < 1e-9 or l2 < 1e-9:
            return 0.0
        cos_theta = max(-1.0, min(1.0, self.dot(other) / (l1 * l2)))
        return math.acos(cos_theta)

    def reflect(self, normal: "Vector2") -> "Vector2":
        n = normal.normalize()
        d = self.dot(n)
        return Vector2(self.x - 2.0 * d * n.x, self.y - 2.0 * d * n.y)

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> "Vector3":
        length = self.length()
        if length < 1e-9:
            return Vector3(0.0, 0.0, 0.0)
        return Vector3(self.x / length, self.y / length, self.z / length)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def distance_to(self, other: "Vector3") -> float:
        return math.hypot(self.x - other.x, self.y - other.y, self.z - other.z)

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


# ── Transform Primitives ──────────────────────────────────────────────────────

@dataclass
class Transform2D:
    x: float = 0.0
    y: float = 0.0
    rotation_deg: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0

    def apply(self, point: tuple[float, float]) -> tuple[float, float]:
        px, py = point
        # Scale
        px *= self.scale_x
        py *= self.scale_y
        # Rotate
        rad = math.radians(self.rotation_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        # Translate
        return (rx + self.x, ry + self.y)


# ── Coordinate & Viewport Remapping ───────────────────────────────────────────

def map_range(
    val: float,
    in_min: float,
    in_max: float,
    out_min: float,
    out_max: float,
    clamp: bool = False,
) -> float:
    """Remap a scalar from input range [in_min, in_max] to output range [out_min, out_max]."""
    if abs(in_max - in_min) < 1e-9:
        return out_min
    t = (val - in_min) / (in_max - in_min)
    if clamp:
        t = max(0.0, min(1.0, t))
    return out_min + t * (out_max - out_min)


def remap_camera_roi_to_game(
    point: tuple[float, float],
    src_bounds: tuple[float, float, float, float],
    dst_bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """
    Remap (x, y) coordinates from vision camera ROI bounds (src_x, src_y, src_w, src_h)
    to gameplay screen/viewport bounds (dst_x, dst_y, dst_w, dst_h).
    """
    sx, sy, sw, sh = src_bounds
    dx, dy, dw, dh = dst_bounds
    px = map_range(point[0], sx, sx + sw, dx, dx + dw, clamp=True)
    py = map_range(point[1], sy, sy + sh, dy, dy + dh, clamp=True)
    return (px, py)


# ── Interpolation & Easing ───────────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by factor t (0 to 1)."""
    return a + (b - a) * max(0.0, min(1.0, t))


def slerp_quaternion(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
    t: float,
) -> tuple[float, float, float, float]:
    """Spherical linear interpolation between unit quaternions q1 and q2."""
    t = max(0.0, min(1.0, t))
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    dot = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2

    if dot < 0.0:
        w2, x2, y2, z2 = -w2, -x2, -y2, -z2
        dot = -dot

    if dot > 0.9995:
        # Linear interp fallback for close orientations
        res = (
            w1 + t * (w2 - w1),
            x1 + t * (x2 - x1),
            y1 + t * (y2 - y1),
            z1 + t * (z2 - z1),
        )
        length = math.sqrt(sum(v * v for v in res))
        return (res[0] / length, res[1] / length, res[2] / length, res[3] / length)

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    sin_theta = math.sin(theta)

    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return (
        s1 * w1 + s2 * w2,
        s1 * x1 + s2 * x2,
        s1 * y1 + s2 * y2,
        s1 * z1 + s2 * z2,
    )


def ease_in_quad(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t


def ease_out_quad(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * (2.0 - t)


def ease_in_out_sine(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return -0.5 * (math.cos(math.pi * t) - 1.0)


def spring(t: float, stiffness: float = 100.0, damping: float = 10.0) -> float:
    """Spring oscillation curve starting at 0 and settling towards 1.0."""
    t = max(0.0, t)
    omega = math.sqrt(max(0.01, stiffness))
    decay = math.exp(-damping * t)
    return 1.0 - decay * math.cos(omega * t)


# ── Physics Primitives & Kinematics ─────────────────────────────────────────

def integrate_euler(
    pos: Vector2, vel: Vector2, accel: Vector2, dt: float
) -> tuple[Vector2, Vector2]:
    """Integrate velocity and position using standard Euler integration."""
    new_vel = Vector2(vel.x + accel.x * dt, vel.y + accel.y * dt)
    new_pos = Vector2(pos.x + new_vel.x * dt, pos.y + new_vel.y * dt)
    return (new_pos, new_vel)


def calculate_drag_force(velocity: Vector2, drag_coeff: float) -> Vector2:
    """Calculate aerodynamic/friction drag force vector opposing velocity."""
    speed = velocity.length()
    if speed < 1e-9:
        return Vector2(0.0, 0.0)
    drag_mag = drag_coeff * speed * speed
    return Vector2(-drag_mag * (velocity.x / speed), -drag_mag * (velocity.y / speed))


def intersect_aabb_aabb(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> bool:
    """AABB collision test for (x, y, width, height) boxes centered at (x,y)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return (
        abs(x1 - x2) * 2.0 <= (w1 + w2) and
        abs(y1 - y2) * 2.0 <= (h1 + h2)
    )


def intersect_circle_circle(
    c1: tuple[float, float, float],
    c2: tuple[float, float, float],
) -> bool:
    """Circle collision test for (x, y, radius)."""
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[1]
    r_sum = c1[2] + c2[2]
    return (dx * dx + dy * dy) <= (r_sum * r_sum)


def intersect_circle_aabb(
    circle: tuple[float, float, float],
    aabb: tuple[float, float, float, float],
) -> bool:
    """Circle (cx, cy, r) vs AABB (bx, by, bw, bh centered at bx,by) collision test."""
    cx, cy, r = circle
    bx, by, bw, bh = aabb
    closest_x = max(bx - bw / 2.0, min(cx, bx + bw / 2.0))
    closest_y = max(by - bh / 2.0, min(cy, by + bh / 2.0))
    dx = cx - closest_x
    dy = cy - closest_y
    return (dx * dx + dy * dy) <= (r * r)


def predict_projectile_trajectory(
    start_pos: tuple[float, float],
    initial_vel: tuple[float, float],
    gravity: float = 9.81,
    time_step: float = 0.05,
    num_steps: int = 30,
) -> list[tuple[float, float]]:
    """Generate array of trajectory arc points (x, y) over time under gravity."""
    points = []
    x, y = start_pos
    vx, vy = initial_vel
    for step in range(num_steps):
        t = step * time_step
        px = x + vx * t
        py = y + vy * t + 0.5 * gravity * (t * t)
        points.append((px, py))
    return points


# ── Timing & Animation Math ──────────────────────────────────────────────────

class FixedTimestepAccumulator:
    """
    Fixed-timestep accumulator to decouple physical engine updates from
    variable display frame rates.
    """
    def __init__(self, target_fps: float = 60.0):
        self.dt = 1.0 / max(1.0, target_fps)
        self.accumulator = 0.0

    def add_frame_time(self, frame_dt: float):
        # Cap max frame time to avoid spiral of death
        self.accumulator += min(0.25, frame_dt)

    def consume_step(self) -> bool:
        if self.accumulator >= self.dt:
            self.accumulator -= self.dt
            return True
        return False

    def get_alpha(self) -> float:
        """Remainder fraction [0, 1) for frame rendering interpolation."""
        return self.accumulator / self.dt


class Tween:
    """Simple linear/eased parameter tween progress manager."""
    def __init__(self, duration: float = 1.0, easing_fn=ease_in_out_sine):
        self.duration = max(1e-5, duration)
        self.elapsed = 0.0
        self.easing_fn = easing_fn

    def update(self, dt: float) -> float:
        self.elapsed = min(self.duration, self.elapsed + max(0.0, dt))
        t = self.elapsed / self.duration
        return self.easing_fn(t)

    def is_finished(self) -> bool:
        return self.elapsed >= self.duration

    def reset(self):
        self.elapsed = 0.0


# ── Structured Randomness ─────────────────────────────────────────────────────

class SeededRNG:
    """Deterministic seeded random number generator."""
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def seed(self, seed: int):
        self._rng.seed(seed)

    def random(self) -> float:
        return self._rng.random()

    def range(self, min_val: float, max_val: float) -> float:
        return min_val + (max_val - min_val) * self._rng.random()

    def choice(self, options: Sequence[T], weights: Sequence[float] | None = None) -> T:
        if weights is not None:
            return self._rng.choices(options, weights=weights, k=1)[0]
        return self._rng.choice(options)


def weighted_choice(options: Sequence[T], weights: Sequence[float]) -> T:
    """Select option according to relative weight probabilities."""
    return random.choices(options, weights=weights, k=1)[0]


def perlin_noise_1d(x: float) -> float:
    """Simple 1D value noise curve for procedural variation [-1.0, 1.0]."""
    x_int = int(math.floor(x))
    x_frac = x - x_int
    # Smooth step
    u = x_frac * x_frac * (3.0 - 2.0 * x_frac)
    # Hash pseudo-random values
    v0 = math.sin(x_int * 12.9898) * 43758.5453
    v1 = math.sin((x_int + 1) * 12.9898) * 43758.5453
    n0 = (v0 - math.floor(v0)) * 2.0 - 1.0
    n1 = (v1 - math.floor(v1)) * 2.0 - 1.0
    return lerp(n0, n1, u)
