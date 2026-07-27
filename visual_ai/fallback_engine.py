import numpy as np


class PythonFallbackEngine:
    """
    Pure Python fallback physics engine used when C++ engine_core extension is not compiled.
    """
    def __init__(self, width: float = 800.0, height: float = 600.0):
        self.width = width
        self.height = height
        self.x = width / 2.0
        self.y = height / 4.0
        self.vx = 120.0
        self.vy = 0.0
        self.gravity = 400.0
        self.radius = 25.0
        self.target_x = width / 2.0
        self.target_y = height / 2.0

    def set_target_position(self, x: float, y: float):
        self.target_x = x
        self.target_y = y

    def update(self, dt: float):
        self.vy += self.gravity * dt
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = float(np.sqrt(dx * dx + dy * dy))
        if dist > 1.0:
            pull_strength = 150.0
            self.vx += (dx / dist) * pull_strength * dt
            self.vy += (dy / dist) * pull_strength * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.99
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx * 0.8
        elif self.x + self.radius > self.width:
            self.x = self.width - self.radius
            self.vx = -self.vx * 0.8
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy * 0.8
        elif self.y + self.radius > self.height:
            self.y = self.height - self.radius
            self.vy = -self.vy * 0.8

    def get_x(self) -> float: return self.x
    def get_y(self) -> float: return self.y
    def get_target_x(self) -> float: return self.target_x
    def get_target_y(self) -> float: return self.target_y
    def get_width(self) -> float: return self.width
    def get_height(self) -> float: return self.height
