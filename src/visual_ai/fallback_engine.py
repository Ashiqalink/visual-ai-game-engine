import numpy as np
from dataclasses import dataclass
import random

@dataclass
class Block:
    x: float
    y: float
    width: float
    height: float
    health: float
    max_health: float
    active: bool

@dataclass
class Debris:
    x: float
    y: float
    vx: float
    vy: float
    width: float
    height: float
    lifespan: float
    active: bool


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
        self.blocks = []
        self.debris = []

    def set_target_position(self, x: float, y: float):
        self.target_x = x
        self.target_y = y

    def add_block(self, x: float, y: float, w: float, h: float, health: float):
        self.blocks.append(Block(x, y, w, h, health, health, True))

    def get_blocks(self):
        return self.blocks

    def get_debris(self):
        return self.debris
        
    def clear_blocks(self):
        self.blocks.clear()
        self.debris.clear()

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

        # Block collisions
        for block in self.blocks:
            if not block.active: continue
            
            closestX = max(block.x - block.width/2.0, min(self.x, block.x + block.width/2.0))
            closestY = max(block.y - block.height/2.0, min(self.y, block.y + block.height/2.0))
            
            distObjX = self.x - closestX
            distObjY = self.y - closestY
            distance = float(np.sqrt(distObjX * distObjX + distObjY * distObjY))
            
            if distance < self.radius:
                if distance > 0:
                    nx = distObjX / distance
                    ny = distObjY / distance
                    
                    self.x = closestX + nx * self.radius
                    self.y = closestY + ny * self.radius
                    
                    impact = float(np.sqrt(self.vx * self.vx + self.vy * self.vy))
                    dotProduct = (self.vx * nx + self.vy * ny)
                    
                    if dotProduct < 0:
                        self.vx -= 1.6 * dotProduct * nx
                        self.vy -= 1.6 * dotProduct * ny
                        
                        if impact > 50.0:
                            block.health -= impact * 0.1
                            if block.health <= 0.0:
                                block.active = False
                                
                                dw = block.width / 2.0
                                dh = block.height / 2.0
                                for i in range(2):
                                    for j in range(2):
                                        dx_offset = -dw/2.0 if i == 0 else dw/2.0
                                        dy_offset = -dh/2.0 if j == 0 else dh/2.0
                                        dvx = (random.random() * 2.0 - 1.0) * 100.0 + self.vx * 0.2
                                        dvy = (random.random() * 2.0 - 1.0) * 100.0 + self.vy * 0.2
                                        
                                        self.debris.append(Debris(
                                            block.x + dx_offset, block.y + dy_offset,
                                            dvx, dvy,
                                            dw, dh,
                                            3.0,
                                            True
                                        ))

        # Update debris
        for d in self.debris:
            if not d.active: continue
            d.vy += self.gravity * dt
            d.x += d.vx * dt
            d.y += d.vy * dt
            d.lifespan -= dt
            
            if d.y + d.height/2.0 > self.height:
                d.y = self.height - d.height/2.0
                d.vy = -d.vy * 0.5
                d.vx *= 0.8
                
            if d.lifespan <= 0.0:
                d.active = False

    def get_x(self) -> float: return self.x
    def get_y(self) -> float: return self.y
    def get_target_x(self) -> float: return self.target_x
    def get_target_y(self) -> float: return self.target_y
    def get_width(self) -> float: return self.width
    def get_height(self) -> float: return self.height
