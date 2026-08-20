import math
from dataclasses import dataclass, field
import random
from typing import List, Optional, Any
from visual_ai.material import Material


@dataclass
class Entity:
    """General-purpose game engine entity representing an in-world object with transform, velocity, and material."""
    id: int
    name: str = "Entity"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    vrx: float = 0.0
    vry: float = 0.0
    vrz: float = 0.0
    width: float = 1.0
    height: float = 1.0
    depth: float = 1.0
    active: bool = True
    material: Material = field(default_factory=Material)
    mesh: Optional[Any] = None



@dataclass
class Block:
    """Legacy Block structure for block-breaker demo compatibility."""
    x: float
    y: float
    width: float
    height: float
    health: float
    max_health: float
    active: bool
    material: Material = field(default_factory=Material)


@dataclass
class Debris:
    """Legacy Debris structure for particle/destructible effects."""
    x: float
    y: float
    vx: float
    vy: float
    width: float
    height: float
    lifespan: float
    active: bool
    material: Material = field(default_factory=Material)


def _coerce_material(material: Optional[Material]) -> Material:
    """
    Resolve a ``material=`` argument the way the C++ binding does.

    Both engines default to a fresh Material when none is given and reject
    anything that is not one: the binding cannot convert a stray dict, and a
    fallback that accepted it silently would only fail later, in the renderer,
    on whichever machine had no compiled core.
    """
    if material is None:
        return Material()
    if not isinstance(material, Material):
        raise TypeError(
            f"material must be a visual_ai.Material, got {type(material).__name__}"
        )
    return material


class PythonFallbackEngine:
    """
    Pure Python fallback physics and scene engine used when C++ engine_core extension is not compiled.
    Supports general Entity objects with PBR Materials as well as legacy Block/Debris components.
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
        self.blocks: List[Block] = []
        self.debris: List[Debris] = []
        self.entities: List[Entity] = []
        self._next_entity_id: int = 1

    def set_target_position(self, x: float, y: float):
        self.target_x = x
        self.target_y = y

    def add_entity(
        self,
        name: str = "Entity",
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        width: float = 1.0,
        height: float = 1.0,
        depth: float = 1.0,
        material: Optional[Material] = None,
        w: Optional[float] = None,
        h: Optional[float] = None,
        d: Optional[float] = None,
    ) -> Entity:
        """
        Add a general-purpose Entity to the game world.

        The size can be given as ``width``/``height``/``depth`` (this class's
        historical spelling) or as ``w``/``h``/``d`` — the names the C++
        engine's binding uses. Accepting both keeps a keyword call working
        regardless of which engine loaded; the short names win if both are
        passed.
        """
        if w is not None:
            width = w
        if h is not None:
            height = h
        if d is not None:
            depth = d
        ent_id = self._next_entity_id
        self._next_entity_id += 1
        mat = _coerce_material(material)
        entity = Entity(
            id=ent_id,
            name=name,
            x=x,
            y=y,
            z=z,
            vx=vx,
            vy=vy,
            vz=vz,
            width=width,
            height=height,
            depth=depth,
            active=True,
            material=mat,
        )
        self.entities.append(entity)
        return entity

    def add_3d_element(
        self,
        name: str = "3DElement",
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        rx: float = 0.0,
        ry: float = 0.0,
        rz: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        vrx: float = 0.0,
        vry: float = 0.0,
        vrz: float = 0.0,
        scale: float = 1.0,
        material: Optional[Material] = None,
        mesh: Optional[Any] = None,
    ) -> Entity:
        """Add a 3D Element entity to the engine scene."""
        ent_id = self._next_entity_id
        self._next_entity_id += 1
        mat = _coerce_material(material)
        entity = Entity(
            id=ent_id,
            name=name,
            x=x,
            y=y,
            z=z,
            rx=rx,
            ry=ry,
            rz=rz,
            vx=vx,
            vy=vy,
            vz=vz,
            vrx=vrx,
            vry=vry,
            vrz=vrz,
            width=scale,
            height=scale,
            depth=scale,
            active=True,
            material=mat,
            mesh=mesh,
        )
        self.entities.append(entity)
        return entity

    def get_entities(self) -> List[Entity]:
        """Get all active entities in the engine."""
        return self.entities

    def clear_entities(self):
        """Remove all general entities."""
        self.entities.clear()

    def add_block(self, x: float, y: float, w: float, h: float, health: float, material: Optional[Material] = None):
        mat = _coerce_material(material)
        self.blocks.append(Block(x, y, w, h, health, health, True, mat))

    def get_blocks(self) -> List[Block]:
        return self.blocks

    def get_debris(self) -> List[Debris]:
        return self.debris

    def clear_blocks(self):
        self.blocks.clear()
        self.debris.clear()

    def update(self, dt: float):
        # Update main player/target sphere
        self.vy += self.gravity * dt
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 1.0:
            pull_strength = 150.0
            self.vx += (dx / dist) * pull_strength * dt
            self.vy += (dy / dist) * pull_strength * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.99

        # Boundary checks for player sphere
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

        # Update general entities
        for entity in self.entities:
            if not entity.active:
                continue
            entity.x += entity.vx * dt
            entity.y += entity.vy * dt
            entity.z += entity.vz * dt
            # math.fmod, not %: the C++ engine wraps with std::fmod, which
            # keeps the sign (-100° stays -100°, not 260°). Python's % is
            # always non-negative, so the two engines reported different
            # values for the same negative spin and threshold comparisons on
            # rx/ry/rz diverged depending on which engine loaded.
            entity.rx = math.fmod(entity.rx + entity.vrx * dt, 360.0)
            entity.ry = math.fmod(entity.ry + entity.vry * dt, 360.0)
            entity.rz = math.fmod(entity.rz + entity.vrz * dt, 360.0)

        # Block collisions (legacy support)
        for block in self.blocks:
            if not block.active:
                continue

            closestX = max(block.x - block.width / 2.0, min(self.x, block.x + block.width / 2.0))
            closestY = max(block.y - block.height / 2.0, min(self.y, block.y + block.height / 2.0))

            distObjX = self.x - closestX
            distObjY = self.y - closestY
            distance = math.sqrt(distObjX * distObjX + distObjY * distObjY)

            if distance < self.radius:
                if distance > 0:
                    nx = distObjX / distance
                    ny = distObjY / distance

                    self.x = closestX + nx * self.radius
                    self.y = closestY + ny * self.radius

                    impact = math.sqrt(self.vx * self.vx + self.vy * self.vy)
                    dotProduct = self.vx * nx + self.vy * ny

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
                                        dx_offset = -dw / 2.0 if i == 0 else dw / 2.0
                                        dy_offset = -dh / 2.0 if j == 0 else dh / 2.0
                                        dvx = (random.random() * 2.0 - 1.0) * 100.0 + self.vx * 0.2
                                        dvy = (random.random() * 2.0 - 1.0) * 100.0 + self.vy * 0.2

                                        self.debris.append(
                                            Debris(
                                                block.x + dx_offset,
                                                block.y + dy_offset,
                                                dvx,
                                                dvy,
                                                dw,
                                                dh,
                                                3.0,
                                                True,
                                                block.material,
                                            )
                                        )

        # Update debris (legacy support)
        for d in self.debris:
            if not d.active:
                continue
            d.vy += self.gravity * dt
            d.x += d.vx * dt
            d.y += d.vy * dt
            d.lifespan -= dt

            if d.y + d.height / 2.0 > self.height:
                d.y = self.height - d.height / 2.0
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
