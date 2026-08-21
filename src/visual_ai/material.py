"""
Material and Shader Pipeline components for Visual AI Game Engine.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ShaderType(Enum):
    PBR_STANDARD = "PBR_Standard"
    UNLIT = "Unlit"
    TRANSPARENT = "Transparent"
    PHONG = "Phong"
    CUSTOM = "Custom"

    @classmethod
    def from_str(cls, val: str) -> "ShaderType":
        if isinstance(val, ShaderType):
            return val
        norm = str(val).upper().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.name == norm or member.value.upper() == norm:
                return member
        return cls.PBR_STANDARD


@dataclass
class Material:
    """
    General-purpose Material representation supporting Physically Based Rendering (PBR)
    and custom shading models.
    """
    name: str = "DefaultMaterial"
    shader_type: ShaderType = ShaderType.PBR_STANDARD
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    normal_map: str = ""
    roughness: float = 0.5
    metallic: float = 0.0
    emission: tuple[float, float, float] = (0.0, 0.0, 0.0)
    opacity: float = 1.0

    def __post_init__(self):
        # Normalize and clamp bounds
        if isinstance(self.shader_type, str):
            self.shader_type = ShaderType.from_str(self.shader_type)
        self.roughness = max(0.0, min(1.0, float(self.roughness)))
        self.metallic = max(0.0, min(1.0, float(self.metallic)))
        self.opacity = max(0.0, min(1.0, float(self.opacity)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize material to dictionary."""
        return {
            "name": self.name,
            "shader_type": self.shader_type.value,
            "base_color": list(self.base_color),
            "normal_map": self.normal_map,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "emission": list(self.emission),
            "opacity": self.opacity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Material":
        """Deserialize material from dictionary."""
        return cls(
            name=data.get("name", "DefaultMaterial"),
            shader_type=ShaderType.from_str(data.get("shader_type", "PBR_Standard")),
            base_color=tuple(data.get("base_color", (1.0, 1.0, 1.0, 1.0))),
            normal_map=data.get("normal_map", ""),
            roughness=data.get("roughness", 0.5),
            metallic=data.get("metallic", 0.0),
            emission=tuple(data.get("emission", (0.0, 0.0, 0.0))),
            opacity=data.get("opacity", 1.0),
        )

    def to_shader_uniforms(self) -> dict[str, Any]:
        """Convert material parameters into GLSL / GPU shader uniform layout."""
        return {
            "u_BaseColor": list(self.base_color),
            "u_HasNormalMap": bool(self.normal_map),
            "u_NormalMap": self.normal_map,
            "u_Roughness": self.roughness,
            "u_Metallic": self.metallic,
            "u_Emission": list(self.emission),
            "u_Opacity": self.opacity,
            "u_ShaderType": self.shader_type.value,
        }

    @classmethod
    def preset(cls, name: str) -> "Material":
        """Retrieve a built-in material preset by name."""
        presets = {
            "gold": cls(
                name="Gold",
                shader_type=ShaderType.PBR_STANDARD,
                base_color=(1.0, 0.766, 0.336, 1.0),
                roughness=0.15,
                metallic=1.0,
            ),
            "plastic": cls(
                name="SmoothPlastic",
                shader_type=ShaderType.PBR_STANDARD,
                base_color=(0.1, 0.5, 0.9, 1.0),
                roughness=0.3,
                metallic=0.0,
            ),
            "metal": cls(
                name="PolishedMetal",
                shader_type=ShaderType.PBR_STANDARD,
                base_color=(0.95, 0.95, 0.95, 1.0),
                roughness=0.1,
                metallic=0.9,
            ),
            "glass": cls(
                name="ClearGlass",
                shader_type=ShaderType.TRANSPARENT,
                base_color=(1.0, 1.0, 1.0, 0.2),
                roughness=0.05,
                metallic=0.1,
                opacity=0.2,
            ),
            "emissive": cls(
                name="NeonGlow",
                shader_type=ShaderType.UNLIT,
                base_color=(0.0, 1.0, 0.8, 1.0),
                emission=(0.0, 2.0, 1.6),
                roughness=1.0,
                metallic=0.0,
            ),
        }
        key = name.lower()
        if key not in presets:
            raise ValueError(f"Unknown preset '{name}'. Available presets: {list(presets.keys())}")
        return presets[key]
