"""
test_3d_character_options.py — Verification test for Visual AI Engine 3D character primitives and factors.
"""

import sys
import os
import numpy as np

# Ensure visual ai game engine is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visual_ai import Renderer3D, Mesh3D, Transform3D, Camera3D, Material

def test_engine_3d_options():
    print("[TEST] Initializing Camera and Renderer...")
    cam = Camera3D(fov=60.0, screen_width=800.0, screen_height=600.0, position=(0.0, 0.0, 500.0))
    renderer = Renderer3D(camera=cam)

    print("[TEST] Creating New 3D Character Mesh Primitives...")
    capsule = Mesh3D.create_capsule(radius=20.0, height=40.0)
    torus = Mesh3D.create_torus(ring_radius=25.0, tube_radius=8.0)
    prism = Mesh3D.create_prism(width=40.0, height=40.0, depth=30.0)
    ico = Mesh3D.create_icosahedron(radius=30.0)

    # Test Z-depth extrusion factor scaling
    extruded = capsule.apply_extrusion_depth(1.8)
    scaled = torus.scale_non_uniform(1.2, 0.8, 1.5)

    assert len(capsule.vertices) > 0, "Capsule mesh has no vertices!"
    assert len(torus.faces) > 0, "Torus mesh has no faces!"
    assert len(prism.vertices) == 6, f"Prism should have 6 vertices, got {len(prism.vertices)}"
    assert len(ico.faces) == 20, f"Icosahedron should have 20 faces, got {len(ico.faces)}"

    canvas = np.zeros((600, 800, 3), dtype=np.uint8)

    # Render capsule
    t1 = Transform3D(x=-180.0, y=0.0, z=0.0, rx=15.0, ry=45.0)
    renderer.render_mesh(canvas, extruded, t1, material=Material.preset("plastic"))

    # Render torus
    t2 = Transform3D(x=-60.0, y=0.0, z=0.0, rx=45.0, ry=20.0)
    renderer.render_mesh(canvas, scaled, t2, material=Material.preset("gold"))

    # Render prism
    t3 = Transform3D(x=60.0, y=0.0, z=0.0, rx=20.0, ry=30.0)
    renderer.render_mesh(canvas, prism, t3, material=Material.preset("metal"))

    # Render icosahedron
    t4 = Transform3D(x=180.0, y=0.0, z=0.0, rx=30.0, ry=60.0)
    renderer.render_mesh(canvas, ico, t4, material=Material.preset("emissive"))

    nonzero = np.count_nonzero(canvas)
    print(f"[SUCCESS] Rendered 4 new 3D character mesh types. Canvas non-zero pixels: {nonzero}")
    assert nonzero > 2000, "Rendered output pixels lower than expected!"

if __name__ == "__main__":
    test_engine_3d_options()
