"""
Visual AI Game Engine - Seamless 3D Elements Demo
Demonstrates rendering, projecting, and interacting with 3D primitives driven
by AI gesture tracking.
"""

import queue
import time

import cv2
import numpy as np

from visual_ai import (
    CPP_ENGINE_AVAILABLE,
    Camera3D,
    GameEngine,
    Material,
    Mesh3D,
    Renderer3D,
    Transform3D,
    VisionPipeline,
)


def main():
    WIDTH, HEIGHT = 800, 600

    print(f"[3D Demo] C++ Core built: {'yes' if CPP_ENGINE_AVAILABLE else 'no'} "
          "(this demo runs on whichever engine loaded)")
    print("[3D Demo] Initializing 3D Scene and AI Vision Pipeline...")

    # Whichever engine is available: the scene below attaches a `mesh` to each
    # element and mutates the returned entities in place (target_ent.x = ...),
    # and both engines support that now. The demo used to pin the Python
    # fallback because the compiled core did neither -- it took no `mesh` and
    # returned an int id -- and picking GameEngine automatically crashed the
    # demo on any machine that had built the extension.
    engine = GameEngine(float(WIDTH), float(HEIGHT))

    # Initialize 3D Camera & Software Renderer
    camera = Camera3D(fov=60.0, screen_width=WIDTH, screen_height=HEIGHT,
                      position=(0.0, 0.0, 500.0))
    renderer = Renderer3D(camera=camera)

    # Spawn 3D Elements with materials, rotation speeds, and geometries
    cube_mesh = Mesh3D.create_cube(size=70.0)
    pyramid_mesh = Mesh3D.create_pyramid(width=60.0, height=80.0)
    sphere_mesh = Mesh3D.create_sphere(radius=35.0, rings=8, sectors=12)
    cylinder_mesh = Mesh3D.create_cylinder(radius=30.0, height=70.0, segments=10)

    # 1. Rotating Gold Cube
    gold_mat = Material.preset("gold")
    engine.add_3d_element(
        name="GoldCube",
        x=-150.0, y=50.0, z=0.0,
        vrx=30.0, vry=60.0, vrz=15.0,
        scale=1.0,
        material=gold_mat,
        mesh=cube_mesh
    )

    # 2. Emissive Neon Pyramid
    neon_mat = Material.preset("emissive")
    engine.add_3d_element(
        name="NeonPyramid",
        x=150.0, y=-50.0, z=0.0,
        vrx=45.0, vry=30.0, vrz=0.0,
        scale=1.0,
        material=neon_mat,
        mesh=pyramid_mesh
    )

    # 3. Smooth Plastic Sphere
    plastic_mat = Material.preset("plastic")
    engine.add_3d_element(
        name="PlasticSphere",
        x=0.0, y=-120.0, z=50.0,
        vrx=20.0, vry=40.0, vrz=0.0,
        scale=1.0,
        material=plastic_mat,
        mesh=sphere_mesh
    )

    # 4. Interactive Target Cylinder (tracked to hand/face)
    glass_mat = Material.preset("glass")
    target_ent = engine.add_3d_element(
        name="TargetCylinder",
        x=0.0, y=0.0, z=0.0,
        vrx=0.0, vry=90.0, vrz=0.0,
        scale=1.0,
        material=glass_mat,
        mesh=cylinder_mesh
    )

    # Queue for AI camera payloads
    ai_queue = queue.Queue(maxsize=2)
    pipeline = VisionPipeline(result_queue=ai_queue, width=WIDTH, height=HEIGHT)
    pipeline.start()

    print("[3D Demo] 3D Scene running. Move hand or face in camera to "
          "interact! Press 'q' or 'ESC' to quit.")

    last_time = time.time()
    current_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now

            # 1. Non-blocking drain of AI Vision queue
            try:
                ai_data = ai_queue.get_nowait()
                target_x = ai_data["target_x"]
                target_y = ai_data["target_y"]
                if ai_data["frame"] is not None:
                    current_frame = ai_data["frame"]

                # Map 2D vision target to 3D world space relative to camera center
                world_tx = target_x - (WIDTH / 2.0)
                world_ty = (HEIGHT / 2.0) - target_y

                engine.set_target_position(target_x, target_y)

                # Move target cylinder based on vision tracking & depth (z_delta or pinch)
                target_ent.x = world_tx
                target_ent.y = world_ty
                if ai_data.get("is_pinching", False):
                    target_ent.vz = -100.0  # Move deeper
                else:
                    target_ent.vz = 50.0 if target_ent.z < 0 else 0.0

            except queue.Empty:
                pass

            # 2. Advance physics & rotation
            engine.update(dt)

            # Keep target element in reasonable Z bounds
            target_ent.z = max(-200.0, min(100.0, target_ent.z))

            # 3. Render 3D Scene onto camera background
            canvas = current_frame.copy()

            # Render all active 3D entities
            entities = engine.get_entities()
            for ent in entities:
                if not getattr(ent, "active", True):
                    continue

                mesh = getattr(ent, "mesh", None)
                if mesh is None:
                    # Default geometry if mesh not explicitly attached
                    if "Cube" in ent.name:
                        mesh = cube_mesh
                    elif "Pyramid" in ent.name:
                        mesh = pyramid_mesh
                    elif "Sphere" in ent.name:
                        mesh = sphere_mesh
                    else:
                        mesh = cylinder_mesh

                rx = getattr(ent, "rx", 0.0)
                ry = getattr(ent, "ry", 0.0)
                rz = getattr(ent, "rz", 0.0)

                transform = Transform3D(
                    x=ent.x,
                    y=ent.y,
                    z=ent.z,
                    rx=rx,
                    ry=ry,
                    rz=rz,
                    sx=ent.width,
                    sy=ent.height,
                    sz=ent.depth
                )

                renderer.render_mesh(canvas, mesh, transform, material=ent.material)

            # Overlay 3D HUD instructions
            cv2.putText(
                canvas,
                "Visual AI 3D Engine Demo",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"3D Elements: {len(entities)} | Tracked Target: "
                f"({int(target_ent.x)}, {int(target_ent.y)}, {int(target_ent.z)})",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Visual AI - 3D Elements Demo", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
