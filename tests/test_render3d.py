import unittest
import numpy as np
import cv2
from visual_ai.render3d import Transform3D, Camera3D, Mesh3D, Renderer3D
from visual_ai.material import Material, ShaderType
from visual_ai.fallback_engine import PythonFallbackEngine, Entity


class TestRender3D(unittest.TestCase):

    def test_transform_points(self):
        t = Transform3D(x=10.0, y=20.0, z=30.0, rx=0.0, ry=90.0, rz=0.0, sx=2.0, sy=2.0, sz=2.0)
        pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        transformed = t.transform_points(pts)
        self.assertEqual(transformed.shape, (1, 3))
        # 1.0 scaled by 2 = 2.0. Rotated 90 deg around Y gives (0, 0, -2) + translation (10, 20, 30) = (10, 20, 28)
        np.testing.assert_allclose(transformed[0], [10.0, 20.0, 28.0], atol=1e-4)

    def test_camera_projection(self):
        cam = Camera3D(fov=60.0, screen_width=800.0, screen_height=600.0, position=(0.0, 0.0, 500.0))
        # Point directly in front of camera
        projected = cam.project_point((0.0, 0.0, 0.0))
        self.assertIsNotNone(projected)
        px, py, depth = projected
        self.assertEqual(px, 400)
        self.assertEqual(py, 300)
        self.assertAlmostEqual(depth, 500.0, places=4)

    def test_mesh_primitives(self):
        cube = Mesh3D.create_cube(size=40.0)
        self.assertEqual(len(cube.vertices), 8)
        self.assertEqual(len(cube.faces), 6)

        pyramid = Mesh3D.create_pyramid(width=40.0, height=50.0)
        self.assertEqual(len(pyramid.vertices), 5)
        self.assertEqual(len(pyramid.faces), 5)

        sphere = Mesh3D.create_sphere(radius=20.0, rings=6, sectors=8)
        self.assertGreater(len(sphere.vertices), 0)

        cylinder = Mesh3D.create_cylinder(radius=15.0, height=40.0, segments=8)
        self.assertGreater(len(cylinder.vertices), 0)

    def test_renderer_3d(self):
        renderer = Renderer3D()
        cube = Mesh3D.create_cube(size=80.0)
        transform = Transform3D(x=0.0, y=0.0, z=0.0, rx=30.0, ry=45.0, rz=0.0)
        canvas = np.zeros((600, 800, 3), dtype=np.uint8)

        mat = Material(base_color=(0.2, 0.8, 0.4, 1.0))
        result = renderer.render_mesh(canvas, cube, transform, material=mat)
        self.assertEqual(result.shape, (600, 800, 3))
        # Non-zero pixels drawn on canvas
        self.assertGreater(np.count_nonzero(result), 0)

    def test_fallback_engine_3d_element(self):
        engine = PythonFallbackEngine(800.0, 600.0)
        cube_mesh = Mesh3D.create_cube(50.0)
        elem = engine.add_3d_element(
            name="Cube3D",
            x=100.0, y=100.0, z=0.0,
            vrx=45.0, vry=90.0, vrz=0.0,
            mesh=cube_mesh
        )
        self.assertEqual(elem.name, "Cube3D")
        self.assertIn(elem, engine.get_entities())

        engine.update(1.0)
        self.assertAlmostEqual(elem.rx, 45.0, places=4)
        self.assertAlmostEqual(elem.ry, 90.0, places=4)


if __name__ == "__main__":
    unittest.main()
