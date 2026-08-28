"""
test_render_scene.py — one depth sort across several meshes.

`render_mesh` sorts and paints one mesh's faces, so two calls put the second
mesh wholly on top of the first however far behind it is. That is fine while a
scene is one object, and wrong the moment a game wants a hand to reach around
something: the fingers that went behind the cup are painted over it.

`render_scene` collects the faces of every mesh and sorts them together. What
is pinned here is that near geometry wins regardless of the order the meshes
were handed over in — and, as the control, that calling `render_mesh` twice
genuinely does not do this.
"""

import unittest

import numpy as np

from visual_ai.three_d import Camera3D, Material, Mesh3D, Renderer3D, Transform3D

WIDTH, HEIGHT = 800, 600
CENTRE = (HEIGHT // 2, WIDTH // 2)          # (row, col)

RED = Material(base_color=(1.0, 0.0, 0.0), opacity=1.0)
GREEN = Material(base_color=(0.0, 1.0, 0.0), opacity=1.0)
BLUE = Material(base_color=(0.0, 0.0, 1.0), opacity=1.0)


def _at(z, size):
    mesh = Mesh3D.create_cube(size=size)
    transform = Transform3D()
    transform.position = (0.0, 0.0, float(z))
    return mesh, transform


def _renderer():
    # Ambient floor at 1.0 so a face's colour is its material's, undimmed —
    # the test is about which face won, not about shading.
    return Renderer3D(camera=Camera3D(screen_width=WIDTH, screen_height=HEIGHT),
                      ambient_intensity=1.0)


def _frame():
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def _dominant(frame):
    """Which channel the centre pixel is: 'b', 'g', 'r' or 'none'."""
    pixel = frame[CENTRE]
    if not pixel.any():
        return "none"
    return "bgr"[int(np.argmax(pixel))]


class TestSceneOrdering(unittest.TestCase):
    """A big red slab, a small green marker in front of it, a blue one behind."""

    def setUp(self):
        self.renderer = _renderer()
        occluder_mesh, occluder_at = _at(0, 200.0)
        near_mesh, near_at = _at(150, 40.0)
        far_mesh, far_at = _at(-150, 40.0)
        self.occluder = (occluder_mesh, occluder_at, RED)
        self.near = (near_mesh, near_at, GREEN)
        self.far = (far_mesh, far_at, BLUE)

    def test_the_markers_really_are_behind_and_in_front(self):
        """A check that could not fail is not a check: verify the fixture."""
        for item, expected in ((self.occluder, "r"), (self.near, "g"),
                               (self.far, "b")):
            frame = _frame()
            self.renderer.render_scene(frame, [item])
            self.assertEqual(_dominant(frame), expected)

    def test_near_geometry_wins_whatever_order_it_is_given_in(self):
        orders = (
            [self.far, self.occluder, self.near],
            [self.near, self.occluder, self.far],
            [self.occluder, self.near, self.far],
        )
        for order in orders:
            frame = _frame()
            self.renderer.render_scene(frame, order)
            self.assertEqual(_dominant(frame), "g")

    def test_geometry_behind_the_slab_is_not_drawn_at_all(self):
        frame = _frame()
        self.renderer.render_scene(frame, [self.near, self.occluder, self.far])
        # Blue is the only source of a blue-dominant pixel in this scene.
        blue_dominant = ((frame[:, :, 0] > frame[:, :, 1])
                         & (frame[:, :, 0] > frame[:, :, 2]))
        self.assertEqual(int(blue_dominant.sum()), 0)

    def test_two_render_mesh_calls_cannot_do_this(self):
        """The control: this is the behaviour `render_scene` exists to replace."""
        frame = _frame()
        for mesh, transform, material in (self.near, self.occluder):
            self.renderer.render_mesh(frame, mesh, transform, material)
        # The occluder was painted second, so it covers the nearer marker.
        self.assertEqual(_dominant(frame), "r")

    def test_render_mesh_still_paints_one_mesh_the_way_it_always_did(self):
        mesh, transform, material = self.occluder
        scene, single = _frame(), _frame()
        self.renderer.render_scene(scene, [(mesh, transform, material)])
        self.renderer.render_mesh(single, mesh, transform, material)
        self.assertTrue(np.array_equal(scene, single))

    def test_an_empty_scene_leaves_the_frame_alone(self):
        frame = _frame()
        self.renderer.render_scene(frame, [])
        self.assertEqual(int(frame.sum()), 0)

    def test_a_translucent_mesh_keeps_its_own_opacity(self):
        """Opacity travels with the face, not with the last mesh collected."""
        glass = Material(base_color=(0.0, 1.0, 0.0), opacity=0.5)
        mesh, transform, _ = self.near
        frame = _frame()
        self.renderer.render_scene(
            frame, [self.occluder, (mesh, transform, glass)])
        pixel = frame[CENTRE]
        # Half green over red: both channels present, neither saturated away.
        self.assertGreater(int(pixel[1]), 0)
        self.assertGreater(int(pixel[2]), 0)


if __name__ == "__main__":
    unittest.main()
