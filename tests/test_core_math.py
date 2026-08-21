import unittest

from visual_ai import (
    FixedTimestepAccumulator,
    GenericStreamFilter,
    SeededRNG,
    Transform2D,
    Tween,
    Vector2,
    Vector3,
    ease_in_quad,
    ease_out_quad,
    get_finger_angle,
    get_hand_center_and_radius,
    get_landmark_distance,
    get_landmark_velocity,
    integrate_euler,
    intersect_aabb_aabb,
    intersect_circle_aabb,
    intersect_circle_circle,
    lerp,
    perlin_noise_1d,
    predict_projectile_trajectory,
    remap_camera_roi_to_game,
)


class TestEngineCoreMath(unittest.TestCase):

    def test_vectors(self):
        v1 = Vector2(3.0, 4.0)
        self.assertAlmostEqual(v1.length(), 5.0)
        v1_norm = v1.normalize()
        self.assertAlmostEqual(v1_norm.length(), 1.0)
        self.assertAlmostEqual(v1.dot(Vector2(1.0, 0.0)), 3.0)

        v3a = Vector3(1.0, 0.0, 0.0)
        v3b = Vector3(0.0, 1.0, 0.0)
        cross = v3a.cross(v3b)
        self.assertAlmostEqual(cross.x, 0.0)
        self.assertAlmostEqual(cross.y, 0.0)
        self.assertAlmostEqual(cross.z, 1.0)

    def test_transform_and_remap(self):
        t2d = Transform2D(x=10.0, y=20.0, rotation_deg=90.0, scale_x=2.0, scale_y=2.0)
        # Point (1, 0) scaled -> (2, 0), rotated 90deg -> (0, 2), translated -> (10, 22)
        px, py = t2d.apply((1.0, 0.0))
        self.assertAlmostEqual(px, 10.0, places=4)
        self.assertAlmostEqual(py, 22.0, places=4)

        # Remap ROI
        src_bounds = (0.0, 0.0, 640.0, 480.0)
        dst_bounds = (0.0, 0.0, 1920.0, 1080.0)
        rx, ry = remap_camera_roi_to_game((320.0, 240.0), src_bounds, dst_bounds)
        self.assertAlmostEqual(rx, 960.0)
        self.assertAlmostEqual(ry, 540.0)

    def test_interpolation_and_tweens(self):
        self.assertAlmostEqual(lerp(0.0, 100.0, 0.5), 50.0)
        self.assertAlmostEqual(ease_in_quad(0.5), 0.25)
        self.assertAlmostEqual(ease_out_quad(0.5), 0.75)

        tween = Tween(duration=2.0)
        val = tween.update(1.0)
        self.assertGreater(val, 0.0)
        self.assertLess(val, 1.0)
        self.assertFalse(tween.is_finished())
        tween.update(1.0)
        self.assertTrue(tween.is_finished())

    def test_physics_primitives(self):
        pos = Vector2(0.0, 0.0)
        vel = Vector2(10.0, 0.0)
        accel = Vector2(0.0, 9.8)
        new_pos, new_vel = integrate_euler(pos, vel, accel, dt=1.0)
        self.assertAlmostEqual(new_vel.y, 9.8)
        self.assertAlmostEqual(new_pos.x, 10.0)
        self.assertAlmostEqual(new_pos.y, 9.8)

        # Collision detection
        self.assertTrue(intersect_aabb_aabb((0, 0, 10, 10), (2, 2, 10, 10)))
        self.assertFalse(intersect_aabb_aabb((0, 0, 10, 10), (20, 20, 10, 10)))

        self.assertTrue(intersect_circle_circle((0, 0, 5), (3, 0, 5)))
        self.assertFalse(intersect_circle_circle((0, 0, 5), (20, 0, 5)))

        self.assertTrue(intersect_circle_aabb((5, 5, 2), (0, 0, 10, 10)))

        traj = predict_projectile_trajectory((0.0, 0.0), (10.0, -10.0), gravity=9.81, time_step=0.1, num_steps=5)
        self.assertEqual(len(traj), 5)

    def test_fixed_timestep_accumulator(self):
        acc = FixedTimestepAccumulator(target_fps=60.0)
        acc.add_frame_time(0.033)  # ~2 steps at 60fps (dt = ~0.0166)
        steps = 0
        while acc.consume_step():
            steps += 1
        self.assertGreaterEqual(steps, 1)

    def test_rng_and_noise(self):
        rng = SeededRNG(seed=123)
        val1 = rng.random()
        rng.seed(123)
        val2 = rng.random()
        self.assertEqual(val1, val2)

        noise_val = perlin_noise_1d(2.5)
        self.assertGreaterEqual(noise_val, -1.0)
        self.assertLessEqual(noise_val, 1.0)

    def test_gesture_math(self):
        p1 = (0.0, 0.0)
        p2 = (3.0, 4.0)
        self.assertAlmostEqual(get_landmark_distance(p1, p2), 5.0)

        # 90 degree angle check
        j_a = (0.0, 1.0)
        j_b = (0.0, 0.0)
        j_c = (1.0, 0.0)
        angle = get_finger_angle(j_a, j_b, j_c)
        self.assertAlmostEqual(angle, 90.0)

        center, radius = get_hand_center_and_radius([(0, 0), (2, 0), (1, 2)])
        self.assertAlmostEqual(center[0], 1.0)
        self.assertAlmostEqual(center[1], 2.0 / 3.0)
        self.assertGreater(radius, 0.0)

        vx, vy = get_landmark_velocity((0.0, 0.0), (10.0, 20.0), dt=2.0)
        self.assertAlmostEqual(vx, 5.0)
        self.assertAlmostEqual(vy, 10.0)

    def test_generic_stream_filter(self):
        flt = GenericStreamFilter(alpha=0.5)
        f1 = flt.filter(10.0)
        self.assertEqual(f1, 10.0)
        f2 = flt.filter(20.0)
        self.assertEqual(f2, 15.0)


if __name__ == "__main__":
    unittest.main()
