import unittest
import math
from visual_ai import GameEngine, PythonFallbackEngine, CPP_ENGINE_AVAILABLE


class TestGameEngine(unittest.TestCase):
    def setUp(self):
        self.width = 800.0
        self.height = 600.0
        self.engine = GameEngine(self.width, self.height)
        self.fallback = PythonFallbackEngine(self.width, self.height)

    def test_initial_state(self):
        """Test initial coordinates and dimensions."""
        self.assertEqual(self.engine.get_width(), self.width)
        self.assertEqual(self.engine.get_height(), self.height)
        self.assertAlmostEqual(self.engine.get_target_x(), self.width / 2.0)
        self.assertAlmostEqual(self.engine.get_target_y(), self.height / 2.0)

    def test_target_position_update(self):
        """Test updating target coordinates."""
        new_x, new_y = 350.0, 420.0
        self.engine.set_target_position(new_x, new_y)
        self.assertAlmostEqual(self.engine.get_target_x(), new_x)
        self.assertAlmostEqual(self.engine.get_target_y(), new_y)

    def test_physics_update(self):
        """Test position update over time step dt."""
        initial_y = self.engine.get_y()
        self.engine.update(0.016)  # 16ms frame step
        # Position should change due to gravity and velocity
        self.assertNotEqual(self.engine.get_y(), initial_y)

    def test_boundary_constraints(self):
        """Test that physics object stays within window boundaries over multiple frames."""
        for _ in range(500):
            self.engine.update(0.016)
            x = self.engine.get_x()
            y = self.engine.get_y()
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, self.width)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, self.height)

    def test_fallback_engine_parity(self):
        """Test that PythonFallbackEngine operates identically to specified behavior."""
        self.assertEqual(self.fallback.get_width(), self.width)
        self.assertEqual(self.fallback.get_height(), self.height)
        self.fallback.set_target_position(100.0, 100.0)
        self.assertEqual(self.fallback.get_target_x(), 100.0)
        self.assertEqual(self.fallback.get_target_y(), 100.0)
        initial_y = self.fallback.get_y()
        self.fallback.update(0.016)
        self.assertNotEqual(self.fallback.get_y(), initial_y)


if __name__ == "__main__":
    unittest.main()
