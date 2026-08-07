import unittest
from visual_ai import GameEngine, PythonFallbackEngine, Material, Entity, ShaderType


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
        self.assertNotEqual(self.engine.get_y(), initial_y)

    def test_entity_management_and_materials(self):
        """Test adding general entities with PBR materials."""
        gold_mat = Material.preset("gold")
        
        # Test Python fallback engine entity adding
        ent_fb = self.fallback.add_entity("PlayerTarget", x=100.0, y=200.0, z=0.0, material=gold_mat)
        self.assertEqual(ent_fb.name, "PlayerTarget")
        self.assertEqual(ent_fb.material.name, "Gold")
        self.assertEqual(len(self.fallback.get_entities()), 1)

        # Test updating fallback engine with entity velocity
        ent_fb.vx = 50.0
        self.fallback.update(1.0)
        self.assertEqual(ent_fb.x, 150.0)

        # Test C++ engine or active backend
        self.engine.add_entity("Obstacle", x=50.0, y=50.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, w=10.0, h=10.0, d=1.0)
        entities = self.engine.get_entities()
        self.assertEqual(len(entities), 1)

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
