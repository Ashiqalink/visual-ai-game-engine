"""
Behavioural parity between the compiled core and the Python fallback.

`visual_ai.GameEngine` is whichever of the two loaded, and which one that is
depends on nothing more than whether an `engine_core*.pyd` happens to sit next
to the package. Anything a game can observe therefore has to behave the same on
both, or the same game code silently does different things on two machines.

These tests run every assertion against both engines. When the C++ core is not
built the compiled half is skipped rather than failing — the fallback half still
guards the contract.
"""
import unittest

from visual_ai import CPP_ENGINE_AVAILABLE, Material, PythonFallbackEngine, ShaderType

ENGINES = [("fallback", PythonFallbackEngine)]
if CPP_ENGINE_AVAILABLE:
    import engine_core

    ENGINES.append(("cpp", engine_core.GameEngine))


class TestEngineParity(unittest.TestCase):
    def test_add_entity_returns_the_entity(self):
        """add_entity() hands back the Entity, not its integer id."""
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                entity = engine.add_entity("Player", x=10.0, y=20.0)
                self.assertNotIsInstance(entity, int)
                self.assertEqual(entity.id, 1)
                self.assertEqual(entity.name, "Player")
                entity.x = 42.0
                self.assertEqual(engine.get_entities()[0].x, 42.0)

    def test_size_keywords(self):
        """w/h/d and width/height/depth are both accepted and mean the same."""
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                entity = engine.add_entity("Sized", w=3.0, h=4.0, d=5.0)
                self.assertEqual((entity.width, entity.height, entity.depth), (3.0, 4.0, 5.0))

    def test_get_entities_returns_live_objects(self):
        """Writes through get_entities() reach the engine instead of a copy."""
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                entity = engine.add_entity("Mover")
                engine.get_entities()[0].y = 123.0
                self.assertEqual(entity.y, 123.0)
                self.assertIs(engine.get_entities()[0], entity)

                engine.clear_entities()
                self.assertEqual(len(engine.get_entities()), 0)

    def test_material_is_the_sdk_material(self):
        """Materials go in and come back out as visual_ai.Material."""
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                gold = Material.preset("gold")
                entity = engine.add_entity("Gilded", material=gold)
                self.assertIsInstance(entity.material, Material)
                self.assertIs(entity.material, gold)
                self.assertEqual(entity.material.name, "Gold")

                entity.material = Material.preset("glass")
                self.assertEqual(
                    engine.get_entities()[0].material.shader_type, ShaderType.TRANSPARENT
                )

                self.assertIsInstance(engine.add_entity("Bare").material, Material)

    def test_material_rejects_other_types(self):
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                with self.assertRaises(TypeError):
                    engine.add_entity("Wrong", material={"base_color": (1.0, 0.0, 0.0, 1.0)})

    def test_3d_element_carries_a_mesh(self):
        """A mesh is a Python object both engines hold on to unchanged."""
        mesh = ["triangles"]
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                element = engine.add_3d_element("Cube", scale=2.0, mesh=mesh)
                self.assertIs(element.mesh, mesh)
                self.assertIs(engine.get_entities()[0].mesh, mesh)
                self.assertEqual(
                    (element.width, element.height, element.depth), (2.0, 2.0, 2.0)
                )
                self.assertIsNone(engine.add_entity("Flat").mesh)

    def test_rotation_wrap_keeps_its_sign(self):
        """Both wrap with fmod semantics, so a negative spin stays negative."""
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                spinning = engine.add_3d_element("Spinner", vry=-100.0)
                engine.update(1.0)
                self.assertAlmostEqual(spinning.ry, -100.0, places=4)
                engine.update(3.0)
                self.assertAlmostEqual(spinning.ry, -40.0, places=4)

    def test_entity_ids_start_at_one_and_increment(self):
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                self.assertEqual(engine.add_entity("A").id, 1)
                self.assertEqual(engine.add_3d_element("B").id, 2)


if __name__ == "__main__":
    unittest.main()
