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


#: The engine's own scalar state. The fallback holds these as plain instance
#: attributes and the bindings expose them as properties; either way a game
#: reads and assigns them by these names.
SCALARS = ("x", "y", "vx", "vy", "gravity", "radius",
           "target_x", "target_y", "width", "height")


class TestScalarStateParity(unittest.TestCase):
    """
    The engine's own position, velocity and tuning, on both engines.

    The fallback kept all ten as ordinary attributes while the bindings
    published six getters and no setters at all, so `engine.gravity = 500` -
    an obvious thing to write against the fallback - raised AttributeError on
    any machine where the compiled core happened to load instead.
    """

    def test_every_scalar_reads_the_same_initial_value(self):
        values = {}
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                for name in SCALARS:
                    self.assertTrue(hasattr(engine, name),
                                    f"{label} engine has no {name!r}")
                values[label] = {n: getattr(engine, n) for n in SCALARS}
        if len(values) == 2:
            self.assertEqual(values["fallback"], values["cpp"])

    def test_every_scalar_is_assignable(self):
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                for i, name in enumerate(SCALARS):
                    setattr(engine, name, 7.5 + i)
                for i, name in enumerate(SCALARS):
                    self.assertEqual(getattr(engine, name), 7.5 + i,
                                     f"{label}: {name} did not take the write")

    def test_getters_still_agree_with_the_properties(self):
        """The six get_*() methods are the older API and stay bound."""
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                engine.x, engine.y = 11.0, 22.0
                engine.target_x, engine.target_y = 33.0, 44.0
                engine.width, engine.height = 55.0, 66.0
                self.assertEqual(engine.get_x(), 11.0)
                self.assertEqual(engine.get_y(), 22.0)
                self.assertEqual(engine.get_target_x(), 33.0)
                self.assertEqual(engine.get_target_y(), 44.0)
                self.assertEqual(engine.get_width(), 55.0)
                self.assertEqual(engine.get_height(), 66.0)

    def test_set_target_position_and_the_properties_are_one_value(self):
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                engine.set_target_position(120.0, 240.0)
                self.assertEqual((engine.target_x, engine.target_y), (120.0, 240.0))
                engine.target_x = 5.0
                self.assertEqual(engine.get_target_x(), 5.0)

    def test_a_written_scalar_changes_what_update_does(self):
        """
        The writes reach the physics, not just a shadow copy.

        Zero gravity, no target pull and no horizontal speed leaves the engine
        exactly where it was put - which is only true if all four assignments
        landed on the state `update()` reads.
        """
        for label, factory in ENGINES:
            with self.subTest(engine=label):
                engine = factory(800.0, 600.0)
                engine.gravity = 0.0
                engine.vx = engine.vy = 0.0
                engine.x, engine.y = 300.0, 300.0
                engine.set_target_position(300.0, 300.0)
                engine.update(0.5)
                self.assertAlmostEqual(engine.x, 300.0, places=4)
                self.assertAlmostEqual(engine.y, 300.0, places=4)


if __name__ == "__main__":
    unittest.main()
