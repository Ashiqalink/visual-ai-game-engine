import unittest

from visual_ai.material import Material, ShaderType


class TestMaterial(unittest.TestCase):
    def test_default_material(self):
        mat = Material()
        self.assertEqual(mat.name, "DefaultMaterial")
        self.assertEqual(mat.shader_type, ShaderType.PBR_STANDARD)
        self.assertEqual(mat.roughness, 0.5)
        self.assertEqual(mat.metallic, 0.0)
        self.assertEqual(mat.base_color, (1.0, 1.0, 1.0, 1.0))

    def test_clamping_bounds(self):
        mat = Material(roughness=1.5, metallic=-0.2, opacity=2.0)
        self.assertEqual(mat.roughness, 1.0)
        self.assertEqual(mat.metallic, 0.0)
        self.assertEqual(mat.opacity, 1.0)

    def test_serialization(self):
        mat = Material(
            name="CustomGold",
            shader_type=ShaderType.PBR_STANDARD,
            base_color=(1.0, 0.8, 0.2, 1.0),
            roughness=0.1,
            metallic=0.9,
        )
        data = mat.to_dict()
        self.assertEqual(data["name"], "CustomGold")
        self.assertEqual(data["shader_type"], "PBR_Standard")
        self.assertEqual(data["roughness"], 0.1)

        restored = Material.from_dict(data)
        self.assertEqual(restored.name, mat.name)
        self.assertEqual(restored.shader_type, mat.shader_type)
        self.assertAlmostEqual(restored.roughness, mat.roughness)

    def test_shader_uniforms(self):
        mat = Material(roughness=0.2, metallic=0.8)
        uniforms = mat.to_shader_uniforms()
        self.assertIn("u_BaseColor", uniforms)
        self.assertEqual(uniforms["u_Roughness"], 0.2)
        self.assertEqual(uniforms["u_Metallic"], 0.8)
        self.assertEqual(uniforms["u_ShaderType"], "PBR_Standard")

    def test_presets(self):
        gold = Material.preset("gold")
        self.assertEqual(gold.name, "Gold")
        self.assertEqual(gold.metallic, 1.0)

        glass = Material.preset("glass")
        self.assertEqual(glass.shader_type, ShaderType.TRANSPARENT)
        self.assertEqual(glass.opacity, 0.2)

        with self.assertRaises(ValueError):
            Material.preset("unknown_material_preset")


if __name__ == "__main__":
    unittest.main()
