"""
test_gesture_mlp.py — Unit tests for GestureMLP and landmarks_to_features.
"""

import os
import tempfile
import unittest

import numpy as np

from visual_ai.gesture_mlp import GestureMLP, landmarks_to_features, NUM_FEATURES, NUM_LANDMARKS


class _FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def _make_hand(offset=(0.0, 0.0, 0.0), scale=1.0, seed=0):
    """21 landmarks scattered around a wrist at `offset`, scaled by `scale` —
    enough to exercise the normalization math without needing real MediaPipe
    output."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.1, 0.1, size=(NUM_LANDMARKS, 3)) * scale
    pts[0] = (0.0, 0.0, 0.0)          # wrist stays the origin of the cluster
    pts[9] = (0.1 * scale, 0.0, 0.0)  # middle MCP sets the palm span
    pts += np.array(offset)
    return [_FakeLandmark(*p) for p in pts]


class TestLandmarksToFeatures(unittest.TestCase):

    def test_shape_and_dtype(self):
        lm = _make_hand()
        feats = landmarks_to_features(lm)
        self.assertEqual(feats.shape, (NUM_FEATURES,))
        self.assertEqual(feats.dtype, np.float32)

    def test_translation_invariant(self):
        """Same hand pose, shifted in frame, must produce identical features
        — a learned classifier must not care where in the image the hand is."""
        lm_a = _make_hand(offset=(0.0, 0.0, 0.0))
        lm_b = _make_hand(offset=(0.3, 0.4, 0.0))
        np.testing.assert_allclose(
            landmarks_to_features(lm_a), landmarks_to_features(lm_b), atol=1e-5
        )

    def test_scale_invariant(self):
        """Same hand pose nearer vs. farther from the camera (bigger/smaller
        in frame) must also normalize to the same features."""
        lm_near = _make_hand(scale=1.0)
        lm_far = _make_hand(scale=0.4)
        np.testing.assert_allclose(
            landmarks_to_features(lm_near), landmarks_to_features(lm_far), atol=1e-5
        )

    def test_degenerate_palm_span_does_not_explode(self):
        """Wrist and middle-MCP collapsed to the same point (a bad detection
        frame) must not divide by zero."""
        lm = [_FakeLandmark(0.0, 0.0, 0.0) for _ in range(NUM_LANDMARKS)]
        feats = landmarks_to_features(lm)
        self.assertTrue(np.all(np.isfinite(feats)))


class TestGestureMLP(unittest.TestCase):

    def test_rejects_fewer_than_two_labels(self):
        with self.assertRaises(ValueError):
            GestureMLP(["only_one"])

    def test_predict_proba_shape_and_normalization(self):
        model = GestureMLP(["fist", "open_palm", "point"], seed=1)
        feats = np.zeros(NUM_FEATURES, dtype=np.float32)
        probs = model.predict_proba(feats)
        self.assertEqual(probs.shape, (3,))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertTrue(np.all(probs >= 0.0))

    def test_predict_returns_known_label(self):
        model = GestureMLP(["fist", "open_palm"], seed=2)
        label, confidence = model.predict(np.zeros(NUM_FEATURES, dtype=np.float32))
        self.assertIn(label, model.labels)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_batched_predict_proba_matches_single(self):
        model = GestureMLP(["fist", "open_palm"], seed=3)
        rng = np.random.default_rng(0)
        batch = rng.standard_normal((5, NUM_FEATURES)).astype(np.float32)
        batched = model.predict_proba(batch)
        self.assertEqual(batched.shape, (5, 2))
        for i in range(5):
            single = model.predict_proba(batch[i])
            np.testing.assert_allclose(single, batched[i], atol=1e-5)

    def test_training_converges_on_separable_toy_data(self):
        """Two trivially separable clusters — training must drive the model
        to near-perfect accuracy, or the backprop math is wrong."""
        rng = np.random.default_rng(42)
        n_per_class = 40
        cluster_a = rng.normal(loc=-3.0, scale=0.3, size=(n_per_class, NUM_FEATURES))
        cluster_b = rng.normal(loc=3.0, scale=0.3, size=(n_per_class, NUM_FEATURES))
        X = np.concatenate([cluster_a, cluster_b]).astype(np.float32)
        y = ["a"] * n_per_class + ["b"] * n_per_class

        model = GestureMLP(["a", "b"], hidden=(16, 8), seed=0)
        losses = model.train(X, y, epochs=150, lr=0.1, batch_size=16, verbose=True)

        self.assertLess(losses[-1], losses[0])

        correct = 0
        for row, label in zip(X, y):
            pred, _ = model.predict(row)
            correct += pred == label
        accuracy = correct / len(y)
        self.assertGreaterEqual(accuracy, 0.95)

    def test_train_rejects_unknown_label(self):
        model = GestureMLP(["a", "b"], seed=0)
        X = np.zeros((2, NUM_FEATURES), dtype=np.float32)
        with self.assertRaises(ValueError):
            model.train(X, ["a", "not_a_label"], epochs=1)

    def test_train_rejects_empty_dataset(self):
        model = GestureMLP(["a", "b"], seed=0)
        X = np.zeros((0, NUM_FEATURES), dtype=np.float32)
        with self.assertRaises(ValueError):
            model.train(X, [], epochs=1)

    def test_save_load_round_trip_preserves_predictions(self):
        rng = np.random.default_rng(7)
        model = GestureMLP(["fist", "open_palm", "point"], hidden=(12, 6), seed=4)
        probe = rng.standard_normal(NUM_FEATURES).astype(np.float32)
        before = model.predict_proba(probe)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.npz")
            model.save(path)
            self.assertTrue(os.path.exists(path))
            loaded = GestureMLP.load(path)

        self.assertEqual(loaded.labels, model.labels)
        after = loaded.predict_proba(probe)
        np.testing.assert_allclose(before, after, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
