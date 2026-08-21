"""
gesture_mlp.py — small numpy MLP for landmark-based hand-sign classification.

`classify_hand_sign()` in pipeline.py is a hand-tuned geometric classifier:
finger-extension booleans looked up in a fixed table. That is robust but
closed-vocabulary — adding a new sign means reasoning out a new geometric
rule by hand. `GestureMLP` is the trainable alternative: it learns a
classifier from recorded (landmarks, label) examples instead, so a new sign
is a data-collection problem rather than a geometry problem.

This is additive, not a replacement. Nothing in the pipeline's automatic
queue payload changes — `hand_sign` still comes from `classify_hand_sign()`
by default. A `GestureMLP` is opt-in: load one and call `.predict()` on
landmarks a consumer already has (from a MediaPipe `Hands` result, or from
`VisionPipeline._extract_gesture`'s `lm` argument if a game embeds the
pipeline directly).

Kept dependency-free (plain numpy, already a hard dependency of this
package) rather than pulling in a training framework — parity with the rest
of `visual_ai`, which mirrors physics between a C++ engine and a numpy
fallback rather than depending on an external ML runtime.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GestureMLP",
    "landmarks_to_features",
    "NUM_LANDMARKS",
    "NUM_FEATURES",
]

NUM_LANDMARKS = 21   # MediaPipe Hands landmark count
NUM_FEATURES = NUM_LANDMARKS * 3   # x, y, z per landmark

_WRIST = 0
_MIDDLE_MCP = 9   # palm-span anchor: roughly constant across hand poses


def landmarks_to_features(lm) -> np.ndarray:
    """
    Flatten MediaPipe hand landmarks into a translation/scale-normalized
    feature vector.

    Every landmark is expressed relative to the wrist and divided by the
    wrist→middle-MCP distance (palm span), the same normalization
    `VisionPipeline._extract_gesture` uses for finger-extension geometry.
    That makes the features invariant to hand size and distance from the
    camera, which matters more for a learned classifier than a hand-coded
    one: a hand-coded threshold can be re-tuned per-feature, but an MLP
    trained on unnormalized coordinates would just memorize the training
    session's camera distance.

    Parameters
    ----------
    lm : sequence of 21 landmarks, each with `.x`, `.y`, `.z` in [0, 1]
        (a MediaPipe `NormalizedLandmarkList.landmark`, or anything
        duck-typed the same way).

    Returns
    -------
    np.ndarray, shape (NUM_FEATURES,), dtype float32
    """
    pts = np.array([(p.x, p.y, p.z) for p in lm], dtype=np.float32)
    wrist = pts[_WRIST]
    palm_span = float(np.linalg.norm(pts[_MIDDLE_MCP] - wrist))
    palm_span = max(palm_span, 1e-6)
    return ((pts - wrist) / palm_span).reshape(-1)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class GestureMLP:
    """
    A minimal two-hidden-layer MLP: NUM_FEATURES -> hidden -> hidden -> classes.

    Forward pass and training are both plain numpy so the model has no
    runtime dependency beyond what `visual_ai` already requires, and a
    trained model is a handful of small arrays that save/load as one .npz
    file — no format version skew to worry about between training and
    inference environments.
    """

    def __init__(self, labels, hidden=(32, 16), seed=0):
        """
        Parameters
        ----------
        labels : sequence[str]
            Class names, in the order the output layer predicts them.
        hidden : tuple[int, int]
            Sizes of the two hidden layers.
        seed : int
            Seed for weight initialization, so an untrained model is
            reproducible in tests rather than flaking on random init.
        """
        self.labels = list(labels)
        if len(self.labels) < 2:
            raise ValueError("GestureMLP needs at least 2 labels to classify between")
        h1, h2 = hidden
        rng = np.random.default_rng(seed)

        def init(fan_in, fan_out):
            # He init: matches the ReLU hidden layers below.
            scale = np.sqrt(2.0 / fan_in)
            return (rng.standard_normal((fan_in, fan_out)) * scale).astype(np.float32)

        self.w1 = init(NUM_FEATURES, h1)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.w2 = init(h1, h2)
        self.b2 = np.zeros(h2, dtype=np.float32)
        self.w3 = init(h2, len(self.labels))
        self.b3 = np.zeros(len(self.labels), dtype=np.float32)

    # ── Inference ────────────────────────────────────────────────────────
    def _forward(self, x: np.ndarray):
        """x: (N, NUM_FEATURES). Returns (probs, cache) for use by both
        predict() and the training step below."""
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0.0)
        z3 = a2 @ self.w3 + self.b3
        probs = _softmax(z3)
        return probs, (x, z1, a1, z2, a2)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """features: (NUM_FEATURES,) or (N, NUM_FEATURES) -> matching-shape
        (num_labels,) or (N, num_labels) probability vector(s)."""
        x = np.asarray(features, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        probs, _ = self._forward(x)
        return probs[0] if single else probs

    def predict(self, features: np.ndarray):
        """Returns (label: str, confidence: float) for a single feature vector."""
        probs = self.predict_proba(features)
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])

    # ── Training ─────────────────────────────────────────────────────────
    def train(self, X, y, epochs=200, lr=0.05, batch_size=32, seed=0, verbose=False):
        """
        Train in-place with plain mini-batch SGD on cross-entropy loss.

        Parameters
        ----------
        X : array-like, shape (N, NUM_FEATURES)
            Feature vectors — typically `landmarks_to_features()` output,
            stacked one row per recorded sample.
        y : sequence[str]
            Label for each row, must be one of `self.labels`.
        epochs, lr, batch_size : training hyperparameters.
        seed : shuffling seed, kept separate from init seed so re-training
            the same model instance is reproducible without re-seeding
            the weights.
        verbose : if True, returns the per-epoch mean loss list.

        Returns
        -------
        list[float] — mean cross-entropy loss per epoch (empty if
        verbose=False, to avoid holding history nobody asked for).
        """
        X = np.asarray(X, dtype=np.float32)
        label_to_idx = {label: i for i, label in enumerate(self.labels)}
        try:
            y_idx = np.array([label_to_idx[label] for label in y], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"unknown label {exc} — not in {self.labels}") from exc

        n = X.shape[0]
        if n == 0:
            raise ValueError("no training samples given")
        rng = np.random.default_rng(seed)
        losses = []

        for _epoch in range(epochs):
            order = rng.permutation(n)
            epoch_losses = []
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb, yb = X[idx], y_idx[idx]
                loss = self._train_step(xb, yb, lr)
                epoch_losses.append(loss)
            if verbose:
                losses.append(float(np.mean(epoch_losses)))
        return losses

    def _train_step(self, xb: np.ndarray, yb: np.ndarray, lr: float) -> float:
        n = xb.shape[0]
        probs, (x, z1, a1, z2, a2) = self._forward(xb)

        eps = 1e-9
        loss = -np.mean(np.log(probs[np.arange(n), yb] + eps))

        # Backprop: cross-entropy + softmax gradient collapses to (probs - one_hot).
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n), yb] = 1.0
        dz3 = (probs - one_hot) / n

        dw3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.w3.T
        dz2 = da2 * (z2 > 0)

        dw2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.w2.T
        dz1 = da1 * (z1 > 0)

        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        self.w3 -= lr * dw3
        self.b3 -= lr * db3
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1

        return float(loss)

    # ── Persistence ──────────────────────────────────────────────────────
    def save(self, path):
        np.savez(
            path,
            labels=np.array(self.labels),
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            w3=self.w3, b3=self.b3,
        )

    @classmethod
    def load(cls, path) -> GestureMLP:
        data = np.load(path, allow_pickle=False)
        labels = [str(label) for label in data["labels"]]
        model = cls(labels, hidden=(data["w1"].shape[1], data["w2"].shape[1]))
        model.w1, model.b1 = data["w1"], data["b1"]
        model.w2, model.b2 = data["w2"], data["b2"]
        model.w3, model.b3 = data["w3"], data["b3"]
        return model
