"""
train_gesture_mlp.py — train a GestureMLP from samples recorded with
collect_gesture_samples.py, and save the resulting model.

    python tools/train_gesture_mlp.py --data gesture_samples.npz --out gesture_mlp.npz

Wiring the result into VisionPipeline:

    from visual_ai import VisionPipeline, GestureMLP
    model = GestureMLP.load("gesture_mlp.npz")
    pipeline = VisionPipeline(result_queue=q, gesture_mlp=model)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from visual_ai.gesture_mlp import GestureMLP


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help=".npz produced by collect_gesture_samples.py")
    parser.add_argument("--out", default="gesture_mlp.npz", help="output model .npz path")
    parser.add_argument("--hidden", default="32,16", help="two hidden layer sizes, comma-separated")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.2,
                         help="fraction of samples held out to report validation accuracy")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hidden = tuple(int(s) for s in args.hidden.split(","))
    if len(hidden) != 2:
        print("[train] --hidden must name exactly two layer sizes, e.g. 32,16", file=sys.stderr)
        return 1

    data = np.load(args.data, allow_pickle=False)
    X, y = data["X"], data["y"]
    labels = sorted(set(y.tolist()))
    if len(labels) < 2:
        print(f"[train] need at least 2 distinct labels, found {labels}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(X))
    X, y = X[order], y[order]
    n_val = int(len(X) * args.val_split)
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    print(f"[train] {len(X_train)} train / {len(X_val)} val samples, labels={labels}")

    model = GestureMLP(labels, hidden=hidden, seed=args.seed)
    losses = model.train(
        X_train, y_train.tolist(),
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, verbose=True,
    )
    print(f"[train] loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    def accuracy(X_set, y_set):
        if len(X_set) == 0:
            return None
        correct = sum(model.predict(x)[0] == label for x, label in zip(X_set, y_set))
        return correct / len(X_set)

    train_acc = accuracy(X_train, y_train)
    val_acc = accuracy(X_val, y_val)
    print(f"[train] train accuracy: {train_acc:.3f}")
    if val_acc is not None:
        print(f"[train] val accuracy:   {val_acc:.3f}")

    model.save(args.out)
    print(f"[train] saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
