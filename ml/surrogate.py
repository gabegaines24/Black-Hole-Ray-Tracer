"""Minimal MLP surrogate model for Phase 2 null geodesic outcomes.

Architecture: 3-layer MLP, 64 hidden units, ReLU activations.
Predicts (status, r_min, termination_r, steps_taken) from 6 normalised inputs.

Training
--------
    uv run python -m ml.surrogate train --data ml/data/train.npz --epochs 100

Inference
---------
    from ml.surrogate import load_surrogate, predict_batch
    model = load_surrogate("ml/models/surrogate.npz")
    Y_pred = predict_batch(model, X_norm)
"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = ["MLPWeights", "init_random_weights", "train", "predict_batch"]
from typing import NamedTuple

import numpy as np

# Only import numpy — no deep-learning framework dependency at import time.


class MLPWeights(NamedTuple):
    W1: np.ndarray   # (hidden, input)
    b1: np.ndarray   # (hidden,)
    W2: np.ndarray   # (hidden, hidden)
    b2: np.ndarray   # (hidden,)
    W3: np.ndarray   # (output, hidden)
    b3: np.ndarray   # (output,)


N_HIDDEN = 64
N_INPUTS = 6
N_OUTPUTS = 4


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def forward(weights: MLPWeights, x: np.ndarray) -> np.ndarray:
    """Run forward pass: x shape (N, 6) → output (N, 4)."""
    h = _relu(x @ weights.W1.T + weights.b1)
    h = _relu(h @ weights.W2.T + weights.b2)
    return h @ weights.W3.T + weights.b3


def init_random_weights(rng: np.random.Generator | None = None) -> MLPWeights:
    if rng is None:
        rng = np.random.default_rng(0)
    scale1 = np.sqrt(2.0 / N_INPUTS)
    scale2 = np.sqrt(2.0 / N_HIDDEN)
    return MLPWeights(
        W1=rng.standard_normal((N_HIDDEN, N_INPUTS)).astype(np.float32) * scale1,
        b1=np.zeros(N_HIDDEN, dtype=np.float32),
        W2=rng.standard_normal((N_HIDDEN, N_HIDDEN)).astype(np.float32) * scale2,
        b2=np.zeros(N_HIDDEN, dtype=np.float32),
        W3=rng.standard_normal((N_OUTPUTS, N_HIDDEN)).astype(np.float32) * scale2,
        b3=np.zeros(N_OUTPUTS, dtype=np.float32),
    )


def _mse_loss(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def _grad_output(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    n = pred.shape[0]
    return 2.0 * (pred - target) / n


def train(
    weights: MLPWeights,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    verbose: bool = True,
) -> MLPWeights:
    """Train the MLP via mini-batch SGD (pure NumPy, no autograd).

    Returns updated weights.
    """
    rng = np.random.default_rng(1)
    n = X.shape[0]

    W1, b1 = weights.W1.copy(), weights.b1.copy()
    W2, b2 = weights.W2.copy(), weights.b2.copy()
    W3, b3 = weights.W3.copy(), weights.b3.copy()

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb = X[batch_idx].astype(np.float32)
            yb = Y[batch_idx].astype(np.float32)

            # Forward
            z1 = xb @ W1.T + b1
            h1 = _relu(z1)
            z2 = h1 @ W2.T + b2
            h2 = _relu(z2)
            out = h2 @ W3.T + b3

            loss = _mse_loss(out, yb)
            total_loss += loss
            n_batches += 1

            # Backward (manual chain rule)
            d_out = _grad_output(out, yb)
            dW3 = d_out.T @ h2
            db3 = d_out.sum(axis=0)

            d_h2 = d_out @ W3
            d_z2 = d_h2 * (z2 > 0).astype(np.float32)
            dW2 = d_z2.T @ h1
            db2 = d_z2.sum(axis=0)

            d_h1 = d_z2 @ W2
            d_z1 = d_h1 * (z1 > 0).astype(np.float32)
            dW1 = d_z1.T @ xb
            db1 = d_z1.sum(axis=0)

            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            W3 -= lr * dW3
            b3 -= lr * db3

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}/{epochs}  loss={total_loss/n_batches:.4f}")

    return MLPWeights(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)


def save_surrogate(weights: MLPWeights, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        W1=weights.W1, b1=weights.b1,
        W2=weights.W2, b2=weights.b2,
        W3=weights.W3, b3=weights.b3,
    )
    print(f"Saved surrogate → {path}")


def load_surrogate(path: str | Path) -> MLPWeights:
    data = np.load(str(path))
    return MLPWeights(
        W1=data["W1"], b1=data["b1"],
        W2=data["W2"], b2=data["b2"],
        W3=data["W3"], b3=data["b3"],
    )


def predict_batch(weights: MLPWeights, X_norm: np.ndarray) -> np.ndarray:
    """Run inference: X_norm shape (N, 6) normalised → output (N, 4)."""
    return forward(weights, X_norm.astype(np.float32))


def main(argv: list[str] | None = None) -> None:
    import sys
    _root = Path(__file__).resolve().parents[1]
    if str(_root / "src") not in sys.path:
        sys.path.insert(0, str(_root / "src"))
    from ml.dataset import load_dataset
    from ml.schema import normalize_inputs

    parser = argparse.ArgumentParser(description="Train or evaluate the MLP surrogate.")
    sub = parser.add_subparsers(dest="cmd")
    train_p = sub.add_parser("train", help="Train from .npz dataset")
    train_p.add_argument("--data", type=str, default="ml/data/train.npz")
    train_p.add_argument("--out", type=str, default="ml/models/surrogate.npz")
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)

    if args.cmd == "train":
        X_raw, Y = load_dataset(args.data)
        X = normalize_inputs(X_raw)
        print(f"Training on {len(X)} rays …")
        weights = init_random_weights()
        weights = train(weights, X, Y, epochs=args.epochs, lr=args.lr)
        save_surrogate(weights, args.out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
