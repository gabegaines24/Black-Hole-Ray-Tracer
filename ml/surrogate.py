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
from typing import Any, NamedTuple, cast

import numpy as np

from ml.schema import (
    INPUT_COLUMNS,
    MODEL_FORMAT,
    MODEL_SCHEMA_VERSION,
    N_INPUTS as SCHEMA_N_INPUTS,
    N_OUTPUTS as SCHEMA_N_OUTPUTS,
    OUTPUT_COLUMNS,
    R_REF,
    SENTINEL_VALUE,
    STATUS_MAPPING,
    V_REF,
)


# Only import numpy — no deep-learning framework dependency at import time.


class MLPWeights(NamedTuple):
    W1: np.ndarray   # (hidden, input)
    b1: np.ndarray   # (hidden,)
    W2: np.ndarray   # (hidden, hidden)
    b2: np.ndarray   # (hidden,)
    W3: np.ndarray   # (output, hidden)
    b3: np.ndarray   # (output,)


N_HIDDEN = 64
N_INPUTS = SCHEMA_N_INPUTS
N_OUTPUTS = SCHEMA_N_OUTPUTS


def validate_weights(weights: MLPWeights) -> MLPWeights:
    """Validate model tensor dtype, shape, and finite-value invariants."""
    expected = {
        "W1": (N_HIDDEN, N_INPUTS),
        "b1": (N_HIDDEN,),
        "W2": (N_HIDDEN, N_HIDDEN),
        "b2": (N_HIDDEN,),
        "W3": (N_OUTPUTS, N_HIDDEN),
        "b3": (N_OUTPUTS,),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, shape in expected.items():
        array = getattr(weights, name)
        if not isinstance(array, np.ndarray):
            raise ValueError(f"Model tensor {name} must be a NumPy array.")
        if array.dtype != np.float32 or array.shape != shape:
            raise ValueError(f"Model tensor {name} must be float32 with shape {shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Model tensor {name} must contain finite values.")
        arrays[name] = array
    return MLPWeights(**arrays)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def forward(weights: MLPWeights, x: np.ndarray) -> np.ndarray:
    """Run forward pass: x shape (N, 6) → output (N, 4)."""
    weights = validate_weights(weights)
    x_array = np.asarray(x, dtype=np.float32)
    if x_array.ndim != 2 or x_array.shape[1] != N_INPUTS:
        raise ValueError(f"Model input must have shape (N, {N_INPUTS}).")
    if not np.all(np.isfinite(x_array)):
        raise ValueError("Model input must contain finite values.")
    h = _relu(x_array @ weights.W1.T + weights.b1)
    h = _relu(h @ weights.W2.T + weights.b2)
    return h @ weights.W3.T + weights.b3


def init_random_weights(rng: np.random.Generator | None = None) -> MLPWeights:
    if rng is None:
        rng = np.random.default_rng(0)
    scale1 = np.float32(np.sqrt(2.0 / N_INPUTS))
    scale2 = np.float32(np.sqrt(2.0 / N_HIDDEN))
    return MLPWeights(
        W1=(rng.standard_normal((N_HIDDEN, N_INPUTS)).astype(np.float32) * scale1).astype(
            np.float32
        ),
        b1=np.zeros(N_HIDDEN, dtype=np.float32),
        W2=(rng.standard_normal((N_HIDDEN, N_HIDDEN)).astype(np.float32) * scale2).astype(
            np.float32
        ),
        b2=np.zeros(N_HIDDEN, dtype=np.float32),
        W3=(rng.standard_normal((N_OUTPUTS, N_HIDDEN)).astype(np.float32) * scale2).astype(
            np.float32
        ),
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

    return validate_weights(MLPWeights(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3))


def _metadata_scalar(data: Any, key: str) -> Any:
    if key not in data:
        raise ValueError(f"Model metadata field {key!r} is missing.")
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"Model metadata field {key!r} must be scalar.")
    return value.item()


def _validate_model_metadata(data: Any) -> None:
    if int(_metadata_scalar(data, "schema_version")) != MODEL_SCHEMA_VERSION:
        raise ValueError("Unsupported model schema version.")
    if str(_metadata_scalar(data, "model_format")) != MODEL_FORMAT:
        raise ValueError("Unsupported model format.")
    required_metadata = (
        "input_columns",
        "output_columns",
        "status_mapping",
        "sentinel_value",
        "r_ref",
        "v_ref",
        "training_max_steps",
    )
    missing = [key for key in required_metadata if key not in data]
    if missing:
        raise ValueError(f"Model metadata fields are missing: {missing}.")
    input_columns = np.asarray(data["input_columns"]).tolist()
    output_columns = np.asarray(data["output_columns"]).tolist()
    if input_columns != INPUT_COLUMNS or output_columns != OUTPUT_COLUMNS:
        raise ValueError("Model column metadata does not match schema.")
    status_mapping = np.asarray(data["status_mapping"]).tolist()
    expected_statuses = [STATUS_MAPPING[name] for name in ("captured", "escaped", "max_steps", "numerical_error")]
    if status_mapping != expected_statuses:
        raise ValueError("Model status metadata does not match schema.")
    if float(_metadata_scalar(data, "sentinel_value")) != SENTINEL_VALUE:
        raise ValueError("Model sentinel metadata does not match schema.")
    if float(_metadata_scalar(data, "r_ref")) != R_REF:
        raise ValueError("Model input normalization metadata is invalid.")
    if float(_metadata_scalar(data, "v_ref")) != V_REF:
        raise ValueError("Model input normalization metadata is invalid.")
    training_max_steps = float(_metadata_scalar(data, "training_max_steps"))
    if not np.isfinite(training_max_steps) or training_max_steps <= 0.0:
        raise ValueError("Model training_max_steps must be finite and positive.")


def save_surrogate(
    weights: MLPWeights,
    path: str | Path,
    *,
    training_max_steps: float = 4000.0,
) -> None:
    weights = validate_weights(weights)
    if not np.isfinite(training_max_steps) or training_max_steps <= 0.0:
        raise ValueError("training_max_steps must be finite and positive.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        W1=weights.W1,
        b1=weights.b1,
        W2=weights.W2,
        b2=weights.b2,
        W3=weights.W3,
        b3=weights.b3,
        schema_version=np.array(MODEL_SCHEMA_VERSION, dtype=np.int64),
        model_format=np.array(MODEL_FORMAT),
        input_columns=np.asarray(INPUT_COLUMNS),
        output_columns=np.asarray(OUTPUT_COLUMNS),
        status_mapping=np.asarray(
            [STATUS_MAPPING[name] for name in ("captured", "escaped", "max_steps", "numerical_error")],
            dtype=np.int32,
        ),
        sentinel_value=np.array(SENTINEL_VALUE, dtype=np.float32),
        r_ref=np.array(R_REF, dtype=np.float32),
        v_ref=np.array(V_REF, dtype=np.float32),
        training_max_steps=np.array(training_max_steps, dtype=np.float32),
    )
    print(f"Saved surrogate → {path}")


def load_surrogate(path: str | Path) -> MLPWeights:
    with np.load(str(path), allow_pickle=False) as data:
        _validate_model_metadata(data)
        required_weights = ("W1", "b1", "W2", "b2", "W3", "b3")
        missing = [key for key in required_weights if key not in data]
        if missing:
            raise ValueError(f"Model tensors are missing: {missing}.")
        weights = MLPWeights(
            W1=np.array(data["W1"], copy=True),
            b1=np.array(data["b1"], copy=True),
            W2=np.array(data["W2"], copy=True),
            b2=np.array(data["b2"], copy=True),
            W3=np.array(data["W3"], copy=True),
            b3=np.array(data["b3"], copy=True),
        )
    return validate_weights(weights)


def predict_batch(weights: MLPWeights, X_norm: np.ndarray) -> np.ndarray:
    """Run inference: X_norm shape (N, 6) normalised → output (N, 4)."""
    return forward(weights, X_norm)


def main(argv: list[str] | None = None) -> None:
    import sys
    _root = Path(__file__).resolve().parents[1]
    if str(_root / "src") not in sys.path:
        sys.path.insert(0, str(_root / "src"))
    from ml.dataset import load_dataset, load_dataset_metadata
    from ml.schema import normalize_inputs, normalize_outputs

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
        metadata = load_dataset_metadata(args.data)
        integrator = cast(dict[str, Any], metadata["integrator"])
        training_max_steps = float(integrator["max_steps"])
        X = normalize_inputs(X_raw)
        Y = normalize_outputs(Y, max_steps=training_max_steps)
        print(f"Training on {len(X)} rays …")
        weights = init_random_weights()
        weights = train(weights, X, Y, epochs=args.epochs, lr=args.lr)
        save_surrogate(weights, args.out, training_max_steps=training_max_steps)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
