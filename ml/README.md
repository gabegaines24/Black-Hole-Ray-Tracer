# `ml/` — ML Surrogate for Geodesic Warm-Starting

This package provides a lightweight **NumPy-only** MLP surrogate that predicts geodesic outcomes
(status, minimum radius, termination radius, steps taken) for rays starting in the low-curvature
region \(r > r_\text{threshold}\).

## Layout

```
ml/
├── __init__.py          # Public exports
├── schema.py            # Data contract, normalisation, constants
├── dataset.py           # Dataset generation, save/load
├── surrogate.py         # MLP architecture, training loop
├── runtime_gate.py      # Runtime dispatch: surrogate vs RK4
├── data/                # Generated datasets (*.npz, gitignored)
└── models/              # Saved model weights (*.npz, gitignored)
```

## Quick Start

```python
from ml.dataset import generate_dataset, save_dataset
from ml.surrogate import init_random_weights, train, predict_batch
from ml.runtime_gate import RuntimeGate

# 1. Generate training data
X, Y = generate_dataset(n_rays=10_000, seed=42)
save_dataset(X, Y, "ml/data/train.npz")

# 2. Train the surrogate
weights = init_random_weights()
weights, history = train(weights, X, Y, epochs=50)

# 3. Wrap in a runtime gate (falls back to RK4 for r < threshold)
gate = RuntimeGate(weights=weights)
result = gate.trace(y0_state, m=1.0, dlambda=0.06, max_steps=8000)
```

## Data Contract

See [`schema.py`](schema.py) for full input/output normalisation.

| Column | Raw unit | Normalised |
|--------|----------|-----------|
| r | M | r / 30 |
| theta | rad | (θ − π/2) / π |
| phi | rad | φ / 2π |
| v_r | geometric | v_r / 1.0 |
| v_theta | geometric | v_θ × 30 |
| v_phi | geometric | v_φ × 30 |

Outputs: `status` (int), `r_min` (M), `termination_r` (M), `steps_taken`.

## RuntimeGate Threshold

The surrogate is **only used** when `ray.r > SURROGATE_R_THRESHOLD` (default 10 M).
Rays closer to the black hole are always integrated with the full RK4 C kernel for accuracy.

## Tests

```bash
uv run pytest tests/test_ml.py -v
```
