# Reference surrogate weights

`surrogate.npz` is a packaged NumPy MLP artifact for experimental Phase 2 routing.
It contains float32 weights plus schema, column, status, normalization, and sentinel metadata.
Runtime gate rejects metadata or tensor mismatches and falls back to RK4.

Regenerate deterministic dataset and weights locally:

```bash
uv run python -m ml.dataset --n-rays 200 --max-steps 4000 --seed 42 --out-path ml/data/train.npz
uv run python -m ml.surrogate train --data ml/data/train.npz --epochs 40 --out ml/models/surrogate.npz
```

Dataset metadata records coordinate convention, input/output columns, status mapping,
sentinel policy, and integrator parameters.

Use ML only through explicit opt-in:

```bash
uv run blackhole-ray-tracer --phase2-render --phase2-ml-gate ml/models/surrogate.npz
```

Default renders use trusted Python or native integration.
Surrogate predictions remain experimental and do not claim image-quality accuracy.
