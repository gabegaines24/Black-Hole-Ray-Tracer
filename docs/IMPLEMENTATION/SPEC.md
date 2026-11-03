# Implementation specification

This file defines observable behavior for current Phase 1–3 code and optional backends.
It complements equations in `docs/PROJECT_CONTRACT.md` and array details in `docs/STATE_API.md`.

## 1. Scope and authority

- Python remains the reference implementation for every phase.
- Phase 2 C and native CPU paths implement the Schwarzschild contract.
- CUDA is optional and must report unavailable toolchains or runtime errors explicitly.
- Phase 3 is a Python-only Kerr prototype until a separate native-port ticket passes parity.
- ML is an opt-in accelerator experiment. Invalid models must fall back to RK4.
- `docs/STATUS.md` records observed qualification. This file does not imply unavailable hardware.

Authority order:

1. `docs/PROJECT_CONTRACT.md`
2. `docs/STATE_API.md`
3. Accepted `docs/DECISIONS/ADR-*.md`
4. This specification
5. `docs/FINAL_ARCHITECTURE.md`
6. `docs/STATUS.md`
7. `tasks/plan.md`

## 2. Coordinate and state contracts

### Phase 1

Use geometric units `G = c = 1` and the shared Python `phase1.rk4_step`.
Equatorial Schwarzschild helpers use their documented `(u, du/dphi)` convention.

### Phase 2 and Phase 3

Use affine parameter `lambda` and state order:

```text
(t, r, theta, phi, vt, vr, vtheta, vphi)
```

Coordinates are spherical Schwarzschild for Phase 2.
Coordinates are Kerr Boyer–Lindquist for Phase 3.
All Python state arrays use `float64` unless an API explicitly states another dtype.

### Status values

`RayStatus` and C `BH_RT_STATUS_*` values are:

| Name | Integer | Meaning |
|---|---:|---|
| `CAPTURED` | 0 | Radius crossed capture or horizon margin |
| `ESCAPED` | 1 | Radius exceeded escape threshold |
| `MAX_STEPS` | 2 | Integrator exhausted step budget |
| `NUMERICAL_ERROR` | 3 | State or derivative became non-finite |

Unknown status values are invalid at API boundaries.

## 3. Integration contract

- Use RK4 with positive finite `dlambda`.
- Reject non-positive masses, invalid radii, invalid step counts, and non-finite controls.
- Record `steps_taken` as completed RK4 steps.
- Record `r_min` over visited finite states; return NaN when no finite radius exists.
- Terminate before another step when capture, escape, or non-finite state is observed.
- Keep output shapes deterministic, including empty batches.

Phase 2 and CUDA use `r < 2M + r_horizon_epsilon`.
Phase 3 uses `r < r_plus + r_horizon_epsilon`, with
`r_plus = M + sqrt(M^2 - (M*spin)^2)`.

## 4. Disk observables

A disk hit is the first strict sign-changing crossing of `theta = pi / 2`.
The implementation linearly interpolates `r`, `vt`, and `vphi` between adjacent states.
A ray captured before crossing has no disk hit.
A tangent or endpoint equality without a sign change is not a crossing.
Disk color code clamps finite RGB values to `[0, 1]`.

## 5. Phase 2 backend contract

Python `phase2_render.py` is the inspectable reference path.
CPU native scalar and batch paths use canonical C sources under `kernel/src/`.
The PyBind11 bridge validates arrays, normalizes contiguous storage, releases the GIL, and returns NumPy arrays.
The batch ABI is row-major input `(N, 8)` and `int32` status/step outputs.
See `docs/STATE_API.md` for every field and dtype.

CUDA uses the same batch layout and returns a signed `int32` CUDA error code.
Zero means success. Nonzero means invalid result and the Python façade raises `RuntimeError`.
The host wrapper checks allocation, copies, kernel launch, synchronization, and cleanup.

Preview backend order is explicit:

1. CUDA batch when requested and available.
2. CPU native batch when requested and available.
3. Python or ML fallback according to configuration.

No backend may silently claim CUDA availability.

## 6. Phase 3 Kerr contract

- Validate finite `m`, `spin`, `r`, `theta`, and `phi`.
- Require `m > 0`, `abs(spin) <= 1`, `r > r_plus`, and `0 < theta < pi`.
- Reject observer coordinate poles where `abs(sin(theta)) < 1e-10`.
- Build initial null directions in a local ZAMO orthonormal frame.
- Enforce the null constraint initially and every fourth RK4 step by adjusting `vr` in both Python and the bounded C single-ray tracer.
- Monitor Killing energy `E`, axial angular momentum `L`, and null Carter constant:

```text
p_theta = Sigma * vtheta
Q = p_theta^2 + cos(theta)^2 * (L^2 / sin(theta)^2 - a^2 * E^2)
a = m * spin
```

- Keep Kerr disk color and rendering Python-only; C algebra, tracer, camera, crossing, and geometry-independent disk policy remain scalar and have no samples, color, bridge, batch, or CUDA behavior.

## 7. ML contract

Dataset and model files require versioned metadata.
Metadata records coordinate convention, state order, status mapping, sentinel policy, normalization, and max steps.
Loaders reject missing metadata, wrong shapes, wrong dtypes, non-finite tensors, and incompatible schemas.
Runtime prediction validates status, radius, and step bounds.
Any invalid artifact or prediction selects the RK4 fallback.
The packaged reference model is experimental and does not qualify physical accuracy.

## 8. Release acceptance

A release candidate must provide:

- `uv run pytest` result with optional backend skips explained.
- `uv run ruff check src tests ml` result.
- `uv run mypy src ml` result.
- `make -C kernel all` and deterministic kernel demo output.
- CLI help plus tiny Phase 2 and Phase 3 render/report smoke results.
- Native build or explicit native skip evidence.
- CUDA build or explicit `nvcc`/GPU unavailability evidence.
- Wheel and sdist inspection proving required package data.
- Documentation links and claim wording consistent with `docs/STATUS.md`.
