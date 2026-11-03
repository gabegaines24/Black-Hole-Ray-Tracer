# Project contract — blackhole_ray_tracer

This document defines the supported numerical and interface contract. Implementation gaps are recorded in [STATUS.md](./STATUS.md); gaps do not silently change the contract.

## Scope

The project traces null geodesics near black holes. Python is the reference implementation. Native CPU and CUDA paths are acceleration backends, not independent physics models.

Supported work is correctness-first Schwarzschild tracing, with a Python-first Kerr prototype.
Rendering and disk coloring, ML inference, and GPU execution remain bounded by `STATUS.md` evidence.

## Conventions

- Use geometric units with `G = c = 1`.
- Treat `m` as mass parameter.
- Use Schwarzschild horizon `r_h = 2m`.
- Use Boyer–Lindquist names for Kerr coordinates and extensions.
- Use radians for angular coordinates.
- Use affine parameter `lambda` for null-geodesic integration.
- Require Python `>=3.12`.
- Do not claim real-time or production-quality rendering for pure Python.

## Phase 2 state

A Phase 2 state has eight `float64` values in this order:

```text
(t, r, theta, phi, v^t, v^r, v^theta, v^phi)
```

The first four values are Schwarzschild spherical coordinates. The final four values are contravariant coordinate derivatives with respect to affine parameter.

Batch input uses row-major shape `(N, 8)`. Backend output arrays use one row per input ray. The complete FFI layout and dtypes are defined in [STATE_API.md](./STATE_API.md).

## Null constraint

Physical initial states satisfy:

```text
g_mu_nu v^mu v^nu = 0
```

Implementations may project the radial velocity back onto the null cone using the documented reference rule. Projection cadence and failure behavior must remain equivalent across backends.

## Integration contract

Each backend receives the same:

- initial state;
- mass `m`;
- affine step `dlambda`;
- integer `max_steps` budget;
- escape radius `r_escape`;
- horizon margin `r_horizon_epsilon`.

The reference integrator uses RK4 in affine parameter and periodically renormalizes the null constraint. Native backends must match reference status, step count, termination radius, minimum radius, and disk observables within documented tolerances.

## Phase 3 Kerr boundary

Kerr Phase 3 uses Boyer–Lindquist coordinates and a local zero-angular-momentum observer (ZAMO) tetrad.
The camera initializes null rays from local orthonormal directions, including frame-dragging azimuthal velocity.
Supported spin is dimensionless `a/M` in `[-1, 1]`; observers must remain outside the outer horizon and away from poles.
Phase 3 has Python validation for spin-zero parity, invariant drift, off-equatorial motion, convergence, and edge cases.
Native C currently covers Kerr metric/RHS, bounded tracing, ZAMO initialization, equatorial crossing,
and geometry-independent disk hit/Doppler observables. Color rendering, bridge, batch, and CUDA remain
separate qualified boundaries.

## Trace result boundary

`GeodesicTraceResult` stores status, step counts, termination radius and affine parameter,
minimum radius, and optional aligned coordinate and velocity samples. It does not store
the full terminal eight-vector or a dedicated disk-crossing event.

T02 kept this scalar result boundary. T05 uses aligned `(theta, r, v^t, v^phi)`
samples to derive first-crossing observables in the renderer. Batch backends expose
the same observables (`disk_r_cross`, `disk_vt`, and `disk_vph`) directly.

## Termination statuses

Status codes are frozen by [STATE_API.md](./STATE_API.md):

| Code | Name | Meaning |
|---:|---|---|
| 0 | `captured` | Radius crossed the capture threshold. |
| 1 | `escaped` | Radius exceeded escape threshold. |
| 2 | `max_steps` | Step budget ended without another terminal condition. |
| 3 | `numerical_error` | State became non-finite or invalid. |

For a single current state, required precedence is:

1. numerical error;
2. capture;
3. escape;
4. maximum steps.

A backend must report the first terminal condition observed at a step boundary. It must not replace an earlier status with a later status during result finalization.

## Disk observables

Disk detection records the first strict sign-changing crossing of the equatorial plane `theta = pi/2` before terminal integration. Crossing values use the same linear interpolation for every backend:

- crossing radius;
- `v^t` at crossing;
- `v^phi` at crossing.

No crossing returns `disk_hit = 0` and NaN observables. Capture before crossing remains a capture without a disk hit.

Disk color and Doppler calculations are rendering observables. They must not alter geodesic integration.

## Backend boundaries

| Backend | Contract role | Current qualification |
|---|---|---|
| Python Phase 1 | Educational equatorial reference | Tested baseline |
| Python Phase 2 | Schwarzschild 3D reference | Tested baseline; disk semantics tested |
| Python Phase 3 | Kerr Boyer–Lindquist prototype | Validated Python reference; disk color and rendering remain Python-only |
| C kernel | Native numerical backend | Canonical Phase 2 scalar/batch implementation; Kerr metric/RHS/camera/single-ray/crossing/disk-policy parity tested |
| PyBind11 bridge | Optional CPU acceleration boundary | Installed locally; marshalling and render parity tested |
| CUDA batch | Optional accelerator | Experimental; unavailable locally without `nvcc` |
| ML runtime gate | Explicit experimental surrogate | Schema/fallback tested; no quality guarantee |

## Compatibility rules

- Keep public state order and status meanings stable.
- Keep Python reference behavior inspectable.
- Do not add a second FFI shape for the same tracer.
- Do not add Kerr disk color/rendering, batch, bridge, or CUDA paths before separate parity gates pass.
- Mark unavailable native, CUDA, and ML evidence as unavailable or experimental.
