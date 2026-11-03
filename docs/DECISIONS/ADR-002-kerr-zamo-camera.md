# ADR-002: Kerr ZAMO camera

## Context

A static Schwarzschild observer is not a valid general Kerr observer frame.
Initial null directions must be orthonormal under the Kerr metric.

## Decision

Build Phase 3 camera directions in a locally non-rotating ZAMO frame.
Use Boyer–Lindquist coordinates and a local radial, polar, and azimuthal tetrad.
Reject observer poles and positions at or inside the outer horizon.
Keep implementation Python-only until native parity is specified.

## Consequences

- Spin-zero camera output can be compared with the Schwarzschild static camera.
- Near-extreme spin remains testable without a false static-frame assumption.
- A future native port must preserve tetrad normalization and coordinate ordering.

## Evidence

- `src/blackhole_ray_tracer/phase3_camera.py`
- `tests/test_phase3.py`
- `docs/KERR_BOYER_LINDQUIST.md`
