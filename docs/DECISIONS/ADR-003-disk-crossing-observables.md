# ADR-003: Disk crossing observables

## Context

Disk color depends on velocity at the equatorial crossing, not velocity at an arbitrary sampled state.
Multiple crossings and capture-before-crossing need deterministic precedence.

## Decision

Record only the first strict sign-changing `theta = pi/2` crossing.
Linearly interpolate `r`, `vt`, and `vphi` between pre-step and post-step states.
Do not record a hit after capture or for tangent/equality-only contacts.
Use interpolated velocities for Doppler disk color.

## Consequences

- Python, C batch, and CUDA output identical observable fields under matching steps.
- Sampling remains aligned across `r`, `theta`, `vt`, and `vphi`.
- A future higher-order crossing solver would require a new decision and parity tests.

## Evidence

- `src/blackhole_ray_tracer/phase2_disk.py`
- `kernel/src/bh_rt_schwarzschild_phase2_batch.c`
- `kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu`
- `tests/test_phase2.py`
- `tests/test_kernel_phase2_parity.py`
