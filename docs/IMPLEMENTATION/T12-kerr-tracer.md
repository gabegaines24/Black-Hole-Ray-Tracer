# T12 — Kerr native single-ray tracer

**Status:** complete; CPU single-ray trace only.

**Dependencies:** T10.

**Scope:** Extend the canonical Kerr C module with one bounded RK4 tracer and explicit termination results.
This slice does not add a camera binding, trajectory sample buffers, disk observables, batch API,
PyBind11 bridge, CUDA kernel, or renderer.

**Contract:**

- Header: `kernel/include/bh_rt_kerr_phase3.h`.
- State order remains `(t, r, theta, phi, vt, vr, vtheta, vphi)`.
- `spin` remains dimensionless `a/M`.
- Inputs require finite `m`, `spin`, `dlambda`, `r_escape`, and `r_horizon_epsilon`; `m > 0`,
  `abs(spin) <= 1`, `dlambda > 0`, `r_horizon_epsilon >= 0`, and `r_escape > r_plus + epsilon`.
- `max_steps >= 0`; zero means no integration steps and returns `MAX_STEPS`.
- Valid control inputs return `BH_RT_KERR_OK`; invalid controls return an explicit Kerr error code.
- Non-finite initial state returns `BH_RT_KERR_OK` with `BH_RT_STATUS_NUMERICAL_ERROR`.
- The result uses shared `BH_RT_STATUS_*` values and mirrors Python termination fields:
  `status`, `steps_taken`, `max_steps`, `termination_r`, `termination_lambda`, and `r_min`.
- Initial and every fourth completed RK4 step project `v^r` back onto the null constraint.
- Capture and escape checks happen before each integration step.
- No samples are retained. Python remains reference for camera, disk, and rendering behavior.

**Acceptance:** C compiles with C99 warnings-as-errors; terminal status, step count, termination values,
minimum radius, invalid controls, and near-horizon capture match the Python tracer through ctypes.

**Evidence:** `tests/test_kernel_phase3_algebra.py`, `tests/test_kernel_phase3_trace.py`, ADR-007,
and final `docs/STATUS.md` T12 evidence.
