# T10 — Kerr native-port foundation

**Status:** complete; probe slice only.

**Dependencies:** T06, T09.

**Scope:** Establish a small C ABI for Kerr Boyer–Lindquist metric evaluation and first-order geodesic RHS evaluation.
This slice does not add a renderer, batch tracer, PyBind11 module, CUDA kernel, or production claim.

**Contract:**

- Coordinate order remains `(t, r, theta, phi)`.
- State order remains `(t, r, theta, phi, vt, vr, vtheta, vphi)`.
- `m > 0`, `abs(spin) <= 1`, `r > r_plus`, `0 < theta < pi`, and finite inputs are required.
- Metric output is row-major `double[16]`.
- RHS output is `double[8]`.
- Return `0` on success; nonzero `BH_RT_KERR_*` code means output is invalid.
- Probe parity compares metric and RHS values with the Python Phase 3 reference.

**Acceptance:** C compiles with the project C toolchain; Python ctypes tests compare spin zero,
moderate positive/negative spin, and near-extreme spin at exterior off-pole states.
Invalid state and horizon inputs return explicit errors.

**Not in scope:** Kerr trajectory termination, disk observables, camera bindings, native rendering,
CUDA, or replacing Python as the Kerr reference.

**Evidence target:** `tests/test_kernel_phase3_algebra.py`, ADR-006, and final `docs/STATUS.md` T10 evidence.
