# T14 — Kerr native ZAMO camera initializer

**Status:** complete; camera state initializer only.

**Dependencies:** T12.

**Scope:** Port Python ZAMO/local-orthonormal camera initialization into the canonical Kerr C module.
The function returns one complete eight-state ray seed for an explicit screen coordinate.
It does not add rendering, disk observables, trajectory samples, batch ownership, bridge, or CUDA.

**Contract:**

- Header: `kernel/include/bh_rt_kerr_phase3.h`.
- Function: `bh_rt_kerr_zamo_initial_state`.
- Output order is `(t, r, theta, phi, vt, vr, vtheta, vphi)` in caller-owned `double[8]`.
- Screen inputs `sx` and `sy` use normalized coordinates before aspect and FOV scaling.
- Camera validation matches Python: finite inputs, `m > 0`, `abs(spin) <= 1`, exterior radius,
  `0 < theta < pi`, pole exclusion, `0 < fov_deg < 180`, and positive integer dimensions.
- Return `0` on success; invalid pointers, controls, frame domains, and numerical output use explicit
  `BH_RT_KERR_*` errors.
- The generated state is directly consumable by `bh_rt_kerr_phase3_trace`.

**Acceptance:** C compiles with C99 warnings-as-errors; generated states match Python ZAMO seeds for
zero, signed moderate, and near-extreme spin; generated states remain finite and null; invalid camera
controls return explicit errors.

**Evidence:** `tests/test_kernel_phase3_camera.py`, ADR-008, and final `docs/STATUS.md` T14 evidence.
