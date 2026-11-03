# T18 — Kerr native disk policy

**Status:** complete; geometry-independent disk observable policy only.

**Dependencies:** T16.

**Scope:** Port Python disk radius gating and Doppler intensity calculation into one C function.
The function consumes an equatorial crossing and does not trace rays or render pixels.

**Contract:**

- Header: `kernel/include/bh_rt_kerr_phase3.h`.
- Function: `bh_rt_kerr_disk_observable`.
- Inputs include `r_cross`, disk `[r_inner, r_outer]`, mass, crossing `v^t`/`v^phi`, and `redshift_scale`.
- NaN `r_cross` means no crossing and returns `hit = 0`, `z_factor = 1`.
- Radius bounds are inclusive, matching Python.
- A crossing inside bounds returns `hit = 1` and the Python Doppler intensity factor.
- The function does not calculate ISCO, color, trajectory, batch ownership, bridge, CUDA, or rendering.

**Acceptance:** C compiles with C99 warnings-as-errors; NaN, outside-range, boundary, finite-velocity,
non-finite-velocity, zero-scale, and invalid-control cases match Python policy.

**Evidence:** `tests/test_kernel_phase3_disk.py`, ADR-010, and final `docs/STATUS.md` T18 evidence.
