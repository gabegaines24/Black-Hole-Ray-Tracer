# T16 — Kerr native equatorial crossing observables

**Status:** complete; first-crossing observables only.

**Dependencies:** T14.

**Scope:** Add a separate C trace entrypoint that records the first strict crossing of
`theta = pi/2` while preserving the T12 scalar trace result. No disk radius policy,
Doppler intensity, color, renderer, batch API, bridge, or CUDA behavior is added.

**Contract:**

- Header: `kernel/include/bh_rt_kerr_phase3.h`.
- Function: `bh_rt_kerr_phase3_trace_crossing`.
- `bh_rt_kerr_trace_result` remains unchanged and retains T12 ABI layout.
- `bh_rt_kerr_equatorial_crossing` returns `hit`, `r_cross`, `vt_cross`, and `vphi_cross`.
- `hit = 0` returns NaN crossing fields.
- A crossing is the first strict sign change of `theta - pi/2` between consecutive completed RK4 states.
- Crossing values use Python's linear interpolation alpha:
  `abs(th0) / (abs(th0) + abs(th1) + 1e-30)`.
- Interpolate `r`, `v^t`, and `v^phi` from the aligned pre-step and post-step states.
- Capture and escape status semantics remain unchanged.

**Acceptance:** C compiles with C99 warnings-as-errors; signed moderate and near-extreme spin
crossings match Python interpolation; no-step traces return empty NaN observables; missing output
pointers return explicit errors.

**Evidence:** `tests/test_kernel_phase3_crossing.py`, ADR-009, and final `docs/STATUS.md` T16 evidence.
