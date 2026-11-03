# ADR-009: Kerr native equatorial crossing observables

## Context

T12 qualified Kerr C termination and T14 qualified C ZAMO seeds.
Python disk rendering still depends on the first equatorial crossing and interpolated photon
velocities, but porting disk color and rendering would add unrelated policy and ownership concerns.

## Decision

Add a separate C entrypoint, `bh_rt_kerr_phase3_trace_crossing`, that preserves the T12 trace
result layout and writes a separate `bh_rt_kerr_equatorial_crossing` result.
Detect only the first strict sign change of `theta - pi/2` between completed RK4 states.
Interpolate radius, `v^t`, and `v^phi` with the existing Python alpha formula.

Do not add disk inner/outer radius policy, Doppler intensity, color, trajectory buffers, batch,
bridge, CUDA, or rendering behavior in this slice.

## Consequences

- Crossing parity becomes testable without breaking T12 trace-result ABI.
- Future disk code can consume identical crossing observables.
- The C result does not claim a disk hit; it only reports geometric plane crossing.
- A future batch API must define crossing output ownership separately.

## Evidence

- `kernel/include/bh_rt_kerr_phase3.h`
- `kernel/src/bh_rt_kerr_phase3.c`
- `tests/test_kernel_phase3_crossing.py`
