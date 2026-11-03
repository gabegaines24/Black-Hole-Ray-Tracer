# ADR-010: Kerr native disk observable policy

## Context

T16 exposes geometric equatorial crossings, but Python rendering still applies disk radius bounds
and Doppler intensity policy. Porting those calculations separately avoids coupling them to ray
storage, image color, or backend ownership.

## Decision

Add `bh_rt_kerr_disk_observable` with caller-supplied disk bounds and redshift scale.
Match Python's inclusive radius checks, NaN no-crossing sentinel, non-finite velocity fallback,
Keplerian angular velocity, ratio sign handling, and `[0, 10]` intensity clamp.

Do not calculate ISCO, RGB color, trajectory state, batch arrays, bridge ownership, CUDA behavior,
or rendering in this slice.

## Consequences

- C can qualify disk hit and Doppler policy independently from crossing integration.
- The function is not a complete disk renderer and does not claim physical radiative transfer.
- Future render work must define color, sample ownership, and backend parity separately.

## Evidence

- `kernel/include/bh_rt_kerr_phase3.h`
- `kernel/src/bh_rt_kerr_phase3.c`
- `tests/test_kernel_phase3_disk.py`
