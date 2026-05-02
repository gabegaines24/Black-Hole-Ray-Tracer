# C kernel (`kernel/`)

Pure C primitives for eventual high-throughput tracing. **No Python headers** here.

Current contents:

| Path | Purpose |
|------|---------|
| `include/bh_rt_rk4.h` | Explicit **RK4** step API (function-pointer RHS) |
| `src/bh_rt_rk4.c` | Implementation (`workspace` = `5 × n` doubles) |
| `include/bh_rt_schwarzschild_u.h` | Equatorial Schwarzschild null ray in \(u(\phi)=1/r\) (`bh_rt_schwarzschild_u_trace`) |
| `src/bh_rt_schwarzschild_u.c` | Matches `phase1.trace_single_schwarzschild_ray` (RK4 order, ICS, thresholds) |
| `src/demo_harmonic.c` | Phase A harmonic oscillator parity demo (CLI: `dt` `total_time` `omega`) |
| `src/demo_schwarzschild_u.c` | Schwarzschild 2D one-line-result demo |

## Build (`make -C kernel`)

From repository root:

```bash
make -C kernel
```

Builds **`harmonic_demo`** and **`schwarzschild_demo`** (when `make` resolves to the default target `all`).

```bash
./kernel/harmonic_demo 0.02 8 1
./kernel/schwarzschild_demo 1 6 0.2 8 0.002 - 80
```

For `schwarzschild_demo`, the 6th argument is optional `r_capture`; use `-` for Phase‑1 default `2·m + 1e-3`.

Or compile manually:

```bash
cc -std=c99 -Wall -Wextra -O2 -I kernel/include \
  kernel/src/bh_rt_rk4.c kernel/src/demo_harmonic.c -o kernel/harmonic_demo -lm

cc -std=c99 -Wall -Wextra -O2 -I kernel/include \
  kernel/src/bh_rt_rk4.c kernel/src/bh_rt_schwarzschild_u.c kernel/src/demo_schwarzschild_u.c \
  -o kernel/schwarzschild_demo -lm
```

On success, `demo_harmonic.c` prints a line prefixed with `BH_RK4_RESULT`; `demo_schwarzschild_u.c` prints `BH_SCHW_U_RESULT …`.

Pytest parity (optional compile): `tests/test_kernel_harmonic_parity.py`, `tests/test_kernel_schwarzschild_u_parity.py` (skipped if no C toolchain or `SKIP_KERNEL_TESTS=1`).

## Next

- 3D Christoffel RHS in kernel (matching `phase2_geodesic.py`).
- SoA batched integration + deterministic termination flags (see [`docs/STATE_API.md`](../docs/STATE_API.md)).
- `bridge/` pybind11 module calling the batch API.
