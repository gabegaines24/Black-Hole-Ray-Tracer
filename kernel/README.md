# C kernel (`kernel/`)

Pure C primitives for eventual high-throughput tracing. **No Python headers** here.

Current contents:

| Path | Purpose |
|------|---------|
| `include/bh_rt_rk4.h` | Explicit **RK4** step API (function-pointer RHS) |
| `src/bh_rt_rk4.c` | Implementation (`workspace` = `5 × n` doubles) |
| `src/demo_harmonic.c` | Phase A harmonic oscillator parity demo (CLI args: `dt` `total_time` `omega`) |

## Build harmonic demo (`harmonic_demo`)

From repository root:

```bash
make -C kernel
./kernel/harmonic_demo 0.02 8 1
```

Or manually:

```bash
cc -std=c99 -Wall -Wextra -O2 -I kernel/include \\
  kernel/src/bh_rt_rk4.c kernel/src/demo_harmonic.c -o kernel/harmonic_demo -lm
```

On success prints a single line prefixed with `BH_RK4_RESULT` (see `demo_harmonic.c`).

## Next

- Schwarzschild 2D RHS in C (matching `phase1.py` \(u,\phi\) system).
- SoA batched integration + deterministic termination flags (see [`docs/STATE_API.md`](../docs/STATE_API.md)).
- `bridge/` pybind11 module calling the batch API.
