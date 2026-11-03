# Implementation plan index

This directory records execution structure. Evidence belongs in [STATUS.md](../STATUS.md).
Normative behavior belongs in [PROJECT_CONTRACT.md](../PROJECT_CONTRACT.md) and [SPEC.md](SPEC.md).
Array details belong in [STATE_API.md](../STATE_API.md). Accepted choices belong in [DECISIONS/](../DECISIONS/).
The detailed dependency plan is [`tasks/plan.md`](../../tasks/plan.md).

## Authority order

1. `docs/PROJECT_CONTRACT.md` — numerical, state, status, and backend contract.
2. `docs/STATE_API.md` — C/Python/CUDA array layout and FFI dtypes.
3. Accepted decision records — choices not implied by equations or interfaces.
4. `docs/IMPLEMENTATION/SPEC.md` — observable implementation behavior.
5. `docs/FINAL_ARCHITECTURE.md` — stable module boundaries.
6. `docs/STATUS.md` — tested, implemented, experimental, unavailable, and planned claims.
7. `tasks/plan.md` — dependency order, acceptance criteria, and verification.
8. README and comments — usage guidance only.

## Ticket order

| Ticket | Scope | Depends on |
|---|---|---|
| [T00](T00-contract.md) | Contract, baseline, status | — |
| [T01](T01-build-ci.md) | Build, packaging, CI | T00 |
| [T02](T02-python-contract.md) | Canonical Python types and validation | T00 |
| [T03](T03-canonical-c.md) | Canonical Schwarzschild C implementation | T01, T02 |
| [T04](T04-native-bridge.md) | Safe native bridge and parity | T03 |
| [T05](T05-disk-observables.md) | Disk observables and render parity | T02, T03, T04 |
| [T06](T06-kerr-prototype.md) | Kerr camera and numerical validation | T02, T05 |
| [T07](T07-ml-surrogate.md) | Experimental ML contract | T02, T06 |
| [T08](T08-cuda-preview.md) | CUDA, preview, and benchmark qualification | T04, T05 |
| [T09](T09-docs-release.md) | Documentation and release evidence | T01, T05, T06, T07, T08 |
| [T10](T10-kerr-native.md) | Incremental Kerr metric/RHS C probe | T06, T09 |
| [T12](T12-kerr-tracer.md) | Bounded Kerr CPU single-ray tracer | T10 |
| [T14](T14-kerr-camera.md) | Kerr native ZAMO camera initializer | T12 |
| [T16](T16-kerr-crossing.md) | Kerr native equatorial crossing observables | T14 |
| [T18](T18-kerr-disk-policy.md) | Kerr native disk hit and Doppler policy | T16 |

## Checkpoints

- **A — contract and toolchain:** install, package, lint, type check, and baseline tests.
- **B — trusted Schwarzschild path:** Python/C/native parity with one canonical C implementation.
- **C — observables and Kerr:** deterministic disk tests and validated Python Kerr prototype.
- **D — optional backends:** ML/CUDA verified or explicitly demoted.
- **E — Kerr native foundation:** C BL metric/RHS probe parity before any native tracer.
- **F — Kerr native termination:** single-ray C RK4 and status parity before camera or disk port.
- **G — Kerr native camera:** C ZAMO seed parity before disk or render port.
- **H — Kerr native crossing:** first strict equatorial crossing parity before disk policy or render port.
- **I — Kerr native disk policy:** radius and Doppler parity before color or render port.
- **Final — release evidence:** command matrix passes and documentation matches evidence.

Do not begin a dependent ticket while its predecessor lacks acceptance evidence. Preserve unrelated or pre-existing worktree edits.
