# Kerr metric in Boyer–Lindquist coordinates

This document records the validated Python Phase 3 boundary and future port requirements.
The current implementation uses Boyer–Lindquist coordinates, a ZAMO camera, and Python RK4.
T10/T12/T14/T16/T18 C code ports metric/RHS algebra, bounded single-ray termination, ZAMO seed,
equatorial crossing, and geometry-independent disk hit/Doppler observables.
**Do not port Kerr disk color, renderer, bridge, batch, or CUDA paths until separate parity tickets define their contracts.**

---

## 1. The Boyer–Lindquist line element

The Kerr metric for a black hole with mass M and spin parameter a (|a| ≤ M) in
Boyer–Lindquist (BL) coordinates (t, r, θ, φ):

```
ds² = -( 1 - 2Mr/Σ ) dt²
      - (4Mar sin²θ/Σ) dt dφ
      + Σ/Δ dr²
      + Σ dθ²
      + ( r² + a² + 2Ma²r sin²θ/Σ ) sin²θ dφ²
```

where:

```
Σ = r² + a² cos²θ
Δ = r² - 2Mr + a²
```

Setting `a = 0` reduces everything to Schwarzschild.

---

## 2. State vector

Identical to Phase 2 (8-component, affine parameter λ):

```
y = ( t, r, θ, φ, v^t, v^r, v^θ, v^φ )
```

The null condition \(g_{\mu\nu}v^\mu v^\nu = 0\) is enforced by the Python tracer at initialization
and every fourth RK4 step by adjusting \(v^r\) while preserving its sign.

---

## 3. Conserved quantities (useful for validation)

For a null geodesic, two conserved Killing quantities exist:

- **Energy** E = -p_t = -g_tt v^t - g_tφ v^φ (time translational symmetry)
- **Angular momentum** L = p_φ = g_φφ v^φ + g_tφ v^t (axial symmetry)
- **Carter constant** Q (null form):
  ```
  p_theta = Σ v^theta
  Q = p_theta² + cos²(theta) [ L² / sin²(theta) - a² E² ]
  ```
  with `a = m * spin`. `Q` is zero for equatorial geodesics.

These constants are monitored in tests to validate numerical drift.

---

## 4. Non-zero Christoffel symbols (BL, Kerr)

Full list in terms of Σ, Δ, and the metric components; many are shared with Schwarzschild.
The key new ones introduced by `a ≠ 0`:

| Symbol | Arises from |
|--------|-------------|
| Γ^t_{tr}, Γ^t_{tθ} | metric coupling to spin |
| Γ^t_{rφ}, Γ^t_{θφ} | frame-dragging cross-terms |
| Γ^φ_{tr}, Γ^φ_{tθ} | dragged azimuthal motion |
| Γ^φ_{rφ}, Γ^φ_{θφ} | radial/polar motion in φ |
| Γ^r_{tt}, Γ^r_{tφ}, Γ^r_{rr}, Γ^r_{θθ}, Γ^r_{φφ} | radial acceleration |
| Γ^θ_{tt}, Γ^θ_{tφ}, Γ^θ_{rθ}, Γ^θ_{φφ} | polar acceleration |

Reference: Bardeen, Press & Teukolsky (1972); MTW §33.5.

---

## 5. Termination criteria

The Python tracer terminates against the outer horizon margin:

| Condition | Criterion |
|-----------|-----------|
| Captured | `r < r_+ + r_horizon_epsilon`, where `r_+ = M + sqrt(M² − a²)` |
| Escaped | `r > r_escape` |
| Numerical error | Any non-finite state component |
| Step budget | `steps_taken >= max_steps` |

The inner horizon `r_-` is not an external-observer termination boundary.

---

## 6. Validated Python prototype and future port

1. Validate BL metric, inverse, derivatives, and Christoffel symbols in `phase3_christoffel.py`.
2. Build null initial directions with `phase3_camera.py` using a local ZAMO orthonormal frame.
3. Trace `phase3_geodesic.py` with shared `phase1.rk4_step`.
4. Render with `phase3_render.py`; apply disk crossing only to stored trajectory samples.
5. Validate spin-zero camera parity, null initialization, Killing quantities, Carter drift,
   off-equatorial motion, horizon termination, pole rejection, near-extreme spin, and convergence.
6. Port only after a new ticket defines C/CUDA state ABI, scalar/batch parity, and error semantics.

---

## 7. Current and future file map

Current Python implementation:

```
src/blackhole_ray_tracer/
  phase3_christoffel.py — Kerr BL metric, inverse, derivatives, Christoffel symbols
  phase3_camera.py      — ZAMO/local orthonormal camera
  phase3_geodesic.py    — Kerr null RK4 trace and conserved quantities
  phase3_types.py       — KerrRenderConfig and spin validation
  phase3_render.py      — Kerr image loop and disk crossing
  phase3_report.py      — Kerr presets and benchmark
  phase3_driver.py      — render/report CLI
```

Native incremental boundary:

```
kernel/
  bh_rt_kerr_phase3.h/.c — Kerr BL metric, RHS, ZAMO seed, single-ray termination,
  equatorial crossing, and geometry-independent disk hit/Doppler policy
  No disk color renderer, bridge, batch, or CUDA behavior
```

Future port scope:

```
bridge/ and kernel/cuda/
  Kerr color/render bindings, batch, bridge, and CUDA paths require separate ABI/parity tickets.
```

---

## 8. References

- Bardeen, J. M., Press, W. H., Teukolsky, S. A. (1972). *ApJ* **178**, 347.
- MTW — Misner, Thorne, Wheeler, *Gravitation*, §33.
- Levin, J. & Perez-Giz, G. (2008). *Phys. Rev. D* **77**, 103005.
  (Taxonomy of geodesics in Kerr spacetime.)
