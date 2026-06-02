# Kerr metric in Boyer–Lindquist coordinates

This document is the design reference for the future `phase3_*` (Kerr) integrator.
**Do not port to C until the Python prototype passes parity tests with known analytic results.**

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

The null condition g_μν v^μ v^ν = 0 must be enforced at the start and periodically
via `v^r` renormalization (same strategy as Phase 2, but with the Kerr metric).

---

## 3. Conserved quantities (useful for validation)

For a null geodesic, two conserved Killing quantities exist:

- **Energy** E = -p_t = -g_tt v^t - g_tφ v^φ (time translational symmetry)
- **Angular momentum** L = p_φ = g_φφ v^φ + g_tφ v^t (axial symmetry)
- **Carter constant** Q (second-order conserved quantity from the Killing tensor):
  ```
  Q = ( v^θ Σ )² + cos²θ [ a² (-v^t² + ...) + (L/sin θ)² ]
  ```
  Q is zero for equatorial geodesics.

These constants can be monitored to validate numerical drift.

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

Same as Phase 2, with updated radii:

| Condition | Criterion |
|-----------|-----------|
| Captured | r < r_+ + ε, where r_+ = M + √(M² − a²) (outer horizon) |
| Escaped | r > r_escape |
| Numerical error | Any NaN / non-finite state component |
| Step budget | steps_taken ≥ max_steps |

The inner horizon r_- = M - √(M² − a²) is not a termination boundary for external observers.

---

## 6. Prototype plan (Python first)

1. Implement `kerr_christoffel(mu, a_idx, b_idx, r, th, m, spin)` in a new
   `phase3_christoffel.py`, mirroring the structure of `phase2_christoffel.py`.
2. Implement `trace_kerr_null_geodesic(x0, v0, m, a, ...)` in `phase3_geodesic.py`,
   reusing `rk4_step` from `phase1.py`.
3. Validation tests:
   - `a=0` traces must match Phase 2 Schwarzschild to within numerical tolerance.
   - Equatorial (θ=π/2, v^θ=0) traces must conserve E and L to 1 part in 10⁴.
   - For known circular orbits, verify trajectory stays on the orbit for N laps.
4. Only after these pass: port the Christoffel function to C, reusing the
   `bh_rt_p2_userdata` / `bh_rt_p2_deriv` pattern from `bh_rt_schwarzschild_phase2.c`.

---

## 7. File map (planned)

```
src/blackhole_ray_tracer/
  phase3_christoffel.py     — Kerr Christoffel symbols
  phase3_geodesic.py        — Kerr null geodesic integrator
  phase3_types.py           — KerrRenderConfig, spin parameter a
  phase3_render.py          — Kerr pinhole renderer (reuses camera layer)

kernel/
  src/bh_rt_kerr_phase3.c           — C Kerr Christoffel + single-ray trace
  src/bh_rt_kerr_phase3_batch.c     — C batch version
  include/bh_rt_kerr_phase3.h
  include/bh_rt_kerr_phase3_batch.h

bridge/
  module_phase3.cpp                 — PyBind11 bindings
```

---

## 8. References

- Bardeen, J. M., Press, W. H., Teukolsky, S. A. (1972). *ApJ* **178**, 347.
- MTW — Misner, Thorne, Wheeler, *Gravitation*, §33.
- Levin, J. & Perez-Giz, G. (2008). *Phys. Rev. D* **77**, 103005.
  (Taxonomy of geodesics in Kerr spacetime.)
