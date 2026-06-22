# Kerr Metric — Boyer-Lindquist Quick Reference

This document summarises the conventions used in `phase3_christoffel.py` and `phase3_geodesic.py`.

## Signature and Coordinates

Metric signature **(-+++)**.  
Coordinates: \(x^0 = t,\; x^1 = r,\; x^2 = \theta,\; x^3 = \varphi\).

Units: **geometric** (\(G = c = 1\)).  Mass \(M\), spin \(a = J/M\) with \(|a| \leq M\).

## Key Scalars

| Symbol | Definition |
|--------|-----------|
| \(\Sigma\) | \(r^2 + a^2 \cos^2\theta\) |
| \(\Delta\) | \(r^2 - 2Mr + a^2\) |
| \(\rho^2\) | same as \(\Sigma\) in this code |

## Horizons

- **Outer (event) horizon**: \(r_+ = M + \sqrt{M^2 - a^2}\)  
- **Inner (Cauchy) horizon**: \(r_- = M - \sqrt{M^2 - a^2}\)  
- \(\Delta = 0\) at both horizons.

## Non-zero Metric Components

In Boyer-Lindquist coordinates:

$$
g_{tt} = -\left(1 - \frac{2Mr}{\Sigma}\right), \quad
g_{t\varphi} = -\frac{2Mar\sin^2\theta}{\Sigma}
$$

$$
g_{rr} = \frac{\Sigma}{\Delta}, \quad
g_{\theta\theta} = \Sigma
$$

$$
g_{\varphi\varphi} = \left(r^2 + a^2 + \frac{2Ma^2 r \sin^2\theta}{\Sigma}\right)\sin^2\theta
$$

## Conserved Quantities (Geodesics)

For a geodesic \(x^\mu(\lambda)\) with 4-velocity \(v^\mu = dx^\mu/d\lambda\):

| Conserved quantity | Expression |
|--------------------|-----------|
| Energy \(E\) | \(-p_t = -(g_{tt} v^t + g_{t\varphi} v^\varphi)\) |
| Angular momentum \(L_z\) | \(p_\varphi = g_{\varphi t} v^t + g_{\varphi\varphi} v^\varphi\) |

## Null Condition

For null geodesics (photons):

$$g_{\mu\nu} v^\mu v^\nu = 0$$

Implemented as a periodic check/re-normalisation in `renormalize_vr_kerr`.

## Schwarzschild Limit

Setting \(a = 0\):

- \(\Sigma = r^2\),  \(\Delta = r^2 - 2Mr = r(r-2M)\)  
- \(g_{t\varphi} = 0\) (no frame dragging)  
- Recovers the standard Schwarzschild metric used in Phase 2.

## ISCO

- **Schwarzschild** (\(a=0\)): \(r_\text{ISCO} = 6M\)  
- **Prograde Kerr** (\(a=M\)): \(r_\text{ISCO} \to M\)  
- **Retrograde Kerr** (\(a=M\)): \(r_\text{ISCO} = 9M\)

## Implementation Notes

- Christoffel symbols: computed analytically from the metric components.
  See `_kerr_geometry()` in `phase3_christoffel.py` for the full table.
- Integrator: 4th-order Runge-Kutta (`phase1.rk4_step`), shared with Phase 2.
- Termination: ray captured when \(r < r_+ + \epsilon\); escaped when \(r > r_\text{escape}\).

## References

1. Bardeen, Press & Teukolsky (1972), ApJ 178, 347 — geodesic equations in Kerr
2. Misner, Thorne & Wheeler (1973), *Gravitation*, §33
3. Chandrasekhar (1983), *The Mathematical Theory of Black Holes*, Ch. 6
4. Boyer & Lindquist (1967), J. Math. Phys. 8, 265 — original BL coordinates
