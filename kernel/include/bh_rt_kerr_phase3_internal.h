#pragma once

/* Scalar helpers shared by Kerr single-ray and SoA batch tracers.
 * Not part of public API — do not include from bridge/ or Python. */

#define BH_RT_K3_N_STATE 8

#ifdef __cplusplus
extern "C" {
#endif

int bh_rt_k3_rhs_components(
    double r, double theta, double vt, double vr, double vtheta, double vphi,
    double m, double spin, int require_exterior, double *dy_out);

double bh_rt_k3_renormalized_vr(
    double r, double theta, double vt, double vr, double vtheta, double vphi,
    double m, double spin);

double bh_rt_k3_outer_horizon(double m, double spin);

#ifdef __cplusplus
}
#endif
