#pragma once

/* Internal helpers shared between single-ray and batch tracers.
 * Not part of the public API — do not include from bridge/ or Python. */

#include "bh_rt_rk4.h"

#define BH_RT_P2_N_STATE 8
#define BH_RT_P2_RK_WORK (5 * BH_RT_P2_N_STATE)

#ifdef __cplusplus
extern "C" {
#endif

/* Schwarzschild metric factor f = 1 - 2M/r. */
double bh_rt_p2_schwarzschild_f(double r, double m);

/* Christoffel symbol Gamma^mu_{ab} (symmetric in a,b) for Schwarzschild. */
double bh_rt_p2_christoffel(int mu, int a, int b, double r, double th, double m);

/* Write geodesic ODE RHS into dy[8]. ud must point to bh_rt_p2_userdata. */
void bh_rt_p2_deriv(double lam, const double *y, double *dy, void *userdata);

/* Return 1 if all 8 state components are finite, else 0. */
int bh_rt_p2_all_finite(const double *y);

/* Project v^r onto the null cone; modifies y[5] in place. */
void bh_rt_p2_renormalize_vr(double *y, double m);

typedef struct bh_rt_p2_userdata {
  double m;
} bh_rt_p2_userdata;

#ifdef __cplusplus
}
#endif
