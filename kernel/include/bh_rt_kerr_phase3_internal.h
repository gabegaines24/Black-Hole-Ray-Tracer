#pragma once

/* Internal helpers shared between Kerr single-ray and batch tracers.
 * Not part of the public API — do not include from bridge/ or Python. */

#include "bh_rt_rk4.h"

#define BH_RT_K3_N_STATE 8
#define BH_RT_K3_RK_WORK (5 * BH_RT_K3_N_STATE)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bh_rt_k3_userdata {
  double m;
  double a;
} bh_rt_k3_userdata;

typedef struct bh_rt_k3_geom {
  int valid;
  double s, c, s2, sc;
  double Sig, Dlt, A;
  double g_tt, g_tph, g_rr, g_thth, g_phph;
  double gup_tt, gup_tph, gup_phph, gup_rr, gup_thth;
  double dg_tt_dr, dg_tt_dth;
  double dg_tph_dr, dg_tph_dth;
  double dg_rr_dr, dg_rr_dth;
  double dg_thth_dr, dg_thth_dth;
  double dg_phph_dr, dg_phph_dth;
} bh_rt_k3_geom;

/* Fill geometry at (r, theta); returns 0 on degenerate points. */
int bh_rt_k3_geometry(double r, double th, double m, double a, bh_rt_k3_geom *g);

/* dv^mu/dlambda = -Gamma^mu_{ab} v^a v^b */
void bh_rt_k3_geodesic_acceleration(const double *y, double m, double a, double *acc);

void bh_rt_k3_deriv(double lam, const double *y, double *dy, void *userdata);

int bh_rt_k3_all_finite(const double *y);

void bh_rt_k3_renormalize_vr(double *y, double m, double a);

double bh_rt_k3_outer_horizon(double m, double a);

#ifdef __cplusplus
}
#endif
