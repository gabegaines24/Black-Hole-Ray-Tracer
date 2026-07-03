#include "bh_rt_kerr_phase3_internal.h"
#include "bh_rt_kerr_phase3.h"
#include "bh_rt_status.h"

#include <math.h>
#include <stddef.h>

#ifndef INFINITY
#define INFINITY (1.0 / 0.0)
#endif

#define I_T 0
#define I_R 1
#define I_TH 2
#define I_PH 3

int bh_rt_k3_geometry(double r, double th, double m, double a, bh_rt_k3_geom *g) {
  if (g == NULL || !(isfinite(r) && isfinite(th) && isfinite(m) && isfinite(a)) || r <= 0.0) {
    if (g)
      g->valid = 0;
    return 0;
  }

  double s = sin(th);
  double c = cos(th);
  double s2 = s * s;
  double sc = s * c;
  double a2 = a * a;

  double Sig = r * r + a2 * c * c;
  double Dlt = r * r - 2.0 * m * r + a2;
  double A = r * r + a2;
  double Sig2 = Sig * Sig;
  double Dlt2 = Dlt * Dlt;

  double Xi = A * Sig + 2.0 * m * r * a2 * s2;

  double det_tph = -Dlt * s2;
  if (fabs(det_tph) < 1e-40) {
    g->valid = 0;
    return 0;
  }

  g->s = s;
  g->c = c;
  g->s2 = s2;
  g->sc = sc;
  g->Sig = Sig;
  g->Dlt = Dlt;
  g->A = A;

  g->g_tt = -(Sig - 2.0 * m * r) / Sig;
  g->g_tph = -2.0 * m * r * a * s2 / Sig;
  g->g_rr = Sig / Dlt;
  g->g_thth = Sig;
  g->g_phph = Xi * s2 / Sig;

  g->gup_tt = g->g_phph / det_tph;
  g->gup_tph = -g->g_tph / det_tph;
  g->gup_phph = g->g_tt / det_tph;
  g->gup_rr = Dlt / Sig;
  g->gup_thth = 1.0 / Sig;

  double dSig_dr = 2.0 * r;
  double dSig_dth = -2.0 * a2 * sc;

  g->dg_tt_dr = 2.0 * m * (a2 * c * c - r * r) / Sig2;
  g->dg_tt_dth = 4.0 * m * r * a2 * sc / Sig2;

  g->dg_tph_dr = 2.0 * m * a * s2 * (r * r - a2 * c * c) / Sig2;
  g->dg_tph_dth = -4.0 * m * r * a * A * sc / Sig2;

  g->dg_rr_dr = (2.0 * r * Dlt - Sig * 2.0 * (r - m)) / Dlt2;
  g->dg_rr_dth = -2.0 * a2 * sc / Dlt;

  g->dg_thth_dr = 2.0 * r;
  g->dg_thth_dth = -2.0 * a2 * sc;

  double dXi_dr = 2.0 * r * (Sig + A) + 2.0 * m * a2 * s2;
  double dXi_dth = -2.0 * a2 * sc * Dlt;

  g->dg_phph_dr = s2 * (Sig * dXi_dr - 2.0 * r * Xi) / Sig2;
  g->dg_phph_dth = 2.0 * sc / Sig2 * (Xi * A - Dlt * Sig * a2 * s2);

  (void)dSig_dr;
  (void)dSig_dth;
  (void)dXi_dth;

  g->valid = 1;
  return 1;
}

void bh_rt_k3_geodesic_acceleration(const double *y, double m, double a, double *acc) {
  double r = y[1], th = y[2];
  double vt = y[4], vr = y[5], vth = y[6], vph = y[7];

  bh_rt_k3_geom geo;
  if (!bh_rt_k3_geometry(r, th, m, a, &geo)) {
    for (int i = 0; i < 4; ++i)
      acc[i] = 0.0;
    return;
  }

  double gr = geo.gup_rr * 0.5;
  double G_r_rr = gr * geo.dg_rr_dr;
  double G_r_rth = gr * geo.dg_rr_dth;
  double G_r_tt = -gr * geo.dg_tt_dr;
  double G_r_tph = -gr * geo.dg_tph_dr;
  double G_r_thth = -gr * geo.dg_thth_dr;
  double G_r_phph = -gr * geo.dg_phph_dr;

  double gth = geo.gup_thth * 0.5;
  double G_th_thth = gth * geo.dg_thth_dth;
  double G_th_thr = gth * geo.dg_thth_dr;
  double G_th_tt = -gth * geo.dg_tt_dth;
  double G_th_tph = -gth * geo.dg_tph_dth;
  double G_th_rr = -gth * geo.dg_rr_dth;
  double G_th_phph = -gth * geo.dg_phph_dth;

  double G_t_tr = 0.5 * (geo.gup_tt * geo.dg_tt_dr + geo.gup_tph * geo.dg_tph_dr);
  double G_t_tth = 0.5 * (geo.gup_tt * geo.dg_tt_dth + geo.gup_tph * geo.dg_tph_dth);
  double G_t_rph = 0.5 * (geo.gup_tt * geo.dg_tph_dr + geo.gup_tph * geo.dg_phph_dr);
  double G_t_thph = 0.5 * (geo.gup_tt * geo.dg_tph_dth + geo.gup_tph * geo.dg_phph_dth);

  double G_ph_tr = 0.5 * (geo.gup_tph * geo.dg_tt_dr + geo.gup_phph * geo.dg_tph_dr);
  double G_ph_tth = 0.5 * (geo.gup_tph * geo.dg_tt_dth + geo.gup_phph * geo.dg_tph_dth);
  double G_ph_rph = 0.5 * (geo.gup_tph * geo.dg_tph_dr + geo.gup_phph * geo.dg_phph_dr);
  double G_ph_thph = 0.5 * (geo.gup_tph * geo.dg_tph_dth + geo.gup_phph * geo.dg_phph_dth);

  acc[I_T] = -(2.0 * G_t_tr * vt * vr + 2.0 * G_t_tth * vt * vth + 2.0 * G_t_rph * vr * vph +
               2.0 * G_t_thph * vth * vph);
  acc[I_R] = -(G_r_tt * vt * vt + 2.0 * G_r_tph * vt * vph + G_r_rr * vr * vr +
               2.0 * G_r_rth * vr * vth + G_r_thth * vth * vth + G_r_phph * vph * vph);
  acc[I_TH] = -(G_th_tt * vt * vt + 2.0 * G_th_tph * vt * vph + G_th_rr * vr * vr +
                2.0 * G_th_thr * vr * vth + G_th_thth * vth * vth + G_th_phph * vph * vph);
  acc[I_PH] = -(2.0 * G_ph_tr * vt * vr + 2.0 * G_ph_tth * vt * vth + 2.0 * G_ph_rph * vr * vph +
                2.0 * G_ph_thph * vth * vph);
}

void bh_rt_k3_deriv(double lam, const double *y, double *dy, void *userdata) {
  (void)lam;
  bh_rt_k3_userdata *ud = (bh_rt_k3_userdata *)userdata;
  dy[0] = y[4];
  dy[1] = y[5];
  dy[2] = y[6];
  dy[3] = y[7];
  bh_rt_k3_geodesic_acceleration(y, ud->m, ud->a, dy + 4);
}

int bh_rt_k3_all_finite(const double *y) {
  for (int i = 0; i < BH_RT_K3_N_STATE; ++i)
    if (!isfinite(y[i]))
      return 0;
  return 1;
}

void bh_rt_k3_renormalize_vr(double *y, double m, double a) {
  double r = y[1], th = y[2];
  double vt = y[4], vr = y[5], vth = y[6], vph = y[7];

  bh_rt_k3_geom geo;
  if (!bh_rt_k3_geometry(r, th, m, a, &geo))
    return;
  if (geo.g_rr <= 0.0 || !isfinite(geo.g_rr))
    return;

  double inner = -(geo.g_tt * vt * vt + 2.0 * geo.g_tph * vt * vph + geo.g_thth * vth * vth +
                   geo.g_phph * vph * vph);
  if (inner < 0.0 && inner > -1e-6)
    inner = 0.0;
  if (inner < 0.0 || !isfinite(inner))
    return;

  double sgn = (vr >= 0.0) ? 1.0 : -1.0;
  y[5] = sgn * sqrt(inner / geo.g_rr);
}

double bh_rt_k3_outer_horizon(double m, double a) {
  double disc = m * m - a * a;
  if (disc < 0.0)
    disc = 0.0;
  return m + sqrt(disc);
}

static void bh_rt_kerr_phase3_trace_impl(const double *y0, double m, double a, double dlambda,
                                         int max_steps, double r_escape, double r_horizon_epsilon,
                                         bh_rt_phase2_trace_result *out) {
  double y[BH_RT_K3_N_STATE];
  for (int i = 0; i < BH_RT_K3_N_STATE; ++i)
    y[i] = y0[i];

  bh_rt_k3_userdata ud;
  ud.m = m;
  ud.a = a;
  double rk_ws[BH_RT_K3_RK_WORK];

  double r_cap = bh_rt_k3_outer_horizon(m, a) + r_horizon_epsilon;
  bh_rt_k3_renormalize_vr(y, m, a);

  double r_min_val = INFINITY;
  double lam = 0.0;
  int status = BH_RT_STATUS_MAX_STEPS;
  double termination_r = NAN;
  double termination_lambda = 0.0;
  int steps_taken = 0;
  int broke = 0;

  for (int step_idx = 0; step_idx < max_steps; ++step_idx) {
    double r = y[1];
    if (!bh_rt_k3_all_finite(y)) {
      status = BH_RT_STATUS_NUMERICAL_ERROR;
      termination_r = r;
      termination_lambda = lam;
      steps_taken = step_idx;
      broke = 1;
      break;
    }
    if (r < r_cap) {
      status = BH_RT_STATUS_CAPTURED;
      termination_r = r;
      termination_lambda = lam;
      steps_taken = step_idx;
      broke = 1;
      break;
    }
    if (r > r_escape) {
      status = BH_RT_STATUS_ESCAPED;
      termination_r = r;
      termination_lambda = lam;
      steps_taken = step_idx;
      broke = 1;
      break;
    }
    if (isfinite(r))
      r_min_val = fmin(r_min_val, r);

    bh_rt_rk4_step(bh_rt_k3_deriv, lam, y, dlambda, &ud, BH_RT_K3_N_STATE, rk_ws);
    if ((step_idx % 4) == 0)
      bh_rt_k3_renormalize_vr(y, m, a);
    lam += dlambda;
    steps_taken = step_idx + 1;
  }

  if (!broke) {
    termination_r = y[1];
    termination_lambda = lam;
  }
  if (status == BH_RT_STATUS_MAX_STEPS) {
    termination_r = y[1];
    termination_lambda = lam;
    {
      double r = y[1];
      if (isfinite(r))
        r_min_val = fmin(r_min_val, r);
    }
    if (!bh_rt_k3_all_finite(y))
      status = BH_RT_STATUS_NUMERICAL_ERROR;
  }
  if (!isfinite(r_min_val))
    r_min_val = NAN;

  out->status = status;
  out->steps_taken = steps_taken;
  out->max_steps = max_steps;
  out->termination_r = termination_r;
  out->termination_lambda = termination_lambda;
  out->r_min = r_min_val;
}

void bh_rt_kerr_phase3_trace(const double *y0, double m, double a, double dlambda, int max_steps,
                             double r_escape, double r_horizon_epsilon,
                             bh_rt_phase2_trace_result *out) {
  bh_rt_kerr_phase3_trace_impl(y0, m, a, dlambda, max_steps, r_escape, r_horizon_epsilon, out);
}
