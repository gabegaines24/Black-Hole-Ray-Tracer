#include "bh_rt_schwarzschild_phase2.h"

#include <math.h>
#include <stddef.h>

#include "bh_rt_rk4.h"

#ifndef INFINITY
#define INFINITY (1.0 / 0.0)
#endif

enum {
  BH_RT_ST_CAPTURED = 0,
  BH_RT_ST_ESCAPED = 1,
  BH_RT_ST_MAX_STEPS = 2,
  BH_RT_ST_NUMERIC = 3
};

#define N_STATE 8
#define RK_WORK (5 * N_STATE)

typedef struct bh_rt_phase2_userdata {
  double m;
} bh_rt_phase2_userdata;

static double schwarzschild_f(double r, double m) { return 1.0 - 2.0 * m / r; }

/* Gamma^mu_ab, symmetric in (a,b). Indices: t=0,r=1,theta=2,phi=3. */
static double christoffel(int mu, int a, int b, double r, double th,
                          double m) {
#define I_T 0
#define I_R 1
#define I_TH 2
#define I_PH 3
  if (r <= 2.0 * m || !(isfinite(r) && isfinite(th)))
    return 0.0;
  double ff = schwarzschild_f(r, m);
  double s = sin(th);
  double c = cos(th);
  double s2 = s * s;
  double r2 = r * r;
  double m_r2f = m / (r2 * ff);

  if (a > b) {
    int tmp = a;
    a = b;
    b = tmp;
  }

  if (mu == I_T) {
    if (a == I_T && b == I_R)
      return m_r2f;
    return 0.0;
  }

  if (mu == I_R) {
    if (a == I_T && b == I_T)
      return m * ff / r2;
    if (a == I_R && b == I_R)
      return m / (r2 * ff);
    if (a == I_TH && b == I_TH)
      return -r * ff;
    if (a == I_PH && b == I_PH)
      return -r * ff * s2;
    return 0.0;
  }

  if (mu == I_TH) {
    if (a == I_R && b == I_TH)
      return 1.0 / r;
    if (a == I_PH && b == I_PH)
      return -s * c;
    return 0.0;
  }

  if (mu == I_PH) {
    if (a == I_R && b == I_PH)
      return 1.0 / r;
    if (a == I_TH && b == I_PH)
      return (fabs(s) > 1e-12) ? (c / s) : 0.0;
    return 0.0;
  }

  return 0.0;
#undef I_T
#undef I_R
#undef I_TH
#undef I_PH
}

static void geodesic_acceleration(const double *y, double m, double *acc) {
  double r = y[1];
  double th = y[2];
  double vt = y[4], vr = y[5], vth = y[6], vph = y[7];
  double vv[4] = {vt, vr, vth, vph};

  for (int mu = 0; mu < 4; ++mu) {
    double sum = 0.0;
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        sum += christoffel(mu, a, b, r, th, m) * vv[a] * vv[b];
      }
    }
    acc[mu] = -sum;
  }
}

static void phase2_deriv(double lam, const double *y, double *dy,
                         void *userdata) {
  (void)lam;
  bh_rt_phase2_userdata *ud = (bh_rt_phase2_userdata *)userdata;

  dy[0] = y[4];
  dy[1] = y[5];
  dy[2] = y[6];
  dy[3] = y[7];
  geodesic_acceleration(y, ud->m, dy + 4);
}

static int all_finite_state(const double *y) {
  for (int i = 0; i < N_STATE; ++i) {
    if (!isfinite(y[i]))
      return 0;
  }
  return 1;
}

static void renormalize_vr_inplace(double *y, double m) {
  double r = y[1];
  double th = y[2];
  double vt = y[4];
  double vr = y[5];
  double vth = y[6];
  double vph = y[7];

  double f = schwarzschild_f(r, m);
  if (f <= 0.0 || !isfinite(f))
    return;
  double s = sin(th);
  double a_part = (r * r) * (vth * vth + s * s * vph * vph);
  double inner = f * f * vt * vt - f * a_part;
  if (inner < 0.0 && inner > -1e-6)
    inner = 0.0;
  if (inner < 0.0 || !isfinite(inner))
    return;
  double sgn = (vr >= 0.0) ? 1.0 : -1.0;
  y[5] = sgn * sqrt(inner);
}

void bh_rt_schwarzschild_phase2_trace(const double *y0, double m, double dlambda,
                                    int max_steps, double r_escape,
                                    double r_horizon_epsilon,
                                    bh_rt_phase2_trace_result *out) {
  double y[N_STATE];
  for (int i = 0; i < N_STATE; ++i)
    y[i] = y0[i];

  bh_rt_phase2_userdata ud;
  ud.m = m;

  double rk_ws[RK_WORK];

  double r_h = 2.0 * m;
  double r_cap = r_h + r_horizon_epsilon;

  renormalize_vr_inplace(y, m);

  double r_min_val = INFINITY;
  double lam = 0.0;
  int status = BH_RT_ST_MAX_STEPS;
  double termination_r = NAN;
  double termination_lambda = 0.0;
  int steps_taken = 0;
  int broke = 0;

  for (int step_idx = 0; step_idx < max_steps; ++step_idx) {
    double r = y[1];

    if (!all_finite_state(y)) {
      status = BH_RT_ST_NUMERIC;
      termination_r = r;
      termination_lambda = lam;
      steps_taken = step_idx;
      broke = 1;
      break;
    }

    if (r < r_cap) {
      status = BH_RT_ST_CAPTURED;
      termination_r = r;
      termination_lambda = lam;
      steps_taken = step_idx;
      broke = 1;
      break;
    }
    if (r > r_escape) {
      status = BH_RT_ST_ESCAPED;
      termination_r = r;
      termination_lambda = lam;
      steps_taken = step_idx;
      broke = 1;
      break;
    }

    if (isfinite(r))
      r_min_val = fmin(r_min_val, r);

    bh_rt_rk4_step(phase2_deriv, lam, y, dlambda, &ud, N_STATE, rk_ws);

    if ((step_idx % 4) == 0)
      renormalize_vr_inplace(y, m);

    lam += dlambda;
    steps_taken = step_idx + 1;
  }

  if (!broke) {
    termination_r = y[1];
    termination_lambda = lam;
  }

  if (status == BH_RT_ST_MAX_STEPS) {
    termination_r = y[1];
    termination_lambda = lam;
    {
      double r = y[1];
      if (isfinite(r))
        r_min_val = fmin(r_min_val, r);
    }
    if (!all_finite_state(y))
      status = BH_RT_ST_NUMERIC;
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
