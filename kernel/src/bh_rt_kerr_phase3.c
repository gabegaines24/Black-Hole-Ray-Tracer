#include "bh_rt_kerr_phase3.h"

#include <math.h>
#include <stddef.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BH_RT_KERR_N_COORD 4
#define BH_RT_KERR_N_STATE 8
#define I_T 0
#define I_R 1
#define I_TH 2
#define I_PH 3

static int all_finite(const double *values, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(values[i]))
      return 0;
  }
  return 1;
}

static double spin_a(double m, double spin) {
  return m * spin;
}

static double sigma_value(double r, double theta, double m, double spin) {
  const double a = spin_a(m, spin);
  const double c = cos(theta);
  return r * r + a * a * c * c;
}

static double delta_value(double r, double m, double spin) {
  const double a = spin_a(m, spin);
  return r * r - 2.0 * m * r + a * a;
}

static double outer_horizon(double m, double spin) {
  const double a = spin_a(m, spin);
  return m + sqrt(fmax(m * m - a * a, 0.0));
}

static int valid_parameters(double r, double theta, double m, double spin) {
  if (!(isfinite(r) && isfinite(theta) && isfinite(m) && isfinite(spin)))
    return 0;
  if (m <= 0.0 || fabs(spin) > 1.0)
    return 0;
  if (r <= outer_horizon(m, spin) || !(theta > 0.0 && theta < M_PI))
    return 0;
  if (fabs(sin(theta)) <= 1e-12)
    return 0;
  return 1;
}
static int valid_algebra_parameters(double r, double theta, double m,
                                    double spin) {
  if (!(isfinite(r) && isfinite(theta) && isfinite(m) && isfinite(spin)))
    return 0;
  if (m <= 0.0 || fabs(spin) > 1.0 || r <= 0.0)
    return 0;
  if (!(theta > 0.0 && theta < M_PI) || fabs(sin(theta)) <= 1e-12)
    return 0;
  return 1;
}


static void zero_metric(double *g) {
  for (int i = 0; i < BH_RT_KERR_N_COORD * BH_RT_KERR_N_COORD; ++i)
    g[i] = 0.0;
}

static int metric_values(double r, double theta, double m, double spin, double *g) {
  const double a = spin_a(m, spin);
  const double s = sin(theta);
  const double s2 = s * s;
  const double sigma = sigma_value(r, theta, m, spin);
  const double delta = delta_value(r, m, spin);

  if (!(sigma > 0.0) || delta == 0.0 || !isfinite(sigma) || !isfinite(delta))
    return 0;

  zero_metric(g);
  g[I_T * 4 + I_T] = -1.0 + 2.0 * m * r / sigma;
  g[I_T * 4 + I_PH] = -2.0 * m * a * r * s2 / sigma;
  g[I_PH * 4 + I_T] = g[I_T * 4 + I_PH];
  g[I_R * 4 + I_R] = sigma / delta;
  g[I_TH * 4 + I_TH] = sigma;
  g[I_PH * 4 + I_PH] =
      (r * r + a * a + 2.0 * m * a * a * r * s2 / sigma) * s2;
  return all_finite(g, 16);
}

static int metric_derivative(int coord_idx, double r, double theta, double m,
                             double spin, double *dg) {
  const double a = spin_a(m, spin);
  const double s = sin(theta);
  const double c = cos(theta);
  const double s2 = s * s;
  const double ds2_dtheta = 2.0 * s * c;
  const double sigma = sigma_value(r, theta, m, spin);
  const double delta = delta_value(r, m, spin);
  const double d_sigma_dr = 2.0 * r;
  const double d_sigma_dtheta = -2.0 * a * a * s * c;

  zero_metric(dg);
  if (coord_idx != I_R && coord_idx != I_TH)
    return 1;
  if (!(sigma > 0.0) || delta == 0.0 || !isfinite(sigma) || !isfinite(delta))
    return 0;

  if (coord_idx == I_R) {
    const double sigma_minus_r_dsigma = sigma - r * d_sigma_dr;
    const double d_delta_dr = 2.0 * r - 2.0 * m;
    dg[I_T * 4 + I_T] =
        2.0 * m * sigma_minus_r_dsigma / (sigma * sigma);
    dg[I_T * 4 + I_PH] =
        -2.0 * m * a * s2 * sigma_minus_r_dsigma / (sigma * sigma);
    dg[I_PH * 4 + I_T] = dg[I_T * 4 + I_PH];
    dg[I_R * 4 + I_R] =
        (d_sigma_dr * delta - sigma * d_delta_dr) / (delta * delta);
    dg[I_TH * 4 + I_TH] = d_sigma_dr;
    dg[I_PH * 4 + I_PH] = s2 *
        (2.0 * r +
         2.0 * m * a * a * s2 * sigma_minus_r_dsigma / (sigma * sigma));
    return all_finite(dg, 16);
  }

  dg[I_T * 4 + I_T] =
      -2.0 * m * r * d_sigma_dtheta / (sigma * sigma);
  dg[I_T * 4 + I_PH] = -2.0 * m * a * r *
      (ds2_dtheta / sigma - s2 * d_sigma_dtheta / (sigma * sigma));
  dg[I_PH * 4 + I_T] = dg[I_T * 4 + I_PH];
  dg[I_R * 4 + I_R] = d_sigma_dtheta / delta;
  dg[I_TH * 4 + I_TH] = d_sigma_dtheta;

  const double base = r * r + a * a + 2.0 * m * a * a * r * s2 / sigma;
  const double d_base_dtheta = 2.0 * m * a * a * r *
      (ds2_dtheta / sigma - s2 * d_sigma_dtheta / (sigma * sigma));
  dg[I_PH * 4 + I_PH] =
      ds2_dtheta * base + s2 * d_base_dtheta;
  return all_finite(dg, 16);
}

static int inverse_metric(const double *g, double *g_inv) {
  zero_metric(g_inv);
  const double denom = g[I_T * 4 + I_T] * g[I_PH * 4 + I_PH] -
      g[I_T * 4 + I_PH] * g[I_T * 4 + I_PH];
  if (fabs(denom) < 1e-30 || fabs(g[I_R * 4 + I_R]) < 1e-30 ||
      fabs(g[I_TH * 4 + I_TH]) < 1e-30)
    return 0;
  g_inv[I_T * 4 + I_T] = g[I_PH * 4 + I_PH] / denom;
  g_inv[I_T * 4 + I_PH] = -g[I_T * 4 + I_PH] / denom;
  g_inv[I_PH * 4 + I_T] = g_inv[I_T * 4 + I_PH];
  g_inv[I_PH * 4 + I_PH] = g[I_T * 4 + I_T] / denom;
  g_inv[I_R * 4 + I_R] = 1.0 / g[I_R * 4 + I_R];
  g_inv[I_TH * 4 + I_TH] = 1.0 / g[I_TH * 4 + I_TH];
  return all_finite(g_inv, 16);
}

static int christoffel(int mu, int a_idx, int b_idx, double r, double theta,
                       double m, double spin, double *value) {
  if (r <= outer_horizon(m, spin)) {
    *value = 0.0;
    return 1;
  }
  double g[16];
  double g_inv[16];
  double d_a[16];
  double d_b[16];

  if (!metric_values(r, theta, m, spin, g) ||
      !inverse_metric(g, g_inv) ||
      !metric_derivative(a_idx, r, theta, m, spin, d_a) ||
      !metric_derivative(b_idx, r, theta, m, spin, d_b))
    return 0;

  double total = 0.0;
  for (int rho = 0; rho < BH_RT_KERR_N_COORD; ++rho) {
    double d_rho[16];
    if (!metric_derivative(rho, r, theta, m, spin, d_rho))
      return 0;
    total += g_inv[mu * 4 + rho] *
        (d_a[b_idx * 4 + rho] + d_b[a_idx * 4 + rho] -
         d_rho[a_idx * 4 + b_idx]);
  }
  *value = 0.5 * total;
  return isfinite(*value);
}

static int geodesic_rhs_values(const double *y, double m, double spin,
                               int require_exterior, double *dy_out) {
  if (!all_finite(y, BH_RT_KERR_N_STATE))
    return 0;
  if (require_exterior) {
    if (!valid_parameters(y[I_R], y[I_TH], m, spin))
      return 0;
  } else if (!valid_algebra_parameters(y[I_R], y[I_TH], m, spin)) {
    return 0;
  }

  dy_out[I_T] = y[4];
  dy_out[I_R] = y[5];
  dy_out[I_TH] = y[6];
  dy_out[I_PH] = y[7];

  const double v[4] = {y[4], y[5], y[6], y[7]};
  for (int mu = 0; mu < BH_RT_KERR_N_COORD; ++mu) {
    double total = 0.0;
    for (int a_idx = 0; a_idx < BH_RT_KERR_N_COORD; ++a_idx) {
      for (int b_idx = 0; b_idx < BH_RT_KERR_N_COORD; ++b_idx) {
        double gamma = 0.0;
        if (!christoffel(mu, a_idx, b_idx, y[I_R], y[I_TH], m, spin,
                         &gamma))
          return 0;
        total += gamma * v[a_idx] * v[b_idx];
      }
    }
    dy_out[4 + mu] = -total;
  }
  return all_finite(dy_out, BH_RT_KERR_N_STATE);
}

static void renormalize_kerr_vr(double *y, double m, double spin) {
  double g[16];
  if (!valid_algebra_parameters(y[I_R], y[I_TH], m, spin) ||
      !metric_values(y[I_R], y[I_TH], m, spin, g))
    return;

  const double g_rr = g[I_R * 4 + I_R];
  if (fabs(g_rr) < 1e-30 || !isfinite(g_rr))
    return;
  const double vt = y[4];
  const double vr = y[5];
  const double vtheta = y[6];
  const double vphi = y[7];
  const double other =
      g[I_T * 4 + I_T] * vt * vt +
      2.0 * g[I_T * 4 + I_PH] * vt * vphi +
      g[I_TH * 4 + I_TH] * vtheta * vtheta +
      g[I_PH * 4 + I_PH] * vphi * vphi;
  double inner = -other / g_rr;
  if (inner < 0.0 && inner > -1e-8)
    inner = 0.0;
  if (inner < 0.0 || !isfinite(inner))
    return;
  y[5] = (vr >= 0.0 ? 1.0 : -1.0) * sqrt(inner);
}

static int rk4_step_kerr(double *y, double dlambda, double m, double spin) {
  double k1[BH_RT_KERR_N_STATE];
  double k2[BH_RT_KERR_N_STATE];
  double k3[BH_RT_KERR_N_STATE];
  double k4[BH_RT_KERR_N_STATE];
  double trial[BH_RT_KERR_N_STATE];

  if (!geodesic_rhs_values(y, m, spin, 0, k1))
    return 0;
  for (int i = 0; i < BH_RT_KERR_N_STATE; ++i)
    trial[i] = y[i] + 0.5 * dlambda * k1[i];
  if (!geodesic_rhs_values(trial, m, spin, 0, k2))
    return 0;
  for (int i = 0; i < BH_RT_KERR_N_STATE; ++i)
    trial[i] = y[i] + 0.5 * dlambda * k2[i];
  if (!geodesic_rhs_values(trial, m, spin, 0, k3))
    return 0;
  for (int i = 0; i < BH_RT_KERR_N_STATE; ++i)
    trial[i] = y[i] + dlambda * k3[i];
  if (!geodesic_rhs_values(trial, m, spin, 0, k4))
    return 0;
  for (int i = 0; i < BH_RT_KERR_N_STATE; ++i) {
    y[i] += (dlambda / 6.0) *
        (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  }
  return all_finite(y, BH_RT_KERR_N_STATE);
}

int32_t bh_rt_kerr_metric(double r, double theta, double m, double spin,
                          double *g_out) {
  if (g_out == NULL)
    return BH_RT_KERR_INVALID_ARGUMENT;
  if (!valid_parameters(r, theta, m, spin))
    return BH_RT_KERR_DOMAIN_ERROR;
  if (!metric_values(r, theta, m, spin, g_out))
    return BH_RT_KERR_NUMERICAL_ERROR;
  return BH_RT_KERR_OK;
}

int32_t bh_rt_kerr_zamo_initial_state(
    double m, double spin, double r, double theta, double phi,
    double fov_deg, int32_t width, int32_t height, double sx, double sy,
    double *y_out) {
  if (y_out == NULL)
    return BH_RT_KERR_INVALID_ARGUMENT;
  if (!(isfinite(m) && isfinite(spin) && isfinite(r) &&
        isfinite(theta) && isfinite(phi) && isfinite(fov_deg) &&
        isfinite(sx) && isfinite(sy)))
    return BH_RT_KERR_INVALID_ARGUMENT;
  if (width <= 0 || height <= 0 || m <= 0.0 || fabs(spin) > 1.0 ||
      r <= outer_horizon(m, spin) || !(theta > 0.0 && theta < M_PI) ||
      fabs(sin(theta)) < 1e-10 || !(fov_deg > 0.0 && fov_deg < 180.0))
    return BH_RT_KERR_DOMAIN_ERROR;
  if (!valid_parameters(r, theta, m, spin))
    return BH_RT_KERR_DOMAIN_ERROR;

  double g[16];
  if (!metric_values(r, theta, m, spin, g))
    return BH_RT_KERR_NUMERICAL_ERROR;
  const double g_rr = g[I_R * 4 + I_R];
  const double g_theta_theta = g[I_TH * 4 + I_TH];
  const double g_phi_phi = g[I_PH * 4 + I_PH];
  const double lapse_squared =
      g_phi_phi > 0.0
          ? -(g[I_T * 4 + I_T] -
              g[I_T * 4 + I_PH] * g[I_T * 4 + I_PH] / g_phi_phi)
          : NAN;
  if (!(isfinite(g_rr) && isfinite(g_theta_theta) &&
        isfinite(g_phi_phi) && isfinite(lapse_squared) &&
        g_rr > 0.0 && g_theta_theta > 0.0 && g_phi_phi > 0.0 &&
        lapse_squared > 0.0))
    return BH_RT_KERR_DOMAIN_ERROR;

  const double tan_half_fov = tan(0.5 * fov_deg * M_PI / 180.0);
  const double local_x = -1.0;
  const double local_y =
      sx * ((double)width / (double)height) * tan_half_fov;
  const double local_z = sy * tan_half_fov;
  const double norm = sqrt(local_x * local_x + local_y * local_y +
                           local_z * local_z);
  if (!(isfinite(norm) && norm >= 1e-15))
    return BH_RT_KERR_DOMAIN_ERROR;

  const double lapse = sqrt(lapse_squared);
  const double frame_dragging =
      -g[I_T * 4 + I_PH] / (lapse * g_phi_phi);
  y_out[0] = 0.0;
  y_out[1] = r;
  y_out[2] = theta;
  y_out[3] = phi;
  y_out[4] = 1.0 / lapse;
  y_out[5] = (local_x / norm) / sqrt(g_rr);
  y_out[6] = (local_y / norm) / sqrt(g_theta_theta);
  y_out[7] = frame_dragging + (local_z / norm) / sqrt(g_phi_phi);
  if (!all_finite(y_out, BH_RT_KERR_N_STATE))
    return BH_RT_KERR_NUMERICAL_ERROR;
  return BH_RT_KERR_OK;
}

static double disk_intensity_factor(
    double r, double m, double vt, double vphi, double scale) {
  if (r <= 2.0 * m || !isfinite(r) || !isfinite(vt) || !isfinite(vphi))
    return 1.0;
  const double f = 1.0 - 2.0 * m / r;
  const double omega = sqrt(m / (r * r * r));
  const double denom = f * vt - r * r * omega * vphi;
  if (!isfinite(denom) || fabs(denom) < 1e-30)
    return 1.0;
  const double ratio = (f * vt) / denom;
  if (!isfinite(ratio))
    return 1.0;
  if (ratio <= 0.0 || scale == 0.0)
    return 0.0;
  const double intensity = pow(ratio, 4.0) * scale;
  if (!isfinite(intensity))
    return intensity > 0.0 ? 10.0 : 0.0;
  return fmin(fmax(intensity, 0.0), 10.0);
}

int32_t bh_rt_kerr_disk_observable(
    double r_cross, double r_inner, double r_outer, double m,
    double vt, double vphi, double redshift_scale,
    bh_rt_kerr_disk_result *out) {
  if (out == NULL)
    return BH_RT_KERR_INVALID_ARGUMENT;
  if (!(isfinite(r_inner) && isfinite(r_outer) && isfinite(m) &&
        isfinite(redshift_scale)))
    return BH_RT_KERR_INVALID_ARGUMENT;
  if (m <= 0.0 || r_inner <= 0.0 || r_outer <= r_inner ||
      redshift_scale < 0.0)
    return BH_RT_KERR_DOMAIN_ERROR;

  out->hit = 0;
  out->z_factor = 1.0;
  if (isnan(r_cross) || r_cross < r_inner || r_cross > r_outer)
    return BH_RT_KERR_OK;
  out->hit = 1;
  out->z_factor =
      disk_intensity_factor(r_cross, m, vt, vphi, redshift_scale);
  return isfinite(out->z_factor)
      ? BH_RT_KERR_OK
      : BH_RT_KERR_NUMERICAL_ERROR;
}

int32_t bh_rt_kerr_geodesic_rhs(const double *y, double m, double spin,
                                double *dy_out) {
  if (y == NULL || dy_out == NULL)
    return BH_RT_KERR_INVALID_ARGUMENT;
  if (!all_finite(y, BH_RT_KERR_N_STATE))
    return BH_RT_KERR_INVALID_ARGUMENT;
  if (!valid_parameters(y[I_R], y[I_TH], m, spin))
    return BH_RT_KERR_DOMAIN_ERROR;
  if (!geodesic_rhs_values(y, m, spin, 1, dy_out))
    return BH_RT_KERR_NUMERICAL_ERROR;
  return BH_RT_KERR_OK;
}

static int32_t trace_kerr_internal(
    const double *y0, double m, double spin, double dlambda,
    int32_t max_steps, double r_escape, double r_horizon_epsilon,
    bh_rt_kerr_trace_result *out,
    bh_rt_kerr_equatorial_crossing *crossing_out) {
  if (y0 == NULL || out == NULL)
    return BH_RT_KERR_INVALID_ARGUMENT;
  if (!(isfinite(m) && isfinite(spin) && isfinite(dlambda) &&
        isfinite(r_escape) && isfinite(r_horizon_epsilon)))
    return BH_RT_KERR_INVALID_ARGUMENT;
  if (m <= 0.0 || fabs(spin) > 1.0 || dlambda <= 0.0 ||
      max_steps < 0 || r_horizon_epsilon < 0.0)
    return BH_RT_KERR_DOMAIN_ERROR;

  const double r_cap = outer_horizon(m, spin) + r_horizon_epsilon;
  if (!isfinite(r_cap) || r_escape <= r_cap)
    return BH_RT_KERR_DOMAIN_ERROR;

  double y[BH_RT_KERR_N_STATE];
  for (int i = 0; i < BH_RT_KERR_N_STATE; ++i)
    y[i] = y0[i];

  out->status = BH_RT_STATUS_MAX_STEPS;
  out->steps_taken = 0;
  out->max_steps = max_steps;
  out->termination_r = NAN;
  out->termination_lambda = 0.0;
  out->r_min = NAN;
  if (crossing_out != NULL) {
    crossing_out->hit = 0;
    crossing_out->r_cross = NAN;
    crossing_out->vt_cross = NAN;
    crossing_out->vphi_cross = NAN;
  }

  if (!all_finite(y, BH_RT_KERR_N_STATE)) {
    out->status = BH_RT_STATUS_NUMERICAL_ERROR;
    return BH_RT_KERR_OK;
  }

  renormalize_kerr_vr(y, m, spin);
  double r_min = INFINITY;
  double lam = 0.0;
  int broke = 0;

  for (int32_t step_idx = 0; step_idx < max_steps; ++step_idx) {
    const double r = y[I_R];
    if (!all_finite(y, BH_RT_KERR_N_STATE)) {
      out->status = BH_RT_STATUS_NUMERICAL_ERROR;
      out->steps_taken = step_idx;
      broke = 1;
      break;
    }
    if (r < r_cap) {
      out->status = BH_RT_STATUS_CAPTURED;
      out->termination_r = r;
      out->termination_lambda = lam;
      out->steps_taken = step_idx;
      broke = 1;
      break;
    }
    if (r > r_escape) {
      out->status = BH_RT_STATUS_ESCAPED;
      out->termination_r = r;
      out->termination_lambda = lam;
      out->steps_taken = step_idx;
      broke = 1;
      break;
    }
    r_min = fmin(r_min, r);

    double previous[BH_RT_KERR_N_STATE];
    for (int i = 0; i < BH_RT_KERR_N_STATE; ++i)
      previous[i] = y[i];
    if (!rk4_step_kerr(y, dlambda, m, spin)) {
      out->status = BH_RT_STATUS_NUMERICAL_ERROR;
      out->steps_taken = step_idx;
      broke = 1;
      break;
    }
    if ((step_idx % 4) == 0)
      renormalize_kerr_vr(y, m, spin);
    if (crossing_out != NULL && !crossing_out->hit &&
        all_finite(y, BH_RT_KERR_N_STATE)) {
      const double th0 = previous[I_TH] - 0.5 * M_PI;
      const double th1 = y[I_TH] - 0.5 * M_PI;
      if (th0 * th1 < 0.0) {
        const double alpha =
            fabs(th0) / (fabs(th0) + fabs(th1) + 1e-30);
        crossing_out->hit = 1;
        crossing_out->r_cross =
            previous[I_R] * (1.0 - alpha) + y[I_R] * alpha;
        crossing_out->vt_cross =
            previous[4] * (1.0 - alpha) + y[4] * alpha;
        crossing_out->vphi_cross =
            previous[7] * (1.0 - alpha) + y[7] * alpha;
      }
    }
    lam += dlambda;
    out->steps_taken = step_idx + 1;
  }

  if (!broke && out->status == BH_RT_STATUS_MAX_STEPS) {
    out->termination_r = y[I_R];
    out->termination_lambda = lam;
  }
  if (out->status == BH_RT_STATUS_MAX_STEPS) {
    out->termination_r = y[I_R];
    out->termination_lambda = lam;
    if (isfinite(y[I_R]))
      r_min = fmin(r_min, y[I_R]);
  }
  if (isfinite(r_min))
    out->r_min = r_min;
  return BH_RT_KERR_OK;
}

int32_t bh_rt_kerr_phase3_trace(
    const double *y0, double m, double spin, double dlambda,
    int32_t max_steps, double r_escape, double r_horizon_epsilon,
    bh_rt_kerr_trace_result *out) {
  return trace_kerr_internal(
      y0, m, spin, dlambda, max_steps, r_escape, r_horizon_epsilon,
      out, NULL);
}

int32_t bh_rt_kerr_phase3_trace_crossing(
    const double *y0, double m, double spin, double dlambda,
    int32_t max_steps, double r_escape, double r_horizon_epsilon,
    bh_rt_kerr_trace_result *trace_out,
    bh_rt_kerr_equatorial_crossing *crossing_out) {
  if (crossing_out == NULL)
    return BH_RT_KERR_INVALID_ARGUMENT;
  return trace_kerr_internal(
      y0, m, spin, dlambda, max_steps, r_escape, r_horizon_epsilon,
      trace_out, crossing_out);
}
