#include "bh_rt_kerr_phase3.h"
#include "bh_rt_kerr_phase3_batch.h"
#include "bh_rt_kerr_phase3_internal.h"
#include "bh_rt_status.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum {
  K3_BATCH_STAGE_GROUPS = 6,
  K3_BATCH_PREVIOUS_VALUES = 4,
  K3_BATCH_WORK_SLOTS =
      K3_BATCH_STAGE_GROUPS * BH_RT_K3_N_STATE + K3_BATCH_PREVIOUS_VALUES,
  K3_BATCH_I_T = 0,
  K3_BATCH_I_R = 1,
  K3_BATCH_I_TH = 2,
  K3_BATCH_I_PH = 3,
  K3_BATCH_I_VT = 4,
  K3_BATCH_I_VR = 5,
  K3_BATCH_I_VTH = 6,
  K3_BATCH_I_VPH = 7,
};

static int state_finite(double *const state[BH_RT_K3_N_STATE], int ray) {
  for (int component = 0; component < BH_RT_K3_N_STATE; ++component) {
    if (!isfinite(state[component][ray]))
      return 0;
  }
  return 1;
}

static int rhs_into_soa(
    double *const values[BH_RT_K3_N_STATE],
    double *const derivatives[BH_RT_K3_N_STATE],
    int ray,
    double m,
    double spin) {
  double dy[BH_RT_K3_N_STATE];
  if (!bh_rt_k3_rhs_components(
          values[K3_BATCH_I_R][ray], values[K3_BATCH_I_TH][ray],
          values[K3_BATCH_I_VT][ray], values[K3_BATCH_I_VR][ray],
          values[K3_BATCH_I_VTH][ray], values[K3_BATCH_I_VPH][ray], m, spin, 0,
          dy))
    return 0;
  for (int component = 0; component < BH_RT_K3_N_STATE; ++component)
    derivatives[component][ray] = dy[component];
  return 1;
}

static int valid_batch_parameters(
    double m, double spin, double dlambda, int max_steps, double r_escape,
    double r_horizon_epsilon) {
  if (!(isfinite(m) && isfinite(spin) && isfinite(dlambda) &&
        isfinite(r_escape) && isfinite(r_horizon_epsilon)))
    return 0;
  if (m <= 0.0 || fabs(spin) > 1.0 || dlambda <= 0.0 || max_steps < 0 ||
      r_horizon_epsilon < 0.0)
    return 0;
  const double r_cap =
      bh_rt_k3_outer_horizon(m, spin) + r_horizon_epsilon;
  return isfinite(r_cap) && r_escape > r_cap;
}

static void fill_batch_error(
    int n, int *out_status, int *out_steps_taken, double *out_termination_r,
    double *out_r_min, double *out_eq_r_cross) {
  for (int ray = 0; ray < n; ++ray) {
    out_status[ray] = BH_RT_STATUS_NUMERICAL_ERROR;
    out_steps_taken[ray] = 0;
    out_termination_r[ray] = NAN;
    out_r_min[ray] = NAN;
    out_eq_r_cross[ray] = NAN;
  }
}

static void mark_numerical_error(
    unsigned char *active, int *out_status, int *out_steps_taken, int ray,
    int step_idx) {
  active[ray] = 0;
  out_status[ray] = BH_RT_STATUS_NUMERICAL_ERROR;
  out_steps_taken[ray] = step_idx;
}

void bh_rt_kerr_phase3_batch_trace(
    const double *y0,
    int n,
    double m,
    double a,
    double dlambda,
    int max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int *out_status,
    int *out_steps_taken,
    double *out_termination_r,
    double *out_r_min,
    double *out_eq_r_cross) {
  if (n <= 0)
    return;
  if (!valid_batch_parameters(
          m, a, dlambda, max_steps, r_escape, r_horizon_epsilon)) {
    fill_batch_error(
        n, out_status, out_steps_taken, out_termination_r, out_r_min,
        out_eq_r_cross);
    return;
  }

  const size_t rays = (size_t)n;
  const size_t work_slots = (size_t)K3_BATCH_WORK_SLOTS;
  if (rays > (size_t)-1 / (work_slots * sizeof(double))) {
    fill_batch_error(
        n, out_status, out_steps_taken, out_termination_r, out_r_min,
        out_eq_r_cross);
    return;
  }

  double *work = (double *)malloc(rays * work_slots * sizeof(double));
  unsigned char *active = (unsigned char *)malloc(rays);
  if (work == NULL || active == NULL) {
    free(work);
    free(active);
    fill_batch_error(
        n, out_status, out_steps_taken, out_termination_r, out_r_min,
        out_eq_r_cross);
    return;
  }

  double *state[BH_RT_K3_N_STATE];
  double *k1[BH_RT_K3_N_STATE];
  double *k2[BH_RT_K3_N_STATE];
  double *k3[BH_RT_K3_N_STATE];
  double *k4[BH_RT_K3_N_STATE];
  double *trial[BH_RT_K3_N_STATE];
  double **groups[K3_BATCH_STAGE_GROUPS] = {state, k1, k2, k3, k4, trial};
  size_t cursor = 0;
  for (int group = 0; group < K3_BATCH_STAGE_GROUPS; ++group) {
    for (int component = 0; component < BH_RT_K3_N_STATE; ++component) {
      groups[group][component] = work + cursor;
      cursor += rays;
    }
  }
  double *previous_r = work + cursor;
  cursor += rays;
  double *previous_theta = work + cursor;
  cursor += rays;
  double *previous_vt = work + cursor;
  cursor += rays;
  double *previous_vphi = work + cursor;

  const double r_cap =
      bh_rt_k3_outer_horizon(m, a) + r_horizon_epsilon;
  for (int ray = 0; ray < n; ++ray) {
    const double *src = y0 + (size_t)ray * BH_RT_K3_N_STATE;
    for (int component = 0; component < BH_RT_K3_N_STATE; ++component)
      state[component][ray] = src[component];

    out_status[ray] = BH_RT_STATUS_MAX_STEPS;
    out_steps_taken[ray] = 0;
    out_termination_r[ray] = NAN;
    out_r_min[ray] = INFINITY;
    out_eq_r_cross[ray] = NAN;
    active[ray] = 1;

    if (!state_finite(state, ray)) {
      mark_numerical_error(
          active, out_status, out_steps_taken, ray, 0);
      out_r_min[ray] = NAN;
      continue;
    }
    state[K3_BATCH_I_VR][ray] = bh_rt_k3_renormalized_vr(
        state[K3_BATCH_I_R][ray], state[K3_BATCH_I_TH][ray],
        state[K3_BATCH_I_VT][ray], state[K3_BATCH_I_VR][ray],
        state[K3_BATCH_I_VTH][ray], state[K3_BATCH_I_VPH][ray], m, a);
  }

  for (int step_idx = 0; step_idx < max_steps; ++step_idx) {
    int ready = 0;
    for (int ray = 0; ray < n; ++ray) {
      if (!active[ray])
        continue;
      if (!state_finite(state, ray)) {
        mark_numerical_error(
            active, out_status, out_steps_taken, ray, step_idx);
        continue;
      }

      const double r = state[K3_BATCH_I_R][ray];
      if (r < r_cap) {
        active[ray] = 0;
        out_status[ray] = BH_RT_STATUS_CAPTURED;
        out_steps_taken[ray] = step_idx;
        out_termination_r[ray] = r;
        continue;
      }
      if (r > r_escape) {
        active[ray] = 0;
        out_status[ray] = BH_RT_STATUS_ESCAPED;
        out_steps_taken[ray] = step_idx;
        out_termination_r[ray] = r;
        continue;
      }

      out_r_min[ray] = fmin(out_r_min[ray], r);
      previous_r[ray] = r;
      previous_theta[ray] = state[K3_BATCH_I_TH][ray];
      previous_vt[ray] = state[K3_BATCH_I_VT][ray];
      previous_vphi[ray] = state[K3_BATCH_I_VPH][ray];
      ready = 1;
    }
    if (!ready)
      break;

    for (int ray = 0; ray < n; ++ray) {
      if (active[ray] &&
          !rhs_into_soa(state, k1, ray, m, a))
        mark_numerical_error(
            active, out_status, out_steps_taken, ray, step_idx);
    }

    for (int ray = 0; ray < n; ++ray) {
      if (!active[ray])
        continue;
      for (int component = 0; component < BH_RT_K3_N_STATE; ++component)
        trial[component][ray] =
            state[component][ray] + 0.5 * dlambda * k1[component][ray];
      if (!rhs_into_soa(trial, k2, ray, m, a))
        mark_numerical_error(
            active, out_status, out_steps_taken, ray, step_idx);
    }

    for (int ray = 0; ray < n; ++ray) {
      if (!active[ray])
        continue;
      for (int component = 0; component < BH_RT_K3_N_STATE; ++component)
        trial[component][ray] =
            state[component][ray] + 0.5 * dlambda * k2[component][ray];
      if (!rhs_into_soa(trial, k3, ray, m, a))
        mark_numerical_error(
            active, out_status, out_steps_taken, ray, step_idx);
    }

    for (int ray = 0; ray < n; ++ray) {
      if (!active[ray])
        continue;
      for (int component = 0; component < BH_RT_K3_N_STATE; ++component)
        trial[component][ray] =
            state[component][ray] + dlambda * k3[component][ray];
      if (!rhs_into_soa(trial, k4, ray, m, a))
        mark_numerical_error(
            active, out_status, out_steps_taken, ray, step_idx);
    }

    for (int ray = 0; ray < n; ++ray) {
      if (!active[ray])
        continue;
      for (int component = 0; component < BH_RT_K3_N_STATE; ++component)
        state[component][ray] +=
            (dlambda / 6.0) *
            (k1[component][ray] + 2.0 * k2[component][ray] +
             2.0 * k3[component][ray] + k4[component][ray]);
      if (!state_finite(state, ray)) {
        mark_numerical_error(
            active, out_status, out_steps_taken, ray, step_idx);
        continue;
      }

      if ((step_idx % 4) == 0)
        state[K3_BATCH_I_VR][ray] = bh_rt_k3_renormalized_vr(
            state[K3_BATCH_I_R][ray], state[K3_BATCH_I_TH][ray],
            state[K3_BATCH_I_VT][ray], state[K3_BATCH_I_VR][ray],
            state[K3_BATCH_I_VTH][ray], state[K3_BATCH_I_VPH][ray], m, a);

      if (isnan(out_eq_r_cross[ray]) && state_finite(state, ray)) {
        const double th0 = previous_theta[ray] - 0.5 * M_PI;
        const double th1 = state[K3_BATCH_I_TH][ray] - 0.5 * M_PI;
        if (th0 * th1 < 0.0) {
          const double alpha =
              fabs(th0) / (fabs(th0) + fabs(th1) + 1e-30);
          out_eq_r_cross[ray] =
              previous_r[ray] * (1.0 - alpha) +
              state[K3_BATCH_I_R][ray] * alpha;
        }
      }
      out_steps_taken[ray] = step_idx + 1;
    }
  }

  for (int ray = 0; ray < n; ++ray) {
    if (active[ray]) {
      out_termination_r[ray] = state[K3_BATCH_I_R][ray];
      if (isfinite(state[K3_BATCH_I_R][ray]))
        out_r_min[ray] = fmin(out_r_min[ray], state[K3_BATCH_I_R][ray]);
    }
    if (!isfinite(out_r_min[ray]))
      out_r_min[ray] = NAN;
  }

  free(active);
  free(work);
}
