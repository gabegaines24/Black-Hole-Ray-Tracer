#pragma once

#include "bh_rt_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Phase 2+ Schwarzschild 3D kernel API.
 *
 * State order mirrors Python phase2_geodesic exactly:
 *   y = [t, r, theta, phi, vt, vr, vtheta, vphi]
 *
 * This scalar trace is the correctness anchor for the future SoA batch API:
 *   t[], r[], theta[], phi[], vt[], vr[], vtheta[], vphi[], status[].
 * Keep the scalar termination and null-renormalization semantics in parity
 * before adding SIMD or bridge/pybind11 layers.
 */

typedef struct bh_rt_schw3d_trace_result {
  int status;
  int steps_taken;
  int max_steps;
  double termination_r;
  double termination_lambda;
  double r_min;
  double final_state[8];
} bh_rt_schw3d_trace_result;

typedef struct bh_rt_schw3d_batch_input {
  int count;
  const double *t0;
  const double *r0;
  const double *theta0;
  const double *phi0;
  const double *vt0;
  const double *vr0;
  const double *vtheta0;
  const double *vphi0;
} bh_rt_schw3d_batch_input;

typedef struct bh_rt_schw3d_batch_output {
  int *status;
  int *steps_taken;
  double *termination_r;
  double *termination_lambda;
  double *r_min;
  double *final_t;
  double *final_r;
  double *final_theta;
  double *final_phi;
  double *final_vt;
  double *final_vr;
  double *final_vtheta;
  double *final_vphi;
} bh_rt_schw3d_batch_output;

double bh_rt_schw3d_metric_invariant(const double y[8], double m);

void bh_rt_schw3d_renormalize_null(double y[8], double m,
                                   int preserve_vr_sign);

void bh_rt_schwarzschild_3d_trace(const double x0[4], const double v0[4],
                                  double m, double dlambda, int max_steps,
                                  double r_escape,
                                  double r_horizon_epsilon,
                                  bh_rt_schw3d_trace_result *out);

void bh_rt_schwarzschild_3d_trace_batch(
    const bh_rt_schw3d_batch_input *in, double m, double dlambda,
    int max_steps, double r_escape, double r_horizon_epsilon,
    bh_rt_schw3d_batch_output *out);

#ifdef __cplusplus
}
#endif
