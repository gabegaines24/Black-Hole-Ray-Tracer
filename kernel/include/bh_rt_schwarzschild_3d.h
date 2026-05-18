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

double bh_rt_schw3d_metric_invariant(const double y[8], double m);

void bh_rt_schw3d_renormalize_null(double y[8], double m,
                                   int preserve_vr_sign);

void bh_rt_schwarzschild_3d_trace(const double x0[4], const double v0[4],
                                  double m, double dlambda, int max_steps,
                                  double r_escape,
                                  double r_horizon_epsilon,
                                  bh_rt_schw3d_trace_result *out);

#ifdef __cplusplus
}
#endif
