#pragma once

#include "bh_rt_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bh_rt_schw_u_trace_result {
  int status;
  double termination_phi;
  double termination_r; /* finite, +inf via IEEE (HUGE_VAL), or NAN */
  int steps_taken;
  double r_min; /* NAN if no samples appended */
} bh_rt_schw_u_trace_result;

/* Integrate Phase 1 equatorial null rays in u(phi)=1/r form; mirrors
 * phase1.trace_single_schwarzschild_ray (same ICS, rk4_step order, thresholds). */
void bh_rt_schwarzschild_u_trace(double m, double b, double phi_start,
                                 double phi_max, double dphi, double r_capture,
                                 double r_escape,
                                 bh_rt_schw_u_trace_result *out);

#ifdef __cplusplus
}
#endif
