#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Launch one CUDA thread per ray and return a CUDA error code. */
int32_t bh_rt_schwarzschild_phase2_batch_trace_cuda(
    const double *y0,
    int n,
    double m,
    double dlambda,
    int max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int *out_status,
    int *out_steps_taken,
    double *out_termination_r,
    double *out_r_min,
    int *out_disk_hit,
    double *out_disk_r_cross,
    double *out_disk_vt,
    double *out_disk_vph);

#ifdef __cplusplus
}
#endif
