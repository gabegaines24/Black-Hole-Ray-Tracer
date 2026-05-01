/* Generic explicit RK4 step for y' = f(x, y).
 * No Python headers — safe to compile as pure C library. */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*bh_rt_deriv_fn)(double x, const double *y, double *out_dy,
                               void *userdata);

/* Advance y[0..n-1] in place by step h. workspace must hold 4*n doubles. */
void bh_rt_rk4_step(bh_rt_deriv_fn f, double x, double *restrict y, double h,
                    void *userdata, int n, double *restrict workspace);

#ifdef __cplusplus
}
#endif
