/*
 * CUDA batch port of bh_rt_schwarzschild_phase2_batch.c
 *
 * Each CUDA thread traces one null geodesic.  Input/output layout is identical
 * to the CPU batch API (docs/STATE_API.md) so the same PyBind11 bridge can
 * dispatch to either backend at runtime.
 *
 *   nvcc -O2 -arch=sm_75 -I kernel/include --compiler-options -fPIC -shared \
 *        -o build/libbh_rt_phase2_cuda.so \
 *        kernel/cuda/bh_rt_schwarzschild_phase2_batch.cu \
 *        -lm
 *
 * Runtime selection: native_phase2_cuda.py loads this optional library and
 * raises an explicit error when a requested CUDA backend is unavailable.
 */

#include "bh_rt_schwarzschild_phase2_cuda.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

/* Status constants (mirror bh_rt_status.h for CUDA compilation). */
#define BH_RT_STATUS_CAPTURED        0
#define BH_RT_STATUS_ESCAPED         1
#define BH_RT_STATUS_MAX_STEPS       2
#define BH_RT_STATUS_NUMERICAL_ERROR 3

#define N_STATE 8
#define RK_WORK (5 * N_STATE)

/* ── device helpers ────────────────────────────────────────────────────────── */

__device__ static double d_schwarzschild_f(double r, double m) {
  return 1.0 - 2.0 * m / r;
}

__device__ static double d_christoffel(int mu, int a, int b,
                                        double r, double th, double m) {
  if (r <= 2.0 * m || !isfinite(r) || !isfinite(th)) return 0.0;
  double ff = d_schwarzschild_f(r, m);
  double s = sin(th), c = cos(th), s2 = s * s;
  double r2 = r * r;
  double m_r2f = m / (r2 * ff);
  if (a > b) { int t = a; a = b; b = t; }
  if (mu == 0) { if (a == 0 && b == 1) return m_r2f; return 0.0; }
  if (mu == 1) {
    if (a == 0 && b == 0) return m * ff / r2;
    if (a == 1 && b == 1) return m / (r2 * ff);
    if (a == 2 && b == 2) return -r * ff;
    if (a == 3 && b == 3) return -r * ff * s2;
    return 0.0;
  }
  if (mu == 2) {
    if (a == 1 && b == 2) return 1.0 / r;
    if (a == 3 && b == 3) return -s * c;
    return 0.0;
  }
  if (mu == 3) {
    if (a == 1 && b == 3) return 1.0 / r;
    if (a == 2 && b == 3) return (fabs(s) > 1e-12) ? (c / s) : 0.0;
    return 0.0;
  }
  return 0.0;
}

__device__ static void d_deriv(double lam, const double *y, double *dy, double m) {
  (void)lam;
  double r = y[1], th = y[2];
  dy[0] = y[4]; dy[1] = y[5]; dy[2] = y[6]; dy[3] = y[7];
  for (int mu = 0; mu < 4; ++mu) {
    double sum = 0.0;
    for (int aa = 0; aa < 4; ++aa)
      for (int bb = 0; bb < 4; ++bb)
        sum += d_christoffel(mu, aa, bb, r, th, m) * y[4+aa] * y[4+bb];
    dy[4 + mu] = -sum;
  }
}

__device__ static int d_all_finite(const double *y) {
  for (int i = 0; i < N_STATE; ++i) if (!isfinite(y[i])) return 0;
  return 1;
}

__device__ static void d_renormalize(double *y, double m) {
  double r = y[1], th = y[2];
  double vt = y[4], vr = y[5], vth = y[6], vph = y[7];
  double f = d_schwarzschild_f(r, m);
  if (f <= 0.0 || !isfinite(f)) return;
  double s = sin(th);
  double a_part = (r * r) * (vth * vth + s * s * vph * vph);
  double inner = f * f * vt * vt - f * a_part;
  if (inner < 0.0 && inner > -1e-6) inner = 0.0;
  if (inner < 0.0 || !isfinite(inner)) return;
  double sgn = (vr >= 0.0) ? 1.0 : -1.0;
  y[5] = sgn * sqrt(inner);
}

/* RK4 step, inline to keep register usage low. */
__device__ static void d_rk4_step(double lam, double *y, double dlambda, double m) {
  double k1[N_STATE], k2[N_STATE], k3[N_STATE], k4[N_STATE], ytmp[N_STATE];
  double h = dlambda, h2 = h * 0.5;

  d_deriv(lam,         y,     k1, m);
  for (int i = 0; i < N_STATE; ++i) ytmp[i] = y[i] + h2 * k1[i];
  d_deriv(lam + h2,    ytmp,  k2, m);
  for (int i = 0; i < N_STATE; ++i) ytmp[i] = y[i] + h2 * k2[i];
  d_deriv(lam + h2,    ytmp,  k3, m);
  for (int i = 0; i < N_STATE; ++i) ytmp[i] = y[i] + h  * k3[i];
  d_deriv(lam + h,     ytmp,  k4, m);
  for (int i = 0; i < N_STATE; ++i)
    y[i] += (h / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
}

/* ── kernel ────────────────────────────────────────────────────────────────── */

extern "C" __global__ void bh_rt_phase2_batch_kernel(
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
    double *out_disk_vph) {
  int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= n)
    return;

  double y[N_STATE];
  const double *src = y0 + ray * N_STATE;
  for (int k = 0; k < N_STATE; ++k)
    y[k] = src[k];

  const double pi_half = 1.5707963267948966;
  const double r_cap = 2.0 * m + r_horizon_epsilon;
  d_renormalize(y, m);

  double r_min_val = INFINITY;
  double lam = 0.0;
  int status = BH_RT_STATUS_MAX_STEPS;
  double termination_r = NAN;
  int steps_taken = 0;
  int broke = 0;
  int disk_hit = 0;
  double disk_r_cross = NAN;
  double disk_vt = NAN;
  double disk_vph = NAN;
  double previous_r = y[1];
  double previous_dth = y[2] - pi_half;
  double previous_vt = y[4];
  double previous_vph = y[7];

  for (int step_idx = 0; step_idx < max_steps; ++step_idx) {
    const double r = y[1];
    const double current_dth = y[2] - pi_half;
    if (!d_all_finite(y)) {
      status = BH_RT_STATUS_NUMERICAL_ERROR;
      termination_r = r;
      steps_taken = step_idx;
      broke = 1;
      break;
    }
    if (r < r_cap) {
      status = BH_RT_STATUS_CAPTURED;
      termination_r = r;
      steps_taken = step_idx;
      broke = 1;
      break;
    }
    if (r > r_escape) {
      status = BH_RT_STATUS_ESCAPED;
      termination_r = r;
      steps_taken = step_idx;
      broke = 1;
      break;
    }
    r_min_val = fmin(r_min_val, r);

    if (step_idx > 0 && !disk_hit && previous_dth * current_dth < 0.0) {
      double alpha =
          fabs(previous_dth) /
          (fabs(previous_dth) + fabs(current_dth) + 1e-30);
      disk_hit = 1;
      disk_r_cross = previous_r * (1.0 - alpha) + r * alpha;
      disk_vt = previous_vt * (1.0 - alpha) + y[4] * alpha;
      disk_vph = previous_vph * (1.0 - alpha) + y[7] * alpha;
    }
    previous_r = r;
    previous_dth = current_dth;
    previous_vt = y[4];
    previous_vph = y[7];

    d_rk4_step(lam, y, dlambda, m);
    if ((step_idx % 4) == 0)
      d_renormalize(y, m);
    lam += dlambda;
    steps_taken = step_idx + 1;
  }

  if (!broke)
    termination_r = y[1];
  if (status == BH_RT_STATUS_MAX_STEPS) {
    termination_r = y[1];
    if (isfinite(y[1]))
      r_min_val = fmin(r_min_val, y[1]);
    if (!d_all_finite(y))
      status = BH_RT_STATUS_NUMERICAL_ERROR;
  }
  if (!isfinite(r_min_val))
    r_min_val = NAN;

  out_status[ray] = status;
  out_steps_taken[ray] = steps_taken;
  out_termination_r[ray] = termination_r;
  out_r_min[ray] = r_min_val;
  out_disk_hit[ray] = disk_hit;
  out_disk_r_cross[ray] = disk_r_cross;
  out_disk_vt[ray] = disk_vt;
  out_disk_vph[ray] = disk_vph;
}

/* ── host-side launcher (C linkage, callable from Python ctypes or bridge) ── */

extern "C" int32_t bh_rt_schwarzschild_phase2_batch_trace_cuda(
    const double *h_y0,
    int n,
    double m,
    double dlambda,
    int max_steps,
    double r_escape,
    double r_horizon_epsilon,
    int *h_status,
    int *h_steps,
    double *h_term_r,
    double *h_r_min,
    int *h_disk_hit,
    double *h_disk_r_cross,
    double *h_disk_vt,
    double *h_disk_vph) {
  if (n < 0 || max_steps < 0 ||
      !(isfinite(m) && isfinite(dlambda) && isfinite(r_escape) &&
        isfinite(r_horizon_epsilon)) ||
      m <= 0.0 || dlambda <= 0.0 || r_escape <= 0.0 ||
      r_horizon_epsilon < 0.0)
    return static_cast<int32_t>(cudaErrorInvalidValue);
  if (n == 0)
    return static_cast<int32_t>(cudaSuccess);
  if (h_y0 == nullptr || h_status == nullptr || h_steps == nullptr ||
      h_term_r == nullptr || h_r_min == nullptr || h_disk_hit == nullptr ||
      h_disk_r_cross == nullptr || h_disk_vt == nullptr ||
      h_disk_vph == nullptr)
    return static_cast<int32_t>(cudaErrorInvalidValue);

  const size_t ray_count = static_cast<size_t>(n);
  const size_t state_bytes = N_STATE * sizeof(double);
  if (ray_count > static_cast<size_t>(-1) / state_bytes)
    return static_cast<int32_t>(cudaErrorInvalidValue);
  const size_t y0_bytes = ray_count * state_bytes;
  const size_t int_bytes = ray_count * sizeof(int);
  const size_t double_bytes = ray_count * sizeof(double);

  double *d_y0 = nullptr;
  int *d_status = nullptr;
  int *d_steps = nullptr;
  double *d_term_r = nullptr;
  double *d_r_min = nullptr;
  int *d_disk_hit = nullptr;
  double *d_disk_r_cross = nullptr;
  double *d_disk_vt = nullptr;
  double *d_disk_vph = nullptr;
  const int block = 128;
  const int grid = static_cast<int>(
      (ray_count + static_cast<size_t>(block) - 1) / static_cast<size_t>(block));
  int32_t result = static_cast<int32_t>(cudaSuccess);
  cudaError_t error = cudaSuccess;

  error = cudaMalloc(reinterpret_cast<void **>(&d_y0), y0_bytes);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMalloc(reinterpret_cast<void **>(&d_status), int_bytes);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMalloc(reinterpret_cast<void **>(&d_steps), int_bytes);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMalloc(reinterpret_cast<void **>(&d_term_r), double_bytes);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMalloc(reinterpret_cast<void **>(&d_r_min), double_bytes);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMalloc(reinterpret_cast<void **>(&d_disk_hit), int_bytes);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMalloc(reinterpret_cast<void **>(&d_disk_r_cross), double_bytes);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMalloc(reinterpret_cast<void **>(&d_disk_vt), double_bytes);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMalloc(reinterpret_cast<void **>(&d_disk_vph), double_bytes);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }

  error = cudaMemcpy(d_y0, h_y0, y0_bytes, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }

  bh_rt_phase2_batch_kernel<<<grid, block>>>(
      d_y0, n, m, dlambda, max_steps, r_escape, r_horizon_epsilon, d_status,
      d_steps, d_term_r, d_r_min, d_disk_hit, d_disk_r_cross, d_disk_vt,
      d_disk_vph);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }

  error = cudaMemcpy(h_status, d_status, int_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMemcpy(h_steps, d_steps, int_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMemcpy(h_term_r, d_term_r, double_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMemcpy(h_r_min, d_r_min, double_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMemcpy(
      h_disk_hit, d_disk_hit, int_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMemcpy(
      h_disk_r_cross, d_disk_r_cross, double_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMemcpy(h_disk_vt, d_disk_vt, double_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }
  error = cudaMemcpy(
      h_disk_vph, d_disk_vph, double_bytes, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    result = static_cast<int32_t>(error);
    goto cleanup;
  }

cleanup:
#define BH_RT_CUDA_FREE(pointer)                                              \
  do {                                                                        \
    if ((pointer) != nullptr) {                                               \
      error = cudaFree(pointer);                                              \
      if (result == static_cast<int32_t>(cudaSuccess) && error != cudaSuccess) \
        result = static_cast<int32_t>(error);                                \
    }                                                                         \
  } while (0)
  BH_RT_CUDA_FREE(d_disk_vph);
  BH_RT_CUDA_FREE(d_disk_vt);
  BH_RT_CUDA_FREE(d_disk_r_cross);
  BH_RT_CUDA_FREE(d_disk_hit);
  BH_RT_CUDA_FREE(d_r_min);
  BH_RT_CUDA_FREE(d_term_r);
  BH_RT_CUDA_FREE(d_steps);
  BH_RT_CUDA_FREE(d_status);
  BH_RT_CUDA_FREE(d_y0);
#undef BH_RT_CUDA_FREE
  return result;
}
