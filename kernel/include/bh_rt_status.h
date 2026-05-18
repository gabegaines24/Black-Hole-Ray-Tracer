/* Shared kernel status codes.
 *
 * Values intentionally match blackhole_ray_tracer.phase1.RayStatus ordering
 * used by Python parity tests:
 *   captured=0, escaped=1, max_steps=2, numerical_error=3.
 */
#pragma once

#define BH_RT_STATUS_CAPTURED         0
#define BH_RT_STATUS_ESCAPED          1
#define BH_RT_STATUS_MAX_STEPS        2
#define BH_RT_STATUS_NUMERICAL_ERROR  3
