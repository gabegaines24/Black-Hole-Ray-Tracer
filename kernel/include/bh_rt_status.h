/* Shared kernel status codes.
 *
 * Values intentionally match blackhole_ray_tracer.phase1.RayStatus ordering
 * and docs/STATE_API.md:
 *   captured=0, escaped=1, max_steps=2, numerical_error=3.
 */
#pragma once

#define BH_RT_STATUS_CAPTURED         0  /* r fell inside capture threshold */
#define BH_RT_STATUS_ESCAPED          1  /* r exceeded escape threshold     */
#define BH_RT_STATUS_MAX_STEPS        2  /* step budget exhausted           */
#define BH_RT_STATUS_NUMERICAL_ERROR  3  /* NaN / non-finite state          */
