"""Top-level package for blackhole_ray_tracer."""

__all__ = [
    "__version__",
    # Phase 1
    "RayStatus",
    "rk4_step",
    # Phase 2 — Schwarzschild
    "Phase2RenderConfig",
    "GeodesicTraceResult",
    "DiskConfig",
    "render_schwarzschild_3d_image",
    "trace_null_geodesic_3d",
    # Phase 3 — Kerr
    "KerrRenderConfig",
    "render_kerr_3d_image",
    "trace_kerr_null_geodesic",
    "kerr_sigma",
    "kerr_delta",
    "kerr_conserved",
]

__version__ = "0.1.0"
