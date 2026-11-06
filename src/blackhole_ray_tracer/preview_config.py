"""Shared Phase 2 preview / dashboard render configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .native_phase2 import batch_native_available
from .native_phase2_cuda import cuda_batch_available
from .phase2_ml import ml_surrogate_available
from .phase2_report import render_config_from_preset
from .phase2_types import DiskConfig, Phase2RenderConfig

PreviewBackend = Literal["auto", "python", "native", "cuda", "ml"]
PREVIEW_BACKENDS: tuple[PreviewBackend, ...] = ("auto", "python", "native", "cuda", "ml")


@dataclass(frozen=True, slots=True)
class PreviewBuildArgs:
    """Inputs for building a ``Phase2RenderConfig`` for preview or dashboard use."""

    preset: str | None = None
    width: int = 48
    height: int = 48
    backend: PreviewBackend = "auto"
    disk: bool = False
    sky_mode: str = "gradient"
    r_observer: float | None = None
    fov_deg: float | None = None
    observer_phi: float = 0.0
    ml_model_path: str | None = None
    supersample: int = 1


def backend_from_phase2_flags(
    *,
    native: bool = False,
    cuda: bool = False,
    ml: bool = False,
) -> PreviewBackend:
    """Map main/phase2_driver render flags to a preview backend choice."""
    if cuda:
        return "cuda"
    if native:
        return "native"
    if ml:
        return "ml"
    return "auto"


def resolve_preview_backend_flags(
    backend: PreviewBackend,
    *,
    ml_model_path: str | None = None,
) -> tuple[bool, bool, bool]:
    """Map a preview backend choice to ``(use_cuda, use_native, use_ml)`` flags.

    For ``auto``, priority matches ``phase2_render``: cuda → native → python.
    ML is never chosen automatically (requires an explicit model path or trained weights).
    """
    if backend == "python":
        return False, False, False
    if backend == "cuda":
        return True, False, False
    if backend == "native":
        return False, True, False
    if backend == "ml":
        return False, False, True
    # auto
    if cuda_batch_available():
        return True, False, False
    if batch_native_available():
        return False, True, False
    return False, False, False


def _apply_backend_fallback(
    use_cuda: bool,
    use_native: bool,
    use_ml: bool,
    *,
    ml_model_path: str | None,
) -> tuple[bool, bool, bool, list[str]]:
    """Disable unavailable backends and collect user-facing warning messages."""
    warnings: list[str] = []
    if use_cuda and not cuda_batch_available():
        warnings.append(
            "CUDA batch library not found — falling back. "
            "Build with `make -C kernel/cuda cuda_batch`."
        )
        use_cuda = False
    if use_native and not batch_native_available():
        warnings.append(
            "Native batch extension not available — falling back. "
            "Build with BLACKHOLE_BUILD_NATIVE=1."
        )
        use_native = False
    if use_ml:
        probe = Phase2RenderConfig(
            width=1,
            height=1,
            use_ml_surrogate=True,
            ml_model_path=ml_model_path,
        )
        if not ml_surrogate_available(probe):
            warnings.append(
                "ML surrogate model not found — falling back. "
                "Train with `uv run python -m ml.surrogate train` or pass --ml-model-path."
            )
            use_ml = False
    if use_cuda:
        use_native = False
        use_ml = False
    elif use_native:
        use_ml = False
    return use_cuda, use_native, use_ml, warnings


def build_preview_config(args: PreviewBuildArgs) -> tuple[Phase2RenderConfig, list[str]]:
    """Build a ``Phase2RenderConfig`` and any backend fallback warnings."""
    use_cuda, use_native, use_ml = resolve_preview_backend_flags(
        args.backend,
        ml_model_path=args.ml_model_path,
    )
    use_cuda, use_native, use_ml, warnings = _apply_backend_fallback(
        use_cuda,
        use_native,
        use_ml,
        ml_model_path=args.ml_model_path,
    )

    if args.preset is not None:
        cfg = render_config_from_preset(
            args.preset,
            sky_mode=args.sky_mode,
            use_native_phase2=use_native,
            use_cuda_batch=use_cuda,
            use_ml_surrogate=use_ml,
            ml_model_path=args.ml_model_path,
        )
    else:
        cfg = Phase2RenderConfig(
            width=args.width,
            height=args.height,
            m=1.0,
            dlambda=0.1,
            max_steps=4000,
            sky_mode=args.sky_mode,
            use_native_phase2=use_native,
            use_cuda_batch=use_cuda,
            use_ml_surrogate=use_ml,
            ml_model_path=args.ml_model_path,
        )

    overrides: dict[str, object] = {}
    if args.r_observer is not None:
        overrides["r_observer"] = args.r_observer
    if args.fov_deg is not None:
        overrides["fov_deg"] = args.fov_deg
    if args.observer_phi != 0.0:
        overrides["observer_phi"] = args.observer_phi
    if args.disk:
        overrides["disk"] = DiskConfig()
    if args.supersample > 1:
        overrides["supersample"] = args.supersample
    if args.sky_mode != cfg.sky_mode:
        overrides["sky_mode"] = args.sky_mode

    if overrides:
        import dataclasses

        cfg = dataclasses.replace(cfg, **overrides)

    return cfg, warnings


def describe_requested_backend(cfg: Phase2RenderConfig) -> str:
    """Human-readable label for the configured backend flags (before render)."""
    if (cfg.use_cuda_batch or cfg.use_cuda_phase2) and cuda_batch_available():
        return "cuda_batch"
    if cfg.use_native_phase2 and batch_native_available():
        return "native_batch"
    if cfg.use_ml_surrogate:
        return "ml_surrogate"
    return "python"


def add_preview_cli_args(parser) -> None:
    """Register shared preview / dashboard CLI flags on an ``ArgumentParser``."""
    parser.add_argument(
        "--preset",
        choices=("fast", "balanced", "quality"),
        default=None,
        help="Use a named render preset (overrides --width/--height).",
    )
    parser.add_argument("--width", type=int, default=48, help="Image width in pixels.")
    parser.add_argument("--height", type=int, default=48, help="Image height in pixels.")
    parser.add_argument(
        "--backend",
        choices=PREVIEW_BACKENDS,
        default="auto",
        help="Render backend: auto (cuda→native→python), or force python/native/cuda/ml.",
    )
    parser.add_argument(
        "--disk",
        action="store_true",
        help="Enable thin equatorial accretion disk overlay.",
    )
    parser.add_argument(
        "--sky",
        choices=("gradient", "flat"),
        default="gradient",
        help="Sky color mapping for escaped rays.",
    )
    parser.add_argument(
        "--r-observer",
        type=float,
        default=None,
        help="Observer radial coordinate (geometric units).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=None,
        help="Vertical field of view in degrees.",
    )
    parser.add_argument(
        "--observer-phi",
        type=float,
        default=0.0,
        help="Observer azimuth φ in radians.",
    )
    parser.add_argument(
        "--ml-model-path",
        type=str,
        default=None,
        help="Path to trained surrogate .npz (with --backend ml).",
    )
    parser.add_argument(
        "--aa",
        "--supersample",
        dest="supersample",
        type=int,
        default=1,
        metavar="S",
        help="Anti-aliasing super-sample factor (render at S× then box-average).",
    )


def preview_args_from_namespace(args) -> PreviewBuildArgs:
    """Convert an ``argparse`` namespace (with ``add_preview_cli_args``) to ``PreviewBuildArgs``."""
    return PreviewBuildArgs(
        preset=getattr(args, "preset", None),
        width=getattr(args, "width", 48),
        height=getattr(args, "height", 48),
        backend=getattr(args, "backend", "auto"),
        disk=getattr(args, "disk", False),
        sky_mode=getattr(args, "sky", "gradient"),
        r_observer=getattr(args, "r_observer", None),
        fov_deg=getattr(args, "fov", None),
        observer_phi=getattr(args, "observer_phi", 0.0),
        ml_model_path=getattr(args, "ml_model_path", None),
        supersample=getattr(args, "supersample", 1),
    )
