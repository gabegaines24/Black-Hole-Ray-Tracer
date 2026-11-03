from typing import Any


def schwarzschild_phase2_trace(
    y0: Any,
    m: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    r_horizon_epsilon: float = ...,
) -> dict[str, Any]: ...


def schwarzschild_phase2_batch_trace(
    y0_batch: Any,
    m: float,
    dlambda: float,
    max_steps: int,
    r_escape: float,
    r_horizon_epsilon: float = ...,
) -> dict[str, Any]: ...
