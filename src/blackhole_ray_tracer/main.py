"""CLI entrypoint for bootstrap and Phase 1 experiments."""

import argparse
from rich.console import Console

from .phase1 import summarize_phase1_a_b


def main() -> None:
    """Run command-line tasks for the project."""
    parser = argparse.ArgumentParser(prog="blackhole-ray-tracer")
    parser.add_argument(
        "--phase1-ab",
        action="store_true",
        help="Run Step A (RK4 sanity) and Step B (single Schwarzschild ray).",
    )
    args = parser.parse_args()

    console = Console()
    if args.phase1_ab:
        console.print(summarize_phase1_a_b())
        return
    console.print("blackhole-ray-tracer bootstrap ready. Use --phase1-ab to run Step A+B.")
