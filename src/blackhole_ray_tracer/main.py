"""CLI entrypoint for project bootstrap validation."""

from rich.console import Console


def main() -> None:
    """Run a minimal startup check."""
    console = Console()
    console.print("blackhole-ray-tracer bootstrap ready.")
