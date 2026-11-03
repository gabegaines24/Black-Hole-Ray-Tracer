# Agent instructions — blackhole_ray_tracer

Human and AI collaborators: read **`docs/OVERVIEW.md`** and **`docs/ROADMAP.md`** before architectural or multi-file changes.

**Cursor:** persistent rules live in **`.cursorrules`** (repo root). Per-session goals: **`docs/AGENT_PROMPT.md`**.

## Quick rules

1. **Truth about layout:** Inspect `kernel/`, `bridge/`, and `ml/` before describing contents. Current source exists in all three; future files remain planned until present.
2. **Phases:** Keep Phase 1 equatorial/shared RK4, Phase 2 3D Schwarzschild, and Phase 3 Python Kerr modular.
3. **Shared integrator:** Prefer `phase1.rk4_step` for Python RK4 unless migrating to C.
4. **Quality bar:** Run `uv run pytest` after substantive Python changes; run `uv run ruff check src tests ml` and `uv run mypy src ml` when convenient.
5. **Artifacts:** Never commit tracked `*.ppm` (ignored in `.gitignore`) or `__pycache__/`.
6. **Diffs:** Smallest correct change only; avoid drive-by refactors.
7. **Uncertainty:** Say when unsure and read the repo rather than hallucinating filenames or formulas.
8. **Future Kerr docs:** Prefer **Boyer–Lindquist** naming when documenting coordinate extensions.

Copy-paste session starter for Claude/composer-sized tasks: **`docs/AGENT_PROMPT.md`**.
