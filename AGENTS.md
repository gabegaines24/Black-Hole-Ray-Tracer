# Agent instructions — blackhole_ray_tracer

Human and AI collaborators: read **`docs/OVERVIEW.md`** and **`docs/ROADMAP.md`** before architectural or multi-file changes.

**Cursor:** persistent rules live in **`.cursorrules`** (repo root). Per-session goals: **`docs/AGENT_PROMPT.md`**.

## Quick rules

1. **Truth about layout:** `kernel/`, `bridge/`, `ml/` are **planned** until source files appear there — do not describe them as implemented.
2. **Phases:** Keep `phase1` (equatorial / shared RK4) and `phase2` (3D prototype) modular; extend via `phaseN_*`, `kernel/`, or documented APIs.
3. **Shared integrator:** Prefer `phase1.rk4_step` for Python RK4 unless migrating to C.
4. **Quality bar:** Run `uv run pytest` after substantive Python changes; `uv run ruff check src tests` when convenient.
5. **Artifacts:** Never commit tracked `*.ppm` (ignored in `.gitignore`) or `__pycache__/`.
6. **Diffs:** Smallest correct change only; avoid drive-by refactors.
7. **Uncertainty:** Say when unsure and read the repo rather than hallucinating filenames or formulas.
8. **Future Kerr docs:** Prefer **Boyer–Lindquist** naming when documenting coordinate extensions.

Copy-paste session starter for Claude/composer-sized tasks: **`docs/AGENT_PROMPT.md`**.
