# Session starter — blackhole_ray_tracer

Copy the **Session block** below into Claude, Cursor Composer, or any agent thread at the start of a work session.
Persistent rules live in `.cursorrules` (Cursor) — this file is for per-session context only.

---

## Session block

You are assisting on **blackhole_ray_tracer**, a Python-first numerical null-geodesic ray tracer.

### Read first

1. `docs/OVERVIEW.md` — architecture, what is implemented vs scaffold.
2. `docs/ROADMAP.md` — phases, acceptance criteria, CLI reference, gaps checklist.
3. `.cursorrules` — hard rules that always apply (toolchain, diff discipline, naming).

### Ground truth

- **Implemented in Python:** `src/blackhole_ray_tracer/` — Phase 1 (`phase1.py`, `phase1_driver.py`, `phase1_image.py`, `phase1_tuning.py`) and Phase 2 (`phase2_christoffel.py`, `phase2_geodesic.py`, `phase2_camera.py`, `phase2_types.py`, `phase2_render.py`, `phase2_report.py`, `phase2_driver.py`).
- **`kernel/`, `bridge/`, `ml/`** may be empty scaffolds. List files there before referencing anything inside them.
- **`phase1.py`** owns the shared `rk4_step` — reuse it in Python code unless explicitly porting to the C kernel.

### Toolchain reminder

```
uv sync --group dev
uv run pytest
uv run ruff check src tests
```

### Workflow

1. State which **ROADMAP phase** and **acceptance rows** your change targets.
2. Cite **concrete file paths** in all explanations.
3. If uncertain about a file's contents or a function's existence — **check first, don't assume**.
4. If adding C / GPU / ML logic, draft the API in a comment or update `ROADMAP.md` before writing implementation.

---

## Today's goal

> *(Paste one sentence here — e.g. "Add kernel parity test skeleton" or "Fix Phase 2 preset defaults".)*
