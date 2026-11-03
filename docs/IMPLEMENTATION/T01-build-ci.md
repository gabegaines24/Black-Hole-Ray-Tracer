# T01 — Build, packaging, and CI

**Status:** complete.

**Dependencies:** T00.

**Scope:** Make `uv` installation, native source paths, package data, CI quality gates, and kernel smoke builds reproducible.

**Acceptance:** CI covers supported Python versions, lint, typing, tests, wheel/sdist build, native smoke, and kernel demo.

**Evidence:** `.github/workflows/ci.yml`, `setup.py`, `MANIFEST.in`, `pyproject.toml`.
