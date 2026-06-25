---
name: biomodals-app-development
description: Biomodals Modal app development standards. Use when Codex is creating, editing, reviewing, or scaffolding files under src/biomodals/app/**/*_app.py, including app discovery, AppConfig usage, Modal image construction, helper APIs, volumes, data flow, local entrypoint CLI docstrings, examples, and smoke-test expectations.
---

# Biomodals App Development

## Core Workflow

Use this skill for Biomodals app files named `*_app.py`. Before non-trivial
changes, read the current repo guidance:

- `references/app-development.md` for the app-development standards and checklist.
- `docs/agents/app-development.md` for repo-level coordination notes and deviation links.
- The closest reference apps for current patterns:
  - `src/biomodals/app/fold/alphafold3_app.py`
  - `src/biomodals/app/bioinfo/rosetta_app.py`
  - `src/biomodals/app/design/boltzgen_app.py`
  - `src/biomodals/app/design/rfdiffusion_app.py` for durable,
    workflow-compatible app outputs backed by an app output volume.
  - `src/biomodals/app/design/ligandmpnn_app.py` for fast,
    workflow-compatible rerunnable app outputs returned as inline zstd bytes.

For new apps or major output changes, use the reference before choosing the
data flow. If workflow compatibility is needed, coordinate with
`biomodals-workflow-development`.

## Core Guardrails

- Keep code compatible with `biomodals app list` and `biomodals app help`.
- Use `AppConfig`, `patch_image_for_helper(...)`, existing `biomodals.helper`
  APIs, and `CONF.mounts(...)` before adding local variants.
- Keep Modal returns primitive when practical. Return paths as strings, not
  `Path` objects.
- Workflow-compatible app functions return `AppRunResult`; standalone local
  entrypoints stay CLI-only.
- Add or update examples and focused tests when behavior or invocation changes.
- Run `prek run --files <changed files>` when practical; for CLI/discovery
  changes also smoke test `uv run biomodals app list` and
  `uv run biomodals app help <app-name>`.
