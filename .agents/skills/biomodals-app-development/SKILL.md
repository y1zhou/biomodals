---
name: biomodals-app-development
description: Biomodals Modal app development standards. Use when Codex is creating, editing, reviewing, or scaffolding files under src/biomodals/app/**/*_app.py, including app discovery, AppConfig usage, Modal image construction, helper APIs, volumes, data flow, local entrypoint CLI docstrings, examples, and smoke-test expectations.
---

# Biomodals App Development

## Core Workflow

Use this skill for Biomodals app files named `*_app.py`.

Before changing behavior, read the current repo guidance:

- `references/app-development.md` for the app-development standards and checklist.
- `docs/agents/app-development.md` for repo-level coordination notes and deviation links.
- The closest reference apps for current patterns:
  - `src/biomodals/app/fold/alphafold3_app.py`
  - `src/biomodals/app/bioinfo/rosetta_app.py`
  - `src/biomodals/app/design/boltzgen_app.py`

For new apps, ask the user which data-flow class applies before choosing architecture: short-lived inference, long-running/cached, or parallel/resumable. If the user already gave enough context, state the classification and proceed.

## Implementation Rules

Keep app code compatible with `biomodals help` and app discovery:

- Name files `<toolname>_app.py` under `src/biomodals/app/<category>/`.
- Use a user-facing module docstring with upstream links, prerequisites, and output behavior.
- Add `# ruff: noqa: PLC0415` near the top.
- Use module-level `CONF = AppConfig(...)` for new apps; pin `repo_commit_hash` or `version`.
- Let `gpu` and `timeout` be overridden from `os.environ`.
- Build runtime images through `patch_image_for_helper(...)`.
- Prefer helpers from `biomodals.helper` and `biomodals.helper.shell` instead of open-coded shell, archive, copy, download, hashing, or warmup logic.
- Name local entrypoints `submit_<toolname>_task(...)` and use Google-style `Args:` docstrings so `biomodals help <app>` renders flags.
- Use `🧬` for local entrypoint status messages and `💊` for remote Modal-container status messages.
- Add or update an example command under `examples/app/` when app behavior or invocation changes.

## Review Checklist

When reviewing or finishing an app change, check:

- Discovery: path, filename, app name, and local entrypoint name match CLI expectations.
- Reproducibility: upstream version or commit is pinned.
- Runtime boundaries: dependencies used only inside Modal images stay lazily imported.
- Volumes: model volumes are read-only for inference unless the tool writes caches there; writable volumes are committed after writes.
- Data flow: quick jobs return `.tar.zst` bytes via `package_outputs(...)`; persistent, resumable, or batch jobs use `CONF.get_out_volume()` or shared volumes.
- Output safety: local output directories are created, existing tarballs are not overwritten accidentally, and final paths or Modal volume locations are printed.
- CLI docs: local entrypoint docstrings use exact Google-style `Args:` formatting with continuation indentation.
- Verification: run `prek run --files <changed files>` when practical, plus `uv run biomodals list` and `uv run biomodals help <app-name>` for CLI or discovery changes.

## Reference

Load `references/app-development.md` when implementing a non-trivial app, reviewing details, or checking exact patterns for AppConfig, image construction, volumes, remote functions, helper APIs, entrypoints, data flow, caching, legacy migration, and examples.
