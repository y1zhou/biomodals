# Biomodals Agent Instructions

## Repository expectations

- This is a Python 3.12+ project for running bioinformatics tools on Modal.
- Prefer `uv run ...` for project commands and `biomodals` CLI smoke tests.
- CI runs `prek` against `.pre-commit-config.yaml`; after code edits, run `prek run --files <changed files>` when practical.
- For CLI or app-discovery changes, smoke test with `uv run biomodals list` and `uv run biomodals help <app-name>` when practical.
- Keep generated archives, large run outputs, Modal result directories, and local test data out of commits unless the user explicitly asks for them.

## Instruction maintenance

- Keep root `AGENTS.md` focused on repo-wide expectations. Put long-form context in linked docs under `docs/agents/` or narrower instruction files when that context only applies to a subtree.
- If app-development instructions and reference apps conflict in a way not covered below, ask for clarification before changing app behavior or updating standards.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for `y1zhou/biomodals`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five-label triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses a single-context domain doc layout. See `docs/agents/domain.md`.

### Modal platform

This repo is built on Modal, a serverless cloud platform for running Python code. See `docs/agents/modal.md`.

### Workflow development

When creating, editing, or reviewing files under `src/biomodals/workflow/`, read `docs/agents/workflow-development.md` first.

## Biomodals app development

When creating, editing, or reviewing files under `src/biomodals/app/**/*_app.py`, read `.github/instructions/app-development.instructions.md` first.
Use these apps as the current implementation standards:

- `src/biomodals/app/fold/alphafold3_app.py`
- `src/biomodals/app/bioinfo/rosetta_app.py`
- `src/biomodals/app/design/boltzgen_app.py`

### Core app conventions

- App files must be named `<toolname>_app.py` and live under the appropriate app category directory.
- Keep module docstrings user-facing; include upstream source links, important prerequisites, and output behavior.
- Add `# ruff: noqa: PLC0415` near the top because Modal functions often need runtime-only imports.
- Use `AppConfig` for new apps. Pin either `repo_commit_hash` or `version`, and allow `gpu` and `timeout` to be overridden through environment variables when applicable.
- Build images with `patch_image_for_helper(...)` so the runtime has Biomodals helper code and shell tooling available.
- Prefer helpers from `biomodals.app.helper` and `biomodals.app.helper.shell` over reimplementing shell execution, packaging, downloads, file copying, warming, and hashing.
- Use `@app.local_entrypoint()` functions named `submit_<toolname>_task(...)`, with Google-style `Args:` docstrings so `biomodals help` can render CLI documentation.
- Use `🧬` for local entrypoint status messages and `💊` for remote Modal-container status messages.

### Reference-app clarifications

The instruction file describes the baseline, but the reference apps show accepted patterns for app-specific tradeoffs:

- Use an `AppInfo` dataclass only when it improves readability by grouping several related constants. For a small number of simple constants, module-level constants like `OUT_VOLUME`, `OUTPUTS_DIR`, or `ROSETTA_DIR` are acceptable and can be easier to maintain.
- Data flow depends on the app category. Short-lived inference should usually send local input bytes to a remote function and return tarball bytes directly. Long-running apps should cache intermediate and final results in Modal volumes. Parallel or interruptible runs should use queues, locks, stable run IDs, and resumable runners where possible.
- Before choosing a data flow for a new app, ask the user whether the app is short-lived inference, long-running/cached, or parallel/resumable.
- Imports required only inside the Modal runtime image, and not declared as dependencies of the `biomodals` package, must stay inside the function or method that uses them. Top-level imports are acceptable when the dependency is part of the `biomodals` package dependencies and is used by multiple local functions.
- Mount model volumes read-only for inference when the inference code only reads model artifacts. Writable model mounts are exceptions for tools that write runtime caches into the model directory, such as AlphaFold3's JAX cache.

### Output and volume patterns

- Return `.tar.zst` bytes with `package_outputs(...)` for quick jobs where direct download is practical.
- Use `CONF.get_out_volume()` or shared volumes for outputs that need persistence, resumability, batching, or later retrieval.
- Commit Modal volumes after writing cache entries, model downloads, uploaded inputs, or intermediate outputs.
- Use deterministic cache keys from `hash_string()` for expensive reusable work, with sharded paths such as `<AppName>/<hash[:2]>/<hash>/`.
- For user-facing local output, resolve `out_dir`, create it if needed, avoid overwriting existing tarballs unless explicitly intended, and print the final path or Modal volume location.

When developing new apps that must violate these conventions for good reason, document the reason for the deviation in the `docs/agents/` directory, and refer to it in this `AGENTS.md` file.
