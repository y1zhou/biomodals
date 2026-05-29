# Biomodals Agent Instructions

## Repository expectations

- This is a Python 3.12+ project for running bioinformatics tools on Modal.
- Prefer `uv run ...` for project commands and `biomodals` CLI smoke tests. Never run `python ...` directly.
- Use `polars` for tabular data parsing and writing in Python code; avoid new
  `csv` or `pandas` parsing unless an upstream tool API specifically requires it.
- Use `orjson` for non-Pydantic JSON serialization and deserialization. For
  Pydantic models, serialize with `model_dump_json()` and parse JSON bytes or
  strings with `model_validate_json(...)`.
- CI runs `prek` against `.pre-commit-config.yaml`; after code edits, run `prek run --files <changed files>` when practical.
- For CLI or app-discovery changes, smoke test with `uv run biomodals app list`, `uv run biomodals app help <app-name>`, and `uv run biomodals workflow list` when practical.
- Keep generated archives, large run outputs, Modal result directories, and local test data out of commits unless the user explicitly asks for them.
- Avoid extracting trivial helper functions that are only a couple of lines and used once or twice; inline the logic and add comments when that is clearer.

## Instruction maintenance

- Keep root `AGENTS.md` focused on repo-wide expectations. Put long-form context in linked docs under `docs/agents/` or narrower instruction files when that context only applies to a subtree.
- If the app-development skill and reference apps conflict in a way not covered by repo docs, ask for clarification before changing app behavior or updating standards.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues for `y1zhou/biomodals`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five-label triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses a single-context domain doc layout. See `docs/agents/domain.md`.

### Modal platform

This repo is built on Modal, a serverless cloud platform for running Python code. See `docs/agents/modal.md`.

### App development

When creating, editing, or reviewing files under `src/biomodals/app/**/*_app.py`, use the repo-local `biomodals-app-development` skill. See `docs/agents/app-development.md`.

### Workflow development

When creating or editing reusable workflow runtime code under
`src/biomodals/workflow/` or shared workflow schemas under
`src/biomodals/schema/`, use the repo-local `biomodals-workflow-development`
skill. See `docs/agents/workflow-development.md`.

## Biomodals app development

The detailed app-development standards are consolidated in `.agents/skills/biomodals-app-development/`.

Use these apps as current implementation references:

- `src/biomodals/app/fold/alphafold3_app.py`
- `src/biomodals/app/bioinfo/rosetta_app.py`
- `src/biomodals/app/design/boltzgen_app.py`

When developing new apps that must violate the skill's conventions for good reason, document the reason for the deviation in `docs/agents/` and link that note from `docs/agents/app-development.md`.

## Biomodals workflow development

The detailed workflow-development standards are consolidated in `.agents/skills/biomodals-workflow-development/`.

Use `src/biomodals/workflow/shortmd_workflow.py` as the current executable workflow reference. Ignore `src/biomodals/workflow/ppiflow_workflow.py` as a reference pattern for now because it is expected to be refactored.
