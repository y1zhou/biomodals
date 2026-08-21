# Biomodals App Development

Detailed app-development instructions for `src/biomodals/app/**/*_app.py` live in the repo-local skill:

- `.agents/skills/biomodals-app-development/SKILL.md`

The skill routes app work to focused references under
`.agents/skills/biomodals-app-development/references/`:

- `quick-app.md` for the common app contract and trust boundaries.
- `multi-module.md` for complex app structure and image source inclusion.
- `execution-kernel.md` for durable scheduling and remote coordinator boundaries.
- `staged-cache.md` for durable identities and publication.
- `fanout-resume.md` for concurrency, durable fan-out, and resumption.
- `upstream-patching.md` for upstream safety and scientific equivalence.
- `testing.md` for the app test pyramid and verification.

## How Agents Should Use It

- Invoke or read the `biomodals-app-development` skill before creating, editing, or reviewing Biomodals app files.
- Treat the skill as the maintained source for app discovery, `AppConfig`, Modal
  image construction, helper usage, trust boundaries, volumes, staged caches,
  data flow, local entrypoint docstrings, examples, and smoke tests.
- When adding workflow-compatible app functions, also follow
  `docs/agents/workflow-development.md`.

## Maintenance

- Update the skill when app-development standards change.
- Keep this document as a pointer and coordination note, not a duplicate copy of the skill.
- If an app needs to intentionally deviate from the skill, add a focused note under `docs/agents/` explaining why and link it from this document.
- Keep local entrypoints CLI-only. Workflow reuse should happen through remote
  app functions that return shared schemas from `biomodals.schema`.

## Approved App-Specific Deviations

- [AlphaFold3 cache integrity and run layout](alphafold3-app-deviations.md)
  documents the intentionally trusted immutable stores, marker-only seed cache,
  and multi-request run layout used by `alphafold3_app.py`.
- [GROMACS analysis checkpoints](gromacs-analysis-checkpoints.md) retain the
  established run-directory and timestamp contract for narrow CSV/PNG restart
  repair. Do not extend that exception to new cache stages.
