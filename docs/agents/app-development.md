# Biomodals App Development

Detailed app-development instructions for `src/biomodals/app/**/*_app.py` live in the repo-local skill:

- `.agents/skills/biomodals-app-development/SKILL.md`
- `.agents/skills/biomodals-app-development/references/app-development.md`

The previous `.github/instructions/app-development.instructions.md` file has been removed so app standards have one maintained source.

## How Agents Should Use It

- Invoke or read the `biomodals-app-development` skill before creating, editing, or reviewing Biomodals app files.
- Treat the skill as the baseline for app discovery, `AppConfig`, Modal image construction, helper usage, volumes, data flow, local entrypoint docstrings, examples, and smoke tests.
- For app model/output volumes, prefer `CONF.mounts(...)`. For shared Modal
  volumes with custom mountpoints, mount only the needed subdirectory with
  `Volume.with_mount_options(sub_path=...)` and combine read-only and subpath
  options in the same call when inference should not write to model artifacts.
- Treat `AppConfig` as a shared schema from `biomodals.schema.app`; keep
  Modal-specific volume and image helpers outside `biomodals.schema`.
- Compare non-trivial app changes against the current reference apps:
  - `src/biomodals/app/fold/alphafold3_app.py`
  - `src/biomodals/app/bioinfo/rosetta_app.py`
  - `src/biomodals/app/design/boltzgen_app.py`
- When adding workflow-compatible app functions, also follow
  `docs/agents/workflow-development.md`.

## Maintenance

- Update the skill when app-development standards change.
- Keep this document as a pointer and coordination note, not a duplicate copy of the skill.
- If an app needs to intentionally deviate from the skill, add a focused note under `docs/agents/` explaining why and link it from this document.
- Keep local entrypoints CLI-only. Workflow reuse should happen through remote
  app functions that return shared schemas from `biomodals.schema`.
