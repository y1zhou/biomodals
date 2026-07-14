# Quick Biomodals App

Apply this common contract, then load specialized references from `SKILL.md`.

## Discovery and module shape

- Put the discoverable entrypoint at
  `src/biomodals/app/<category>/<tool>_app.py`; the CLI strips `_app`.
- Expose module-level `app`, Modal functions, and `submit_<tool>_task` there.
- Make the module docstring useful in `biomodals app help <tool>`: include the
  upstream URL, prerequisites, caveats, and outputs.
- Use a Google-style entrypoint docstring with signature-exact names under
  `Args:`; underscores become kebab-case flags.
- Keep top-level imports locally available. Import image-only packages inside
  Modal functions and mark intentional runtime imports for Ruff.

## Configuration and image

- Define `CONF = AppConfig(...)`, pin upstream, and reuse its environment, paths,
  GPU, and timeout.
- Build with `patch_image_for_helper(...)`, a Debian slim or pinned registry
  base, `.env(...)`, and `.uv_pip_install(...)`.
- Use `copy_patch_files=True` only for build-time helper imports. Include sibling
  modules as described in `multi-module.md`.
- Create `app = modal.App(CONF.name, image=image, tags=CONF.tags)`.
- Declare realistic timeout/CPU/memory; add GPU only where needed and bound
  environment overrides by those resources.

## Trust boundaries

Validate before the first remote call and again before upstream consumption:

- Require regular files, size/count limits, valid encoding, parseable structure,
  and the accepted alphabet/schema.
- Reject empty or duplicate logical IDs. Preserve display labels but derive
  stable internal IDs for paths and joins.
- Use `biomodals.helper.shell.sanitize_filename` for a user-derived path
  component, then check uniqueness. It is not shell escaping.
- Resolve derived paths below their intended root before writes, recursive
  deletion, or extraction.
- Pass argv lists; never interpolate user headers, names, paths, or sequences
  into shell commands.

## Volumes and outputs

- Use `CONF.mounts(model_volume=True)` and allow model writes only in setup; use
  the output volume for persistent run artifacts. Prefer app-specific subpaths
  and read-only inference mounts. Set custom `read_only` and `sub_path` together.
- Use `AppRunLayout`. Return primitive payloads and string paths. Workflow
  functions return `AppRunResult` with `VolumePath`; reserve compressed
  `InlineBytes` for small rerunnable archives. Keep entrypoints CLI-only.
- For short jobs, send bytes, work in a temporary directory, and
  `package_outputs(...)`. Use staged caches for resumable work.
- Derive a safe default run name, build local paths with
  `build_local_output_path`, and reject accidental overwrites.
- Prefix remote/local logs with `💊`/`🧬 `. Use non-capturing `run_command` modes
  when output is not consumed.

## Existing helpers and references

Prefer existing `biomodals.helper` APIs (`run_command`, `package_outputs`,
`warmup_directory`, `download_files`, `hash_string`, shared local-output helpers)
over local variants. Do not extract trivial one-use helpers.

Use these valid relative paths as examples:

- [AlphaFold3](../../../../src/biomodals/app/fold/alphafold3_app.py) and
  [Rosetta](../../../../src/biomodals/app/bioinfo/rosetta_app.py): conventional.
- [RFdiffusion](../../../../src/biomodals/app/design/rfdiffusion_app.py): durable
  workflow `VolumePath` output.
- [LigandMPNN](../../../../src/biomodals/app/design/ligandmpnn_app.py): small
  workflow `InlineBytes` output.

Set `depends_on_apps` only when composing apps through the workflow runtime.
