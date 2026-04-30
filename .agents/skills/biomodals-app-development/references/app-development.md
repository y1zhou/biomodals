# Biomodals App Development Reference

This reference is the maintained app-development standard for files under `src/biomodals/app/**/*_app.py`.

## Discovery And File Shape

Biomodals apps are self-contained Modal applications wrapping bioinformatics tools. They live under `src/biomodals/app/<category>/`.

- Name app files `<toolname>_app.py`; the `_app.py` suffix is how `cli.py` discovers apps with `APP_HOME.glob("*/*_app.py")`.
- Place apps in an appropriate category such as `fold/`, `design/`, `score/`, or `bioinfo/`.
- The CLI app name is the filename stem with `_app` stripped, for example `protenix_app.py` becomes `protenix`.
- Use section banners to keep modules scan-friendly:
  - module docstring
  - imports
  - `# Modal configs`
  - `# Image and app definitions`
  - optional fetch/download setup
  - inference functions
  - local entrypoint

## Module Docstring

The module docstring is rendered verbatim by `biomodals help <app>` as Markdown. Keep it user-facing and include the upstream source URL, important prerequisites, caveats, and output behavior.

Typical shape:

```python
"""<Tool name> source repo: <https://github.com/...>.

## Configuration

## Additional notes

## Outputs
"""
```

Use optional configuration tables only when the local entrypoint docstring is insufficient.

## Imports And Ruff

- Add `# ruff: noqa: PLC0415` near the top because Modal functions often need runtime-only imports inside function bodies.
- Keep top-level imports to stdlib, `modal`, `Path`, and Biomodals app/helper dependencies when possible.
- Imports required only inside the Modal runtime image, and not declared as Biomodals package dependencies, must stay inside the function or method that uses them.
- Top-level imports are acceptable when the dependency is in Biomodals package dependencies and used by multiple local functions.

## AppConfig

New apps should define module-level `CONF = AppConfig(...)`.

```python
from biomodals.app.config import AppConfig

CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="ToolName",
    repo_url="https://github.com/...",
    repo_commit_hash="abc123...",
    package_name="toolname",
    version="1.0.0",
    python_version="3.11",
    cuda_version="cu128",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "3600")),
)
```

Rules:

- Pin either `repo_commit_hash` or `version`, or both.
- Let `gpu` and `timeout` be overridden by environment variables with sensible defaults.
- Use `CONF.default_env` when setting image environment variables. It provides standard UV, Hugging Face, Torch, and torch backend environment.
- Use `CONF.model_dir`, `CONF.git_clone_dir`, `CONF.model_volume_mountpoint`, and related fields instead of hardcoded paths.

Use an `AppInfo` dataclass only when grouping several related app constants improves readability. For a few simple constants, module-level constants such as `OUT_VOLUME` or `OUTPUTS_DIR` are acceptable.

## Image Construction

- Build images through `patch_image_for_helper(...)` from `biomodals.helper`; it injects helper modules plus shell tooling needed by helpers such as `package_outputs`.
- Prefer `modal.Image.debian_slim()` or `modal.Image.from_registry()` as a base.
- Use `.env(CONF.default_env | {...})` to merge app-specific environment with defaults.
- Use `.uv_pip_install(...)` for Python dependencies.
- Pass `copy_patch_files=True` only when later image build steps depend on helper code.

Pattern:

```python
runtime_image = patch_image_for_helper(
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .apt_install("git", "build-essential", "zstd")
    .env(CONF.default_env | {"CUSTOM_VAR": "value"})
    .uv_pip_install(f"git+{CONF.repo_url}@{CONF.repo_commit_hash}")
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
```

## Volumes

- Import shared volumes from `biomodals.app.constant`, such as `MODEL_VOLUME` or `MSA_CACHE_VOLUME`.
- Mount model weights read-only for inference when the function only reads model artifacts.
- Use `CONF.model_volume_mountpoint` for model volume mount paths.
- Commit volume changes explicitly after writes with `VOLUME.commit()`.
- Use `CONF.get_out_volume()` for app-specific persistent outputs.

## Remote Functions

Always specify a timeout with `CONF.timeout` or `MAX_TIMEOUT`. Add resource hints:

- GPU inference: `gpu=CONF.gpu`, `cpu=(min, max)`, `memory=(min, max)`.
- CPU-only data pipelines omit `gpu`.
- Unless there is a reason to reserve more, CPU minimum should be `0.125`.
- Long-running GPU inference should usually use `MAX_TIMEOUT` and read-only model mounts.
- Use `TemporaryDirectory` or `mkdtemp` for working directories.
- Return output tarball bytes with `package_outputs(output_dir)` for quick jobs.

Resource pattern:

```python
@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=MAX_TIMEOUT,
    volumes={CONF.model_volume_mountpoint: MODEL_VOLUME.read_only()},
)
```

## Helper APIs

Prefer existing helpers instead of reimplementing common behavior:

- `run_command(cmd)` from `biomodals.helper.shell` for streaming shell commands.
- `run_command_with_log(cmd, log_file)` for command logging.
- `run_background_command(cmd)` for non-blocking subprocesses.
- `package_outputs(root)` for `.tar.zst` bytes.
- `copy_files(mapping)` for parallel file copying.
- `find_with_fd(dir, pattern)` for fast file search.
- `warmup_directory(dir)` for pre-caching files.
- `softlink_dir(src, dst)` for tool-expected symlink layouts.
- `download_files(urls)` from `biomodals.helper.web` for async concurrent downloads.
- `struct2seq(path)` from `biomodals.helper.structure` for PDB/CIF sequence extraction.
- `hash_string(s)` from `biomodals.helper` for cache keys.
- `patch_image_for_helper(image)` from `biomodals.helper` for Modal images.

## Local Entrypoint

The `@app.local_entrypoint()` function is the user-facing orchestration layer on the local machine.

- Name it `submit_<toolname>_task(...)`.
- Validate input paths before remote calls.
- Default `run_name` from input stem when applicable.
- Resolve and create `out_dir`; avoid accidental overwrites.
- Read local inputs as bytes and pass bytes to remote functions for quick jobs.
- Write returned tarball bytes locally.
- Print final local path or Modal volume location.

Docstring rules for `biomodals help`:

- Use Google-style docstrings with an `Args:` section.
- Put `Args:` on its own line.
- Start each argument line at the first indentation level as `name: description`.
- Indent continuation lines at double the argument indentation.
- Match docstring argument names to the function signature.
- Signature uses underscores; CLI flags become kebab-case.

## Data Flow

Standard quick-job flow:

1. User passes file paths to the local entrypoint.
2. Entrypoint reads files as bytes and calls remote function(s).
3. Remote function writes bytes to a temp directory, runs the tool, packages outputs with `package_outputs()`, and returns bytes.
4. Entrypoint writes returned `.tar.zst` bytes to local output.

Choose architecture by job type:

- Short-lived inference usually sends local input bytes to remote functions and returns tarball bytes directly.
- Long-running apps should cache intermediate and final results in Modal volumes.
- Parallel or interruptible runs should use queues, locks, stable run IDs, and resumable runners where possible.

Before choosing data flow for a new app, ask whether it is short-lived inference, long-running/cached, or parallel/resumable unless already clear from the request.

## Caching

- Use `hash_string()` on input sequences or content for deterministic cache keys.
- Store cached artifacts under sharded paths such as `<AppName>/<hash[:2]>/<hash>/`.
- Check cache before expensive work and return early when possible.
- Commit volume changes after writing cache entries.

## Legacy Apps

Older apps can use raw constants such as `GPU`, `TIMEOUT`, and `APP_NAME`. When touching legacy files, prefer migrating to `AppConfig` plus optional `AppInfo` if the change scope permits. New apps should use `AppConfig`.

## Examples And Verification

- When app development changes invocation or adds a new app, add or update an example bash script under `examples/app/` using `biomodals run`.
- Use small example inputs under `examples/data/` only when existing data is insufficient.
- After edits, run `prek run --files <changed files>` when practical.
- For CLI or app discovery changes, smoke test `uv run biomodals list` and `uv run biomodals help <app-name>` when practical.
