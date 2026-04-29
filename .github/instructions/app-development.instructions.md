---
description: 'Use when creating, editing, or reviewing biomodals app files (*_app.py). Covers Modal app structure, AppConfig usage, helper imports, image building, entrypoint docstring format, and CLI compatibility. Use when: writing a new app, adding features to an existing app, reviewing app code, or scaffolding a Modal function.'
applyTo: src/biomodals/app/**/*_app.py
---

# Biomodals App Development Guidelines

Every biomodals app is a self-contained Modal application that wraps a bioinformatics tool. Apps live under `src/biomodals/app/<category>/` and follow a strict structure so the CLI (`biomodals help`, `biomodals run`) can discover and document them automatically.

## File Naming and Discovery

- App files **must** be named `<toolname>_app.py` (the `_app.py` suffix is how `cli.py` discovers apps via `APP_HOME.glob("*/*_app.py")`).
- Place the file under the appropriate category subfolder: `fold/`, `design/`, `score/`, or `bioinfo/`.
- The app name exposed to the CLI is the filename stem with `_app` stripped (e.g. `protenix_app.py` → `protenix`).

## Module Docstring

The module docstring is rendered verbatim by `biomodals help <app>` as Markdown. Structure it as:

```python
"""<Tool name> source repo: <https://github.com/...>.

## Configuration          # (optional) table of CLI flags - only for apps
                          # where the local_entrypoint docstring is insufficient

## Additional notes       # (optional) prerequisites, caveats, known limitations

## Outputs                # describe what the user receives (tarball contents, etc.)
"""
```

Keep it user-facing: explain what the tool does, what inputs it needs, and what outputs it produces. Reference URLs for upstream docs.

## Ruff Noqa and Imports

- Always add `# ruff: noqa: PLC0415` near the top to suppress "import not at top of file" warnings — Modal functions need lazy imports inside function bodies.
- Top-level imports: `os`, `modal`, `Path`, and anything from `biomodals.app.*`.
- Imports required only inside the Modal runtime image, and not declared as dependencies of the `biomodals` package, must stay inside the function or method that uses them. Top-level imports are acceptable when the dependency is part of the `biomodals` package dependencies and is used by multiple local functions.

## Section Order and Separator Comments

Organise the file into clearly separated sections using `##########################################` banners:

```
1. Module docstring
2. Top-level imports (stdlib, modal, biomodals helpers)
3. # Modal configs              — CONF, AppInfo dataclass, derived constants
4. # Image and app definitions  — runtime_image, app = modal.App(...)
5. # Fetch model weights        — (optional) download function
6. # Inference functions         — the core remote functions
7. # Entrypoint for ephemeral usage — @app.local_entrypoint()
```

## AppConfig (CONF)

Always define a module-level `CONF` using `AppConfig` from `biomodals.app.config`:

```python
from biomodals.app.config import AppConfig

CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},   # auto-tags by category folder
    name="ToolName",                               # human-readable, used as Modal App name
    repo_url="https://github.com/...",             # upstream source
    repo_commit_hash="abc123...",                  # pinned commit for reproducibility
    package_name="toolname",                       # PyPI / pip install name (if any)
    version="1.0.0",                               # upstream version string
    python_version="3.11",                         # Python version for the container
    cuda_version="cu128",                          # UV torch backend (cu121, cu128, cu130…)
    gpu=os.environ.get("GPU", "L40S"),             # allow override via env var
    timeout=int(os.environ.get("TIMEOUT", "3600")),# allow override via env var
)
```

Key rules:

- **Always pin** either `repo_commit_hash` or `version` (or both) for reproducibility.
- Let `gpu` and `timeout` be overridable via `os.environ.get(...)` with sensible defaults.
- Use `CONF.default_env` when setting environment variables on the image — it provides `UV_COMPILE_BYTECODE`, `HF_HOME`, `TORCH_HOME`, and `UV_TORCH_BACKEND` automatically.
- Use `CONF.model_dir`, `CONF.git_clone_dir`, `CONF.model_volume_mountpoint` etc. instead of hardcoding paths.

## AppInfo Dataclass

When an app has additional constants beyond what `AppConfig` provides (volume mount paths, download URLs, supported model lists, data caches), wrap them in a `@dataclass` named `AppInfo`:

```python
from dataclasses import dataclass

@dataclass
class AppInfo:
    """Container for app-specific configuration and constants."""
    msa_cache_dir: str = "/tool-msa-cache"
    model_dir: Path = CONF.model_dir
    supported_models: Sequence[str] = ("model_v1", "model_v2")
```

Instantiate as `APP_INFO = AppInfo()` right after the class definition. This keeps the namespace clean and groups related constants.
Use an `AppInfo` dataclass only when it improves readability by grouping several related constants. For a small number of simple constants, module-level constants like `OUT_VOLUME` or `OUTPUTS_DIR` are acceptable and can be easier to maintain.

## Image Building

- Always wrap image construction with `patch_image_for_helper(...)` from `biomodals.app.helper` — this injects the helper modules, `zstd`, and `fd-find` into the image so that `package_outputs`, `run_command`, etc. work at runtime in the container.
- Prefer `modal.Image.debian_slim()` or `modal.Image.from_registry()` as base.
- Use `.env(CONF.default_env | {...})` to merge app-specific env vars with the defaults.
- Use `.uv_pip_install(...)` for Python dependencies (Modal's built-in uv integration).
- Pass `copy_patch_files=True` to `patch_image_for_helper` only when subsequent image build steps depend on the helper code.

```python
runtime_image = patch_image_for_helper(
    modal.Image.debian_slim(python_version=CONF.python_version)
    .apt_install("git", "build-essential", "zstd")
    .env(CONF.default_env | {"CUSTOM_VAR": "value"})
    .uv_pip_install(f"git+{CONF.repo_url}@{CONF.repo_commit_hash}")
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
```

## Volumes

- Import shared volumes from `biomodals.app.constant` (e.g. `MODEL_VOLUME`, `MSA_CACHE_VOLUME`).
- Mount model weights as read-only (`.read_only()`) in inference functions when the function only reads from the volume.
- Use `CONF.model_volume_mountpoint` as the mount path for model volumes.
- Commit volume changes explicitly after writes: `MODEL_VOLUME.commit()`.
- For app-specific output volumes, use `CONF.get_out_volume()` which creates a `<AppName>-outputs` volume automatically.

## Remote Functions (@app.function)

- Always specify `timeout` (use `CONF.timeout` or `MAX_TIMEOUT` from constants).
- Specify resource hints: `gpu=CONF.gpu`, `cpu=(min, max)`, `memory=(min, max)`. Unless absolutely required, the CPU minimum should always be `0.125`.
- CPU-only functions (e.g. data pipelines, MSA search) omit the `gpu` parameter.
- GPU inference functions should use `MAX_TIMEOUT` for long-running tasks and mount model volumes read-only.
- Use `TemporaryDirectory` or `mkdtemp` for working directories — they are cleaned on container exit.
- Return results as `bytes` (tarball) using `package_outputs(output_dir)` from the shell helper.

Standard resource annotations pattern:

```python
@app.function(
    gpu=CONF.gpu,
    cpu=(1.125, 16.125),      # burst for tar compression
    memory=(1024, 65536),      # reserve 1GB, OOM at 64GB
    timeout=MAX_TIMEOUT,
    volumes={
        CONF.model_volume_mountpoint: MODEL_VOLUME.read_only(),
    },
)
```

## Helper Module Usage

Always prefer helpers from `biomodals.app.helper` over reimplementing:

| Helper                                | Module             | Use For                                       |
| ------------------------------------- | ------------------ | --------------------------------------------- |
| `run_command(cmd)`                    | `helper.shell`     | Run a shell command with streaming output     |
| `run_command_with_log(cmd, log_file)` | `helper.shell`     | Run and log to a file (inference runs)        |
| `run_background_command(cmd)`         | `helper.shell`     | Non-blocking subprocess                       |
| `package_outputs(root)`               | `helper.shell`     | Bundle outputs into a `.tar.zst` bytes object |
| `copy_files(mapping)`                 | `helper.shell`     | Parallel file copying (volume → local SSD)    |
| `find_with_fd(dir, pattern)`          | `helper.shell`     | Find files using fd/fdfind                    |
| `warmup_directory(dir)`               | `helper.shell`     | Pre-cache files into memory                   |
| `softlink_dir(src, dst)`              | `helper.shell`     | Create symlinks for tool-expected paths       |
| `download_files(urls)`                | `helper.web`       | Async concurrent downloads with retries       |
| `struct2seq(path)`                    | `helper.structure` | PDB/CIF → list of (chain_id, sequence)        |
| `hash_string(s)`                      | `helper.__init__`  | SHA-256 hash for cache keys                   |
| `patch_image_for_helper(image)`       | `helper.__init__`  | Inject helpers into Modal image               |

## Local Entrypoint

The `@app.local_entrypoint()` function runs on the user's machine and orchestrates remote calls. This is the primary user-facing interface.

### Naming Convention

Name it `submit_<toolname>_task(...)`. This makes it easy to identify in logs and when using `biomodals run <app>::<entrypoint>`.

### Docstring Format (Critical for CLI)

The CLI's `biomodals help` command parses the local entrypoint docstring using `_docstring_to_markdown_table()`. This parser expects **Google-style docstrings** with an `Args:` section:

```python
@app.local_entrypoint()
def submit_tool_task(
    input_file: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    ...
) -> None:
    """Short one-line summary of what this entrypoint does.

    Args:
        input_file: Path to the input file.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in the current working directory.
        run_name: Optional run name for output files. Defaults to input
            filename stem.
        some_flag: Description of this parameter. Multi-line descriptions
            must be indented with double the indentation of the arg name.
    """
```

Rules for the parser:

- The `Args:` keyword must appear on its own line.
- Each argument line starts at the first indentation level with `name: description`.
- Continuation lines must be indented at **double** the indent level of the argument name.
- The parser matches argument names to the function signature to extract defaults.
- Argument names use underscores in the signature; the CLI flag is auto-converted to kebab-case (`--input-file`).

### Standard Parameters

Most entrypoints should include these common parameters:

```python
def submit_tool_task(
    input_file: str,                    # path to input (always first positional-ish arg)
    out_dir: str | None = None,         # local output directory, defaults to CWD
    run_name: str | None = None,        # output naming, defaults to input stem
    # ... tool-specific params ...
    download_models: bool = False,      # (if applicable) download-only mode
    force_redownload: bool = False,     # (if applicable) force re-download
) -> None:
```

### Standard Entrypoint Logic

Follow this pattern in the entrypoint body:

```python
# 1. Validate inputs (file existence, supported values)
input_path = Path(input_file).expanduser().resolve()
if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

# 2. Set run name (default to input stem)
if run_name is None:
    run_name = input_path.stem

# 3. Set output path and check for conflicts
local_out_dir = (
    Path(out_dir).expanduser().resolve() if out_dir is not None else Path.cwd()
)
out_file = local_out_dir / f"{run_name}.tar.zst"
if out_file.exists():
    raise FileExistsError(f"Output file already exists: {out_file}")

# 4. (Optional) Ensure models are downloaded
download_data.remote(model_name=model_name, force=force_redownload)

# 5. Read input and call remote function(s)
input_bytes = input_path.read_bytes()
tarball_bytes = run_inference.remote(input_bytes=input_bytes, ...)

# 6. Write results locally
local_out_dir.mkdir(parents=True, exist_ok=True)
out_file.write_bytes(tarball_bytes)
print(f"🧬 Run complete! Results saved to {out_file}")
```

### Print Emoji Convention

Use emoji prefixes for user-facing status messages:

- `🧬` for entrypoint messages (local machine).
- `💊` for remote function messages (Modal container).

## Data Flow Pattern

The standard data flow is:

1. **User** passes file path(s) to the local entrypoint.
2. **Entrypoint** reads the file as `bytes` and passes to remote function(s).
3. **Remote function** writes bytes to a temp directory, runs the tool, packages outputs with `package_outputs()`, returns `bytes`.
4. **Entrypoint** writes the returned tarball bytes to the local output directory.

This avoids mounting local filesystems into containers. Only `bytes` cross the local ↔ remote boundary.

Data flow depends on the app category. Short-lived inference should usually send local input bytes to a remote function and return tarball bytes directly, as described above.
Long-running apps should cache intermediate and final results in Modal volumes.
Parallel or interruptible runs should use queues, locks, stable run IDs, and resumable runners where possible.

## Caching Strategy

- Use `hash_string()` on input sequences/content to create deterministic cache keys.
- Store cached artifacts in Modal volumes under `<AppName>/<hash[:2]>/<hash>/` for sharded directory structure.
- Check for cached results before running expensive operations; return early if found.
- Commit volume changes after writing cache entries: `VOLUME.commit()`.

## Apps That Don't Use AppConfig (Legacy)

Older apps (e.g. `abcfold2_app.py`, `rfdiffusion_app.py`, `gromacs_app.py`) use raw constants (`GPU`, `TIMEOUT`, `APP_NAME`) and manual volume definitions. When touching these files, prefer migrating to `AppConfig` + `AppInfo` pattern if the change scope permits. New apps **must** use `AppConfig` unless there are very few constants.

## Examples

When the app development is done, generate an example bash script under `examples/app/` that demonstrates how to call the local entrypoint with `biomodals run`.
If data files under `examples/data/` are not sufficient, add small example input files (e.g. a short FASTA) for testing purposes.
