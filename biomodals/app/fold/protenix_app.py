"""Protenix source repo: <https://github.com/bytedance/Protenix>.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input-json` | **Required** | Path to input JSON file. For a description of the JSON schema, see https://github.com/bytedance/Protenix/blob/main/docs/infer_json_format.md. |
| `--out-dir` | `$CWD` | Optional local output directory. If not specified, outputs will be saved in the current working directory. |
| `--run-name` | stem name of `--input-json` | Optional run name used to name output files. |
| `--model-name` | `protenix_base_default_v1.0.0` | Model checkpoint name. Supported models: `protenix_base_default_v1.0.0`, `protenix_base_20250630_v1.0.0`. |
| `--seeds` | `"101"` | Comma-separated random seeds for inference. |
| `--cycle` | `10` | Pairformer cycle number. |
| `--step` | `200` | Number of diffusion steps. |
| `--sample` | `5` | Number of samples per seed. |
| `--dtype` | `"bf16"` | Inference dtype (`bf16` or `fp32`). |
| `--use-msa`/`--no-use-msa` | `--use-msa` | Whether to use MSA features. |
| `--msa-server-mode` | `protenix` | MSA search mode: `protenix` (remote server, no local DB needed) or `colabfold`. |
| `--use-template`/`--no-use-template` | `--no-use-template` | Whether to search for and use templates. |
| `--use-rna-msa`/`--no-use-rna-msa` | `--no-use-rna-msa` | Whether to use RNA MSA features. |
| `--download-models`/`--no-download-models` | `--no-download-models` | Whether to download model weights and data caches, then exit without running inference. |
| `--force-redownload`/`--no-force-redownload` | `--no-force-redownload` | Whether to force re-download even if files already exist. |
| `--extra-args` | `None` | Additional CLI arguments passed directly to `protenix pred` as a single string. |

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `Protenix` | Name of the Modal app to use. |
| `GPU` | `L40S` | Type of GPU to use. See https://modal.com/docs/guide/gpu for details. |
| `TIMEOUT` | `3600` | Timeout for each Modal function in seconds. |

## Additional notes

* The default `--msa-server-mode protenix` uses the Protenix remote MSA server,
  so no local MSA databases are required. Switch to `colabfold` if you have a
  pre-populated database volume.
* Templates are only used when `--use-template` is passed. Template support
  requires the v1.0.0 model checkpoints.
* RNA MSA is only supported by v1.0.0 model checkpoints.
* The `protenix_base_constraint_v0.5.0` model supports pocket, contact, and
  substructure constraints specified in the input JSON.
* For large structures (>2000 tokens), consider using an A100 (80GB) or H100
  GPU by setting the `GPU` environment variable.

## Outputs

* Results will be saved to the specified `--out-dir` as
  `<run-name>_protenix_outputs.tar.zst`.
* The tarball contains predicted `.cif` structure files and
  `*_summary_confidence.json` files with pLDDT, pAE, and ranking scores.
"""

# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603

import os
import shlex
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
# T4: 16GB, L4: 24GB, A10G: 24GB, L40S: 48GB, A100-40G, A100-80G, H100: 80GB
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", "3600"))  # seconds
APP_NAME = os.environ.get("MODAL_APP", "Protenix")

# Volume for model weights and data caches
DATA_VOLUME_NAME = "protenix-data"
DATA_VOLUME = Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
DATA_DIR = "/protenix-data"

# Supported model checkpoints
SUPPORTED_MODELS = ("protenix_base_default_v1.0.0", "protenix_base_20250630_v1.0.0")

# Base URL for downloading checkpoints and data caches
PROTENIX_DOWNLOAD_BASE = "https://protenix.tos-cn-beijing.volces.com"

# CCD and other data caches required for inference
DATA_CACHE_FILES = (
    "common/components.cif",
    "common/components.cif.rdkit_mol.pkl",
    "common/clusters-by-entity-40.txt",
    "common/obsolete_release_date.csv",
)

# Additional files needed when templates are enabled
TEMPLATE_CACHE_FILES = (
    "common/obsolete_to_successor.json",
    "common/release_date_cache.json",
)

# Repository and commit hash
REPO_URL = "https://github.com/bytedance/Protenix"
REPO_COMMIT = "a0ff3fe5e6395f19e6134b8c40eab5a298bba30d"

##########################################
# Image and app definitions
##########################################

# https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image
cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

runtime_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",
        "build-essential",
        "zstd",
        "hmmer",
        "kalign",
        "wget",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            # https://modal.com/docs/guide/cuda
            "UV_TORCH_BACKEND": "cu126",
        }
    )
    .uv_pip_install(f"git+{REPO_URL}.git@{REPO_COMMIT}")
    # Trigger kernel compilation
    .run_commands(
        "python /usr/local/lib/python3.11/site-packages/protenix/model/layer_norm/layer_norm.py",
        gpu=GPU,
    )
)

app = App(APP_NAME, image=runtime_image)


##########################################
# Helper functions
##########################################
def package_outputs(
    dir: str, tar_args: list[str] | None = None, num_threads: int = 16
) -> bytes:
    """Package directory into a tar.zst archive and return as bytes."""
    import subprocess as sp
    from pathlib import Path

    dir_path = Path(dir)
    cmd = ["tar", "-I", f"zstd -T{num_threads}"]  # ZSTD_NBTHREADS
    if tar_args is not None:
        cmd.extend(tar_args)
    cmd.extend(["-c", dir_path.name])

    return sp.check_output(cmd, cwd=dir_path.parent)


def run_command(cmd: list[str], **kwargs) -> None:
    """Run a shell command and stream output to stdout."""
    import subprocess as sp

    print(f"💊 Running command: {' '.join(cmd)}")
    # Set default kwargs for sp.Popen
    kwargs.setdefault("stdout", sp.PIPE)
    kwargs.setdefault("stderr", sp.STDOUT)
    kwargs.setdefault("bufsize", 1)
    kwargs.setdefault("encoding", "utf-8")

    with sp.Popen(cmd, **kwargs) as p:
        if p.stdout is None:
            raise RuntimeError("stdout is None")

        buffered_output = None
        while (buffered_output := p.stdout.readline()) != "" or p.poll() is None:
            print(buffered_output, end="", flush=True)

        if p.returncode != 0:
            raise RuntimeError(f"Command failed with return code {p.returncode}")


def download_from_url(url: str, local_path: Path, force: bool = False) -> None:
    """Download a file from a URL if it doesn't already exist."""
    import urllib.request

    if local_path.exists() and not force:
        print(f"  ✓ Already exists: {local_path.name}")
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ⬇ Downloading: {url}")
    urllib.request.urlretrieve(url, str(local_path))  # noqa: S310
    print(f"  ✓ Saved to: {local_path}")


##########################################
# Fetch model weights and data caches
##########################################
@app.function(volumes={DATA_DIR: DATA_VOLUME}, timeout=TIMEOUT)
def download_protenix_data(
    model_name: str = "protenix_base_default_v1.0.0",
    force: bool = False,
    include_templates: bool = False,
) -> None:
    """Download Protenix model checkpoint and shared data caches.

    Args:
        model_name: Name of the model checkpoint to download.
        force: Force re-download even if files already exist.
        include_templates: Also download template-related data files.
    """
    data_root = Path(DATA_DIR)

    # Download common data caches
    print("💊 Downloading Protenix data caches...")
    for rel_path in DATA_CACHE_FILES:
        url = f"{PROTENIX_DOWNLOAD_BASE}/{rel_path}"
        download_from_url(url, data_root / rel_path, force=force)

    # Download template data if requested
    if include_templates:
        print("💊 Downloading template data files...")
        for rel_path in TEMPLATE_CACHE_FILES:
            url = f"{PROTENIX_DOWNLOAD_BASE}/{rel_path}"
            download_from_url(url, data_root / rel_path, force=force)

    # Download model checkpoint
    print(f"💊 Downloading model checkpoint: {model_name}")
    ckpt_url = f"{PROTENIX_DOWNLOAD_BASE}/checkpoint/{model_name}.pt"
    download_from_url(
        url=ckpt_url,
        local_path=data_root / "checkpoint" / f"{model_name}.pt",
        force=force,
    )

    DATA_VOLUME.commit()
    print("💊 Download complete")


##########################################
# Inference functions
##########################################
@app.function(
    gpu=GPU,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=TIMEOUT,
    volumes={DATA_DIR: DATA_VOLUME},
)
def run_protenix(
    json_str: bytes,
    run_name: str,
    model_name: str = "protenix_base_default_v1.0.0",
    seeds: str = "101",
    cycle: int = 10,
    step: int = 200,
    sample: int = 5,
    dtype: str = "bf16",
    use_msa: bool = True,
    msa_server_mode: str = "protenix",
    use_template: bool = False,
    use_rna_msa: bool = False,
    extra_args: str | None = None,
) -> bytes:
    """Run Protenix structure prediction.

    Args:
        json_str: Input JSON content as bytes.
        run_name: Name for this run (used for output directory).
        model_name: Model checkpoint name.
        seeds: Comma-separated random seeds.
        cycle: Pairformer cycle number.
        step: Number of diffusion steps.
        sample: Number of samples per seed.
        dtype: Inference dtype (bf16 or fp32).
        use_msa: Whether to use MSA features.
        msa_server_mode: MSA search mode (protenix or colabfold).
        use_template: Whether to use templates.
        use_rna_msa: Whether to use RNA MSA.
        extra_args: Additional CLI arguments as a string.

    Returns:
        Tarball bytes of inference outputs (CIF files + confidence JSONs).
    """
    import tempfile

    DATA_VOLUME.reload()

    # Write input JSON to a temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_json_path = tmpdir_path / f"{run_name}.json"
        input_json_path.write_bytes(json_str)

        out_dir = tmpdir_path / "output"
        out_dir.mkdir()

        # Build protenix pred command
        cmd = [
            "protenix",
            "pred",
            "-i",
            str(input_json_path),
            "-o",
            str(out_dir),
            "-s",
            seeds,
            "-c",
            str(cycle),
            "-p",
            str(step),
            "-e",
            str(sample),
            "-d",
            dtype,
            "-n",
            model_name,
            "--use_msa",
            str(use_msa),
            "--msa_server_mode",
            msa_server_mode,
            "--use_template",
            str(use_template),
            "--use_rna_msa",
            str(use_rna_msa),
        ]

        if extra_args:
            cmd.extend(shlex.split(extra_args))

        # Set PROTENIX_ROOT_DIR so protenix finds cached data and checkpoints
        env_override = {
            "PROTENIX_ROOT_DIR": DATA_DIR,
        }
        import os as _os

        run_env = _os.environ.copy()
        run_env.update(env_override)

        run_command(cmd, env=run_env, cwd=out_dir)

        # Package outputs
        print("💊 Packaging Protenix results...")
        tarball_bytes = package_outputs(str(out_dir))
        print("💊 Packaging complete.")

    return tarball_bytes


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_protenix_task(
    input_json: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    model_name: str = "protenix_base_default_v1.0.0",
    seeds: str = "101",
    cycle: int = 10,
    step: int = 200,
    sample: int = 5,
    dtype: str = "bf16",
    use_msa: bool = True,
    msa_server_mode: str = "protenix",
    use_template: bool = False,
    use_rna_msa: bool = False,
    download_models: bool = False,
    force_redownload: bool = False,
    extra_args: str | None = None,
) -> None:
    """Run Protenix structure prediction on Modal and fetch results to `out_dir`.

    Args:
        input_json: Path to input JSON file
        out_dir: Optional output directory (defaults to $CWD)
        run_name: Optional run name (defaults to input filename stem)
        model_name: Model checkpoint name
        seeds: Comma-separated random seeds for inference
        cycle: Pairformer cycle number
        step: Number of diffusion steps
        sample: Number of samples per seed
        dtype: Inference dtype (bf16 or fp32)
        use_msa: Whether to use MSA features
        msa_server_mode: MSA search mode (protenix or colabfold)
        use_template: Whether to use templates
        use_rna_msa: Whether to use RNA MSA features
        download_models: Whether to download model weights and skip running
        force_redownload: Whether to force re-download of model weights
        extra_args: Additional CLI arguments passed to protenix pred
    """
    from pathlib import Path

    # Validate model name
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    # Handle download-only mode
    if download_models:
        print(f"🧬 Downloading Protenix model: {model_name}")
        download_protenix_data.remote(
            model_name=model_name,
            force=force_redownload,
            include_templates=use_template,
        )
        print("🧬 Download complete!")
        return

    # Validate and read input
    input_path = Path(input_json).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    if run_name is None:
        run_name = input_path.stem

    json_str = input_path.read_bytes()

    local_out_dir = (
        Path(out_dir).expanduser().resolve() if out_dir is not None else Path.cwd()
    )
    out_file = local_out_dir / f"{run_name}_protenix_outputs.tar.zst"
    if out_file.exists():
        raise FileExistsError(f"Output file already exists: {out_file}")

    # Ensure models and data caches are available
    print(f"🧬 Checking Protenix model and data caches for {model_name}...")
    download_protenix_data.remote(
        model_name=model_name,
        force=False,
        include_templates=use_template,
    )

    # Run inference
    print(f"🧬 Running Protenix inference with {model_name}...")
    tarball_bytes = run_protenix.remote(
        json_str=json_str,
        run_name=run_name,
        model_name=model_name,
        seeds=seeds,
        cycle=cycle,
        step=step,
        sample=sample,
        dtype=dtype,
        use_msa=use_msa,
        msa_server_mode=msa_server_mode,
        use_template=use_template,
        use_rna_msa=use_rna_msa,
        extra_args=extra_args,
    )

    # Save results locally
    local_out_dir.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(tarball_bytes)
    print(f"🧬 Protenix run complete! Results saved to {out_file}")
