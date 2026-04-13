"""AlphaFold3 source repo: <https://github.com/google-deepmind/alphafold3>.

Note that this script only provides a runtime for AlphaFold3.
To acquire the model weights and MSA databases, please follow instructions at:

<https://github.com/google-deepmind/alphafold3#obtaining-model-parameters>

Make sure the model checkpoint is available at `/biomodals-store/AlphaFold3/af3.bin`,
and the MSA databases are available at `/AlphaFold3-msa-db/`.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|


| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `GPU` | `L40S` | Type of GPU to use. See https://modal.com/docs/guide/gpu for details. |
| `TIMEOUT` | `3600` | Timeout for each Modal function in seconds. |

## Additional notes

See <https://github.com/google-deepmind/alphafold3/tree/main/docs>.

## Outputs

See <https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md>.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
from pathlib import Path

from modal import App, Image

from biomodals.app.config import AppConfig
from biomodals.app.constant import (
    AF3_MSA_DB_VOLUME,
    MAX_TIMEOUT,
    MODEL_VOLUME,
    MSA_CACHE_VOLUME,
)
from biomodals.app.helper import patch_image_for_helper
from biomodals.app.helper.shell import run_command_with_log

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    name="AlphaFold3",
    repo_url="https://github.com/google-deepmind/alphafold3",
    repo_commit_hash="87bd9e678d9acacc4aa9baa05e820f32b80e1b49",
    package_name="alphafold3",
    version="3.0.1",
    python_version="3.12",
    cuda_version="cu130",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "3600")),
)

# Volume for genetic serach databases
MSA_DB_DIR = "/AlphaFold3-msa-db"
MSA_CACHE_DIR = "/biomodals-msa-cache"

# Volume for prediction outputs (enables skip/resume across interrupted runs)
OUTPUTS_VOLUME_NAME, OUTPUTS_VOLUME = CONF.out_volume()
OUTPUTS_DIR = CONF.output_volume_mountpoint
MODEL_DIR = CONF.model_dir

##########################################
# Image and app definitions
##########################################

# Ref: https://github.com/google-deepmind/alphafold3/blob/main/docker/Dockerfile
runtime_image = patch_image_for_helper(
    Image.debian_slim(python_version=CONF.python_version)
    .apt_install("git", "build-essential", "zstd", "zlib1g-dev", "wget")
    .env(
        CONF.default_env
        | {
            "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_CLIENT_MEM_FRACTION": "0.95",
        }
    )
    .run_commands(
        " && ".join(
            (
                # Clone AlphaFold3 repo
                f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
                f"cd {CONF.git_clone_dir}",
                f"git checkout {CONF.repo_commit_hash}",
                # Download, check hash, and extract HMMER
                "mkdir /hmmer_build",
                "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz --directory-prefix /hmmer_build",
                "cd /hmmer_build",
                "echo 'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 hmmer-3.4.tar.gz' | sha256sum --check",
                "tar zxf hmmer-3.4.tar.gz",
                "rm hmmer-3.4.tar.gz",
                # Apply the --seq_limit patch to HMMER
                "cd /hmmer_build",
                f"patch -p0 < {CONF.git_clone_dir}/docker/jackhmmer_seq_limit.patch",
                # Build and install HMMER
                "cd /hmmer_build/hmmer-3.4",
                "./configure --prefix=/hmmer",
                "make -j",
                "make install",
                "cd /hmmer_build/hmmer-3.4/easel",
                "make install",
                "rm -rf /hmmer_build",
            )
        )
    )
    .workdir(str(CONF.git_clone_dir))
    # .uv_sync(frozen=True, extra_options="--no-editable")
    .uv_pip_install(str(CONF.git_clone_dir))
    .run_commands("build_data")  # installed in the previous step
    .env({"PATH": "/hmmer/bin:$PATH"})
)
app = App(CONF.name, image=runtime_image)


##########################################
# Inference functions
##########################################
@app.function(
    cpu=8,
    memory=(1024, 131072),  # reserve 1GB, OOM at 128GB
    timeout=CONF.timeout,
    volumes={
        CONF.model_volume_mountpoint: MODEL_VOLUME,
        MSA_DB_DIR: AF3_MSA_DB_VOLUME,
        MSA_CACHE_DIR: MSA_CACHE_VOLUME,
        OUTPUTS_DIR: OUTPUTS_VOLUME,
    },
)
def run_data_pipeline(json_file: str | Path) -> Path:
    """Run AlphaFold3 data pipeline (CPU-only)."""
    import sys

    json_path = Path(json_file).resolve()
    out_dir = Path(OUTPUTS_DIR) / json_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: more performant runs when multiple inputs share same chains
    # https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md
    cmd = [
        sys.executable,
        str(CONF.git_clone_dir / "run_alphafold.py"),
        "--run_inference=false",
        f"--json_path={json_path}",
        f"--output_dir={out_dir}",
        f"--model_dir={CONF.model_dir}",
        f"--db_dir={MSA_DB_DIR}",
        f"--jax_compilation_cache_dir={CONF.model_dir / 'jax_cache'}",
    ]
    run_command_with_log(cmd, log_file=out_dir / "data_pipeline.log")
    return out_dir


@app.function(
    gpu=CONF.gpu,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 131072),  # reserve 1GB, OOM at 128GB
    timeout=MAX_TIMEOUT,
    volumes={
        CONF.model_volume_mountpoint: MODEL_VOLUME.read_only(),
        MSA_CACHE_DIR: MSA_CACHE_VOLUME,
        OUTPUTS_DIR: OUTPUTS_VOLUME,
    },
)
def run_inference_pipeline() -> bytes:
    """Run AlphaFold3 structure prediction.

    Returns:
        Tarball bytes of inference outputs (CIF files + confidence JSONs).

    """
    pass


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_alphafold3_task(
    input_json: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    seeds: str = "101",
    cycle: int = 10,
    step: int = 200,
    sample: int = 5,
) -> None:
    """Run AlphaFold3 on Modal and fetch results to `out_dir`.

    Args:
        input_json: Path to input JSON file.
        out_dir: Optional output directory (defaults to $CWD)
        run_name: Optional run name (defaults to input filename stem)
        run_data_pipeline: Whether to run MSA and template search.
        seeds: Comma-separated random seeds for inference
        cycle: Pairformer cycle number
        step: Number of diffusion steps
        sample: Number of samples per seed
    """
    # Validate and read input
    input_path = Path(input_json).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if run_name is None:
        run_name = input_path.stem

    json_str = input_path.read_bytes()

    local_out_dir = (
        Path(out_dir).expanduser().resolve() if out_dir is not None else Path.cwd()
    )
    out_file = local_out_dir / f"{run_name}.tar.zst"
    if out_file.exists():
        raise FileExistsError(f"Output file already exists: {out_file}")

    # Run inference
    print(f"🧬 Running {CONF.name} data pipeline...")
    with OUTPUTS_VOLUME.batch_upload() as batch:
        batch.put_file(input_path, f"/{run_name}/{run_name}.json")

    remote_run_dir = run_data_pipeline.remote(
        f"{OUTPUTS_DIR}/{run_name}/{run_name}.json"
    )
    print(f"🧬 Data pipeline results generated in remote directory: {remote_run_dir}")

    # Save results locally
    # local_out_dir.mkdir(parents=True, exist_ok=True)
    # out_file.write_bytes(tarball_bytes)
    # print(f"🧬 {CONF.name} run complete! Results saved to {out_file}")
