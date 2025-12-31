"""ABCFold2 source repo: <https://github.com/y1zhou/ABCFold/tree/feat/schema>.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input-yaml` | **Required** | Path to YAML design specification file. For a detailed description of the YAML schema, see https://github.com/y1zhou/ABCFold/blob/feat/schema/abcfold/schema.py. |
| `--out-dir` | `$CWD` | Optional local output directory. If not specified, outputs will be saved in the current working directory. |
| `--run-name` | stem name of `--input-yaml` | Optional run name used to name output directory. |
| `--msa-chains` | None (all chains) | Optional comma-separated list of chains to search MSAs for. If not specified, MSAs will be searched for all chains. To skip MSA searches entirely, pass 'empty'. |
| `--search-templates`/`--no-search-templates` | `--no-search-templates` | Whether to search for templates and add to input YAML. |
| `--download-models`/`--no-download-models` | `--no-download-models` | Whether to download model weights and skip running. |
| `--force-redownload` | `--no-force-redownload` | Whether to force re-download of model weights even if they exist. |
| `--run-boltz`/`--no-run-boltz` | `--run-boltz` | Whether to run Boltz inference. |
| `--run-chai`/`--no-run-chai` | `--run-chai` | Whether to run Chai inference. |

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `MODAL_APP` | `ABCFold2` | Name of the Modal app to use. |
| `GPU` | `A10G` | Type of GPU to use. See https://modal.com/docs/guide/gpu for details. |
| `TIMEOUT` | `1800` | Timeout for each Modal function in seconds. |

## Additional notes on input flags

* MSAs will *always* be searched automatically, since omitting MSAs for Boltz/Chai translates to worse performance in most cases.
* Templates will be searched only if the `--search-templates` flag is passed. When multiple templates are found, only the top four will be used.
* The `--run-boltz` and `--run-chai` flags control whether to run structure prediction with the respective model. Inputs for both models will always be prepared for convenience.

## Outputs

* Results will be saved to the specified `--out-dir` under a subdirectory named after the `--run-name`.
* The output directory will contain a `run-config.json` file with the run parameters used.
* Inference results will be saved as `<model-name>_models.tar.zst` files. Extract them and analyze results using `abcfold2 postprocess`.
"""
# Ignore ruff warnings about import location and unsafe subprocess usage
# ruff: noqa: PLC0415, S603

import os
from pathlib import Path

from modal import App, Image, Volume

##########################################
# Modal configs
##########################################
# T4: 16GB, L4: 24GB, A10G: 24GB, L40S: 48GB, A100-40G, A100-80G, H100: 80GB
# https://modal.com/docs/guide/gpu
GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", "1800"))  # seconds
APP_NAME = os.environ.get("MODAL_APP", "ABCFold2")

# Volume for model cache
CHAI_VOLUME_NAME = "chai-models"
CHAI_VOLUME = Volume.from_name(CHAI_VOLUME_NAME, create_if_missing=True)
CHAI_MODEL_DIR = "/chai-models"

BOLTZ_VOLUME_NAME = "boltz-models"
BOLTZ_VOLUME = Volume.from_name(BOLTZ_VOLUME_NAME, create_if_missing=True)
BOLTZ_MODEL_DIR = "/boltz-models"

# Volume for outputs
OUTPUTS_VOLUME_NAME = "abcfold2-outputs"
OUTPUTS_VOLUME = Volume.from_name(
    OUTPUTS_VOLUME_NAME, create_if_missing=True, version=2
)
OUTPUTS_DIR = "/abcfold2-outputs"

# Repositories and commit hashes
ABCFOLD_DIR = "/opt/ABCFold"
ABCFOLD_REPO = "https://github.com/y1zhou/ABCFold"
ABCFOLD_COMMIT = "fcfdd49fbec0db73eb38dfad49f9649e81147337"

CHAI_DIR = "/opt/chai-lab"
CHAI_REPO = "https://github.com/y1zhou/chai-lab"
CHAI_COMMIT = "0ac68311911bfcd28b118fc289437bf3eff8ac97"

BOLTZ_DIR = "/opt/boltz"
BOLTZ_REPO = "https://github.com/jwohlwend/boltz"
BOLTZ_COMMIT = "cb04aeccdd480fd4db707f0bbafde538397fa2ac"
BOLTZ_MODEL_HASH = "6fdef46d763fee7fbb83ca5501ccceff43b85607"  # HF revision

##########################################
# Image and app definitions
##########################################
download_image = (
    Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]==0.26.3")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # speed up downloads
)

runtime_image = (
    Image.debian_slim()
    .apt_install("git", "build-essential")
    .env(
        {
            # "UV_COMPILE_BYTECODE": "1",  # slower image build, faster runtime
            # https://modal.com/docs/guide/cuda
            "UV_TORCH_BACKEND": "cu128",  # find best torch and CUDA versions
            "CHAI_DOWNLOADS_DIR": str(CHAI_MODEL_DIR),  # store chai model weights
            "BOLTZ_CACHE": str(BOLTZ_MODEL_DIR),  # store boltz model weights
        }
    )
    .run_commands(
        " && ".join(
            (
                # Clone Boltz and Chai
                f"git clone {BOLTZ_REPO} {BOLTZ_DIR}",
                f"cd {BOLTZ_DIR}",
                f"git checkout {BOLTZ_COMMIT}",
                f"git clone {CHAI_REPO} {CHAI_DIR}",
                f"cd {CHAI_DIR}",
                f"git checkout {CHAI_COMMIT}",
                # Setup ABCFold2 environment
                f"git clone {ABCFOLD_REPO} {ABCFOLD_DIR}",
                f"cd {ABCFOLD_DIR}",
                f"git checkout {ABCFOLD_COMMIT}",
                "uv venv --python 3.12",
                f"uv pip install {BOLTZ_DIR}[cuda] {CHAI_DIR}",
                "uv pip install .",
            ),
        )
    )
    .env({"PATH": f"{ABCFOLD_DIR}/.venv/bin:$PATH"})
    .apt_install(
        "kalign",  # for Chai templates
        "zstd",  # for packaging outputs
    )
    .workdir(ABCFOLD_DIR)
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
    cmd = ["tar", "--zstd"]
    if tar_args is not None:
        cmd.extend(tar_args)
    cmd.extend(["-cf", "-", dir_path.name])

    return sp.check_output(
        cmd, cwd=dir_path.parent, env={"ZSTD_NBTHREADS": str(num_threads)}
    )  # noqa: S603


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes={BOLTZ_MODEL_DIR: BOLTZ_VOLUME}, timeout=TIMEOUT, image=download_image
)
def download_boltz_models(force: bool = False) -> None:
    """Download Boltz models into the mounted volume.

    From: https://modal.com/docs/examples/boltz_predict.
    """
    import tarfile

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="boltz-community/boltz-2",
        revision=BOLTZ_MODEL_HASH,
        local_dir=BOLTZ_MODEL_DIR,
        force_download=force,
    )
    tar_mols = Path(BOLTZ_MODEL_DIR) / "mols.tar"
    if not (Path(BOLTZ_MODEL_DIR) / "mols").exists():
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(BOLTZ_MODEL_DIR)  # noqa: S202

    BOLTZ_VOLUME.commit()


async def download_file(session, url: str, local_path: Path):
    """Download a file asynchronously using aiohttp."""
    async with session.get(url) as response:
        response.raise_for_status()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            while chunk := await response.content.read(8192):
                f.write(chunk)


@app.function(
    volumes={CHAI_MODEL_DIR: CHAI_VOLUME}, timeout=TIMEOUT, image=download_image
)
async def download_chai_models(force=False):
    """From https://modal.com/docs/examples/chai1."""
    import asyncio

    import aiohttp

    base_url = "https://chaiassets.com/chai1-inference-depencencies/"  # sic
    inference_dependencies = [
        "conformers_v1.apkl",
        "models_v2/trunk.pt",
        "models_v2/token_embedder.pt",
        "models_v2/feature_embedding.pt",
        "models_v2/diffusion_module.pt",
        "models_v2/confidence_head.pt",
        "models_v2/bond_loss_input_proj.pt",
        "esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # launch downloads concurrently
    chai_model_dir = Path(CHAI_MODEL_DIR)
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for dep in inference_dependencies:
            local_path = chai_model_dir / dep
            if force or not local_path.exists():
                url = base_url + dep
                print(f"🧬 downloading {dep} to {local_path}")
                tasks.append(download_file(session, url, local_path))

        # run all of the downloads and await their completion
        await asyncio.gather(*tasks)

    # Special treatment for ESM
    esm2_path = chai_model_dir / "esm2" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
    esm_path = chai_model_dir / "esm" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
    if esm2_path.exists() and not esm_path.exists():
        esm_path.parent.mkdir(parents=True, exist_ok=True)
        esm_path.symlink_to(esm2_path)

    CHAI_VOLUME.commit()  # ensures models are visible on remote filesystem before exiting, otherwise takes a few seconds, racing with inference


##########################################
# Inference functions
##########################################
def load_params_from_run_yaml(yaml_path: Path) -> dict:
    """Load run parameters from ABCFold2 YAML config."""
    from abcfold.schema import load_abcfold_config

    conf = load_abcfold_config(yaml_path)
    return {
        "seeds": conf.seeds,
        "num_trunk_recycles": conf.num_trunk_recycles,
        "num_diffn_timesteps": conf.num_diffn_timesteps,
        "num_diffn_samples": conf.num_diffn_samples,
        "num_trunk_samples": conf.num_trunk_samples,
        "boltz_additional_cli_args": conf.boltz_additional_cli_args,
    }


@app.function(image=runtime_image, timeout=TIMEOUT)
def get_run_id(yaml_str: bytes) -> str:
    """Get content-based run ID from ABCFold2 config."""
    import tempfile

    from abcfold.schema import load_abcfold_config

    # Determine content-based run ID
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_yaml_path = Path(tmpdir) / "abcfold-config.yaml"
        tmp_yaml_path.write_bytes(yaml_str)
        conf = load_abcfold_config(tmp_yaml_path)

    return conf.hash


@app.function(
    image=runtime_image,
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME, BOLTZ_MODEL_DIR: BOLTZ_VOLUME},
)
def prepare_abcfold2(
    yaml_str: bytes, search_templates: bool, msa_chains: str | None = None
) -> dict[str, str | list[int] | int | list[str] | None]:
    """Prepare inputs to Boltz and Chai using ABCFold2 config."""
    import tempfile
    from pathlib import Path

    from abcfold.cli.prepare import prepare_boltz, prepare_chai, search_msa

    run_id: str = get_run_id.local(yaml_str=yaml_str)
    if not search_templates:
        run_id = f"{run_id}-no-tmpl"
    out_dir_full: Path = Path(OUTPUTS_DIR) / run_id[:2] / run_id
    out_dir_full.mkdir(parents=True, exist_ok=True)

    # Check if MSA and templates were already generated for a previous run with same ID
    yaml_path = out_dir_full / f"{run_id}.yaml"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_yaml_path = Path(tmpdir) / f"{run_id}.yaml"
        tmp_yaml_path.write_bytes(yaml_str)
        new_conf = load_params_from_run_yaml(tmp_yaml_path)

        if not yaml_path.exists():
            # Run MSA and template search
            search_msa(
                conf_file=tmp_yaml_path,
                out_dir=out_dir_full,
                force=True,
                chains=msa_chains,
                search_templates=search_templates,
                template_cache_dir=Path(OUTPUTS_DIR) / ".cache" / "rcsb",
            )
            OUTPUTS_VOLUME.commit()

    # Generate inputs for Boltz and Chai
    if not (out_dir_full / "boltz_models" / f"{run_id}.yaml").exists():
        _ = prepare_boltz(conf_file=yaml_path, out_dir=out_dir_full)
        OUTPUTS_VOLUME.commit()
    if not (out_dir_full / "chai_models" / f"{run_id}.yaml").exists():
        _ = prepare_chai(
            conf_file=yaml_path,
            out_dir=out_dir_full,
            ccd_lib_dir=Path(BOLTZ_MODEL_DIR) / "mols",
        )
        OUTPUTS_VOLUME.commit()

    # Pull run parameters from YAML
    conf = load_params_from_run_yaml(yaml_path)
    conf["run_id"] = run_id
    conf["workdir"] = str(out_dir_full)
    conf["seeds"] = new_conf["seeds"]  # ensure seeds are up to date
    return conf


@app.function(
    cpu=(0.125, 16.125),  # burst for tar compression
    image=runtime_image,
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME, BOLTZ_MODEL_DIR: BOLTZ_VOLUME},
)
def collect_abcfold2_boltz_data(
    run_conf: dict[str, str | list[int] | int | list[str] | None],
):
    """Manage Boltz runs and return all Boltz results."""
    from pathlib import Path

    work_path = Path(run_conf["workdir"]).expanduser().resolve()
    run_id = run_conf["run_id"]
    work_path = work_path / "boltz_models"
    boltz_conf_path = work_path / f"{run_id}.yaml"
    OUTPUTS_VOLUME.reload()

    if not boltz_conf_path.exists():
        raise FileNotFoundError(f"Boltz config file not found: {boltz_conf_path}")

    random_seeds = run_conf.get("seeds", [])
    seeds_to_run = []
    for seed in random_seeds:
        boltz_run_dir = work_path / f"boltz_results_seed-{seed}"
        if not boltz_run_dir.exists():
            seeds_to_run.append(seed)

    if seeds_to_run:
        # modal function map results need to be consumed to actually run
        for boltz_run_dir in run_abcfold2_boltz.map(seeds_to_run, kwargs=run_conf):
            print(f"Boltz run complete: {boltz_run_dir}")

    OUTPUTS_VOLUME.reload()
    print("Packaging Boltz results...")
    boltz_tarball_bytes = package_outputs(
        str(work_path),
        [
            "--exclude",
            "boltz_msa",
            "--exclude",
            "lightning_logs",
            "--exclude",
            "processed",
            "--exclude",
            "msa",
        ],
    )
    print("Packaging complete.")
    return boltz_tarball_bytes


@app.function(
    gpu=GPU,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME, BOLTZ_MODEL_DIR: BOLTZ_VOLUME},
)
def run_abcfold2_boltz(
    seed: int,
    workdir: str | Path,
    run_id: str,
    num_trunk_recycles: int,  # recycling_steps
    num_diffn_timesteps: int,  # sampling_steps
    num_diffn_samples: int,  # diffusion_samples
    boltz_additional_cli_args: list[str] | None,
    **kwargs,  # ignore extra items from run config
) -> str:
    """Run Boltz with the given ABCFold2 configuration."""
    from abcfold.boltz.run_boltz_abcfold import run_boltz

    OUTPUTS_VOLUME.reload()
    work_path = Path(workdir).expanduser().resolve()
    work_path = work_path / "boltz_models"
    boltz_conf_path = work_path / f"{run_id}.yaml"
    if not boltz_conf_path.exists():
        raise FileNotFoundError(f"Boltz config file not found: {boltz_conf_path}")

    boltz_run_dir = run_boltz(
        output_dir=work_path,
        boltz_yaml_file=boltz_conf_path,
        seed=seed,
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        num_diffn_samples=num_diffn_samples,
        boltz_additional_cli_args=boltz_additional_cli_args,
    )
    OUTPUTS_VOLUME.commit()
    return str(boltz_run_dir)


@app.function(
    cpu=(0.125, 16.125),  # burst for tar compression
    image=runtime_image,
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME, CHAI_MODEL_DIR: CHAI_VOLUME},
)
def collect_abcfold2_chai_data(
    run_conf: dict[str, str | list[int] | int | list[str] | None],
):
    """Manage Chai runs and return all Chai results."""
    from pathlib import Path

    work_path = Path(run_conf["workdir"]).expanduser().resolve()
    run_id = run_conf["run_id"]
    work_path = work_path / "chai_models"
    chai_conf_path = work_path / f"{run_id}.yaml"
    OUTPUTS_VOLUME.reload()

    if not chai_conf_path.exists():
        raise FileNotFoundError(f"Chai config file not found: {chai_conf_path}")

    random_seeds = run_conf.get("seeds", [])
    seeds_to_run = []
    for seed in random_seeds:
        chai_run_dir = work_path / f"chai_seed-{seed}"
        if not chai_run_dir.exists():
            seeds_to_run.append(seed)

    if seeds_to_run:
        # modal function map results need to be consumed to actually run
        for chai_run_dir in run_abcfold2_chai.map(seeds_to_run, kwargs=run_conf):
            print(f"Chai run complete: {chai_run_dir}")

    OUTPUTS_VOLUME.reload()
    print("Packaging Chai results...")
    chai_tarball_bytes = package_outputs(str(work_path))
    print("Packaging complete.")
    return chai_tarball_bytes


@app.function(
    gpu=GPU,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=TIMEOUT,
    volumes={OUTPUTS_DIR: OUTPUTS_VOLUME, CHAI_MODEL_DIR: CHAI_VOLUME},
)
def run_abcfold2_chai(
    seed: int,
    workdir: str | Path,
    run_id: str,
    num_trunk_recycles: int,
    num_diffn_timesteps: int,
    num_diffn_samples: int,
    num_trunk_samples: int,
    **kwargs,  # ignore extra items from run config
) -> str:
    """Run Chai with the given ABCFold2 configuration."""
    from abcfold.chai1.run_chai1_abcfold import run_chai

    OUTPUTS_VOLUME.reload()
    work_path = Path(workdir).expanduser().resolve()
    chai_work_path = work_path / "chai_models"
    chai_conf_path = chai_work_path / f"{run_id}.yaml"
    if not chai_conf_path.exists():
        raise FileNotFoundError(f"Chai config file not found: {chai_conf_path}")

    template_hits_path = work_path / "msa" / "all_chain_templates.m8"
    if not template_hits_path.exists():
        template_hits_path = None
    chai_run_dir = run_chai(
        output_dir=chai_work_path,
        chai_yaml_file=chai_conf_path,
        seed=seed,
        template_hits_path=template_hits_path,
        template_cif_dir=work_path / "msa" / "templates",
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        num_diffn_samples=num_diffn_samples,
        num_trunk_samples=num_trunk_samples,
    )
    OUTPUTS_VOLUME.commit()
    return str(chai_run_dir)


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def submit_abcfold2_task(
    input_yaml: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    msa_chains: str | None = None,
    search_templates: bool = False,
    download_models: bool = False,
    force_redownload: bool = False,
    run_boltz: bool = True,
    run_chai: bool = True,
) -> None:
    """Run ABCFold2 on modal and fetch results to `out_dir`.

    Note that MSAs will be searched automatically. Templates will be searched only if
    `search_templates` is True.

    Args:
        input_yaml: Path to YAML design specification file
        out_dir: Optional output directory (defaults to $CWD)
        run_name: Optional run name (defaults to {input filename stem})
        msa_chains: Optional comma-separated list of chains to search MSAs for (defaults to all chains)
        search_templates: Whether to search for templates and add to input YAML
        download_models: Whether to download model weights before running
        force_redownload: Whether to force re-download of model weights
        run_boltz: Whether to run Boltz inference
        run_chai: Whether to run Chai inference
    """
    import json
    from pathlib import Path

    # Load input and find its hash
    yaml_path = Path(input_yaml).expanduser().resolve()
    yaml_str = yaml_path.read_bytes()

    if run_name is None:
        run_name = yaml_path.stem
    if not search_templates:
        run_name = f"{run_name}-no-tmpl"

    local_out_dir = (
        Path(out_dir) / run_name if out_dir is not None else Path.cwd() / run_name
    )
    if local_out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {local_out_dir}")

    print("🧬 Starting ABCFold2 run...")
    run_conf = prepare_abcfold2.remote(
        yaml_str=yaml_str, search_templates=search_templates, msa_chains=msa_chains
    )
    local_out_dir.mkdir(parents=True, exist_ok=True)
    with open(local_out_dir / "run-config.json", "w") as f:
        json.dump(run_conf, f, indent=2)

    if download_models:
        print("🧬 Checking Boltz inference dependencies...")
        download_boltz_models.remote(force=force_redownload)

        print("🧬 Checking Chai inference dependencies...")
        download_chai_models.remote(force=force_redownload)

    # Run Boltz for each seed
    if run_boltz:
        out_path = local_out_dir / "boltz_models.tar.zst"
        print(f"🧬 Running Boltz and collecting results to {out_path}")
        boltz_data = collect_abcfold2_boltz_data.remote(run_conf=run_conf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(boltz_data)

    # Run Chai for each seed
    if run_chai:
        out_path = local_out_dir / "chai_models.tar.zst"
        print(f"🧬 Running Chai and collecting results to {out_path}")
        chai_data = collect_abcfold2_chai_data.remote(run_conf=run_conf)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(chai_data)

    print(f"🧬 ABCFold2 run complete! Results saved to {local_out_dir}")
