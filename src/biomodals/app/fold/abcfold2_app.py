"""ABCFold2 source repo: <https://github.com/y1zhou/ABCFold/tree/feat/schema>.

## Additional notes on input flags

* MSAs will *always* be searched automatically, since omitting MSAs for Boltz/Chai translates to worse performance in most cases.
* Templates will be searched only if the `--search-templates` flag is passed. When multiple templates are found, only the top four will be used.
* The `--run-boltz` and `--run-chai` flags control whether to run structure prediction with the respective model. Inputs for both models will always be prepared for convenience.

## Outputs

* Results will be saved to the specified `--out-dir` under a subdirectory named after the `--run-name`.
* When `--no-search-templates` is passed, `-no-tmpl` will be appended to the run name.
* The output directory will contain a `run-config.json` file with the run parameters used.
* Inference results will be saved as `<model-name>_models.tar.zst` files. Extract them and analyze results using `abcfold2 postprocess`.
"""
# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
from dataclasses import dataclass
from pathlib import Path

import modal

from biomodals.app.config import AppConfig
from biomodals.app.constant import MODEL_VOLUME
from biomodals.helper import patch_image_for_helper
from biomodals.helper.shell import package_outputs
from biomodals.helper.web import download_files

##########################################
# Modal configs
##########################################
# TODO: migrate to uniaf3
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="ABCFold2",
    repo_url="https://github.com/y1zhou/ABCFold",
    repo_commit_hash="fcfdd49fbec0db73eb38dfad49f9649e81147337",
    package_name="abcfold",
    version="0.2.0",
    python_version="3.12",
    cuda_version="cu128",
    gpu=os.environ.get("GPU", "A10G"),
    timeout=int(os.environ.get("TIMEOUT", "3600")),
)
ChaiConf = AppConfig(
    name="Chai-1",
    repo_url="https://github.com/y1zhou/chai-lab",
    repo_commit_hash="0ac68311911bfcd28b118fc289437bf3eff8ac97",
    package_name="chai_lab",
    version="0.6.1",
)
BoltzConf = AppConfig(
    name="Boltz",
    repo_url="https://github.com/jwohlwend/boltz",
    repo_commit_hash="cb04aeccdd480fd4db707f0bbafde538397fa2ac",
    package_name="boltz",
    version="2.2.1",
)


@dataclass
class AppInfo:
    """Container for ABCFold2-specific configuration and constants."""

    abcfold_dir: str = str(CONF.git_clone_dir)
    boltz_model_hash: str = "6fdef46d763fee7fbb83ca5501ccceff43b85607"


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()

# Volumes
OUTPUTS_VOLUME = CONF.get_out_volume()
OUTPUTS_VOLUME_NAME = OUTPUTS_VOLUME.name
OUTPUTS_DIR = CONF.output_volume_mountpoint

download_image = patch_image_for_helper(
    modal.Image.debian_slim()
    .uv_pip_install("huggingface_hub>=1.10")
    .env(
        CONF.default_env
        | {
            "CHAI_DOWNLOADS_DIR": str(ChaiConf.model_dir),
            "BOLTZ_CACHE": str(BoltzConf.model_dir),
        }
    )
)

runtime_image = patch_image_for_helper(
    modal.Image.debian_slim()
    .apt_install("git", "build-essential")
    .env(
        CONF.default_env
        | {
            "CHAI_DOWNLOADS_DIR": str(ChaiConf.model_dir),
            "BOLTZ_CACHE": str(BoltzConf.model_dir),
        }
    )
    .run_commands(
        " && ".join(
            (
                # Clone Boltz and Chai
                f"git clone {BoltzConf.repo_url} {BoltzConf.git_clone_dir}",
                f"cd {BoltzConf.git_clone_dir}",
                f"git checkout {BoltzConf.repo_commit_hash}",
                f"git clone {ChaiConf.repo_url} {ChaiConf.git_clone_dir}",
                f"cd {ChaiConf.git_clone_dir}",
                f"git checkout {ChaiConf.repo_commit_hash}",
                # Setup ABCFold2 environment
                f"git clone {CONF.repo_url} {APP_INFO.abcfold_dir}",
                f"cd {APP_INFO.abcfold_dir}",
                f"git checkout {CONF.repo_commit_hash}",
                "uv venv --python 3.12",
                f"uv pip install {BoltzConf.git_clone_dir}[cuda] {ChaiConf.git_clone_dir}",
                "uv pip install .",
            ),
        )
    )
    .env({"PATH": f"{APP_INFO.abcfold_dir}/.venv/bin:$PATH"})
    .apt_install("kalign")  # for Chai templates
    .workdir(APP_INFO.abcfold_dir)
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes={str(BoltzConf.model_volume_mountpoint): MODEL_VOLUME},
    timeout=CONF.timeout,
    image=download_image,
)
def download_boltz_models(force: bool = False) -> None:
    """Download Boltz models into the mounted volume.

    From: https://modal.com/docs/examples/boltz_predict.
    """
    import tarfile

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="boltz-community/boltz-2",
        revision=APP_INFO.boltz_model_hash,
        local_dir=BoltzConf.model_dir,
        force_download=force,
    )
    boltz_download_dir = BoltzConf.model_dir
    tar_mols = boltz_download_dir / "mols.tar"
    if not (boltz_download_dir / "mols").exists():
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(boltz_download_dir)  # noqa: S202

    MODEL_VOLUME.commit()


@app.function(
    volumes={str(ChaiConf.model_volume_mountpoint): MODEL_VOLUME},
    timeout=CONF.timeout,
    image=download_image,
)
async def download_chai_models(force=False):
    """From https://modal.com/docs/examples/chai1."""
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

    # launch downloads concurrently
    chai_model_dir = ChaiConf.model_dir
    download_tasks = {
        f"{base_url}{dep}": chai_model_dir / dep for dep in inference_dependencies
    }
    download_files(download_tasks, progress_bar_desc="Downloading Chai models")

    # Special treatment for ESM
    esm2_path = chai_model_dir / "esm2" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
    esm_path = chai_model_dir / "esm" / "traced_sdpa_esm2_t36_3B_UR50D_fp16.pt"
    if esm2_path.exists() and not esm_path.exists():
        esm_path.parent.mkdir(parents=True, exist_ok=True)
        esm_path.symlink_to(esm2_path)

    # ensures models are visible on remote filesystem before exiting,
    # otherwise takes a few seconds, racing with inference
    MODEL_VOLUME.commit()


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


@app.function(image=runtime_image, timeout=CONF.timeout)
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
    timeout=CONF.timeout,
    volumes={
        OUTPUTS_DIR: OUTPUTS_VOLUME,
        BoltzConf.model_volume_mountpoint: MODEL_VOLUME,
    },
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
            ccd_lib_dir=BoltzConf.model_dir / "mols",
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
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=CONF.timeout,
    volumes={
        OUTPUTS_DIR: OUTPUTS_VOLUME,
        BoltzConf.model_volume_mountpoint: MODEL_VOLUME,
    },
)
def collect_abcfold2_boltz_data(
    run_conf: dict[str, str | list[int] | int | list[str] | None],
):
    """Manage Boltz runs and return all Boltz results."""
    from pathlib import Path

    work_path = Path(str(run_conf["workdir"])).expanduser().resolve()
    run_id = run_conf["run_id"]
    work_path = work_path / "boltz_models"
    boltz_conf_path = work_path / f"{run_id}.yaml"
    OUTPUTS_VOLUME.reload()

    if not boltz_conf_path.exists():
        raise FileNotFoundError(f"Boltz config file not found: {boltz_conf_path}")

    random_seeds = run_conf.get("seeds", [])
    if not isinstance(random_seeds, list):
        if random_seeds is None:
            random_seeds = []
        else:
            random_seeds = [int(random_seeds)]
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
    print("💊 Packaging Boltz results...")
    boltz_tarball_bytes = package_outputs(
        work_path,
        tar_args=[
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
    return boltz_tarball_bytes


@app.function(
    gpu=CONF.gpu,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=CONF.timeout,
    volumes={
        OUTPUTS_DIR: OUTPUTS_VOLUME,
        BoltzConf.model_volume_mountpoint: MODEL_VOLUME,
    },
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
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=CONF.timeout,
    volumes={
        OUTPUTS_DIR: OUTPUTS_VOLUME,
        ChaiConf.model_volume_mountpoint: MODEL_VOLUME,
    },
)
def collect_abcfold2_chai_data(
    run_conf: dict[str, str | list[int] | int | list[str] | None],
):
    """Manage Chai runs and return all Chai results."""
    from pathlib import Path

    work_path = Path(str(run_conf["workdir"])).expanduser().resolve()
    run_id = run_conf["run_id"]
    work_path = work_path / "chai_models"
    chai_conf_path = work_path / f"{run_id}.yaml"
    OUTPUTS_VOLUME.reload()

    if not chai_conf_path.exists():
        raise FileNotFoundError(f"Chai config file not found: {chai_conf_path}")

    random_seeds = run_conf.get("seeds", [])
    if not isinstance(random_seeds, list):
        if random_seeds is None:
            random_seeds = []
        else:
            random_seeds = [int(random_seeds)]
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
    print("💊 Packaging Chai results...")
    chai_tarball_bytes = package_outputs(work_path)
    return chai_tarball_bytes


@app.function(
    gpu=CONF.gpu,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=CONF.timeout,
    volumes={
        OUTPUTS_DIR: OUTPUTS_VOLUME,
        ChaiConf.model_volume_mountpoint: MODEL_VOLUME,
    },
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

    Note that MSAs will be searched automatically. Templates will be searched
    only if `search_templates` is True.

    Args:
        input_yaml: Path to YAML design specification file. For a detailed
            description of the YAML schema, see
            <https://github.com/y1zhou/ABCFold/blob/feat/schema/abcfold/schema.py>.
        out_dir: Optional output directory. If not specified, outputs will
            be saved in the current working directory.
        run_name: Optional name for the output directory. Defaults to the
            stem of the input YAML file.
        msa_chains: Optional comma-separated list of chains to search MSAs for.
            If not specified, MSAs will be searched for all chains.
        search_templates: Whether to search for templates and add to input YAML.
        download_models: Whether to download model weights and skip running.
        force_redownload: Whether to force re-download of model weights.
        run_boltz: Whether to run Boltz inference.
        run_chai: Whether to run Chai inference.
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
    inference_tasks: list[modal.FunctionCall] = []
    output_paths: list[Path] = []
    if run_boltz:
        out_path = local_out_dir / "boltz_models.tar.zst"
        print(f"🧬 Running Boltz and collecting results to {out_path}")
        boltz_task = collect_abcfold2_boltz_data.spawn(run_conf=run_conf)
        inference_tasks.append(boltz_task)
        output_paths.append(out_path)

    # Run Chai for each seed
    if run_chai:
        out_path = local_out_dir / "chai_models.tar.zst"
        print(f"🧬 Running Chai and collecting results to {out_path}")
        chai_task = collect_abcfold2_chai_data.spawn(run_conf=run_conf)
        inference_tasks.append(chai_task)
        output_paths.append(out_path)

    if not inference_tasks:
        print("🧬 No inference tasks specified, exiting...")
        return

    inference_data = modal.FunctionCall.gather(*inference_tasks)
    for out_path, data in zip(output_paths, inference_data, strict=True):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)

    print(f"🧬 ABCFold2 run complete! Results saved to {local_out_dir}")
