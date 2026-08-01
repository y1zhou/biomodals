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
from hashlib import sha256
from pathlib import Path, PurePosixPath
from stat import S_ISREG
from uuid import UUID, uuid4

import modal
import orjson

from biomodals.app.config import AppConfig
from biomodals.app.fold.abcfold2_execution import (
    ABCFold2ExecutionCoordinator,
    ABCFold2ExecutionRequest,
    load_execution_request,
    run_config_from_snapshot,
    stage_execution_request,
)
from biomodals.execution import DeploymentIdentity, ExecutionSnapshot, RunStatus
from biomodals.execution.modal import (
    ModalCallDriver,
    deployed_execution_coordinator,
    development_modal_call_driver,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.app_run import AppRunLayout
from biomodals.helper.constant import MODEL_VOLUME
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

download_image = (
    modal.Image
    .debian_slim()
    .uv_pip_install("huggingface_hub>=1.10")
    .env(
        CONF.default_env
        | {
            "CHAI_DOWNLOADS_DIR": ChaiConf.model_volume_mountpoint,
            "BOLTZ_CACHE": BoltzConf.model_volume_mountpoint,
        }
    )
    .pipe(patch_image_for_helper)
)

runtime_image = (
    modal.Image
    .debian_slim()
    .apt_install("git", "build-essential")
    .env(
        CONF.default_env
        | {
            "CHAI_DOWNLOADS_DIR": ChaiConf.model_volume_mountpoint,
            "BOLTZ_CACHE": BoltzConf.model_volume_mountpoint,
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
    .pipe(patch_image_for_helper)
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
ABCFOLD2_OUTPUT_CLAIMS = modal.Dict.from_name(
    f"{CONF.name}-output-claims",
    create_if_missing=True,
)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_abcfold2_task"})
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8
_DEFAULT_MAX_ACTIVE_PROVIDER_CALLS = 64


##########################################
# Fetch model weights
##########################################
@app.function(
    volumes=BoltzConf.mounts(model_volume=True, model_ro=False, is_huggingface=True),
    timeout=CONF.timeout,
    image=download_image,
)
def download_boltz_models(force: bool = False) -> None:
    """Download Boltz models into the mounted volume.

    From: https://modal.com/docs/examples/boltz_predict.
    """
    import tarfile

    from huggingface_hub import snapshot_download  # type: ignore[ty:unresolved-import]

    boltz_download_dir = Path(BoltzConf.model_volume_mountpoint)
    snapshot_download(
        repo_id="boltz-community/boltz-2",
        revision=APP_INFO.boltz_model_hash,
        local_dir=boltz_download_dir,
        force_download=force,
    )
    MODEL_VOLUME.commit()

    tar_mols = boltz_download_dir / "mols.tar"
    if not (boltz_download_dir / "mols").exists():
        with tarfile.open(str(tar_mols), "r") as tar:
            tar.extractall(boltz_download_dir)  # noqa: S202
    MODEL_VOLUME.commit()


@app.function(
    volumes=ChaiConf.mounts(model_volume=True, model_ro=False),
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
    chai_model_dir = Path(ChaiConf.model_volume_mountpoint)
    download_tasks = {
        f"{base_url}{dep}": chai_model_dir / dep for dep in inference_dependencies
    }
    download_files(download_tasks, progress_bar_desc="Downloading Chai models")
    MODEL_VOLUME.commit()

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
    from abcfold.schema import load_abcfold_config  # type: ignore[ty:unresolved-import]

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

    from abcfold.schema import load_abcfold_config  # type: ignore[ty:unresolved-import]

    # Determine content-based run ID
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_yaml_path = Path(tmpdir) / "abcfold-config.yaml"
        tmp_yaml_path.write_bytes(yaml_str)
        conf = load_abcfold_config(tmp_yaml_path)

    return conf.hash


@app.function(
    image=runtime_image,
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True) | BoltzConf.mounts(model_volume=True),
)
def prepare_abcfold2(
    yaml_str: bytes, search_templates: bool, msa_chains: str | None = None
) -> dict[str, str | list[int] | int | list[str] | None]:
    """Prepare inputs to Boltz and Chai using ABCFold2 config."""
    import tempfile
    from pathlib import Path

    from abcfold.cli.prepare import (  # type: ignore[ty:unresolved-import]
        prepare_boltz,
        prepare_chai,
        search_msa,
    )

    run_id: str = get_run_id.local(yaml_str=yaml_str)
    if not search_templates:
        run_id = f"{run_id}-no-tmpl"
    layout = AppRunLayout.from_run_root(
        Path(CONF.output_volume_mountpoint) / run_id[:2] / run_id
    )
    # TODO: ABCFold2 upstream writes msa/, boltz_models/, and chai_models/
    # below the run root. A workflow-compatible wrapper could map these into
    # inputs/prep/outputs without changing the cached upstream workdir.
    out_dir_full = layout.run_root
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
                template_cache_dir=Path(CONF.output_volume_mountpoint)
                / ".cache"
                / "rcsb",
            )
            CONF.output_volume.commit()

    # Generate inputs for Boltz and Chai
    if not (out_dir_full / "boltz_models" / f"{run_id}.yaml").exists():
        _ = prepare_boltz(conf_file=yaml_path, out_dir=out_dir_full)
        CONF.output_volume.commit()
    if not (out_dir_full / "chai_models" / f"{run_id}.yaml").exists():
        _ = prepare_chai(
            conf_file=yaml_path,
            out_dir=out_dir_full,
            ccd_lib_dir=Path(BoltzConf.model_volume_mountpoint) / "mols",
        )
        CONF.output_volume.commit()

    # Pull run parameters from YAML
    conf = load_params_from_run_yaml(yaml_path)
    conf["run_id"] = run_id
    conf["workdir"] = str(out_dir_full)
    conf["seeds"] = new_conf["seeds"]  # ensure seeds are up to date
    return conf


def _archive_path(workdir: str | Path, model_name: str) -> Path:
    """Return one model branch's stable archive publication path."""
    return Path(workdir) / f"{model_name}_models.tar.zst"


def _publication_dir(workdir: str | Path) -> Path:
    return Path(workdir) / ".biomodals"


def _model_publication_key(model_name: str, run_conf: dict[str, object]) -> str:
    """Identify model outputs whose upstream run ID intentionally omits seeds."""
    payload = {
        "model": model_name,
        "run_id": run_conf.get("run_id"),
        "seeds": run_conf.get("seeds"),
        "num_trunk_recycles": run_conf.get("num_trunk_recycles"),
        "num_diffn_timesteps": run_conf.get("num_diffn_timesteps"),
        "num_diffn_samples": run_conf.get("num_diffn_samples"),
        "num_trunk_samples": run_conf.get("num_trunk_samples"),
        "boltz_additional_cli_args": run_conf.get("boltz_additional_cli_args"),
        "abcfold2_version": CONF.repo_commit_hash or CONF.version,
        "model_version": (
            BoltzConf.repo_commit_hash or BoltzConf.version
            if model_name == "boltz"
            else ChaiConf.repo_commit_hash or ChaiConf.version
        ),
    }
    return sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _archive_ready(
    workdir: str | Path,
    model_name: str,
    publication_key: str,
) -> bool:
    """Return whether a complete atomic archive publication exists."""
    path = _archive_path(workdir, model_name)
    marker_path = _publication_dir(workdir) / f"{model_name}-archive.json"
    try:
        marker = orjson.loads(marker_path.read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("publication_key") == publication_key
        and marker.get("archive_path") == str(path)
        and _publication_file_matches(
            path,
            marker.get("size"),
            marker.get("sha256"),
        )
    )


def _seed_ready(
    workdir: str | Path,
    model_name: str,
    seed: int,
    publication_key: str,
) -> bool:
    """Return whether a worker published one complete model seed."""
    model_dir = Path(workdir) / f"{model_name}_models"
    directory = (
        model_dir / f"boltz_results_seed-{seed}"
        if model_name == "boltz"
        else model_dir / f"chai_seed-{seed}"
    )
    marker_path = _publication_dir(workdir) / f"{model_name}-seed-{seed}.json"
    try:
        marker = orjson.loads(marker_path.read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("publication_key") == publication_key
        and marker.get("result_path") == str(directory)
        and _directory_publication_matches(directory, marker.get("artifacts"))
    )


def _write_publication_marker(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(orjson.dumps(value, option=orjson.OPT_SORT_KEYS))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _publication_file_matches(
    path: Path,
    expected_size: object,
    expected_digest: object,
) -> bool:
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
        or not isinstance(expected_digest, str)
        or len(expected_digest) != 64
    ):
        return False
    try:
        if path.is_symlink():
            return False
        stat = path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return (
        S_ISREG(stat.st_mode)
        and stat.st_size == expected_size
        and _sha256_file(path) == expected_digest
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_artifacts(directory: Path) -> list[dict[str, str | int]]:
    artifacts = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("ABCFold2 seed output contains a symbolic link")
        if path.is_dir():
            continue
        stat = path.stat()
        if not S_ISREG(stat.st_mode):
            raise RuntimeError("ABCFold2 seed output contains a non-regular file")
        artifacts.append({
            "path": path.relative_to(directory).as_posix(),
            "size": stat.st_size,
            "sha256": _sha256_file(path),
        })
    if not artifacts:
        raise RuntimeError("ABCFold2 seed output contains no files")
    return artifacts


def _directory_publication_matches(directory: Path, raw_artifacts: object) -> bool:
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        return False
    seen: set[str] = set()
    for artifact in raw_artifacts:
        if not isinstance(artifact, dict):
            return False
        relative_text = artifact.get("path")
        if not isinstance(relative_text, str) or relative_text in seen:
            return False
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            return False
        seen.add(relative_text)
        if not _publication_file_matches(
            directory.joinpath(*relative.parts),
            artifact.get("size"),
            artifact.get("sha256"),
        ):
            return False
    return True


def _publish_archive(
    workdir: Path,
    model_name: str,
    content: bytes,
    publication_key: str,
) -> dict[str, str | int]:
    """Atomically publish one bounded provider result into the output volume."""
    path = _archive_path(workdir, model_name)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha256(content).hexdigest()
    _write_publication_marker(
        _publication_dir(workdir) / f"{model_name}-archive.json",
        {
            "publication_key": publication_key,
            "archive_path": str(path),
            "size": len(content),
            "sha256": digest,
        },
    )
    CONF.output_volume.commit()
    return {
        "archive_path": str(path),
        "size": len(content),
        "sha256": digest,
    }


@app.function(
    cpu=(0.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True) | BoltzConf.mounts(model_volume=True),
)
def collect_abcfold2_boltz_data(
    run_conf: dict[str, str | list[int] | int | list[str] | None],
    publication_key: str,
) -> dict[str, str | int]:
    """Package completed Boltz seed results into the output volume."""
    from pathlib import Path

    work_path = Path(str(run_conf["workdir"])).expanduser().resolve()
    run_id = run_conf["run_id"]
    work_path = work_path / "boltz_models"
    boltz_conf_path = work_path / f"{run_id}.yaml"
    CONF.output_volume.reload()

    if not boltz_conf_path.exists():
        raise FileNotFoundError(f"Boltz config file not found: {boltz_conf_path}")

    print("💊 Packaging Boltz results...")
    archive = package_outputs(
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
    return _publish_archive(
        Path(str(run_conf["workdir"])),
        "boltz",
        archive,
        publication_key,
    )


@app.function(
    gpu=CONF.gpu,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True) | BoltzConf.mounts(model_volume=True),
)
def run_abcfold2_boltz(
    seed: int,
    workdir: str | Path,
    run_id: str,
    num_trunk_recycles: int,  # recycling_steps
    num_diffn_timesteps: int,  # sampling_steps
    num_diffn_samples: int,  # diffusion_samples
    boltz_additional_cli_args: list[str] | None,
    publication_key: str,
    **kwargs,  # ignore extra items from run config
) -> str:
    """Run Boltz with the given ABCFold2 configuration."""
    from abcfold.boltz.run_boltz_abcfold import (  # type: ignore[ty:unresolved-import]
        run_boltz,
    )

    CONF.output_volume.reload()
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
    _write_publication_marker(
        _publication_dir(workdir) / f"boltz-seed-{seed}.json",
        {
            "publication_key": publication_key,
            "result_path": str(boltz_run_dir),
            "artifacts": _directory_artifacts(Path(boltz_run_dir)),
        },
    )
    CONF.output_volume.commit()
    return str(boltz_run_dir)


@app.function(
    cpu=(0.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True) | ChaiConf.mounts(model_volume=True),
)
def collect_abcfold2_chai_data(
    run_conf: dict[str, str | list[int] | int | list[str] | None],
    publication_key: str,
) -> dict[str, str | int]:
    """Package completed Chai seed results into the output volume."""
    from pathlib import Path

    work_path = Path(str(run_conf["workdir"])).expanduser().resolve()
    run_id = run_conf["run_id"]
    work_path = work_path / "chai_models"
    chai_conf_path = work_path / f"{run_id}.yaml"
    CONF.output_volume.reload()

    if not chai_conf_path.exists():
        raise FileNotFoundError(f"Chai config file not found: {chai_conf_path}")

    print("💊 Packaging Chai results...")
    archive = package_outputs(work_path)
    return _publish_archive(
        Path(str(run_conf["workdir"])),
        "chai",
        archive,
        publication_key,
    )


@app.function(
    gpu=CONF.gpu,
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    image=runtime_image,
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True) | ChaiConf.mounts(model_volume=True),
)
def run_abcfold2_chai(
    seed: int,
    workdir: str | Path,
    run_id: str,
    num_trunk_recycles: int,
    num_diffn_timesteps: int,
    num_diffn_samples: int,
    num_trunk_samples: int,
    publication_key: str,
    **kwargs,  # ignore extra items from run config
) -> str:
    """Run Chai with the given ABCFold2 configuration."""
    from abcfold.chai1.run_chai1_abcfold import (  # type: ignore[ty:unresolved-import]
        run_chai,
    )

    CONF.output_volume.reload()
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
    _write_publication_marker(
        _publication_dir(workdir) / f"chai-seed-{seed}.json",
        {
            "publication_key": publication_key,
            "result_path": str(chai_run_dir),
            "artifacts": _directory_artifacts(Path(chai_run_dir)),
        },
    )
    CONF.output_volume.commit()
    return str(chai_run_dir)


##########################################
# Deployment-local execution coordinator
##########################################
@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    max_containers=1,
    volumes=CONF.mounts(output_volume=True),
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with ABCFold2 functions."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh output state before accepting lifecycle methods."""
        self._coordinator_adapter = None
        self._development = None
        self._identity()
        CONF.output_volume.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionSnapshot:
        """Drive one staged root App Run until it stops."""
        return self._adapter(development=development).run()

    @modal.method()
    def status(self) -> ExecutionSnapshot:
        """Read this Run's durable kernel snapshot."""
        return self._adapter().status()

    @modal.method()
    def cancel(self) -> ExecutionSnapshot:
        """Request idempotent cancellation for this Run."""
        return self._adapter().cancel()

    @modal.method()
    def resume(self) -> ExecutionSnapshot:
        """Resume this Run without retrying failed Tasks."""
        return self._adapter().resume()

    @modal.method()
    def restart(
        self,
        predecessor_execution_run_id: str,
        predecessor_deployment_environment: str,
        predecessor_deployment_name: str,
        predecessor_deployment_version: int,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> ExecutionSnapshot:
        """Create and drive one compatible Successor Run."""
        return self._adapter().restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=DeploymentIdentity(
                predecessor_deployment_environment,
                predecessor_deployment_name,
                predecessor_deployment_version,
            ),
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
        )

    @modal.method()
    def restart_from(
        self,
        predecessor_execution_run_id: str,
    ) -> ExecutionSnapshot:
        """Create a compatible Successor while inferring predecessor identity."""
        return self._adapter().restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            candidate_request=load_execution_request(
                CONF.output_volume_mountpoint,
                UUID(self.execution_run_id),
            ),
        )

    @modal.exit()
    def exit(self) -> None:
        """Close local state without cancelling attached calls."""
        adapter = getattr(self, "_coordinator_adapter", None)
        if adapter is not None:
            adapter.close()

    def _identity(self) -> tuple[UUID, DeploymentIdentity]:
        return (
            UUID(self.execution_run_id),
            DeploymentIdentity(
                self.deployment_environment,
                self.deployment_name,
                self.deployment_version,
            ),
        )

    def _adapter(
        self,
        *,
        development: bool | None = None,
    ) -> ABCFold2ExecutionCoordinator:
        adapter = getattr(self, "_coordinator_adapter", None)
        selected_mode = getattr(self, "_development", None)
        if adapter is not None:
            if development is not None and selected_mode != development:
                raise ValueError("Coordinator execution mode cannot change in place")
            return adapter
        execution_run_id, deployment = self._identity()
        selected_mode = False if development is None else development
        adapter = ABCFold2ExecutionCoordinator(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=Path(CONF.output_volume_mountpoint),
            output_volume=CONF.output_volume,
            output_claims=ABCFOLD2_OUTPUT_CLAIMS,
            modal_driver=_coordinator_modal_driver(development=selected_mode),
        )
        self._coordinator_adapter = adapter
        self._development = selected_mode
        return adapter


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    """Resolve exact deployed functions or current-source handles."""
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {
            "prepare_abcfold2": prepare_abcfold2,
            "download_boltz_models": download_boltz_models,
            "download_chai_models": download_chai_models,
            "run_abcfold2_boltz": run_abcfold2_boltz,
            "collect_abcfold2_boltz_data": collect_abcfold2_boltz_data,
            "run_abcfold2_chai": run_abcfold2_chai,
            "collect_abcfold2_chai_data": collect_abcfold2_chai_data,
        },
        workload_name="ABCFold2",
    )


def _execution_coordinator_handle(
    *,
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
    use_deployed_coordinator: bool,
):
    """Resolve this run's exact deployed or current-source coordinator."""
    if use_deployed_coordinator:
        return deployed_execution_coordinator(
            execution_run_id=execution_run_id,
            deployment=deployment,
        )
    return ExecutionCoordinator(
        execution_run_id=str(execution_run_id),
        deployment_environment=deployment.environment,
        deployment_name=deployment.deployment_name,
        deployment_version=deployment.deployment_version,
    )


##########################################
# Local entrypoint client
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
    max_parallel_children: int | None = None,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
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
        max_parallel_children: Maximum number of child inference containers to
            run at once in each ABCFold2 coordinator.
        use_deployed_coordinator: Target the exact deployed coordinator. The
            Biomodals CLI supplies this for normal runs.
        deployment_environment: Modal Environment containing the coordinator.
        deployment_name: Exact deployed Modal app name.
        deployment_version: Exact numeric deployment version.
        restart_from: Optional predecessor Execution Run ID for a Successor Run.
    """
    yaml_path = Path(input_yaml).expanduser().resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"ABCFold2 YAML not found: {yaml_path}")
    predecessor_execution_run_id = None if restart_from is None else UUID(restart_from)
    run_name = run_name or yaml_path.stem
    if not search_templates:
        run_name = f"{run_name}-no-tmpl"
    capacity = (
        _DEFAULT_MAX_ACTIVE_PROVIDER_CALLS
        if max_parallel_children is None
        else max_parallel_children
    )
    request = ABCFold2ExecutionRequest(
        run_name=run_name,
        yaml_content=yaml_path.read_bytes(),
        msa_chains=msa_chains,
        search_templates=search_templates,
        download_models=download_models,
        force_redownload=force_redownload,
        run_boltz=run_boltz,
        run_chai=run_chai,
        max_active_provider_calls=capacity,
        app_version=CONF.repo_commit_hash or CONF.version or "unknown",
        boltz_version=BoltzConf.repo_commit_hash or BoltzConf.version or "unknown",
        chai_version=ChaiConf.repo_commit_hash or ChaiConf.version or "unknown",
    )

    local_out_dir = (
        Path(out_dir) / run_name if out_dir is not None else Path.cwd() / run_name
    )
    if local_out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {local_out_dir}")

    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        deployment_environment,
        deployment_name,
        deployment_version,
    )
    stage_execution_request(CONF.output_volume, execution_run_id, request)
    coordinator = _execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
    )
    if predecessor_execution_run_id is None:
        call = coordinator.run.spawn(development=not use_deployed_coordinator)
    else:
        call = coordinator.restart_from.spawn(
            predecessor_execution_run_id=str(predecessor_execution_run_id),
        )
    print(f"Execution Run ID: {execution_run_id}")
    print(
        "Deployment Identity: "
        f"{deployment.environment}/{deployment.deployment_name}/"
        f"v{deployment.deployment_version}"
    )
    print(f"Coordinator FunctionCall ID: {call.object_id}")
    snapshot = call.get()
    if snapshot.run.status != RunStatus.SUCCEEDED:
        diagnostic = snapshot.run.status_message or (
            snapshot.run.status_reason.value
            if snapshot.run.status_reason is not None
            else snapshot.run.status.value
        )
        raise RuntimeError(
            f"{CONF.name} Execution Run ended as "
            f"{snapshot.run.status.value}: {diagnostic}"
        )

    run_conf = run_config_from_snapshot(snapshot)
    local_out_dir.mkdir(parents=True, exist_ok=True)
    local_run_conf = run_conf.as_kwargs()
    local_run_conf["max_parallel_children"] = max_parallel_children
    (local_out_dir / "run-config.json").write_bytes(
        orjson.dumps(local_run_conf, option=orjson.OPT_INDENT_2),
    )
    for model_name, enabled in (
        ("boltz", request.run_boltz),
        ("chai", request.run_chai),
    ):
        if not enabled:
            continue
        remote_path = _archive_path(run_conf.workdir, model_name)
        relative_path = remote_path.relative_to(CONF.output_volume_mountpoint)
        data = b"".join(CONF.output_volume.read_file(relative_path.as_posix()))
        (local_out_dir / remote_path.name).write_bytes(data)

    print(f"🧬 ABCFold2 run complete! Results saved to {local_out_dir}")
