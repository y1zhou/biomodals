"""Protenix source repo: <https://github.com/y1zhou/Protenix>.

## Notes

* The default `--msa-server-mode protenix` uses the Protenix remote MSA server,
  so no local MSA databases are required. Switch to `colabfold` if you have a
  pre-populated database volume.
* MSA/template preprocessing is run in a CPU-only Modal function and cached in a
  persistent Modal volume before GPU inference.
* Templates are only used when `--use-template` is passed. Template support
  requires the v1.0.0 model checkpoints.
* RNA MSA is only supported by v1.0.0 model checkpoints.
* The `protenix_base_constraint_v0.5.0` model supports pocket, contact, and
  substructure constraints specified in the input JSON.
* For large structures (>2000 tokens), consider using an A100 (80GB) or H100
  GPU by setting the `GPU` environment variable.

## Outputs

* Results will be saved to the specified `--out-dir` as `<run-name>.tar.zst`.
* For prediction runs, the tarball contains predicted `.cif` structure files and
  `*_summary_confidence.json` files with pLDDT, pAE, and ranking scores.
* For `--score-only` runs, the tarball contains per-structure confidence JSON
  files produced by `protenixscore score`.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from stat import S_ISREG
from uuid import UUID, uuid4

import modal

from biomodals.app.config import AppConfig
from biomodals.app.fold.protenix_execution import (
    ProtenixExecutionCoordinator,
    ProtenixExecutionRequest,
    ProtenixMsaTaskSpec,
    ProtenixPreparationPlan,
    load_execution_request,
    stage_execution_request,
)
from biomodals.execution import DeploymentIdentity, ExecutionSnapshot, RunStatus
from biomodals.execution.modal import (
    ModalCallDriver,
    development_modal_call_driver,
    execution_coordinator_adapter,
    execution_coordinator_identity,
    initialize_execution_coordinator_host,
)
from biomodals.execution.modal import (
    execution_coordinator_handle as _execution_coordinator_handle,
)
from biomodals.helper import hash_string, patch_image_for_helper
from biomodals.helper.constant import (
    MAX_TIMEOUT,
    MODEL_VOLUME,
    MSA_CACHE_VOLUME,
)
from biomodals.helper.io import (
    build_local_output_path,
    resolve_local_output_dir,
    write_local_tarball,
)
from biomodals.helper.shell import (
    package_outputs,
    run_command,
    sanitize_filename,
)
from biomodals.helper.structure import struct2seq
from biomodals.helper.web import download_files

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="Protenix",
    repo_url="https://github.com/y1zhou/Protenix",
    repo_commit_hash="7e1de70749910c401339dd49aa62735510c22959",
    package_name="protenix",
    version="2.0.0",
    python_version="3.11",
    cuda_version="cu128",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "3600")),
)


@dataclass
class AppInfo:
    """Container for app-specific configuration and constants."""

    # https://modal.com/docs/guide/cuda#for-more-complex-setups-use-an-officially-supported-cuda-image
    # CUDA version should be no greater than host CUDA version
    # "devel" image includes the full CUDA toolkit, which is required for
    # building custom LayerNorm kernels
    cuda_tag = f"{CONF.cuda_version_numeric}-devel-ubuntu24.04"

    # Volume for preprocessed MSA/template intermediates (MSA_CACHE_VOLUME)
    msa_cache_volume_subdir: str = f"/{CONF.name}"
    msa_cache_mountpoint: str = "/msa-cache"

    # Base URL for downloading checkpoints and data caches
    # https://github.com/bytedance/Protenix/blob/main/protenix/web_service/dependency_url.py
    base_url: str = "https://protenix.tos-cn-beijing.volces.com"

    # Supported model checkpoints
    supported_models: Sequence[str] = (
        "protenix_base_default_v1.0.0",
        "protenix_base_20250630_v1.0.0",
        # "protenix-v2.pt",  # TODO: keep an eye on protenix-v2
    )
    # CCD and other data caches required for inference
    data_cache: Sequence[str] = (
        "common/components.cif",
        "common/components.cif.rdkit_mol.pkl",
        "common/clusters-by-entity-40.txt",
        "common/obsolete_release_date.csv",
    )
    # Additional files needed when templates are enabled
    template_cache: Sequence[str] = (
        "common/obsolete_to_successor.json",
        "common/release_date_cache.json",
    )


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()
runtime_image = (
    modal.Image
    .from_registry(f"nvidia/cuda:{APP_INFO.cuda_tag}", add_python=CONF.python_version)
    .entrypoint([])  # remove verbose logging in the base image
    .apt_install("git", "build-essential", "zstd", "hmmer", "kalign", "wget")
    .env(
        CONF.default_env
        | {
            "PYTHONUNBUFFERED": "1",
            "PROTENIX_ROOT_DIR": CONF.model_volume_mountpoint,
            "PROTENIX_CHECKPOINT_DIR": str(
                Path(CONF.model_volume_mountpoint) / "checkpoint"
            ),
        }
    )
    .uv_pip_install(
        f"{CONF.package_name}[{CONF.cuda_version}] @ "
        f"git+{CONF.repo_url}@{CONF.repo_commit_hash}"
    )
    # Trigger kernel compilation
    .run_commands(
        "python -m protenix.model.layer_norm.layer_norm",
        gpu=CONF.gpu,
        env={"LAYERNORM_TYPE": "fast_layernorm"},  # default, but just in case
    )
    .pipe(patch_image_for_helper)
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
PROTENIX_OUTPUT_CLAIMS = modal.Dict.from_name(
    f"{CONF.name}-output-claims",
    create_if_missing=True,
)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_protenix_task"})
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8
_DEFAULT_MAX_ACTIVE_PROVIDER_CALLS = 64


def _msa_task_marker_path(task: ProtenixMsaTaskSpec) -> Path:
    return Path(task.output_dir) / f".{task.task_key}.complete.json"


def _msa_task_ready(task: ProtenixMsaTaskSpec) -> bool:
    """Return whether one query published its expected updated JSON."""
    import orjson

    try:
        marker = orjson.loads(_msa_task_marker_path(task).read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return False
    expected = Path(task.expected_json_path)
    return (
        isinstance(marker, dict)
        and marker.get("publication_key") == task.publication_key
        and marker.get("expected_json_path") == str(expected)
        and _publication_file_matches(
            expected,
            marker.get("size"),
            marker.get("sha256"),
        )
    )


def _prepared_marker_path(plan: ProtenixPreparationPlan) -> Path:
    return Path(plan.prepared_json_path).with_suffix(".complete.json")


def _prepared_ready(plan: ProtenixPreparationPlan) -> bool:
    """Return whether all searched tasks were assembled into prepared JSON."""
    import orjson

    path = Path(plan.prepared_json_path)
    try:
        marker = orjson.loads(_prepared_marker_path(plan).read_bytes())
    except (
        FileNotFoundError,
        IsADirectoryError,
        NotADirectoryError,
        orjson.JSONDecodeError,
    ):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("preparation_key") == plan.preparation_key
        and _publication_file_matches(
            path,
            marker.get("size"),
            marker.get("sha256"),
        )
    )


def _result_path(result_key: str, run_name: str) -> Path:
    return (
        Path(CONF.output_volume_mountpoint)
        / "cache"
        / result_key[:2]
        / result_key
        / f"{run_name}.tar.zst"
    )


def _result_ready(result_key: str, run_name: str) -> bool:
    """Return whether one atomic result archive matches its completion marker."""
    import orjson

    path = _result_path(result_key, run_name)
    marker_path = path.with_suffix(f"{path.suffix}.complete.json")
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
        and marker.get("result_key") == result_key
        and _publication_file_matches(
            path,
            marker.get("size"),
            marker.get("sha256"),
        )
    )


def _publication_file_matches(
    path: Path,
    expected_size: object,
    expected_digest: object,
) -> bool:
    """Validate one regular artifact without hiding inconclusive I/O errors."""
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
    if not S_ISREG(stat.st_mode) or stat.st_size != expected_size:
        return False
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == expected_digest


def _publish_result(
    result_key: str,
    run_name: str,
    content: bytes,
) -> dict[str, str | int]:
    """Atomically publish one result archive and its completion evidence."""
    import orjson

    path = _result_path(result_key, run_name)
    _atomic_write(path, content)
    digest = sha256(content).hexdigest()
    _atomic_write(
        path.with_suffix(f"{path.suffix}.complete.json"),
        orjson.dumps({
            "result_key": result_key,
            "size": len(content),
            "sha256": digest,
        }),
    )
    CONF.output_volume.commit()
    return {"result_path": str(path), "size": len(content), "sha256": digest}


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


##########################################
# Fetch model weights and data caches
##########################################
@app.function(
    volumes=CONF.mounts(model_volume=True, model_ro=False), timeout=CONF.timeout
)
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
    data_root = Path(CONF.model_volume_mountpoint)
    files_to_download: dict[str, str | Path] = {}

    # Download common data caches
    data_caches = {
        f"{APP_INFO.base_url}/{rel_path}": data_root / rel_path
        for rel_path in APP_INFO.data_cache
    }
    files_to_download = files_to_download | data_caches

    # Download template data if requested
    if include_templates:
        template_caches = {
            f"{APP_INFO.base_url}/{rel_path}": data_root / rel_path
            for rel_path in APP_INFO.template_cache
        }
        files_to_download = files_to_download | template_caches

    # TODO: https://github.com/bytedance/Protenix/blob/main/scripts/database/download_protenix_data.sh

    # Download model checkpoint
    ckpt_url = f"{APP_INFO.base_url}/checkpoint/{model_name}.pt"
    files_to_download = files_to_download | {
        ckpt_url: data_root / "checkpoint" / f"{model_name}.pt"
    }
    download_files(
        files_to_download, force=force, progress_bar_desc="💊 Downloading Protenix data"
    )
    MODEL_VOLUME.commit()
    print("💊 Download complete")


##########################################
# Inference functions
##########################################
@app.function(
    timeout=CONF.timeout,
    volumes={
        APP_INFO.msa_cache_mountpoint: MSA_CACHE_VOLUME.with_mount_options(
            sub_path=APP_INFO.msa_cache_volume_subdir
        )
    },
)
def query_protenix_msa_server(task: ProtenixMsaTaskSpec) -> None:
    """Query the Protenix remote MSA server with the given command."""
    import orjson
    from uniaf3.schema import ProtenixConfig

    if _msa_task_ready(task):
        return

    cmd = [
        "protenix",
        task.query_command,
        f"--input={task.input_json_path}",
        f"--out_dir={task.output_dir}",
        f"--msa_server_mode={task.msa_server_mode}",
    ]
    run_command(cmd)

    # Move the searched files out of the run_name subdir such that future
    # runs with different names could hit the same cache
    out_path = Path(task.output_dir)
    msa_out_dir = next(out_path.glob("*/msa"))
    run_name_dir = msa_out_dir.parent
    run_name = run_name_dir.name
    for subdir in run_name_dir.iterdir():
        subdir.rename(out_path / subdir.name)

    # Also need to update the file paths in the JSON to reflect new locations
    def _get_new_location(old_path: str | None) -> str | None:
        if old_path is None:
            return
        old_path_file = Path(old_path)
        if old_path_file.is_relative_to(run_name_dir):
            return str(out_path / old_path_file.relative_to(run_name_dir))
        return old_path

    for conf_json in out_path.glob(f"{run_name}-*.json"):
        conf = ProtenixConfig.from_file(conf_json)
        for updated_task in conf.root:
            for seq in updated_task.sequences:
                if seq.proteinChain is not None:
                    seq.proteinChain.unpairedMsaPath = _get_new_location(
                        seq.proteinChain.unpairedMsaPath
                    )
                    seq.proteinChain.pairedMsaPath = _get_new_location(
                        seq.proteinChain.pairedMsaPath
                    )
                if seq.rnaSequence is not None:
                    seq.rnaSequence.unpairedMsaPath = _get_new_location(
                        seq.rnaSequence.unpairedMsaPath
                    )
        conf.to_files(out_path, conf_json.stem)

    run_name_dir.rmdir()
    expected = Path(task.expected_json_path)
    if not expected.is_file() or expected.stat().st_size == 0:
        raise FileNotFoundError(f"Expected MSA output not found: {expected}")
    _atomic_write(
        _msa_task_marker_path(task),
        orjson.dumps({
            "publication_key": task.publication_key,
            "expected_json_path": str(expected),
            "size": expected.stat().st_size,
            "sha256": sha256(expected.read_bytes()).hexdigest(),
        }),
    )
    MSA_CACHE_VOLUME.commit()


@app.function(
    timeout=CONF.timeout,
    volumes={
        APP_INFO.msa_cache_mountpoint: MSA_CACHE_VOLUME.with_mount_options(
            sub_path=APP_INFO.msa_cache_volume_subdir
        )
    },
)
def plan_protenix_inputs(
    input_bytes: bytes,
    msa_server_mode: str = "protenix",
    use_template: bool = False,
    use_rna_msa: bool = False,
) -> ProtenixPreparationPlan:
    """Discover content-addressed per-input MSA/template Tasks."""
    from tempfile import mkdtemp

    from uniaf3.schema import ProtenixConfig

    tmpdir = Path(mkdtemp(prefix="protenix_prep_"))  # cleaned on container exit
    tmp_json_path = tmpdir / "input.json"
    tmp_json_path.write_bytes(input_bytes)

    # `protenix prep` (inputprep) runs MSA + template + RNA MSA search.
    # It first produces `input-update-msa.json`, then (if template or RNA
    # MSA updates were actually made) renames it to `input-final-updated.json`.
    # `protenix mt` skips the RNA MSA search.
    # `protenix msa` only runs protein sequence MSA search.
    match (use_template, use_rna_msa):
        case (True, True):
            protenix_command = "prep"
            updated_suffix = "final-updated"
        case (True, False):
            protenix_command = "mt"
            updated_suffix = "final-updated"
        case (False, False):
            protenix_command = "msa"
            updated_suffix = "update-msa"
        case _:
            raise ValueError("RNA MSA without templates is not supported")

    conf = ProtenixConfig.from_file(tmp_json_path)
    tasks: list[ProtenixMsaTaskSpec] = []
    for task_idx, task in enumerate(conf.root):
        protein_seqs: list[str] = []
        rna_seqs: list[str] = []
        for seq in task.sequences:
            if (prot_chain := seq.proteinChain) is not None:
                protein_seqs.append(prot_chain.sequence)
            elif (rna_chain := seq.rnaSequence) is not None:
                rna_seqs.append(rna_chain.sequence)

        hash_key = (
            hash_string(":".join(protein_seqs + rna_seqs))
            if use_rna_msa
            else hash_string(":".join(protein_seqs))
        )
        cache_dir = (
            Path(APP_INFO.msa_cache_mountpoint)
            / msa_server_mode
            / hash_key[:2]
            / hash_key
        )
        cache_dir.mkdir(parents=True, exist_ok=True)

        task_name = task.name
        ProtenixConfig([task]).to_files(cache_dir, task_name)
        input_json_path = cache_dir / f"{task_name}.json"
        expected_json_path = cache_dir / f"{task_name}-{updated_suffix}.json"
        publication_key = hash_string(
            "\n".join((
                CONF.repo_commit_hash or CONF.version or "unknown",
                protenix_command,
                msa_server_mode,
                sha256(input_json_path.read_bytes()).hexdigest(),
            ))
        )
        tasks.append(
            ProtenixMsaTaskSpec(
                task_key=f"{task_idx:04d}-{task_name}",
                input_name=task_name,
                query_command=protenix_command,
                input_json_path=str(input_json_path),
                output_dir=str(cache_dir),
                msa_server_mode=msa_server_mode,
                expected_json_path=str(expected_json_path),
                publication_key=publication_key,
            )
        )

    preparation_key = hash_string(
        "\n".join((
            CONF.repo_commit_hash or CONF.version or "unknown",
            sha256(input_bytes).hexdigest(),
            msa_server_mode,
            str(use_template),
            str(use_rna_msa),
        ))
    )
    prepared_json_path = (
        Path(APP_INFO.msa_cache_mountpoint)
        / "prepared"
        / preparation_key[:2]
        / preparation_key
        / "input.json"
    )
    MSA_CACHE_VOLUME.commit()
    return ProtenixPreparationPlan(
        preparation_key=preparation_key,
        prepared_json_path=str(prepared_json_path),
        tasks=tuple(tasks),
    )


@app.function(
    timeout=CONF.timeout,
    volumes={
        APP_INFO.msa_cache_mountpoint: MSA_CACHE_VOLUME.with_mount_options(
            sub_path=APP_INFO.msa_cache_volume_subdir
        )
    },
)
def finalize_protenix_inputs(
    input_bytes: bytes,
    plan: ProtenixPreparationPlan,
) -> dict[str, str | int]:
    """Assemble completed MSA Tasks into one prepared input publication."""
    from tempfile import mkdtemp

    import orjson
    from uniaf3.schema import ProtenixConfig

    MSA_CACHE_VOLUME.reload()
    if _prepared_ready(plan):
        path = Path(plan.prepared_json_path)
        return {"prepared_json_path": str(path), "size": path.stat().st_size}

    tmpdir = Path(mkdtemp(prefix="protenix_finalize_"))
    input_path = tmpdir / "input.json"
    input_path.write_bytes(input_bytes)
    conf = ProtenixConfig.from_file(input_path)
    if len(conf.root) != len(plan.tasks):
        raise ValueError("Protenix preparation plan does not match its input")

    for task_idx, task in enumerate(conf.root):
        if task.name != plan.tasks[task_idx].input_name:
            raise ValueError("Protenix preparation task order changed")
        updated_json_path = Path(plan.tasks[task_idx].expected_json_path)
        if not _msa_task_ready(plan.tasks[task_idx]):
            raise FileNotFoundError(
                f"Expected MSA output not found: {updated_json_path}"
            )
        updated_conf = ProtenixConfig.from_file(updated_json_path)
        conf.root[task_idx] = updated_conf.root[0]

    content = conf.to_json().encode()
    path = Path(plan.prepared_json_path)
    _atomic_write(path, content)
    _atomic_write(
        _prepared_marker_path(plan),
        orjson.dumps({
            "preparation_key": plan.preparation_key,
            "size": len(content),
            "sha256": sha256(content).hexdigest(),
        }),
    )
    MSA_CACHE_VOLUME.commit()
    return {"prepared_json_path": str(path), "size": len(content)}


@app.function(
    gpu=CONF.gpu,
    cpu=(1.125, 16.125),  # burst for tar compression
    memory=(1024, 65536),  # reserve 1GB, OOM at 64GB
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(model_volume=True, output_volume=True)
    | {
        APP_INFO.msa_cache_mountpoint: MSA_CACHE_VOLUME.with_mount_options(
            sub_path=APP_INFO.msa_cache_volume_subdir
        )
    },
)
def run_protenix(
    input_bytes: bytes | None,
    run_name: str,
    result_key: str,
    prepared_input_path: str | None = None,
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
    use_tfg_guidance: bool = False,
    use_fast_layernorm: bool = True,
    extra_args: str | None = None,
    score_only: bool = False,
) -> dict[str, str | int]:
    """Run Protenix structure prediction or confidence scoring.

    Args:
        input_bytes: Input JSON for prediction, or PDB/CIF in `score_only` mode.
        run_name: Name for this run (used for output directory).
        result_key: Content address for the published result archive.
        prepared_input_path: Optional prepared JSON in the MSA cache volume.
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
        use_tfg_guidance: Enable Training-Free Guidance (TFG) for refined sampling.
        use_fast_layernorm: Whether to enable the custom CUDA layernorm kernel.
        extra_args: Additional CLI arguments as a string.
        score_only: When True, score an existing PDB/CIF structure using
            ``protenixscore score`` instead of running diffusion prediction.

    Returns:
        Metadata for the published inference or scoring result archive.

    """
    from tempfile import mkdtemp

    if prepared_input_path is not None:
        MSA_CACHE_VOLUME.reload()
        input_bytes = Path(prepared_input_path).read_bytes()
    if input_bytes is None:
        raise ValueError("Protenix input bytes or prepared input path is required")

    run_env = os.environ.copy()
    if use_fast_layernorm:
        run_env["LAYERNORM_TYPE"] = "fast_layernorm"

    tmpdir_path = Path(mkdtemp(prefix="protenix_run_"))  # cleaned on container exit
    out_dir = tmpdir_path / run_name
    out_dir.mkdir()

    # Score an existing structure with the Protenix confidence head
    if score_only:
        # Detect CIF vs PDB format: CIF files start with 'data_' (after
        # any leading comment lines starting with '#'), PDB files with
        # record types like HEADER/ATOM/REMARK.
        input_ext = ".pdb"
        for _line in input_bytes.splitlines():
            stripped = _line.strip()
            if not stripped or stripped.startswith(b"#"):
                continue
            if stripped.startswith(b"data_"):
                input_ext = ".cif"
            break
        input_file = tmpdir_path / f"{run_name}{input_ext}"
        input_file.write_bytes(input_bytes)

        # Map use_msa → --use_msas (both | false)
        # ProtenixScore's use_msas controls which chain roles receive MSAs.
        use_msas_val = "both" if use_msa else "false"

        # Map msa_server_mode → --msa_host_url.
        # The protenix remote server URL matches what `protenix msa` uses
        # (MMSEQS_SERVICE_HOST_URL in protenix/web_service/colab_request_parser.py).
        msa_host_url = (
            "https://protenix-server.com/api/msa"
            if msa_server_mode == "protenix"
            else "https://api.colabfold.com"
        )

        # Cache fetched MSAs so they can be reused across runs
        input_seqs = struct2seq(input_file)
        cache_key = hash_string(":".join(x[1] for x in input_seqs))
        score_msa_cache_dir = (
            Path(APP_INFO.msa_cache_mountpoint)
            / "score"
            / msa_server_mode
            / cache_key[:2]
            / cache_key
        )
        score_msa_cache_dir.mkdir(parents=True, exist_ok=True)
        # TODO: split MSA search and score steps
        cmd = [
            "protenixscore",
            "score",
            f"--input={input_file}",
            f"--output={tmpdir_path}",
            f"--model_name={model_name}",
            f"--dtype={dtype}",
            f"--use_msas={use_msas_val}",
            f"--msa_host_url={msa_host_url}",
            f"--msa_cache_dir={score_msa_cache_dir}",
            "--msa_cache_mode=readwrite",
        ]
        run_command(
            cmd,
            output_mode="tee",
            log_file=out_dir / f"{run_name}.log",
            env=run_env,
            cwd=tmpdir_path,
        )

        # Persist MSA cache back to the volume for reuse in future runs
        MSA_CACHE_VOLUME.commit()
        print("💊 Packaging ProtenixScore results...")
        tarball_bytes = package_outputs(out_dir)
        return _publish_result(result_key, run_name, tarball_bytes)

    # --- Prediction mode ---
    input_json_path = tmpdir_path / f"{run_name}.json"
    input_json_path.write_bytes(input_bytes)
    cmd = [
        "protenix",
        "pred",
        f"--input={input_json_path}",
        f"--out_dir={tmpdir_path}",
        f"--seeds={seeds}",
        f"--cycle={cycle}",
        f"--step={step}",
        f"--sample={sample}",
        f"--dtype={dtype}",
        f"--model_name={model_name}",
        f"--use_msa={use_msa}",
        f"--msa_server_mode={msa_server_mode}",
        f"--use_template={use_template}",
        f"--use_rna_msa={use_rna_msa}",
        f"--use_tfg_guidance={use_tfg_guidance}",
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    run_command(
        cmd,
        output_mode="tee",
        log_file=out_dir / f"{run_name}.log",
        env=run_env,
        cwd=tmpdir_path,
    )

    # Package outputs
    print("💊 Packaging Protenix results...")
    tarball_bytes = package_outputs(out_dir)
    return _publish_result(result_key, run_name, tarball_bytes)


##########################################
# Deployment-local execution coordinator
##########################################
@app.cls(
    cpu=(0.125, 4.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    max_containers=1,
    volumes=CONF.mounts(output_volume=True)
    | {
        APP_INFO.msa_cache_mountpoint: MSA_CACHE_VOLUME.with_mount_options(
            sub_path=APP_INFO.msa_cache_volume_subdir
        )
    },
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with Protenix functions."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh mounted state before accepting lifecycle methods."""
        initialize_execution_coordinator_host(self)
        self._identity()
        CONF.output_volume.reload()
        MSA_CACHE_VOLUME.reload()

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
        return execution_coordinator_identity(self)

    def _adapter(
        self,
        *,
        development: bool | None = None,
    ) -> ProtenixExecutionCoordinator:
        execution_run_id, deployment = self._identity()
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected_mode: ProtenixExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=Path(CONF.output_volume_mountpoint),
                output_volume=CONF.output_volume,
                msa_cache_volume=MSA_CACHE_VOLUME,
                output_claims=PROTENIX_OUTPUT_CLAIMS,
                modal_driver=_coordinator_modal_driver(development=selected_mode),
            ),
        )


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    """Resolve exact deployed functions or current-source handles."""
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {
            "download_protenix_data": download_protenix_data,
            "plan_protenix_inputs": plan_protenix_inputs,
            "query_protenix_msa_server": query_protenix_msa_server,
            "finalize_protenix_inputs": finalize_protenix_inputs,
            "run_protenix": run_protenix,
        },
        workload_name="Protenix",
    )


##########################################
# Local entrypoint client
##########################################
@app.local_entrypoint()
def submit_protenix_task(
    input_file: str,
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
    use_tfg_guidance: bool = False,
    use_fast_layernorm: bool = True,
    force_redownload: bool = False,
    extra_args: str | None = None,
    score_only: bool = False,
    max_parallel_msa: int | None = None,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Run Protenix structure prediction on Modal and fetch results to `out_dir`.

    Args:
        input_file: Path to input JSON file, or a PDB/CIF file in `score_only` mode.
            For a description of the JSON schema, see
            <https://github.com/y1zhou/Protenix/blob/main/docs/infer_json_format.md>.
        out_dir: Optional local output directory. If not specified, outputs
            will be saved in the current working directory.
        run_name: Optional run name for output files. Defaults to input filename stem.
        model_name: Model checkpoint name. See `APP_INFO.supported_models`
            for available models.
        seeds: Comma-separated random seeds for inference.
        cycle: Pairformer cycle number.
        step: Number of diffusion steps.
        sample: Number of samples per seed.
        dtype: Inference dtype (bf16 or fp32).
        use_msa: Whether to use MSA features. Pass `--no-use-msa` to disable.
        msa_server_mode: MSA search mode (`protenix` or `colabfold`).
        use_template: Whether to use templates. Requires Protenix data files.
        use_rna_msa: Whether to use RNA MSA features.
        use_tfg_guidance: Enable Training-Free Guidance (TFG) for refined sampling.
        use_fast_layernorm: Whether to enable the custom CUDA layernorm kernel.
        force_redownload: Whether to force re-download of model weights.
        extra_args: Additional CLI arguments passed to `protenix pred`.
        score_only: When True, score an existing PDB/CIF structure using
            ``protenixscore score`` instead of running prediction.
        max_parallel_msa: Maximum number of MSA search containers to run at once.
        use_deployed_coordinator: Target the exact deployed coordinator. The
            Biomodals CLI supplies this for normal runs.
        deployment_environment: Modal Environment containing the coordinator.
        deployment_name: Exact deployed Modal app name.
        deployment_version: Exact numeric deployment version.
        restart_from: Optional predecessor Execution Run ID for a Successor Run.
    """
    input_path = Path(input_file).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    predecessor_execution_run_id = None if restart_from is None else UUID(restart_from)
    capacity = (
        _DEFAULT_MAX_ACTIVE_PROVIDER_CALLS
        if max_parallel_msa is None
        else max_parallel_msa
    )
    request = ProtenixExecutionRequest(
        run_name=sanitize_filename(run_name or input_path.stem),
        input_content=input_path.read_bytes(),
        model_name=model_name,
        seeds=seeds,
        cycle=cycle,
        step=step,
        sample=sample,
        dtype=dtype,
        use_msa=use_msa,
        msa_server_mode=("colabfold" if score_only else msa_server_mode),
        use_template=use_template,
        use_rna_msa=use_rna_msa,
        use_tfg_guidance=use_tfg_guidance,
        use_fast_layernorm=use_fast_layernorm,
        force_redownload=force_redownload,
        extra_args=extra_args,
        score_only=score_only,
        max_active_provider_calls=capacity,
        app_version=CONF.repo_commit_hash or CONF.version or "unknown",
    )
    if request.model_name not in APP_INFO.supported_models:
        raise ValueError(
            f"Unsupported model: {request.model_name}. "
            f"Supported models: {', '.join(APP_INFO.supported_models)}"
        )

    local_out_dir = resolve_local_output_dir(out_dir)
    out_file = build_local_output_path(
        local_out_dir,
        run_name=request.run_name,
        suffix=f"_{CONF.name}",
    )

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
        local_coordinator=ExecutionCoordinator,
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

    result_path = _result_path(request.result_key, request.run_name)
    relative_path = result_path.relative_to(CONF.output_volume_mountpoint)
    tarball_bytes = b"".join(CONF.output_volume.read_file(relative_path.as_posix()))
    write_local_tarball(out_file, tarball_bytes)
    print(f"🧬 Protenix run complete! Results saved to {out_file}")
