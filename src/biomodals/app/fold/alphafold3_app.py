"""AlphaFold3 source repo: <https://github.com/google-deepmind/alphafold3>.

## Additional notes

This app provides the AlphaFold3 runtime and a separate, plan-only-by-default
entrypoint for building its fixed sharded genetic-database profiles. Follow
upstream's instructions to acquire the model weights and source databases:

<https://github.com/google-deepmind/alphafold3#obtaining-model-parameters>

<https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#obtaining-genetic-databases>

The model checkpoint must be available at `/AlphaFold3/af3.bin` in the
`biomodals-store` Volume. Put the upstream genetic database files in
`AlphaFold3-msa-db`, then use `setup_sharded_databases` to populate the
separate `AlphaFold3-msa-db-sharded` Volume before running searches.

See <https://github.com/google-deepmind/alphafold3/tree/main/docs> for general docs.

## Outputs

See <https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md>.
"""

# Ignore ruff warnings about import location
# ruff: noqa: PLC0415

import os
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import cast

import modal
import orjson
from uniaf3.schema.alphafold3 import (
    AF3Config,
    AF3Template,
)

from biomodals.app.config import AppConfig
from biomodals.app.fold.alphafold3.inference_inputs import (
    ALPHAFOLD3_APP_VERSION,
    PreparedInferenceRun,
    materialize_local_input,
    prepare_inference_run,
    serialize_af3_input,
    validate_af3_config,
    validate_inference_parameters,
)
from biomodals.app.fold.alphafold3.msa_search import (
    MSA_SEARCH_CLAIM_DICT_NAME,
    SEARCH_MAX_PARALLEL_SHARDS,
    SEARCH_N_CPU,
    ChainMsaState,
    MsaAssemblyTask,
    Polymer,
    RawSearchTask,
    SearchRuntime,
    assemble_and_publish_msas,
    field_is_populated,
    inspect_raw_searches,
    plan_msa_resolution,
    run_database_search,
    sequence_hash,
)
from biomodals.app.fold.alphafold3.profile_builder import (
    ProfileBuilderRuntime,
    build_profile,
    cleanup_profile_workspace,
    inspect_profile_registry,
    plan_missing_profile_builds,
)
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    ALPHAFOLD3_REPOSITORY,
    BUILD_MEMORY_MIB,
    BUILD_TIMEOUT_SECONDS,
    DATABASE_PROFILE_SPECS,
    DEFAULT_SEQKIT_THREADS,
    PROFILE_BUILD_CLAIM_DICT_NAME,
    PROFILE_BUILD_MAX_CONTAINERS,
    SEQKIT_VERSION,
    SHARDED_DB_VOLUME_NAME,
    SOURCE_DB_VOLUME_NAME,
    SourcePolicy,
    profile_build_slot_budget,
    validate_seqkit_threads,
    validate_source_policy,
)
from biomodals.app.fold.alphafold3.request_results import (
    RequestPublication,
    create_request_archive,
    publish_request_results,
    sanitize_presentation_name,
)
from biomodals.app.fold.alphafold3.seed_predictions import (
    SEED_PREDICTION_CLAIM_DICT_NAME,
    ClaimedSeed,
    InferenceRuntime,
    SeedWorkerTask,
    canonical_output_name,
    claim_seed_predictions,
    claimed_seed_from_dict,
    finalize_run_summary,
    inspect_seed_predictions,
    partition_claimed_seeds,
    run_seed_prediction_worker,
    seed_claim_plan_from_dict,
)
from biomodals.app.fold.alphafold3.sharding import (
    CONTAINER_NATIVE_SOURCE_DIR,
    NATIVE_SOURCE_DIR_ENV,
    utc_now,
    write_json_atomic,
)
from biomodals.app.fold.alphafold3.template_search import (
    DEFAULT_MAX_TEMPLATE_DATE,
    TemplateRuntime,
    TemplateTask,
    inspect_template_entries,
)
from biomodals.app.fold.alphafold3.template_search import (
    run_template_search as run_resumable_template_search,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import (
    AF3_MSA_DB_VOLUME,
    MAX_TIMEOUT,
    MSA_CACHE_VOLUME,
    MSA_CACHE_VOLUME_NAME,
)
from biomodals.helper.io import resolve_local_output_dir
from biomodals.helper.shell import run_command
from biomodals.helper.task_budget import bounded_map

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="AlphaFold3",
    repo_url=ALPHAFOLD3_REPOSITORY,
    repo_commit_hash=ALPHAFOLD3_COMMIT,
    package_name="alphafold3",
    version=ALPHAFOLD3_APP_VERSION,
    python_version="3.12",
    cuda_version="cu130",
    gpu=os.environ.get("GPU", "L40S"),
    timeout=int(os.environ.get("TIMEOUT", "21600")),
)


@dataclass
class AppInfo:
    """Container for AlphaFold3-specific information and configurations."""

    # Volume mount path for genetic search databases
    msa_db_dir: str = f"/{CONF.name}-msa-db"
    # Volume mount path for immutable sharded genetic databases
    sharded_msa_db_dir: str = f"/{SHARDED_DB_VOLUME_NAME}"
    # Volume mount path for MSA output cache
    msa_cache_dir: str = f"/{MSA_CACHE_VOLUME_NAME}"
    msa_cache_volume_subdir: str = f"/{CONF.name}"
    # Durable setup evidence below the app's output Volume
    profile_build_evidence_dir: str = "msa-profile-builds"


##########################################
# Image and app definitions
##########################################
APP_INFO = AppInfo()
SHARDED_MSA_DB_VOLUME = modal.Volume.from_name(
    SHARDED_DB_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
PROFILE_BUILD_CLAIMS = modal.Dict.from_name(
    PROFILE_BUILD_CLAIM_DICT_NAME,
    create_if_missing=True,
)
MSA_SEARCH_CLAIMS = modal.Dict.from_name(
    MSA_SEARCH_CLAIM_DICT_NAME,
    create_if_missing=True,
)
INFERENCE_CLAIMS = modal.Dict.from_name(
    SEED_PREDICTION_CLAIM_DICT_NAME,
    create_if_missing=True,
)

# Ref: https://github.com/google-deepmind/alphafold3/blob/main/docker/Dockerfile
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
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
        " && ".join((
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
        ))
    )
    .workdir(str(CONF.git_clone_dir))
    # .uv_sync(frozen=True, extra_options="--no-editable")
    .uv_pip_install(str(CONF.git_clone_dir))
    .run_commands("build_data")  # installed in the previous step
    .env({"PATH": "/hmmer/bin:$PATH"})
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.app.fold.alphafold3")
)

# Database preparation does not need the AlphaFold runtime or model stack.
sharding_image = (
    modal.Image
    .micromamba(python_version=CONF.python_version)
    .apt_install("build-essential", "zstd")
    .micromamba_install(
        f"seqkit={SEQKIT_VERSION}",
        channels=["conda-forge", "bioconda"],
    )
    .run_commands("seqkit version")
    .add_local_dir(
        Path(__file__).parent / "alphafold3" / "native",
        str(CONTAINER_NATIVE_SOURCE_DIR),
        copy=True,
    )
    .env({NATIVE_SOURCE_DIR_ENV: str(CONTAINER_NATIVE_SOURCE_DIR)})
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.app.fold.alphafold3")
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)
_CONTAINER_INSTANCE_ID = uuid.uuid4().hex


##########################################
# Helper functions
##########################################
def _profile_builder_runtime() -> ProfileBuilderRuntime:
    """Bind the shared builder to production mounts and persistence objects."""
    return ProfileBuilderRuntime(
        source_root=Path(APP_INFO.msa_db_dir),
        sharded_root=Path(APP_INFO.sharded_msa_db_dir),
        output_root=Path(CONF.output_volume_mountpoint),
        evidence_relpath=APP_INFO.profile_build_evidence_dir,
        source_volume=AF3_MSA_DB_VOLUME,
        sharded_volume=SHARDED_MSA_DB_VOLUME,
        output_volume=CONF.output_volume,
        claims=PROFILE_BUILD_CLAIMS,
        container_id=_CONTAINER_INSTANCE_ID,
    )


@app.function(
    image=sharding_image,
    cpu=(0.125, 32.125),
    memory=BUILD_MEMORY_MIB,
    timeout=BUILD_TIMEOUT_SECONDS,
    max_containers=PROFILE_BUILD_MAX_CONTAINERS,
    volumes={
        APP_INFO.msa_db_dir: AF3_MSA_DB_VOLUME,
        APP_INFO.sharded_msa_db_dir: SHARDED_MSA_DB_VOLUME,
        CONF.output_volume_mountpoint: CONF.output_volume,
    },
)
def build_sharded_database(
    database_id: str,
    seqkit_threads: int = DEFAULT_SEQKIT_THREADS,
    source_policy: SourcePolicy = "keep",
) -> dict[str, object]:
    """Build one fixed immutable database profile."""
    return build_profile(
        _profile_builder_runtime(),
        database_id,
        seqkit_threads,
        source_policy,
    )


@app.function(
    image=sharding_image,
    cpu=0.125,
    memory=1024,
    timeout=600,
    max_containers=1,
    volumes={
        APP_INFO.sharded_msa_db_dir: SHARDED_MSA_DB_VOLUME.with_mount_options(
            read_only=True,
            sub_path="/",
        ),
    },
)
def inspect_sharded_database_profiles() -> dict[str, object]:
    """Inspect all fixed profile manifests without expensive digest scans."""
    SHARDED_MSA_DB_VOLUME.reload()
    return inspect_profile_registry(Path(APP_INFO.sharded_msa_db_dir))


@app.function(
    image=sharding_image,
    cpu=0.125,
    memory=1024,
    timeout=600,
    max_containers=1,
    volumes={
        APP_INFO.sharded_msa_db_dir: SHARDED_MSA_DB_VOLUME,
        CONF.output_volume_mountpoint: CONF.output_volume,
    },
)
def finalize_sharded_database_setup() -> dict[str, object]:
    """Clean abandoned and unselected profiles after all builders complete."""
    SHARDED_MSA_DB_VOLUME.reload()
    CONF.output_volume.reload()
    result = cleanup_profile_workspace(
        Path(APP_INFO.sharded_msa_db_dir),
        PROFILE_BUILD_CLAIMS,
    )
    SHARDED_MSA_DB_VOLUME.commit()
    setup_id = uuid.uuid4().hex
    evidence_root = (
        Path(CONF.output_volume_mountpoint)
        / APP_INFO.profile_build_evidence_dir
        / "setup"
        / setup_id
    )
    completed = result | {
        "setup_id": setup_id,
        "completed_at": utc_now(),
    }
    write_json_atomic(evidence_root / "inventory.json", completed)
    CONF.output_volume.commit()
    write_json_atomic(
        evidence_root / "done.json",
        {
            "status": "complete",
            "setup_id": setup_id,
            "completed_at": utc_now(),
        },
    )
    CONF.output_volume.commit()
    return completed


def _sharded_database_setup_plan(
    seqkit_threads: int,
    source_policy: str,
) -> dict[str, object]:
    """Build the cost-free plan for production profile setup."""
    threads = validate_seqkit_threads(seqkit_threads)
    policy = validate_source_policy(source_policy)
    return {
        "operation": "setup-sharded-databases",
        "profiles": [
            {
                "database_id": spec.database_id,
                "profile_id": spec.profile_id,
                "source_filename": spec.source_filename,
                "shard_count": spec.shard_count,
                "polymer": spec.polymer,
            }
            for spec in DATABASE_PROFILE_SPECS
        ],
        "builder": {
            "function": "build_sharded_database",
            "seqkit_threads": threads,
            "source_policy": policy,
            "cpu": [0.125, 32.125],
            "memory_mib": list(BUILD_MEMORY_MIB),
            "timeout_seconds": BUILD_TIMEOUT_SECONDS,
        },
        "fanout_budget": profile_build_slot_budget(
            len(DATABASE_PROFILE_SPECS),
            threads,
        ),
        "coordination": [
            "inspect-fixed-profile-manifests",
            "submit-all-missing-profiles-concurrently",
            "wait-for-all-builders",
            "final-inventory-and-workspace-cleanup",
        ],
        "volumes": {
            "source": SOURCE_DB_VOLUME_NAME,
            "shards": SHARDED_DB_VOLUME_NAME,
            "evidence": CONF.output_volume_name,
        },
        "cleanup": {
            "barrier": "all-selected-profiles-valid-and-no-active-claims",
            "remove_generation_workspaces": [".staging", ".orphaned"],
            "remove_unselected_profile_directories": True,
        },
    }


def _fill_missing_msa_for_inference(conf: AF3Config) -> AF3Config:
    """Mark bare sequences as single-sequence inference inputs."""
    for seq in conf.sequences:
        if (protein := seq.protein) is not None:
            if not field_is_populated(
                protein.unpairedMsa,
                protein.unpairedMsaPath,
            ):
                protein.unpairedMsa = ""
            if not field_is_populated(
                protein.pairedMsa,
                protein.pairedMsaPath,
            ):
                protein.pairedMsa = ""
            if not protein.templates:
                protein.templates = []
        elif (rna := seq.rna) is not None:
            if not field_is_populated(rna.unpairedMsa, rna.unpairedMsaPath):
                rna.unpairedMsa = ""
    return conf


##########################################
# MSA search functions
##########################################
@app.function(
    cpu=0.125,
    memory=1024,
    timeout=600,
    max_containers=1,
    volumes={
        APP_INFO.sharded_msa_db_dir: (
            SHARDED_MSA_DB_VOLUME.with_mount_options(
                read_only=True,
                sub_path="/",
            )
        ),
        APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME.with_mount_options(
            read_only=True, sub_path=APP_INFO.msa_cache_volume_subdir
        ),
    },
)
def inspect_msa_search_cache(
    inputs: list[tuple[str, str]],
) -> list[dict[str, object]]:
    """Inspect raw markers so cache hits consume no HMMER workers."""
    SHARDED_MSA_DB_VOLUME.reload()
    MSA_CACHE_VOLUME.reload()
    tasks = tuple(
        RawSearchTask(database_id=database_id, sequence=sequence)
        for database_id, sequence in inputs
    )
    return inspect_raw_searches(
        Path(APP_INFO.sharded_msa_db_dir),
        Path(APP_INFO.msa_cache_dir),
        tasks,
    )


def _msa_search_runtime(
    *,
    maximum_age_seconds: int | float,
    wait_timeout_seconds: int | float,
) -> SearchRuntime:
    """Bind shared search code to production mounts and persistence objects."""
    return SearchRuntime(
        sharded_root=Path(APP_INFO.sharded_msa_db_dir),
        cache_root=Path(APP_INFO.msa_cache_dir),
        sharded_volume=SHARDED_MSA_DB_VOLUME,
        cache_volume=MSA_CACHE_VOLUME,
        claims=MSA_SEARCH_CLAIMS,
        container_id=_CONTAINER_INSTANCE_ID,
        maximum_age_seconds=maximum_age_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
        function_call_id=modal.current_function_call_id(),
    )


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131_072),
    timeout=CONF.timeout,
    volumes={
        APP_INFO.sharded_msa_db_dir: (
            SHARDED_MSA_DB_VOLUME.with_mount_options(
                read_only=True,
                sub_path="/",
            )
        ),
        APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME.with_mount_options(
            read_only=False, sub_path=APP_INFO.msa_cache_volume_subdir
        ),
    },
)
def search_database_msa(database_id: str, sequence: str) -> dict[str, object]:
    """Search one fixed sharded database with database-level resume."""
    return run_database_search(
        _msa_search_runtime(
            maximum_age_seconds=CONF.timeout + 900,
            wait_timeout_seconds=max(60, CONF.timeout - 60),
        ),
        database_id,
        sequence,
    )


@app.function(
    cpu=(0.125, 4.125),
    memory=(1024, 32_768),
    timeout=1800,
    volumes={
        APP_INFO.sharded_msa_db_dir: (
            SHARDED_MSA_DB_VOLUME.with_mount_options(
                read_only=True,
                sub_path="/",
            )
        ),
        APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME.with_mount_options(
            read_only=False, sub_path=APP_INFO.msa_cache_volume_subdir
        ),
    },
)
def assemble_sequence_msas(
    polymer: Polymer,
    sequence: str,
    include_unpaired: bool,
    include_paired: bool,
) -> dict[str, object]:
    """Assemble requested fields with pinned upstream deduplication."""
    if polymer not in {"protein", "rna"}:
        raise ValueError(f"Unsupported polymer: {polymer!r}")
    return assemble_and_publish_msas(
        _msa_search_runtime(
            maximum_age_seconds=2700,
            wait_timeout_seconds=1740,
        ),
        MsaAssemblyTask(
            polymer=polymer,
            sequence=sequence,
            include_unpaired=include_unpaired,
            include_paired=include_paired,
        ),
    )


@app.function(
    cpu=0.125,
    memory=1024,
    timeout=600,
    max_containers=1,
    volumes={
        APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME.with_mount_options(
            read_only=True,
            sub_path=APP_INFO.msa_cache_volume_subdir,
        ),
    },
)
def inspect_protein_template_cache(
    inputs: list[tuple[str, str, str]],
) -> list[dict[str, object]]:
    """Inspect canonical template markers without consuming search workers."""
    MSA_CACHE_VOLUME.reload()
    return inspect_template_entries(
        Path(APP_INFO.msa_cache_dir),
        tuple(inputs),
    )


def _template_runtime() -> TemplateRuntime:
    """Bind shared template search to production mounts and claims."""
    return TemplateRuntime(
        source_root=Path(APP_INFO.msa_db_dir),
        cache_root=Path(APP_INFO.msa_cache_dir),
        source_volume=AF3_MSA_DB_VOLUME,
        cache_volume=MSA_CACHE_VOLUME,
        claims=MSA_SEARCH_CLAIMS,
        container_id=_CONTAINER_INSTANCE_ID,
        maximum_age_seconds=CONF.timeout + 900,
        wait_timeout_seconds=max(60, CONF.timeout - 60),
        function_call_id=modal.current_function_call_id(),
    )


@app.function(
    cpu=(0.125, 8.125),
    memory=(1024, 32_768),
    timeout=CONF.timeout,
    volumes={
        APP_INFO.msa_db_dir: AF3_MSA_DB_VOLUME.with_mount_options(
            read_only=True,
            sub_path="/",
        ),
        APP_INFO.msa_cache_dir: MSA_CACHE_VOLUME.with_mount_options(
            read_only=False, sub_path=APP_INFO.msa_cache_volume_subdir
        ),
    },
)
def search_protein_templates(
    sequence: str,
    unpaired_msa: str,
    publish_canonical: bool,
    max_template_date: str = DEFAULT_MAX_TEMPLATE_DATE,
) -> dict[str, object]:
    """Search templates from one resolved protein unpaired MSA."""
    return run_resumable_template_search(
        _template_runtime(),
        TemplateTask(
            sequence=sequence,
            unpaired_msa=unpaired_msa,
            publish_canonical=publish_canonical,
            max_template_date=max_template_date,
        ),
    )


def _validate_search_worker_budget(max_parallel_search_workers: int) -> int:
    """Validate the request-wide remote-worker budget."""
    if (
        isinstance(max_parallel_search_workers, bool)
        or not isinstance(max_parallel_search_workers, int)
        or not 1 <= max_parallel_search_workers <= 32
    ):
        raise ValueError("max_parallel_search_workers must be between 1 and 32")
    return max_parallel_search_workers


def _validate_max_num_gpus(max_num_gpus: int) -> int:
    """Validate the GPU-worker cap before any cost-incurring remote work."""
    if (
        isinstance(max_num_gpus, bool)
        or not isinstance(max_num_gpus, int)
        or max_num_gpus < 1
    ):
        raise ValueError("max_num_gpus must be a positive integer")
    return max_num_gpus


def _chain_msa_states(conf: AF3Config) -> tuple[ChainMsaState, ...]:
    """Describe caller-supplied MSA fields without sharing caller evidence."""
    states: list[ChainMsaState] = []
    for index, entry in enumerate(conf.sequences):
        if (protein := entry.protein) is not None:
            states.append(
                ChainMsaState(
                    chain_index=index,
                    polymer="protein",
                    sequence=protein.sequence,
                    unpaired_present=field_is_populated(
                        protein.unpairedMsa,
                        protein.unpairedMsaPath,
                    ),
                    paired_present=field_is_populated(
                        protein.pairedMsa,
                        protein.pairedMsaPath,
                    ),
                )
            )
        elif (rna := entry.rna) is not None:
            states.append(
                ChainMsaState(
                    chain_index=index,
                    polymer="rna",
                    sequence=rna.sequence,
                    unpaired_present=field_is_populated(
                        rna.unpairedMsa,
                        rna.unpairedMsaPath,
                    ),
                    paired_present=False,
                )
            )
    return tuple(states)


def _remote_search_outcome(
    task: RawSearchTask,
) -> dict[str, object] | Exception:
    """Return a remote search result or its surfaced exception."""
    try:
        return search_database_msa.remote(task.database_id, task.sequence)
    except Exception as exc:
        return exc


def _remote_assembly_outcome(
    task: MsaAssemblyTask,
) -> dict[str, object] | Exception:
    """Return a remote assembly result or its surfaced exception."""
    try:
        return assemble_sequence_msas.remote(
            task.polymer,
            task.sequence,
            task.include_unpaired,
            task.include_paired,
        )
    except Exception as exc:
        return exc


def _remote_template_outcome(
    task: TemplateTask,
) -> dict[str, object] | Exception:
    """Return a remote template result or its surfaced exception."""
    try:
        return search_protein_templates.remote(
            task.sequence,
            task.unpaired_msa,
            task.publish_canonical,
            task.max_template_date,
        )
    except Exception as exc:
        return exc


def _resolved_msa_text(
    inline_value: str | None,
    path_value: str | None,
    *,
    field_name: str,
) -> str:
    """Read one resolved MSA field for downstream template search."""
    if inline_value and path_value:
        raise ValueError(f"{field_name} cannot set both inline and path forms")
    if inline_value:
        return inline_value
    if path_value:
        value = Path(path_value).read_text()
        if not value:
            raise ValueError(f"{field_name} path is empty: {path_value}")
        return value
    raise ValueError(f"{field_name} is unresolved")


def _plan_template_tasks(
    conf: AF3Config,
    states: tuple[ChainMsaState, ...],
    canonical_combined_sequences: set[tuple[Polymer, str]],
) -> tuple[tuple[TemplateTask, ...], dict[str, tuple[int, ...]]]:
    """Deduplicate missing templates without publishing caller MSA evidence."""
    tasks: dict[str, TemplateTask] = {}
    chain_indices: dict[str, list[int]] = {}
    for state in states:
        if state.polymer != "protein":
            continue
        protein = conf.sequences[state.chain_index].protein
        if protein is None:
            raise RuntimeError("Protein MSA state no longer matches its chain")
        if protein.templates:
            continue
        unpaired_msa = _resolved_msa_text(
            protein.unpairedMsa,
            protein.unpairedMsaPath,
            field_name=f"sequences[{state.chain_index}].protein.unpairedMsa",
        )
        publish_canonical = (
            not state.unpaired_present
            and ("protein", state.sequence) in canonical_combined_sequences
        )
        candidate = TemplateTask(
            sequence=state.sequence,
            unpaired_msa=unpaired_msa,
            publish_canonical=publish_canonical,
        )
        identity = candidate.template_identity
        if existing := tasks.get(identity):
            if (
                existing.sequence != candidate.sequence
                or existing.unpaired_msa != candidate.unpaired_msa
            ):
                raise RuntimeError("Protein template identity collision")
            if publish_canonical and not existing.publish_canonical:
                tasks[identity] = TemplateTask(
                    sequence=existing.sequence,
                    unpaired_msa=existing.unpaired_msa,
                    publish_canonical=True,
                    max_template_date=existing.max_template_date,
                )
        else:
            tasks[identity] = candidate
        chain_indices.setdefault(identity, []).append(state.chain_index)
    return (
        tuple(tasks.values()),
        {identity: tuple(indices) for identity, indices in chain_indices.items()},
    )


def _validated_template_result(
    task: TemplateTask,
    outcome: dict[str, object],
    *,
    allowed_statuses: frozenset[str],
) -> tuple[AF3Template, ...]:
    """Validate a cache or worker result before applying it to input chains."""
    if (
        outcome.get("status") not in allowed_statuses
        or outcome.get("sequence_sha256") != sequence_hash(task.sequence)
        or outcome.get("unpaired_msa_sha256") != task.unpaired_msa_sha256
        or outcome.get("template_identity") != task.template_identity
    ):
        raise RuntimeError(f"Invalid protein template result: {outcome!r}")
    raw_templates = outcome.get("templates")
    if not isinstance(raw_templates, list) or not all(
        isinstance(template, dict) for template in raw_templates
    ):
        raise RuntimeError(f"Invalid protein template payload: {outcome!r}")
    return tuple(AF3Template.model_validate(template) for template in raw_templates)


def search_msa_and_templates(
    config: AF3Config | str | Path,
    *,
    search_msa: bool = True,
    search_protein_templates: bool = True,
    max_parallel_search_workers: int = 4,
) -> AF3Config:
    """Resolve MSA fields with bounded resumable database workers."""
    worker_budget = _validate_search_worker_budget(max_parallel_search_workers)
    conf = (
        validate_af3_config(config)
        if isinstance(config, AF3Config)
        else AF3Config.from_file(config)
    )
    if not search_msa:
        return validate_af3_config(_fill_missing_msa_for_inference(conf))

    states = _chain_msa_states(conf)
    plan = plan_msa_resolution(states)
    raw_inputs = [(task.database_id, task.sequence) for task in plan.raw_searches]
    cache_statuses = inspect_msa_search_cache.remote(raw_inputs) if raw_inputs else []
    if len(cache_statuses) != len(plan.raw_searches):
        raise RuntimeError("MSA cache inspection returned the wrong result count")
    missing_raw: list[RawSearchTask] = []
    for task, status in zip(plan.raw_searches, cache_statuses, strict=True):
        if (
            status.get("database_id") != task.database_id
            or status.get("sequence_sha256") != task.sequence_hash
            or status.get("status") not in {"missing", "reused"}
        ):
            raise RuntimeError(f"Invalid MSA cache inspection result: {status}")
        if status["status"] == "missing":
            missing_raw.append(task)

    print(
        "🧬 Sharded MSA search plan: "
        f"{len(cache_statuses) - len(missing_raw)} cached, "
        f"{len(missing_raw)} missing, worker cap {worker_budget}; each database "
        f"worker runs at most {SEARCH_MAX_PARALLEL_SHARDS} shard searches "
        f"with {SEARCH_N_CPU} HMMER CPUs each (request-wide theoretical cap "
        f"{worker_budget * SEARCH_MAX_PARALLEL_SHARDS} shard searches / "
        f"{worker_budget * SEARCH_MAX_PARALLEL_SHARDS * SEARCH_N_CPU} HMMER "
        "CPU slots)."
    )
    search_outcomes = bounded_map(
        missing_raw,
        _remote_search_outcome,
        max_parallel=worker_budget,
    )
    search_failures = [
        {
            "database_id": task.database_id,
            "polymer": task.polymer,
            "sequence_sha256": task.sequence_hash,
            "error_type": type(outcome).__name__,
            "message": str(outcome),
        }
        for task, outcome in zip(missing_raw, search_outcomes, strict=True)
        if isinstance(outcome, Exception)
    ]
    if search_failures:
        raise RuntimeError(
            "Incomplete Raw Database MSA tasks; rerun to reuse successful "
            f"siblings: {search_failures}"
        )

    assembly_outcomes = bounded_map(
        plan.assemblies,
        _remote_assembly_outcome,
        max_parallel=worker_budget,
    )
    assembly_failures = [
        {
            "polymer": task.polymer,
            "sequence_sha256": sequence_hash(task.sequence),
            "error_type": type(outcome).__name__,
            "message": str(outcome),
        }
        for task, outcome in zip(
            plan.assemblies,
            assembly_outcomes,
            strict=True,
        )
        if isinstance(outcome, Exception)
    ]
    if assembly_failures:
        raise RuntimeError(
            "Incomplete MSA assembly tasks; raw database results remain "
            f"reusable: {assembly_failures}"
        )

    assembly_by_sequence: dict[tuple[Polymer, str], dict[str, str]] = {}
    canonical_combined_sequences: set[tuple[Polymer, str]] = set()
    for task, outcome in zip(plan.assemblies, assembly_outcomes, strict=True):
        if not isinstance(outcome, dict):
            raise RuntimeError(f"Invalid MSA assembly result: {outcome!r}")
        status = outcome.get("status")
        if (
            status not in {"published", "reused", "request-local"}
            or outcome.get("polymer") != task.polymer
            or outcome.get("sequence_sha256") != sequence_hash(task.sequence)
        ):
            raise RuntimeError(f"Invalid MSA assembly result: {outcome!r}")
        raw_fields = outcome.get("fields")
        if not isinstance(raw_fields, dict):
            raise RuntimeError(f"Invalid MSA assembly result: {outcome!r}")
        expected_fields = {
            field
            for field, include in (
                ("unpairedMsa", task.include_unpaired),
                ("pairedMsa", task.include_paired),
            )
            if include
        }
        if set(raw_fields) != expected_fields:
            raise RuntimeError(f"Invalid MSA assembly fields: {raw_fields!r}")
        fields: dict[str, str] = {}
        for field, value in raw_fields.items():
            if not isinstance(field, str) or not isinstance(value, str) or not value:
                raise RuntimeError(f"Invalid MSA assembly fields: {raw_fields!r}")
            fields[field] = value
        if status in {"published", "reused"}:
            combined_identity = outcome.get("combined_identity")
            if (
                not isinstance(combined_identity, str)
                or len(combined_identity) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in combined_identity
                )
            ):
                raise RuntimeError(f"Invalid MSA assembly result: {outcome!r}")
            canonical_combined_sequences.add((task.polymer, task.sequence))
        assembly_by_sequence[(task.polymer, task.sequence)] = fields

    for state in states:
        fields = assembly_by_sequence.get((state.polymer, state.sequence), {})
        entry = conf.sequences[state.chain_index]
        if (protein := entry.protein) is not None:
            if not state.unpaired_present:
                protein.unpairedMsa = fields["unpairedMsa"]
                protein.unpairedMsaPath = None
            if not state.paired_present:
                protein.pairedMsa = fields["pairedMsa"]
                protein.pairedMsaPath = None
            if not search_protein_templates and not protein.templates:
                protein.templates = []
        elif (rna := entry.rna) is not None and not state.unpaired_present:
            rna.unpairedMsa = fields["unpairedMsa"]
            rna.unpairedMsaPath = None

    if not search_protein_templates:
        return validate_af3_config(conf)

    template_tasks, template_chain_indices = _plan_template_tasks(
        conf,
        states,
        canonical_combined_sequences,
    )
    canonical_tasks = tuple(task for task in template_tasks if task.publish_canonical)
    cache_inputs = [
        (
            task.sequence,
            task.unpaired_msa_sha256,
            task.max_template_date,
        )
        for task in canonical_tasks
    ]
    template_statuses = (
        inspect_protein_template_cache.remote(cache_inputs) if cache_inputs else []
    )
    if len(template_statuses) != len(canonical_tasks):
        raise RuntimeError(
            "Protein template cache inspection returned the wrong result count"
        )

    templates_by_identity: dict[str, tuple[AF3Template, ...]] = {}
    missing_canonical: list[TemplateTask] = []
    for task, status in zip(
        canonical_tasks,
        template_statuses,
        strict=True,
    ):
        if status.get("status") == "missing":
            if (
                status.get("sequence_sha256") != sequence_hash(task.sequence)
                or status.get("unpaired_msa_sha256") != task.unpaired_msa_sha256
                or status.get("template_identity") != task.template_identity
            ):
                raise RuntimeError(f"Invalid protein template cache result: {status!r}")
            missing_canonical.append(task)
        else:
            templates_by_identity[task.template_identity] = _validated_template_result(
                task,
                status,
                allowed_statuses=frozenset({"reused"}),
            )

    request_local_tasks = tuple(
        task for task in template_tasks if not task.publish_canonical
    )
    worker_tasks = tuple(missing_canonical) + request_local_tasks
    print(
        "🧬 Protein template search plan: "
        f"{len(canonical_tasks) - len(missing_canonical)} cached, "
        f"{len(missing_canonical)} missing canonical, "
        f"{len(request_local_tasks)} request-local, "
        f"worker cap {worker_budget}."
    )
    template_outcomes = bounded_map(
        worker_tasks,
        _remote_template_outcome,
        max_parallel=worker_budget,
    )
    template_failures: list[dict[str, object]] = []
    for task, outcome in zip(worker_tasks, template_outcomes, strict=True):
        if isinstance(outcome, Exception):
            template_failures.append({
                "sequence_sha256": sequence_hash(task.sequence),
                "unpaired_msa_sha256": task.unpaired_msa_sha256,
                "publish_canonical": task.publish_canonical,
                "error_type": type(outcome).__name__,
                "message": str(outcome),
            })
            continue
        if not isinstance(outcome, dict):
            error = RuntimeError(f"Invalid protein template result: {outcome!r}")
        else:
            try:
                templates_by_identity[task.template_identity] = (
                    _validated_template_result(
                        task,
                        outcome,
                        allowed_statuses=(
                            frozenset({"published", "reused"})
                            if task.publish_canonical
                            else frozenset({"request-local"})
                        ),
                    )
                )
            except Exception as exc:
                error = exc
            else:
                continue
        template_failures.append({
            "sequence_sha256": sequence_hash(task.sequence),
            "unpaired_msa_sha256": task.unpaired_msa_sha256,
            "publish_canonical": task.publish_canonical,
            "error_type": type(error).__name__,
            "message": str(error),
        })
    if template_failures:
        raise RuntimeError(
            "Incomplete protein template tasks; completed canonical results "
            f"remain reusable: {template_failures}"
        )

    for identity, chain_indices in template_chain_indices.items():
        templates = templates_by_identity.get(identity)
        if templates is None:
            raise RuntimeError(f"Protein template task produced no result: {identity}")
        for chain_index in chain_indices:
            protein = conf.sequences[chain_index].protein
            if protein is None:
                raise RuntimeError("Protein template plan no longer matches its chain")
            protein.templates = [
                template.model_copy(deep=True) for template in templates
            ]
    return validate_af3_config(conf)


##########################################
# Inference functions
##########################################


def _stage_inference_run(prepared: PreparedInferenceRun) -> None:
    """Upload normalized inputs and custom templates to the output Volume."""
    with CONF.output_volume.batch_upload(force=True) as batch:
        for upload in prepared.uploads:
            batch.put_file(
                BytesIO(upload.content),
                f"/{upload.relative_path.as_posix()}",
            )


def _inference_runtime() -> InferenceRuntime:
    """Bind seed publication to the app-owned output Volume and claims."""
    return InferenceRuntime(
        output_root=Path(CONF.output_volume_mountpoint),
        volume=CONF.output_volume,
        claims=INFERENCE_CLAIMS,
        container_id=_CONTAINER_INSTANCE_ID,
        maximum_age_seconds=MAX_TIMEOUT + 900,
        wait_timeout_seconds=max(60, MAX_TIMEOUT - 60),
        function_call_id=modal.current_function_call_id(),
    )


@app.function(
    cpu=0.125,
    memory=1024,
    timeout=600,
    volumes={
        CONF.output_volume_mountpoint: CONF.output_volume.with_mount_options(
            read_only=True,
            sub_path="/",
        )
    },
)
def inspect_seed_prediction_cache(
    run_id: str,
    seeds: list[int],
    sample_count: int,
) -> list[dict[str, object]]:
    """Inspect seed markers without scanning prediction directories."""
    return inspect_seed_predictions(
        _inference_runtime(),
        run_id,
        tuple(seeds),
        sample_count=sample_count,
    )


@app.function(
    cpu=0.125,
    memory=1024,
    timeout=600,
    volumes={
        CONF.output_volume_mountpoint: CONF.output_volume.with_mount_options(
            read_only=True,
            sub_path="/",
        )
    },
)
def claim_seed_prediction_work(
    run_id: str,
    seeds: list[int],
    sample_count: int,
) -> dict[str, object]:
    """Reuse or atomically claim one request's currently incomplete seeds."""
    return claim_seed_predictions(
        _inference_runtime(),
        run_id,
        tuple(seeds),
        sample_count=sample_count,
    ).to_dict()


@app.function(
    gpu=CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 131072),
    timeout=MAX_TIMEOUT,
    volumes=CONF.mounts(
        output_volume=True,
        model_volume=True,
        model_ro=False,
    ),
)
def run_inference_pipeline(
    json_bytes: bytes,
    run_id: str,
    recycle: int,
    sample: int,
    claimed_seed_records: list[dict[str, object]],
) -> dict[str, object]:
    """Run one disjoint seed group and publish per-seed markers."""
    import sys

    claimed_seeds = tuple(
        claimed_seed_from_dict(record) for record in claimed_seed_records
    )
    base_conf = AF3Config.model_validate_json(json_bytes)

    def execute(
        worker_root: Path,
        canonical_name: str,
        seeds: tuple[int, ...],
    ) -> None:
        conf = base_conf.model_copy(deep=True)
        conf.name = canonical_name
        conf.modelSeeds = list(seeds)
        conf = _fill_missing_msa_for_inference(conf)
        input_json_path = worker_root / "input.json"
        input_json_path.write_bytes(serialize_af3_input(conf))
        print(f"💊 Running inference for {canonical_name} with seeds {list(seeds)}")
        model_dir = Path(CONF.model_volume_mountpoint)
        cmd = [
            sys.executable,
            str(CONF.git_clone_dir / "run_alphafold.py"),
            "--run_inference=true",
            "--run_data_pipeline=false",
            f"--json_path={input_json_path}",
            f"--output_dir={worker_root}",
            f"--model_dir={model_dir}",
            f"--jax_compilation_cache_dir={model_dir / 'jax_cache'}",
            f"--num_recycles={recycle}",
            f"--num_diffusion_samples={sample}",
        ]
        run_command(
            cmd,
            output_mode="tee",
            log_file=worker_root / "run.log",
        )

    return run_seed_prediction_worker(
        _inference_runtime(),
        SeedWorkerTask(
            run_id=run_id,
            sample_count=sample,
            claimed_seeds=claimed_seeds,
        ),
        execute,
    )


@app.function(
    cpu=(0.125, 2.125),
    memory=(1024, 16384),
    timeout=3600,
    volumes=CONF.mounts(output_volume=True),
)
def finalize_inference_summary(
    json_bytes: bytes,
    run_id: str,
    sample_count: int,
) -> dict[str, object]:
    """Rebuild the non-regressing accumulated run summary."""
    from alphafold3.common import (  # type: ignore[ty:unresolved-import]
        folding_input,
    )

    base_conf = AF3Config.model_validate_json(json_bytes)

    def build_data_json(seeds: tuple[int, ...]) -> bytes:
        conf = base_conf.model_copy(deep=True)
        conf.name = canonical_output_name(run_id)
        conf.modelSeeds = list(seeds)
        fold_input = folding_input.Input.from_json(serialize_af3_input(conf).decode())
        return fold_input.to_json().encode()

    return finalize_run_summary(
        _inference_runtime(),
        run_id,
        sample_count=sample_count,
        build_data_json=build_data_json,
    )


@app.function(
    cpu=(0.125, 2.125),
    memory=(1024, 16384),
    timeout=3600,
    volumes=CONF.mounts(output_volume=True),
)
def finalize_inference_request(
    run_id: str,
    request_id: str,
    submitted_seeds: list[int],
    normalized_seeds: list[int],
    sample_count: int,
    display_name: str,
    reused_seeds: list[int],
    published_seeds: list[int],
) -> dict[str, object]:
    """Publish one manifest-last view over the request's completed seeds."""
    return publish_request_results(
        _inference_runtime(),
        RequestPublication(
            run_id=run_id,
            request_id=request_id,
            submitted_seeds=tuple(submitted_seeds),
            normalized_seeds=tuple(normalized_seeds),
            sample_count=sample_count,
            display_name=display_name,
            reused_seeds=tuple(reused_seeds),
            published_seeds=tuple(published_seeds),
        ),
    )


def _claimed_seed_record(item: ClaimedSeed) -> dict[str, object]:
    return {
        "seed": item.seed,
        "claim": {
            "scope_key": item.claim.scope_key,
            "generation_id": item.claim.generation_id,
            "owner": item.claim.owner,
        },
    }


def _completed_seed_set(
    run_id: str,
    seeds: tuple[int, ...],
    sample_count: int,
) -> set[int]:
    statuses = inspect_seed_prediction_cache.remote(
        run_id,
        list(seeds),
        sample_count,
    )
    if len(statuses) != len(seeds):
        raise RuntimeError("Seed marker inspection returned the wrong result count")
    completed: set[int] = set()
    for seed, status in zip(seeds, statuses, strict=True):
        if status.get("run_id") != run_id or status.get("seed") != seed:
            raise RuntimeError(f"Invalid seed marker inspection result: {status!r}")
        if status.get("status") == "reused":
            completed.add(seed)
        elif status.get("status") != "missing":
            raise RuntimeError(f"Invalid seed marker inspection result: {status!r}")
    return completed


def _run_claimed_seed_batches(
    prepared: PreparedInferenceRun,
    claimed_seeds: tuple[ClaimedSeed, ...],
    *,
    recycle: int,
    sample: int,
    max_workers: int,
    poll_timeout: int,
) -> tuple[set[int], set[int], list[dict[str, object]]]:
    batches = partition_claimed_seeds(claimed_seeds, max_workers)
    json_bytes = serialize_af3_input(prepared.worker_config)
    calls: dict[int, tuple[modal.FunctionCall, tuple[ClaimedSeed, ...]]] = {}
    for index, batch in enumerate(batches):
        calls[index] = (
            run_inference_pipeline.spawn(
                json_bytes,
                prepared.run_id,
                recycle,
                sample,
                [_claimed_seed_record(item) for item in batch],
            ),
            batch,
        )

    published: set[int] = set()
    reused: set[int] = set()
    failures: list[dict[str, object]] = []
    while calls:
        for index, (function_call, batch) in calls.copy().items():
            try:
                result = function_call.get(timeout=poll_timeout)
            except TimeoutError:
                continue
            except Exception as exc:
                failures.append({
                    "seeds": [item.seed for item in batch],
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })
                del calls[index]
                continue
            if not isinstance(result, dict) or result.get("run_id") != prepared.run_id:
                failures.append({
                    "seeds": [item.seed for item in batch],
                    "error_type": "InvalidWorkerResult",
                    "message": repr(result),
                })
            else:
                raw_published = result.get("published_seeds")
                raw_reused = result.get("reused_seeds")
                if (
                    not isinstance(raw_published, list)
                    or not isinstance(raw_reused, list)
                    or any(
                        isinstance(seed, bool) or not isinstance(seed, int)
                        for seed in [*raw_published, *raw_reused]
                    )
                    or not set([*raw_published, *raw_reused]).issubset({
                        item.seed for item in batch
                    })
                ):
                    failures.append({
                        "seeds": [item.seed for item in batch],
                        "error_type": "InvalidWorkerResult",
                        "message": repr(result),
                    })
                else:
                    published.update(raw_published)
                    reused.update(raw_reused)
            del calls[index]
    return published, reused, failures


def predict_structures(
    prepared: PreparedInferenceRun,
    recycle: int,
    sample: int,
    num_containers: int,
    *,
    poll_timeout: int = 30,
) -> dict[str, object]:
    """Reconcile, run once, and summarize the requested seed predictions."""
    if (
        isinstance(num_containers, bool)
        or not isinstance(num_containers, int)
        or num_containers < 1
    ):
        raise ValueError("num_containers must be a positive integer")
    requested = prepared.normalized_seeds
    pending = set(requested)
    reused: set[int] = set()
    published: set[int] = set()
    failures: list[dict[str, object]] = []
    attempted: set[int] = set()
    deadline = time.monotonic() + MAX_TIMEOUT + 900

    while pending:
        raw_plan = claim_seed_prediction_work.remote(
            prepared.run_id,
            sorted(pending),
            sample,
        )
        plan = seed_claim_plan_from_dict(raw_plan)
        reused.update(plan.reused_seeds)
        pending.difference_update(plan.reused_seeds)
        if plan.owned:
            owned_seeds = {item.seed for item in plan.owned}
            if attempted.intersection(owned_seeds):
                raise RuntimeError("Refusing to retry a surfaced seed failure")
            attempted.update(owned_seeds)
            (
                batch_published,
                batch_reused,
                batch_failures,
            ) = _run_claimed_seed_batches(
                prepared,
                plan.owned,
                recycle=recycle,
                sample=sample,
                max_workers=num_containers,
                poll_timeout=poll_timeout,
            )
            completed_owned = _completed_seed_set(
                prepared.run_id,
                tuple(sorted(owned_seeds)),
                sample,
            )
            published.update(batch_published)
            published.update(completed_owned - batch_reused)
            reused.update(batch_reused)
            pending.difference_update(owned_seeds)
            failures.extend(batch_failures)
            for seed in sorted(owned_seeds - completed_owned):
                if not any(
                    isinstance(
                        failure_seeds := failure.get("seeds"),
                        list,
                    )
                    and seed in failure_seeds
                    for failure in batch_failures
                ):
                    failures.append({
                        "seeds": [seed],
                        "error_type": "IncompleteSeedPrediction",
                        "message": "Worker returned without a valid seed marker",
                    })

        active_seeds = {item.seed for item in plan.active}
        if pending and pending != active_seeds:
            raise RuntimeError("Seed claim plan did not account for every pending seed")
        if pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failures.append({
                    "seeds": sorted(pending),
                    "error_type": "ActiveSeedTimeout",
                    "message": "Timed out waiting for concurrent seed owners",
                })
                break
            time.sleep(min(30, remaining))

    completed = _completed_seed_set(
        prepared.run_id,
        requested,
        sample,
    )
    reused.update(completed.difference(reused, published))
    incomplete = set(requested) - completed
    summary: dict[str, object] | None = None
    if completed:
        summary = finalize_inference_summary.remote(
            serialize_af3_input(prepared.worker_config),
            prepared.run_id,
            sample,
        )
    result: dict[str, object] = {
        "run_id": prepared.run_id,
        "request_id": prepared.request_id,
        "requested_seeds": list(requested),
        "reused_seeds": sorted(reused),
        "published_seeds": sorted(published),
        "completed_seeds": sorted(completed),
        "incomplete_seeds": sorted(incomplete),
        "failures": failures,
        "summary": summary,
    }
    if incomplete:
        raise RuntimeError(
            "Incomplete AlphaFold3 seed predictions; completed siblings remain "
            f"reusable and no failed seed was retried: {result}"
        )
    result["request"] = finalize_inference_request.remote(
        prepared.run_id,
        prepared.request_id,
        list(prepared.submitted_seeds),
        list(prepared.normalized_seeds),
        sample,
        prepared.display_name,
        sorted(reused),
        sorted(published),
    )
    return result


##########################################
# Entrypoint for ephemeral usage
##########################################
@app.local_entrypoint()
def setup_sharded_databases(
    seqkit_threads: int = DEFAULT_SEQKIT_THREADS,
    source_policy: str = "keep",
    submit: bool = False,
) -> None:
    """Plan or build every missing fixed sharded database profile.

    Args:
        seqkit_threads: SeqKit/native-helper threads per builder, default 8.
        source_policy: Post-publication source action: keep, compress, or delete.
        submit: Submit Modal work. Defaults to false and only prints the plan.
    """
    plan = _sharded_database_setup_plan(seqkit_threads, source_policy)
    print(
        orjson.dumps(
            plan,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        ).decode()
    )
    if not submit:
        print("🧬 Plan only; no Modal function was submitted.")
        return

    print("🧬 Inspecting fixed sharded database profiles...")
    inventory = inspect_sharded_database_profiles.remote()
    inputs = plan_missing_profile_builds(
        inventory,
        seqkit_threads,
        source_policy,
    )
    missing = [database_id for database_id, _, _ in inputs]
    budget = profile_build_slot_budget(len(inputs), seqkit_threads)
    print(
        "🧬 Effective profile-build fanout: "
        f"{budget['builder_containers']} containers, "
        f"{budget['maximum_effective_worker_slots']} configured worker slots."
    )

    results: list[dict[str, object] | BaseException] = []
    if missing:
        print(
            "🧬 Submitting missing database profiles concurrently: "
            f"{', '.join(missing)}"
        )
        results = list(
            build_sharded_database.starmap(
                inputs,
                return_exceptions=True,
            )
        )
        failures = [
            {
                "database_id": database_id,
                "error_type": type(result).__name__,
                "message": str(result),
            }
            for database_id, result in zip(missing, results, strict=True)
            if isinstance(result, BaseException)
        ]
        if failures:
            raise RuntimeError(
                "One or more sharded database builders failed after all "
                f"submitted builders completed: {failures}"
            )
    else:
        print("🧬 All fixed profiles are already valid; no builders submitted.")

    print("🧬 Running final inventory and workspace cleanup...")
    final_inventory = finalize_sharded_database_setup.remote()
    summary = {
        "status": "complete",
        "initial_inventory": inventory,
        "builder_results": results,
        "final_inventory": final_inventory,
    }
    print(
        orjson.dumps(
            summary,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        ).decode()
    )


@app.local_entrypoint()
def submit_alphafold3_task(
    input_json: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    search_msa: bool = True,
    search_protein_templates: bool = True,
    max_parallel_search_workers: int = 4,
    max_num_gpus: int = 1,
    recycle: int = 10,
    sample: int = 5,
) -> None:
    """Run AlphaFold3 on Modal and fetch results to `out_dir`.

    Args:
        input_json: Path to input JSON file.
        out_dir: Optional local archive directory (defaults to $CWD).
        run_name: Optional display name used in downloaded output basenames.
            Defaults to `name` in the AF3 JSON config.
        search_msa: Populate missing protein and RNA MSA fields.
        search_protein_templates: Populate missing protein templates after MSA
            resolution. Non-empty caller fields are always preserved.
        max_parallel_search_workers: Request-wide cap for database and template
            workers. Database workers internally use 16 shards by two HMMER
            CPUs.
        max_num_gpus: Maximum number of disjoint seed workers to run during
            inference.
        recycle: Number of Pairformer recycles to use during inference.
        sample: Number of diffusion samples to generate per seed.

    """
    max_num_gpus = _validate_max_num_gpus(max_num_gpus)
    _validate_search_worker_budget(max_parallel_search_workers)
    validate_inference_parameters(recycle, sample)

    # Validate and read input
    input_path = Path(input_json).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    local_input = materialize_local_input(input_path)
    conf = local_input.config
    if run_name is None:
        run_name = conf.name
    sanitize_presentation_name(run_name)
    conf.name = run_name

    print(f"🧬 Resolving {CONF.name} MSA and template fields...")
    enriched_conf = search_msa_and_templates(
        conf,
        search_msa=search_msa,
        search_protein_templates=search_protein_templates,
        max_parallel_search_workers=max_parallel_search_workers,
    )

    enriched_conf.name = run_name
    enriched_conf.modelSeeds = conf.modelSeeds
    prepared = prepare_inference_run(
        enriched_conf,
        local_input.custom_templates,
        output_mount_root=Path(CONF.output_volume_mountpoint),
        recycle=recycle,
        sample=sample,
    )
    if prepared.submitted_seeds != prepared.normalized_seeds:
        print(
            "🧬 Normalized duplicate model seeds: "
            f"{list(prepared.submitted_seeds)} -> "
            f"{list(prepared.normalized_seeds)}"
        )
    _stage_inference_run(prepared)
    print(
        "🧬 Staged inference input: "
        f"run_id={prepared.run_id}, request_id={prepared.request_id}"
    )

    num_seeds = len(prepared.normalized_seeds)
    num_containers = min(max_num_gpus, num_seeds)
    print(f"🧬 Running {CONF.name} inference pipeline with {num_containers=}...")
    result = predict_structures(
        prepared,
        recycle,
        sample,
        num_containers,
    )
    request_manifest = result.get("request")
    if not isinstance(request_manifest, dict):
        raise RuntimeError(
            f"AlphaFold3 request publication returned invalid metadata: {result!r}"
        )
    archive_path = create_request_archive(
        CONF.output_volume,
        cast(dict[str, object], request_manifest),
        output_dir=resolve_local_output_dir(out_dir),
        display_name=run_name,
    )
    print(
        f"🧬 {CONF.name} results saved to {archive_path}. Durable seed "
        f"predictions remain in {CONF.output_volume_name}:/{prepared.run_root}."
    )
