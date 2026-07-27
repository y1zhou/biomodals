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
import uuid
from pathlib import Path, PurePosixPath

import modal
import orjson
from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.config import AppConfig
from biomodals.app.fold.alphafold3.inference_inputs import (
    ALPHAFOLD3_APP_VERSION,
    PreparedInferenceRun,
    materialize_local_input,
    prepare_inference_run,
    sanitize_af3_name,
    validate_inference_parameters,
    validate_inference_worker_budget,
)
from biomodals.app.fold.alphafold3.inference_pipeline import coordinate_seed_predictions
from biomodals.app.fold.alphafold3.modal_adapters import (
    ModalInferenceExecutor,
    ModalSearchExecutor,
    execute_profile_setup,
    stage_inference_run,
)
from biomodals.app.fold.alphafold3.msa_search import (
    MSA_SEARCH_CLAIM_DICT_NAME,
    MsaAssemblyTask,
    Polymer,
    RawSearchTask,
    SearchRuntime,
    assemble_and_publish_msas,
    inspect_raw_searches,
    run_database_search,
)
from biomodals.app.fold.alphafold3.profile_builder import (
    ProfileBuilderRuntime,
    build_profile,
    finalize_profile_setup,
    inspect_profile_registry,
)
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    ALPHAFOLD3_REPOSITORY,
    BUILD_MEMORY_MIB,
    BUILD_TIMEOUT_SECONDS,
    DEFAULT_SEQKIT_THREADS,
    PROFILE_BUILD_CLAIM_DICT_NAME,
    PROFILE_BUILD_CPU,
    PROFILE_BUILD_MAX_CONTAINERS,
    SEQKIT_VERSION,
    SHARDED_DB_VOLUME_NAME,
    SourcePolicy,
    plan_profile_setup,
)
from biomodals.app.fold.alphafold3.request_results import (
    RequestPublication,
    create_request_archive,
    publish_request_results,
    request_manifest_from_result,
)
from biomodals.app.fold.alphafold3.search_pipeline import (
    resolve_msa_and_templates,
    validate_search_worker_budget,
)
from biomodals.app.fold.alphafold3.seed_predictions import (
    SEED_PREDICTION_CLAIM_DICT_NAME,
    InferenceRuntime,
    claim_seed_predictions,
    inspect_seed_predictions,
)
from biomodals.app.fold.alphafold3.sharding import (
    CONTAINER_NATIVE_SOURCE_DIR,
    NATIVE_SOURCE_DIR_ENV,
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
from biomodals.app.fold.alphafold3.upstream_inference import (
    UpstreamInferenceRuntime,
    finalize_upstream_run_summary,
    run_upstream_seed_worker,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import (
    AF3_MSA_DB_VOLUME,
    MAX_TIMEOUT,
    MSA_CACHE_VOLUME,
)
from biomodals.helper.io import resolve_local_output_dir

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

##########################################
# Image and app definitions
##########################################
SHARDED_MSA_DB_VOLUME = modal.Volume.from_name(
    SHARDED_DB_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
_JAX_CACHE_MOUNTPOINT = PurePosixPath(f"/{CONF.name}-jax-cache")
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
        output_root=Path(CONF.output_volume_mountpoint),
        source_volume=AF3_MSA_DB_VOLUME,
        sharded_volume=SHARDED_MSA_DB_VOLUME,
        output_volume=CONF.output_volume,
        claims=PROFILE_BUILD_CLAIMS,
        container_id=_CONTAINER_INSTANCE_ID,
    )


@app.function(
    image=sharding_image,
    cpu=PROFILE_BUILD_CPU,
    memory=BUILD_MEMORY_MIB,
    timeout=BUILD_TIMEOUT_SECONDS,
    max_containers=PROFILE_BUILD_MAX_CONTAINERS,
    volumes={
        ProfileBuilderRuntime.SOURCE_MOUNT: AF3_MSA_DB_VOLUME,
        ProfileBuilderRuntime.SHARDED_MOUNT: SHARDED_MSA_DB_VOLUME,
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
        ProfileBuilderRuntime.SHARDED_MOUNT: (
            SHARDED_MSA_DB_VOLUME.with_mount_options(
                read_only=True,
                sub_path="/",
            )
        ),
    },
)
def inspect_sharded_database_profiles() -> dict[str, object]:
    """Inspect all fixed profile manifests without expensive digest scans."""
    SHARDED_MSA_DB_VOLUME.reload()
    return inspect_profile_registry(Path(ProfileBuilderRuntime.SHARDED_MOUNT))


@app.function(
    image=sharding_image,
    cpu=0.125,
    memory=1024,
    timeout=600,
    max_containers=1,
    volumes={
        ProfileBuilderRuntime.SHARDED_MOUNT: SHARDED_MSA_DB_VOLUME,
        CONF.output_volume_mountpoint: CONF.output_volume,
    },
)
def finalize_sharded_database_setup() -> dict[str, object]:
    """Clean abandoned and unselected profiles after all builders complete."""
    return finalize_profile_setup(_profile_builder_runtime())


##########################################
# MSA search functions
##########################################
@app.function(
    cpu=0.125,
    memory=1024,
    timeout=600,
    max_containers=1,
    volumes={
        SearchRuntime.SHARDED_MOUNT: (
            SHARDED_MSA_DB_VOLUME.with_mount_options(
                read_only=True,
                sub_path="/",
            )
        ),
        SearchRuntime.CACHE_MOUNT: MSA_CACHE_VOLUME.with_mount_options(
            read_only=True,
            sub_path=SearchRuntime.CACHE_VOLUME_SUBPATH,
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
        Path(SearchRuntime.SHARDED_MOUNT),
        Path(SearchRuntime.CACHE_MOUNT),
        tasks,
    )


def _msa_search_runtime(
    *,
    maximum_age_seconds: int | float,
    wait_timeout_seconds: int | float,
) -> SearchRuntime:
    """Bind shared search code to production mounts and persistence objects."""
    return SearchRuntime(
        sharded_volume=SHARDED_MSA_DB_VOLUME,
        cache_volume=MSA_CACHE_VOLUME,
        claims=MSA_SEARCH_CLAIMS,
        container_id=_CONTAINER_INSTANCE_ID,
        maximum_age_seconds=maximum_age_seconds,
        wait_timeout_seconds=wait_timeout_seconds,
    )


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131_072),
    timeout=CONF.timeout,
    volumes={
        SearchRuntime.SHARDED_MOUNT: (
            SHARDED_MSA_DB_VOLUME.with_mount_options(
                read_only=True,
                sub_path="/",
            )
        ),
        SearchRuntime.CACHE_MOUNT: MSA_CACHE_VOLUME.with_mount_options(
            read_only=False,
            sub_path=SearchRuntime.CACHE_VOLUME_SUBPATH,
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
        SearchRuntime.SHARDED_MOUNT: (
            SHARDED_MSA_DB_VOLUME.with_mount_options(
                read_only=True,
                sub_path="/",
            )
        ),
        SearchRuntime.CACHE_MOUNT: MSA_CACHE_VOLUME.with_mount_options(
            read_only=False,
            sub_path=SearchRuntime.CACHE_VOLUME_SUBPATH,
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
        TemplateRuntime.CACHE_MOUNT: MSA_CACHE_VOLUME.with_mount_options(
            read_only=True,
            sub_path=TemplateRuntime.CACHE_VOLUME_SUBPATH,
        ),
    },
)
def inspect_protein_template_cache(
    inputs: list[tuple[str, str, str]],
) -> list[dict[str, object]]:
    """Inspect canonical template markers without consuming search workers."""
    MSA_CACHE_VOLUME.reload()
    return inspect_template_entries(
        Path(TemplateRuntime.CACHE_MOUNT),
        tuple(inputs),
    )


def _template_runtime() -> TemplateRuntime:
    """Bind shared template search to production mounts and claims."""
    return TemplateRuntime(
        source_volume=AF3_MSA_DB_VOLUME,
        cache_volume=MSA_CACHE_VOLUME,
        claims=MSA_SEARCH_CLAIMS,
        container_id=_CONTAINER_INSTANCE_ID,
        maximum_age_seconds=CONF.timeout + 900,
        wait_timeout_seconds=max(60, CONF.timeout - 60),
    )


@app.function(
    cpu=(0.125, 8.125),
    memory=(1024, 32_768),
    timeout=CONF.timeout,
    volumes={
        TemplateRuntime.SOURCE_MOUNT: AF3_MSA_DB_VOLUME.with_mount_options(
            read_only=True,
            sub_path="/",
        ),
        TemplateRuntime.CACHE_MOUNT: MSA_CACHE_VOLUME.with_mount_options(
            read_only=False,
            sub_path=TemplateRuntime.CACHE_VOLUME_SUBPATH,
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


def _search_msa_and_templates(
    config: AF3Config,
    *,
    search_msa: bool = True,
    search_templates: bool = True,
    max_parallel_search_workers: int = 4,
) -> AF3Config:
    """Resolve MSA/template fields through the production Modal adapter."""
    return resolve_msa_and_templates(
        config,
        ModalSearchExecutor(
            inspect_raw_function=inspect_msa_search_cache,
            raw_search_function=search_database_msa,
            msa_assembly_function=assemble_sequence_msas,
            inspect_templates_function=inspect_protein_template_cache,
            template_search_function=search_protein_templates,
        ),
        search_msa=search_msa,
        search_protein_templates=search_templates,
        max_parallel_search_workers=max_parallel_search_workers,
    )


##########################################
# Inference functions
##########################################


def _inference_runtime() -> InferenceRuntime:
    """Bind seed publication to the app-owned output Volume and claims."""
    return InferenceRuntime(
        output_root=Path(CONF.output_volume_mountpoint),
        volume=CONF.output_volume,
        claims=INFERENCE_CLAIMS,
        container_id=_CONTAINER_INSTANCE_ID,
        maximum_age_seconds=MAX_TIMEOUT + 900,
        wait_timeout_seconds=max(60, MAX_TIMEOUT - 60),
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
        model_ro=True,
    )
    | {
        _JAX_CACHE_MOUNTPOINT: modal.Volume.from_name(
            _JAX_CACHE_MOUNTPOINT.name,
            create_if_missing=True,
            version=2,
        )
    },
)
def run_inference_pipeline(
    json_bytes: bytes,
    run_id: str,
    recycle: int,
    sample: int,
    claimed_seed_records: list[dict[str, object]],
) -> dict[str, object]:
    """Run one disjoint seed group and publish per-seed markers."""
    return run_upstream_seed_worker(
        UpstreamInferenceRuntime(
            predictions=_inference_runtime(),
            source_root=CONF.git_clone_dir,
            model_root=Path(CONF.model_volume_mountpoint),
            jax_cache_dir=Path(_JAX_CACHE_MOUNTPOINT) / ALPHAFOLD3_COMMIT,
        ),
        json_bytes,
        run_id,
        recycle,
        sample,
        claimed_seed_records,
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
    return finalize_upstream_run_summary(
        _inference_runtime(),
        json_bytes,
        run_id,
        sample_count,
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


def _predict_structures(
    prepared: PreparedInferenceRun,
    recycle: int,
    sample: int,
    num_containers: int,
    *,
    poll_timeout: int = 30,
) -> dict[str, object]:
    """Reconcile predictions through the production Modal adapter."""
    return coordinate_seed_predictions(
        prepared,
        ModalInferenceExecutor(
            claim_function=claim_seed_prediction_work,
            inspect_function=inspect_seed_prediction_cache,
            worker_function=run_inference_pipeline,
            summary_function=finalize_inference_summary,
            request_function=finalize_inference_request,
        ),
        recycle=recycle,
        sample=sample,
        num_containers=num_containers,
        active_wait_timeout_seconds=MAX_TIMEOUT + 900,
        worker_poll_timeout_seconds=poll_timeout,
    )


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
    plan = plan_profile_setup(
        seqkit_threads,
        source_policy,
        evidence_volume_name=CONF.output_volume_name,
    )
    print(
        orjson.dumps(
            plan,
            option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
        ).decode()
    )
    if not submit:
        print("🧬 Plan only; no Modal function was submitted.")
        return

    summary = execute_profile_setup(
        inspect_sharded_database_profiles,
        build_sharded_database,
        finalize_sharded_database_setup,
        seqkit_threads=seqkit_threads,
        source_policy=source_policy,
    )
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
    max_num_gpus = validate_inference_worker_budget(max_num_gpus)
    validate_search_worker_budget(max_parallel_search_workers)
    validate_inference_parameters(recycle, sample)

    input_path = Path(input_json).expanduser()
    local_input = materialize_local_input(input_path)
    conf = local_input.config
    if run_name is None:
        run_name = conf.name
    sanitize_af3_name(run_name)
    conf.name = run_name

    print(f"🧬 Resolving {CONF.name} MSA and template fields...")
    enriched_conf = _search_msa_and_templates(
        conf,
        search_msa=search_msa,
        search_templates=search_protein_templates,
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
    stage_inference_run(CONF.output_volume, prepared)
    print(
        "🧬 Staged inference input: "
        f"run_id={prepared.run_id}, request_id={prepared.request_id}"
    )

    num_seeds = len(prepared.normalized_seeds)
    num_containers = min(max_num_gpus, num_seeds)
    print(f"🧬 Running {CONF.name} inference pipeline with {num_containers=}...")
    result = _predict_structures(
        prepared,
        recycle,
        sample,
        num_containers,
    )
    archive_path = create_request_archive(
        CONF.output_volume,
        request_manifest_from_result(result),
        output_dir=resolve_local_output_dir(out_dir),
        display_name=run_name,
    )
    print(
        f"🧬 {CONF.name} results saved to {archive_path}. Durable seed "
        f"predictions remain in {CONF.output_volume_name}:/{prepared.run_root}."
    )
