"""AlphaFold3 upstream: <https://github.com/google-deepmind/alphafold3>.

Biomodals runs its pinned fork: <https://github.com/y1zhou/alphafold3>.

## Additional notes

This app automatically prepares the model and reference assets required by
each request before running its fixed execution graph.

<https://github.com/google-deepmind/alphafold3#obtaining-model-parameters>

<https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md#obtaining-genetic-databases>

## Examples

Run prediction and download the request-scoped archive:

`uv run biomodals app run alphafold3::submit_alphafold3_task -- --input-json input.json --out-dir outputs`

See <https://github.com/google-deepmind/alphafold3/tree/main/docs> for general docs.

## Outputs

See <https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md>.
"""

import os
import uuid
import warnings
from pathlib import Path, PurePosixPath
from uuid import UUID

import modal

from biomodals.app.config import AppConfig
from biomodals.app.fold.alphafold3.environment import (
    ENVIRONMENT_SETUP_CLAIM_DICT_NAME,
    ENVIRONMENT_SETUP_TIMEOUT_SECONDS,
    EnvironmentAsset,
    EnvironmentRuntime,
    prepare_environment_asset,
)
from biomodals.app.fold.alphafold3.execution_coordinator import (
    AlphaFold3ExecutionCoordinator,
)
from biomodals.app.fold.alphafold3.execution_publications import (
    publish_execution_result,
)
from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
    stage_execution_request,
)
from biomodals.app.fold.alphafold3.inference_inputs import (
    ALPHAFOLD3_APP_VERSION,
    MAX_SEED_SAMPLE_PAIRS,
    LoadedInferenceInput,
    load_staged_inference_input,
    materialize_local_input,
    normalize_model_seeds,
    sanitize_af3_name,
)
from biomodals.app.fold.alphafold3.invocation_cache import (
    load_invocation_manifest,
)
from biomodals.app.fold.alphafold3.msa_search import (
    MSA_SEARCH_CLAIM_DICT_NAME,
    MsaArtifactReference,
    MsaAssemblyTask,
    Polymer,
    SearchRuntime,
    assemble_and_publish_msas,
    run_database_search,
    sequence_cache_relpath,
)
from biomodals.app.fold.alphafold3.profile_builder import (
    ProfileBuilderRuntime,
    build_profile,
)
from biomodals.app.fold.alphafold3.profiles import (
    ALPHAFOLD3_COMMIT,
    ALPHAFOLD3_REPOSITORY,
    BUILD_MEMORY_MIB,
    DEFAULT_SEQKIT_THREADS,
    PROFILE_BUILD_CPU,
    PROFILE_BUILD_MAX_CONTAINERS,
    SEQKIT_VERSION,
    SHARDED_DB_VOLUME_NAME,
)
from biomodals.app.fold.alphafold3.request_results import (
    RequestPublication,
    create_request_archive,
    publish_request_results,
    request_publication_from_manifest,
)
from biomodals.app.fold.alphafold3.seed_predictions import (
    SEED_PREDICTION_CLAIM_DICT_NAME,
    InferenceRuntime,
    claim_seed_predictions,
    guard_seed_prediction_claims,
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
    run_template_search,
)
from biomodals.app.fold.alphafold3.upstream_inference import (
    UpstreamInferenceRuntime,
    finalize_upstream_run_summary,
    run_upstream_seed_worker,
)
from biomodals.execution import (
    COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    DeploymentIdentity,
    ExecutionOverview,
    ProviderCallDiagnostic,
    ProviderCallPage,
)
from biomodals.execution.modal import (
    ModalCallDriver,
    development_modal_call_driver,
    execution_coordinator_adapter,
    execution_coordinator_identity,
    initialize_execution_coordinator_host,
    resolve_provider_call_limits,
    submit_staged_execution_run,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.constant import (
    AF3_MSA_DB_VOLUME,
    MAX_TIMEOUT,
    MODEL_VOLUME,
    MSA_CACHE_VOLUME,
)
from biomodals.helper.io import resolve_local_output_dir

##########################################
# Modal configs
##########################################
CONF = AppConfig(
    tags={"group": Path(__file__).parent.name, "biomodals_tool": "alphafold3"},
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
JAX_CACHE_MOUNTPOINT = PurePosixPath(f"/{CONF.name}-jax-cache")
JAX_CACHE_VOLUME = modal.Volume.from_name(
    JAX_CACHE_MOUNTPOINT.name,
    create_if_missing=True,
    version=2,
)
_SUMMARY_TIMEOUT_SECONDS = 3600
_MAX_CONCURRENT_COORDINATOR_INPUTS = 8
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_alphafold3_task"})
MSA_SEARCH_CLAIMS = modal.Dict.from_name(
    MSA_SEARCH_CLAIM_DICT_NAME,
    create_if_missing=True,
)
ENVIRONMENT_SETUP_CLAIMS = modal.Dict.from_name(
    ENVIRONMENT_SETUP_CLAIM_DICT_NAME,
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
_PROFILE_BUILDER_RUNTIME = ProfileBuilderRuntime(
    output_root=Path(CONF.output_volume_mountpoint),
    sharded_volume=SHARDED_MSA_DB_VOLUME,
    output_volume=CONF.output_volume,
)
_ENVIRONMENT_RUNTIME = EnvironmentRuntime(
    model_volume=MODEL_VOLUME,
    source_volume=AF3_MSA_DB_VOLUME,
    sharded_volume=SHARDED_MSA_DB_VOLUME,
    claims=ENVIRONMENT_SETUP_CLAIMS,
    container_id=_CONTAINER_INSTANCE_ID,
)
_INFERENCE_RUNTIME = InferenceRuntime(
    output_root=Path(CONF.output_volume_mountpoint),
    volume=CONF.output_volume,
    claims=modal.Dict.from_name(
        SEED_PREDICTION_CLAIM_DICT_NAME,
        create_if_missing=True,
    ),
    container_id=_CONTAINER_INSTANCE_ID,
    maximum_age_seconds=MAX_TIMEOUT + 900,
    summary_maximum_age_seconds=_SUMMARY_TIMEOUT_SECONDS + 900,
    wait_timeout_seconds=max(60, MAX_TIMEOUT - 60),
)


def _coordinator_result(
    result: dict[str, object],
    execution_result_path: str | None,
) -> dict[str, object]:
    """Return direct output or publish a small coordinator result reference."""
    if execution_result_path is None:
        return result
    return {
        "execution_result": publish_execution_result(
            Path(CONF.output_volume_mountpoint),
            CONF.output_volume,
            execution_result_path,
            result,
        )
    }


@app.function(
    image=sharding_image,
    cpu=PROFILE_BUILD_CPU,
    memory=BUILD_MEMORY_MIB,
    timeout=ENVIRONMENT_SETUP_TIMEOUT_SECONDS,
    max_containers=PROFILE_BUILD_MAX_CONTAINERS,
    volumes={
        EnvironmentRuntime.MODEL_MOUNT: MODEL_VOLUME,
        EnvironmentRuntime.SOURCE_MOUNT: AF3_MSA_DB_VOLUME,
        ProfileBuilderRuntime.SHARDED_MOUNT: SHARDED_MSA_DB_VOLUME,
        CONF.output_volume_mountpoint: CONF.output_volume,
    },
)
def prepare_alphafold3_environment_asset(
    asset_record: dict[str, object],
    generation_id: str,
) -> dict[str, object]:
    """Prepare one model, database profile, or template-reference asset."""
    return prepare_environment_asset(
        _ENVIRONMENT_RUNTIME,
        EnvironmentAsset.from_record(asset_record),
        generation_id,
        build_profile=lambda database_id, selected_generation, source_path: (
            build_profile(
                _PROFILE_BUILDER_RUNTIME,
                database_id,
                DEFAULT_SEQKIT_THREADS,
                generation_id=selected_generation,
                source_path=source_path,
            )
        ),
    )


##########################################
# MSA search functions
##########################################
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
        CONF.output_volume_mountpoint: CONF.output_volume,
    },
)
def search_database_msa(
    database_id: str,
    sequence: str,
    generation_id: str | None = None,
    superseded_generation_ids: tuple[str, ...] = (),
    execution_result_path: str | None = None,
) -> dict[str, object]:
    """Search one fixed sharded database with database-level resume."""
    result = run_database_search(
        _msa_search_runtime(
            maximum_age_seconds=CONF.timeout + 900,
            wait_timeout_seconds=max(60, CONF.timeout - 60),
        ),
        database_id,
        sequence,
        generation_id=generation_id,
        superseded_generation_ids=superseded_generation_ids,
    )
    return _coordinator_result(result, execution_result_path)


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
        CONF.output_volume_mountpoint: CONF.output_volume,
    },
)
def assemble_sequence_msas(
    polymer: Polymer,
    sequence: str,
    include_unpaired: bool,
    include_paired: bool,
    generation_id: str | None = None,
    superseded_generation_ids: tuple[str, ...] = (),
    execution_result_path: str | None = None,
) -> dict[str, object]:
    """Assemble requested fields with pinned upstream deduplication."""
    if polymer not in {"protein", "rna"}:
        raise ValueError(f"Unsupported polymer: {polymer!r}")
    result = assemble_and_publish_msas(
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
        generation_id=generation_id,
        superseded_generation_ids=superseded_generation_ids,
    )
    return _coordinator_result(result, execution_result_path)


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
        CONF.output_volume_mountpoint: CONF.output_volume,
    },
)
def search_protein_templates(
    sequence: str,
    unpaired_msa: str | None,
    unpaired_msa_reference: dict[str, object] | None,
    publish_canonical: bool,
    max_template_date: str = DEFAULT_MAX_TEMPLATE_DATE,
    generation_id: str | None = None,
    superseded_generation_ids: tuple[str, ...] = (),
    execution_result_path: str | None = None,
) -> dict[str, object]:
    """Search templates from one resolved protein unpaired MSA."""
    result = run_template_search(
        TemplateRuntime(
            source_volume=AF3_MSA_DB_VOLUME,
            cache_volume=MSA_CACHE_VOLUME,
            claims=MSA_SEARCH_CLAIMS,
            container_id=_CONTAINER_INSTANCE_ID,
            maximum_age_seconds=CONF.timeout + 900,
            wait_timeout_seconds=max(60, CONF.timeout - 60),
        ),
        TemplateTask(
            sequence=sequence,
            unpaired_msa=unpaired_msa,
            unpaired_msa_reference=(
                MsaArtifactReference.from_record(
                    unpaired_msa_reference,
                    expected_path=(
                        sequence_cache_relpath("protein", sequence) / "unpaired.a3m"
                    ),
                )
                if unpaired_msa_reference is not None
                else None
            ),
            publish_canonical=publish_canonical,
            max_template_date=max_template_date,
        ),
        generation_id=generation_id,
        superseded_generation_ids=superseded_generation_ids,
    )
    return _coordinator_result(result, execution_result_path)


##########################################
# Inference functions
##########################################


def _load_staged_request(
    run_id: str,
    request_id: str,
    staged_input_record: dict[str, object],
) -> LoadedInferenceInput:
    """Reload and validate one marker-bound inference request."""
    CONF.output_volume.reload()
    return load_staged_inference_input(
        Path(CONF.output_volume_mountpoint),
        run_id=run_id,
        request_id=request_id,
        staged_input_record=staged_input_record,
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
    allow_large_inference: bool = False,
) -> list[dict[str, object]]:
    """Inspect seed markers without scanning prediction directories."""
    return inspect_seed_predictions(
        _INFERENCE_RUNTIME,
        run_id,
        tuple(seeds),
        sample_count=sample_count,
        allow_large_inference=allow_large_inference,
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
    allow_large_inference: bool = False,
) -> dict[str, object]:
    """Reuse or atomically claim one request's currently incomplete seeds."""
    return claim_seed_predictions(
        _INFERENCE_RUNTIME,
        run_id,
        tuple(seeds),
        sample_count=sample_count,
        allow_large_inference=allow_large_inference,
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
        model_mount_subdir=False,
    )
    | {
        JAX_CACHE_MOUNTPOINT: JAX_CACHE_VOLUME,
    },
)
def run_inference_pipeline(
    run_id: str,
    request_id: str,
    staged_input_record: dict[str, object],
    claimed_seed_records: list[dict[str, object]],
    allow_large_inference: bool = False,
    execution_result_path: str | None = None,
) -> dict[str, object]:
    """Run one disjoint seed group and publish per-seed markers."""
    with guard_seed_prediction_claims(
        _INFERENCE_RUNTIME,
        run_id,
        claimed_seed_records,
    ) as claimed_seeds:
        staged = _load_staged_request(run_id, request_id, staged_input_record)
        result = run_upstream_seed_worker(
            UpstreamInferenceRuntime(
                predictions=_INFERENCE_RUNTIME,
                source_root=CONF.git_clone_dir,
                model_root=Path(CONF.model_volume_mountpoint) / CONF.name,
                jax_cache_dir=Path(JAX_CACHE_MOUNTPOINT) / ALPHAFOLD3_COMMIT,
            ),
            staged.config,
            run_id,
            staged.recycle,
            staged.sample_count,
            claimed_seeds,
            allow_large_inference=allow_large_inference,
        )
        return _coordinator_result(result, execution_result_path)


@app.function(
    cpu=(0.125, 2.125),
    memory=(1024, 16384),
    timeout=_SUMMARY_TIMEOUT_SECONDS,
    volumes=CONF.mounts(output_volume=True),
)
def finalize_inference_summary(
    run_id: str,
    request_id: str,
    staged_input_record: dict[str, object],
    generation_id: str | None = None,
    superseded_generation_ids: tuple[str, ...] = (),
    execution_result_path: str | None = None,
) -> dict[str, object]:
    """Rebuild the non-regressing accumulated run summary."""
    staged = _load_staged_request(run_id, request_id, staged_input_record)
    result = finalize_upstream_run_summary(
        _INFERENCE_RUNTIME,
        staged.config,
        run_id,
        staged.sample_count,
        generation_id=generation_id,
        superseded_generation_ids=superseded_generation_ids,
    )
    return _coordinator_result(result, execution_result_path)


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
    execution_result_path: str | None = None,
) -> dict[str, object]:
    """Publish one manifest-last view over the request's completed seeds."""
    result = publish_request_results(
        _INFERENCE_RUNTIME,
        RequestPublication(
            run_id=run_id,
            request_id=request_id,
            submitted_seeds=tuple(submitted_seeds),
            normalized_seeds=tuple(normalized_seeds),
            sample_count=sample_count,
            display_name=display_name,
        ),
    )
    return _coordinator_result(result, execution_result_path)


##########################################
# Deployment-local execution coordinator
##########################################


@app.cls(
    cpu=(0.125, 4.125),
    memory=(256, 65536),
    timeout=MAX_TIMEOUT,
    max_containers=1,
    scaledown_window=COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
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
        TemplateRuntime.SOURCE_MOUNT: AF3_MSA_DB_VOLUME.with_mount_options(
            read_only=True,
            sub_path="/",
        ),
        EnvironmentRuntime.MODEL_MOUNT: MODEL_VOLUME.with_mount_options(
            read_only=True,
            sub_path="/",
        ),
        CONF.output_volume_mountpoint: CONF.output_volume,
    },
)
@modal.concurrent(max_inputs=_MAX_CONCURRENT_COORDINATOR_INPUTS)
class ExecutionCoordinator:
    """Run-scoped single writer deployed with AlphaFold3's worker functions."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()
    development: bool = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh output state before accepting lifecycle methods."""
        initialize_execution_coordinator_host(self)
        self._identity()
        CONF.output_volume.reload()

    @modal.method()
    def run(self, development: bool = False) -> ExecutionOverview:
        """Drive a staged root App Run until it stops."""
        return self._adapter(development=development).run()

    @modal.method()
    def status(self) -> ExecutionOverview:
        """Read this Run's durable kernel overview."""
        return self._adapter().status()

    @modal.method()
    def provider_calls(
        self,
        node_key: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        newest_first: bool = False,
    ) -> ProviderCallPage:
        """Read one bounded page of calls for service diagnostics."""
        return self._adapter().provider_calls(
            node_key=node_key,
            cursor=None if cursor is None else UUID(cursor),
            limit=limit,
            newest_first=newest_first,
        )

    @modal.method()
    def provider_call(self, provider_call_id: str) -> ProviderCallDiagnostic | None:
        """Read one call selected by service diagnostics."""
        return self._adapter().provider_call(UUID(provider_call_id))

    @modal.method()
    def cancel(self) -> ExecutionOverview:
        """Request idempotent cancellation for this Run."""
        return self._adapter().cancel()

    @modal.method()
    def resume(self) -> ExecutionOverview:
        """Resume this Run without retrying conclusive failures."""
        return self._adapter().resume()

    @modal.method()
    def prepare_restart(
        self,
        predecessor_execution_run_id: str,
        predecessor_deployment_environment: str,
        predecessor_deployment_name: str,
        predecessor_deployment_version: int,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
    ) -> None:
        """Persist a validated Successor request without driving it."""
        self._adapter().prepare_restart(
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
    def drive_prepared(self) -> ExecutionOverview:
        """Drive one previously prepared root or Successor Run."""
        return self._adapter().drive_prepared()

    @modal.method()
    def restart_from(
        self,
        predecessor_execution_run_id: str,
        candidate_request_bytes: bytes,
    ) -> ExecutionOverview:
        """Create a Successor Run while inferring predecessor identity."""
        adapter = self._adapter()
        adapter.prepare_restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            candidate_request=AlphaFold3ExecutionRequest.from_bytes(
                candidate_request_bytes
            ),
        )
        return adapter.drive_prepared()

    @modal.exit()
    def exit(self) -> None:
        """Close local state without cancelling attached child calls."""
        adapter = getattr(self, "_coordinator_adapter", None)
        if adapter is not None:
            adapter.close()

    def _identity(self) -> tuple[UUID, DeploymentIdentity]:
        return execution_coordinator_identity(self)

    def _adapter(
        self,
        *,
        development: bool | None = None,
    ) -> AlphaFold3ExecutionCoordinator:
        execution_run_id, deployment = self._identity()
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected_mode: AlphaFold3ExecutionCoordinator(
                execution_run_id=execution_run_id,
                deployment=deployment,
                volume_root=Path(CONF.output_volume_mountpoint),
                output_volume=CONF.output_volume,
                output_volume_name=CONF.output_volume_name,
                provider_driver=_coordinator_modal_driver(development=selected_mode),
                search_runtime=_msa_search_runtime(
                    maximum_age_seconds=CONF.timeout + 900,
                    wait_timeout_seconds=max(60, CONF.timeout - 60),
                ),
                template_runtime=TemplateRuntime(
                    source_volume=AF3_MSA_DB_VOLUME,
                    cache_volume=MSA_CACHE_VOLUME,
                    claims=MSA_SEARCH_CLAIMS,
                    container_id=_CONTAINER_INSTANCE_ID,
                    maximum_age_seconds=CONF.timeout + 900,
                    wait_timeout_seconds=max(60, CONF.timeout - 60),
                ),
                inference_runtime=_INFERENCE_RUNTIME,
                environment_runtime=_ENVIRONMENT_RUNTIME,
            ),
        )


def _coordinator_modal_driver(*, development: bool) -> ModalCallDriver:
    """Resolve exact deployed functions or current-source development handles."""
    if not development:
        return ModalCallDriver()
    return development_modal_call_driver(
        {
            "prepare_alphafold3_environment_asset": (
                prepare_alphafold3_environment_asset
            ),
            "search_database_msa": search_database_msa,
            "assemble_sequence_msas": assemble_sequence_msas,
            "search_protein_templates": search_protein_templates,
            "run_inference_pipeline": run_inference_pipeline,
            "finalize_inference_summary": finalize_inference_summary,
            "finalize_inference_request": finalize_inference_request,
        },
        workload_name="AlphaFold3",
    )


##########################################
# Local entrypoints
##########################################
@app.local_entrypoint()
def submit_alphafold3_task(
    input_json: str,
    out_dir: str | None = None,
    run_name: str | None = None,
    search_msa: bool = True,
    search_protein_templates: bool = True,
    max_containers: int | None = None,
    max_gpu_containers: int | None = None,
    allow_large_inference: bool = False,
    recycle: int = 10,
    sample: int = 5,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
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
        max_containers: Maximum active workload containers for this Run.
        max_gpu_containers: Maximum active GPU workload containers within the
            total container limit.
        allow_large_inference: Continue after warning when the request exceeds
            the default seed/sample prediction limit. Other limits still apply.
        recycle: Number of Pairformer recycles to use during inference.
        sample: Number of diffusion samples to generate per seed.
        use_deployed_coordinator: Target the exact deployed coordinator. The
            `biomodals app run` client supplies this for normal runs.
        deployment_environment: Modal Environment containing the coordinator.
        deployment_name: Exact deployed Modal app name.
        deployment_version: Exact numeric Modal deployment version.
        restart_from: Optional predecessor Execution Run ID for a Successor Run.

    """
    conf = materialize_local_input(input_json)
    if run_name is None:
        run_name = conf.name
    sanitize_af3_name(run_name)
    conf.name = run_name
    prediction_count = len(normalize_model_seeds(conf.modelSeeds)) * sample
    if allow_large_inference and prediction_count > MAX_SEED_SAMPLE_PAIRS:
        warnings.warn(
            "AlphaFold3 request will create "
            f"{prediction_count:,} seed/sample predictions, exceeding the "
            f"default limit of {MAX_SEED_SAMPLE_PAIRS:,}; continuing because "
            "--allow-large-inference was set.",
            stacklevel=2,
        )
    total_limit, gpu_limit = resolve_provider_call_limits(
        default_max_containers=4,
        default_max_gpu_containers=1,
        max_containers=max_containers,
        max_gpu_containers=max_gpu_containers,
    )
    request = AlphaFold3ExecutionRequest.prepare(
        conf,
        search_msa=search_msa,
        search_protein_templates=search_protein_templates,
        max_active_provider_calls=total_limit,
        max_active_gpu_provider_calls=gpu_limit,
        allow_large_inference=allow_large_inference,
        recycle=recycle,
        sample=sample,
    )
    execution_run_id = uuid.uuid4()
    deployment = DeploymentIdentity(
        deployment_environment,
        deployment_name,
        deployment_version,
    )
    predecessor_execution_run_id = None if restart_from is None else UUID(restart_from)
    stage_execution_request(CONF.output_volume, execution_run_id, request)
    submit_staged_execution_run(
        CONF.output_volume,
        execution_run_id=execution_run_id,
        deployment=deployment,
        predecessor_execution_run_id=predecessor_execution_run_id,
        use_deployed_coordinator=use_deployed_coordinator,
        local_coordinator=ExecutionCoordinator,
        workload_name=CONF.name,
        restart_kwargs={"candidate_request_bytes": request.to_bytes()},
    )
    manifest = load_invocation_manifest(
        CONF.output_volume,
        request.invocation,
    )
    if manifest is None:
        raise RuntimeError(
            "Successful AlphaFold3 Execution Run has no invocation manifest"
        )
    publication = request_publication_from_manifest(manifest)
    archive_path = create_request_archive(
        CONF.output_volume,
        manifest,
        output_dir=resolve_local_output_dir(out_dir),
        display_name=run_name,
    )
    print(
        f"🧬 {CONF.name} results saved to {archive_path}. Durable seed "
        f"predictions remain in {CONF.output_volume_name}:/"
        f"{PurePosixPath(publication.run_id[:2]) / publication.run_id}."
    )
