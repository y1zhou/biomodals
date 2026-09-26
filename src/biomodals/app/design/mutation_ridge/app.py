"""Additive mutation ridge: https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression.

Fit a sparse additive regressor to measurements and score every compatible,
unmeasured combination through a mutation-count bound. Inputs are mutations,label
CSV and chain-ID parental FASTA; positions are one-based raw coordinates. Results
are candidates.csv, not a serialized fitted model. Ridge does not model epistasis.
No pretrained weights or GPU are required. Deploy for recoverable CLI execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import modal

from biomodals.app.config import AppConfig
from biomodals.app.design.mutation_ridge.execution import (
    OPERATION,
    RUNTIME_IDENTITY,
    RidgeRequest,
    ridge_graph,
)
from biomodals.app.design.mutation_ridge.inputs import MAX_INPUT_BYTES
from biomodals.app.design.mutation_ridge.runtime import run_ridge
from biomodals.execution import DeploymentIdentity
from biomodals.execution.artifact_availability import (
    ArtifactAvailability,
    check_external_artifact_status,
)
from biomodals.execution.modal import orchestrator, resolve_provider_call_limits
from biomodals.helper import patch_image_for_helper
from biomodals.helper.artifacts import read_bounded_file_bytes, sha256_file
from biomodals.helper.modal_volume import download_modal_volume_files
from biomodals.schema import AppRunResult, AppRunStatus, ExecutionArtifact, VolumePath
from biomodals.workflow.display import print_workflow_dag

CONF = AppConfig(
    name="MutationRidge",
    package_name="scikit-learn",
    version="1.9.1",
    python_version="3.12",
    timeout=3600,
    tags={"group": "design"},
)
image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(
        CONF.default_env
        | {
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "POLARS_MAX_THREADS": "2",
        }
    )
    .uv_pip_install(
        "scikit-learn==1.9.1", "numpy==2.5.1", "scipy==1.18.1", "polars==1.44.2"
    )
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.app.design.mutation_ridge")
)
app = modal.App(CONF.name, image=image, tags=CONF.tags).include(
    orchestrator.app, inherit_tags=True
)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_mutation_ridge_task"})


@app.function(
    cpu=2,
    memory=(2048, 16384),
    timeout=CONF.timeout,
    volumes=CONF.mounts(output_volume=True),
)
def mutation_ridge_score(
    request_json: str, output_key: str, runtime_identity: str
) -> AppRunResult:
    """Execute one parent-owned task and commit its complete content-bound CSV."""
    CONF.output_volume.reload()
    result = run_ridge(
        request_json,
        output_key=output_key,
        runtime_identity=runtime_identity,
        volume_root=Path(CONF.output_volume_mountpoint),
        volume_name=CONF.output_volume_name,
    )
    CONF.output_volume.commit()
    return result


@app.function(cpu=0.25, timeout=CONF.timeout, volumes=CONF.mounts(output_volume=True))
def check_mutation_ridge_artifact(artifact: ExecutionArtifact) -> ArtifactAvailability:
    """Inspect app-owned files before reusing a recorded publication."""
    CONF.output_volume.reload()
    return check_external_artifact_status(
        artifact,
        artifact_volume_name=orchestrator.OUT_VOLUME_NAME,
        volume_roots={CONF.output_volume_name: CONF.output_volume_mountpoint},
    )


@app.local_entrypoint()
def submit_mutation_ridge_task(
    input_csv: str,
    parental_fasta: str,
    output_csv: str = "candidates.csv",
    max_mutations: int = 2,
    candidate_budget: int = 1_000_000,
    alpha: float = 1.0,
    higher_is_better: bool = True,
    seed: int = 0,
    dry_run: bool = False,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
    max_containers: int | None = None,
    max_gpu_containers: int | None = None,
) -> None:
    """Fit measurements and download a complete, prediction-sorted candidate CSV.

    Args:
        input_csv: Measurement table with mutations,label and optional id columns.
        parental_fasta: Multi-record FASTA with mutation-table chain IDs.
        output_csv: New local candidate CSV; existing files are never overwritten.
        max_mutations: Maximum total parent-relative substitutions in candidates.
        candidate_budget: Exhaustive admission ceiling; never silently sampled.
        alpha: Positive ridge penalty on unscaled binary substitution features.
        higher_is_better: Sort descending if true, ascending otherwise.
        seed: Reproducible support-aware validation split seed.
        dry_run: Validate data and show the execution graph without remote calls.
        use_deployed_coordinator: Use the exact pinned deployment for durable runs.
        deployment_environment: Modal environment containing that deployment.
        deployment_name: Exact deployed app name.
        deployment_version: Exact numeric deployment version.
        restart_from: Compatible predecessor execution Run for explicit recovery.
        max_containers: Global total provider-call limit.
        max_gpu_containers: Global GPU subset limit; this CPU app uses none.
    """
    content = read_bounded_file_bytes(
        Path(input_csv).expanduser(),
        field_name="measurements",
        max_bytes=MAX_INPUT_BYTES,
    )
    fasta = read_bounded_file_bytes(
        Path(parental_fasta).expanduser(),
        field_name="parental FASTA",
        max_bytes=MAX_INPUT_BYTES,
    )
    request = RidgeRequest(
        measurements_csv=content.decode("utf-8"),
        parental_fasta=fasta.decode("utf-8"),
        max_mutations=max_mutations,
        candidate_budget=candidate_budget,
        alpha=alpha,
        higher_is_better=higher_is_better,
        seed=seed,
    )
    graph = ridge_graph(request)
    if dry_run:
        print_workflow_dag(graph.validate())
        return
    destination = Path(output_csv).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    total, gpu = resolve_provider_call_limits(
        max_containers=max_containers,
        max_gpu_containers=max_gpu_containers,
        default_max_containers=1,
        default_max_gpu_containers=0,
    )
    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        deployment_environment, deployment_name, deployment_version
    )
    coordinator = orchestrator.execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
    )
    arguments: dict[str, Any] = {
        "graph": graph,
        "workload_run_key": str(execution_run_id),
        "max_active_provider_calls": total,
        "max_active_gpu_provider_calls": gpu,
        "max_parallel_nodes": 1,
        "strict_external_artifact_checks": True,
        "external_artifact_checker_function_name": "check_mutation_ridge_artifact",
    }
    if not use_deployed_coordinator:
        arguments["development_function_handles"] = {
            OPERATION: mutation_ridge_score,
            "check_mutation_ridge_artifact": check_mutation_ridge_artifact,
        }
    call = orchestrator.submit_workflow_run(
        coordinator,
        execution_run_id=execution_run_id,
        deployment=deployment,
        predecessor_execution_run_id=None
        if restart_from is None
        else UUID(restart_from),
        coordinator_kwargs=arguments,
    )
    result = AppRunResult.model_validate(call.get())
    if result.status != AppRunStatus.SUCCEEDED:
        raise RuntimeError(f"Mutation ridge finished with {result.status}")
    output = next(output for output in result.outputs if output.name == "candidates")
    if (
        not isinstance(output.storage, VolumePath)
        or output.storage.volume_name != CONF.output_volume_name
    ):
        raise ValueError("Unexpected candidate result storage")
    download_modal_volume_files(
        CONF.output_volume, [(output.storage.path, destination)], concurrency=1
    )
    record = output.metadata["files"][0]
    if (
        destination.stat().st_size != record["size_bytes"]
        or sha256_file(destination) != record["content_sha256"]
    ):
        raise ValueError("Downloaded candidate CSV failed integrity verification")
    print(f"🧬 Candidates saved to {destination}; runtime {RUNTIME_IDENTITY}")
