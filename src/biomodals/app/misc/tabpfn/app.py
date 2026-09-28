"""Native tabular regression with TabPFN: https://github.com/PriorLabs/TabPFN.

Fit a numeric target and predict another table in one invocation. A JSON feature
schema explicitly selects ordered numerical/categorical columns and an optional
identifier. Outputs are predictions.csv, never a reloadable fitted model.
Pinned foundation weights are provisioned in a tracked CPU stage; GPU inference
uses verified local weights with network downloads disabled. Check upstream
software and model licensing before deployment, including hosted-service terms.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import modal

from biomodals.app.config import AppConfig
from biomodals.app.misc.tabpfn.execution import (
    OPERATION,
    PREPARE_OPERATION,
    TableFile,
    TabPFNRequest,
    tabpfn_graph,
)
from biomodals.app.misc.tabpfn.models import (
    RUNTIME_IDENTITY,
    TABPFN_VERSION,
    provision_checkpoint,
)
from biomodals.app.misc.tabpfn.runtime import run_tabpfn
from biomodals.app.misc.tabpfn.tables import (
    MAX_TABLE_BYTES,
    TableSchema,
    read_regression_tables,
)
from biomodals.execution import DeploymentIdentity
from biomodals.execution.artifact_availability import (
    ArtifactAvailability,
    check_external_artifact_status,
)
from biomodals.execution.modal import orchestrator, resolve_provider_call_limits
from biomodals.helper import patch_image_for_helper
from biomodals.helper.artifacts import read_bounded_file_bytes, sha256_file
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.helper.modal_volume import download_modal_volume_files
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    ExecutionArtifact,
    InlineBytes,
    VolumePath,
)
from biomodals.workflow.display import print_workflow_dag

CONF = AppConfig(
    name="TabPFN",
    package_name="tabpfn",
    version=TABPFN_VERSION,
    python_version="3.12",
    cuda_version="cu130",
    gpu="L40S",
    timeout=12 * 3600,
    tags={"group": "misc"},
)
MODEL_ROOT = Path(CONF.model_volume_mountpoint) / "tabpfn"
base_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(
        CONF.default_env
        | {
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TABPFN_NO_BROWSER": "1",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "POLARS_MAX_THREADS": "4",
        }
    )
    .uv_pip_install(
        "tabpfn==9.0.0",
        "torch==2.11.0",
        "scikit-learn==1.9.1",
        "numpy==2.5.1",
        "scipy==1.18.1",
        "polars==1.44.2",
        "huggingface-hub==0.36.2",
        "safetensors==0.8.0",
    )
)
setup_image = base_image.pipe(patch_image_for_helper).add_local_python_source(
    "biomodals.app.misc.tabpfn"
)
image = (
    base_image
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    .pipe(patch_image_for_helper)
    .add_local_python_source("biomodals.app.misc.tabpfn")
)
app = modal.App(CONF.name, image=image, tags=CONF.tags).include(
    orchestrator.app, inherit_tags=True
)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_tabpfn_task"})


@app.function(
    image=setup_image,
    cpu=2,
    memory=4096,
    timeout=CONF.timeout,
    max_containers=1,
    volumes=CONF.mounts(model_volume=True, model_ro=False, model_mount_subdir=False),
)
def prepare_tabpfn_models(runtime_identity: str = RUNTIME_IDENTITY) -> AppRunResult:
    """Verify and commit immutable foundation weights without fitting user data."""
    if runtime_identity != RUNTIME_IDENTITY:
        raise ValueError("Target deployment changed TabPFN scientific versions")
    MODEL_VOLUME.reload()
    provision_checkpoint(MODEL_ROOT)
    MODEL_VOLUME.commit()
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="model_identity",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(
                    data=runtime_identity.encode(),
                    media_type="text/plain",
                    filename="model-identity.txt",
                ),
            )
        ],
    )


@app.function(
    gpu=CONF.gpu,
    cpu=4,
    memory=(8192, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(
        output_volume=True, model_volume=True, model_mount_subdir=False
    ),
)
def tabpfn_fit_predict(
    request_json: str, output_key: str, runtime_identity: str
) -> AppRunResult:
    """Native fit/predict inside the owning Run; no child scheduler or model export."""
    CONF.output_volume.reload()
    MODEL_VOLUME.reload()
    result = run_tabpfn(
        request_json,
        output_key=output_key,
        runtime_identity=runtime_identity,
        volume_root=Path(CONF.output_volume_mountpoint),
        volume_name=CONF.output_volume_name,
        model_root=MODEL_ROOT,
    )
    CONF.output_volume.commit()
    return result


@app.function(cpu=0.25, timeout=CONF.timeout, volumes=CONF.mounts(output_volume=True))
def check_tabpfn_artifact(artifact: ExecutionArtifact) -> ArtifactAvailability:
    """Reconcile app-owned immutable prediction files at the normal boundary."""
    CONF.output_volume.reload()
    return check_external_artifact_status(
        artifact,
        artifact_volume_name=orchestrator.OUT_VOLUME_NAME,
        volume_roots={CONF.output_volume_name: CONF.output_volume_mountpoint},
    )


@app.local_entrypoint()
def submit_tabpfn_task(
    training_csv: str,
    inference_csv: str,
    schema_json: str,
    output_csv: str = "predictions.csv",
    seed: int = 0,
    n_estimators: int = 8,
    batch_size: int = 256,
    dry_run: bool = False,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
    max_containers: int | None = None,
    max_gpu_containers: int | None = None,
) -> None:
    """Validate two tables, stage them, fit once and download native predictions.

    Args:
        training_csv: CSV containing declared features, target and optional ID.
        inference_csv: CSV containing the same features and optional ID, without target.
        schema_json: JSON with target, optional identifier and features name/kind list.
        output_csv: New prediction CSV destination; never overwrite an existing file.
        seed: Native reproducible random seed.
        n_estimators: Number of native foundation-model ensemble configurations.
        batch_size: Maximum inference rows per native predict call.
        dry_run: Validate input and print the graph without any remote calls.
        use_deployed_coordinator: Use an exact deployed app for durable execution.
        deployment_environment: Modal environment containing that app.
        deployment_name: Exact deployed app name.
        deployment_version: Exact numeric deployment version.
        restart_from: Compatible predecessor Run for explicit recovery.
        max_containers: Global total provider-call ceiling.
        max_gpu_containers: Global GPU provider-call subset ceiling.
    """
    from hashlib import sha256

    schema = TableSchema.model_validate_json(
        read_bounded_file_bytes(
            Path(schema_json).expanduser(),
            field_name="feature schema",
            max_bytes=4 * 1024 * 1024,
        )
    )
    train = read_bounded_file_bytes(
        Path(training_csv).expanduser(),
        field_name="training CSV",
        max_bytes=MAX_TABLE_BYTES,
    )
    infer = read_bounded_file_bytes(
        Path(inference_csv).expanduser(),
        field_name="inference CSV",
        max_bytes=MAX_TABLE_BYTES,
    )
    read_regression_tables(train, infer, schema)
    namespace = uuid4().hex
    request = TabPFNRequest(
        training=TableFile(
            path=f"inputs/{namespace}/training.csv",
            sha256=sha256(train).hexdigest(),
            size_bytes=len(train),
        ),
        inference=TableFile(
            path=f"inputs/{namespace}/inference.csv",
            sha256=sha256(infer).hexdigest(),
            size_bytes=len(infer),
        ),
        table_schema=schema,
        seed=seed,
        n_estimators=n_estimators,
        batch_size=batch_size,
    )
    graph = tabpfn_graph(request)
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
        default_max_gpu_containers=1,
    )
    with CONF.output_volume.batch_upload() as batch:
        batch.put_file(BytesIO(train), "/" + request.training.path)
        batch.put_file(BytesIO(infer), "/" + request.inference.path)
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
        "external_artifact_checker_function_name": "check_tabpfn_artifact",
    }
    if not use_deployed_coordinator:
        arguments["development_function_handles"] = {
            OPERATION: tabpfn_fit_predict,
            PREPARE_OPERATION: prepare_tabpfn_models,
            "check_tabpfn_artifact": check_tabpfn_artifact,
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
        raise RuntimeError(f"TabPFN finished with {result.status}")
    output = next(output for output in result.outputs if output.name == "predictions")
    if (
        not isinstance(output.storage, VolumePath)
        or output.storage.volume_name != CONF.output_volume_name
    ):
        raise ValueError("Unexpected TabPFN result storage")
    download_modal_volume_files(
        CONF.output_volume, [(output.storage.path, destination)], concurrency=1
    )
    record = output.metadata["files"][0]
    if (
        destination.stat().st_size != record["size_bytes"]
        or sha256_file(destination) != record["content_sha256"]
    ):
        raise ValueError("Downloaded predictions failed integrity validation")
    print(f"🧬 Predictions saved to {destination}; runtime {RUNTIME_IDENTITY}")
