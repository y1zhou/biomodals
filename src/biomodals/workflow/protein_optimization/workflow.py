"""Optimize arbitrary protein chains with additive combinations or ESMC/TabPFN.

Combination enumerates experimentally supported substitutions without GPU work.
Exploration samples novel substitutions, encodes full independent chains with
ESMC600M and fits TabPFN. Both publish only novel scored candidates as CSV.
Foundation assets are prepared by tracked CPU work before GPU tasks. Native
installation, checkpoint access, license suitability and GPU resource limits
must be verified before production rollout; offline checks do not establish them.
"""

from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import modal

from biomodals.app.config import AppConfig
from biomodals.app.design.mutation_ridge import app as ridge_app
from biomodals.app.design.mutation_ridge.inputs import MAX_INPUT_BYTES
from biomodals.app.misc.tabpfn import app as tabpfn_app
from biomodals.app.misc.tabpfn.models import provision_checkpoint
from biomodals.execution import (
    COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    DeploymentIdentity,
    ExecutionOverview,
    ProviderCallDiagnostic,
    ProviderCallPage,
)
from biomodals.execution.artifact_availability import check_external_artifact_status
from biomodals.execution.modal import (
    ModalCallDriver,
    development_modal_call_driver,
    execution_coordinator_adapter,
    execution_coordinator_handle,
    execution_coordinator_identity,
    initialize_execution_coordinator_host,
    orchestrator,
    resolve_provider_call_limits,
    stage_execution_launch,
)
from biomodals.helper import patch_image_for_helper
from biomodals.helper.artifacts import read_bounded_file_bytes
from biomodals.helper.catalog import include_dependency_apps
from biomodals.helper.constant import MODEL_VOLUME
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    InlineBytes,
    VolumePath,
)
from biomodals.workflow.display import print_workflow_dag
from biomodals.workflow.protein_optimization.design import (
    SCIENTIFIC_VERSIONS,
    OptimizationDesign,
)
from biomodals.workflow.protein_optimization.execution import (
    REQUEST_FILE,
    OptimizationExecutionCoordinator,
    OptimizationExecutionRequest,
    load_execution_request,
)
from biomodals.workflow.protein_optimization.feature_stage import feature_tables
from biomodals.workflow.protein_optimization.features import provision_encoder
from biomodals.workflow.protein_optimization.nodes import (
    FEATURE_OPERATION,
    PREPARE_OPERATION,
    optimization_graph,
)
from biomodals.workflow.protein_optimization.settings import (
    OptimizationSettings,
    mode_defaults,
)

CONF = AppConfig(
    name="ProteinOptimizationWorkflow",
    package_name="biomodals-protein-optimization",
    version="1.0.0",
    python_version="3.12",
    cuda_version="cu130",
    gpu="L40S",
    timeout=12 * 3600,
    depends_on_apps=("mutation_ridge", "tabpfn"),
    tags={
        "depends_on": "mutation_ridge-tabpfn",
        "biomodals_tool": "protein_optimization",
    },
)
OUT_VOLUME = orchestrator.OUT_VOLUME
OUT_VOLUME_NAME = orchestrator.OUT_VOLUME_NAME
OUT_VOLUME_MOUNTPOINT = orchestrator.CONF.output_volume_mountpoint
ENCODER_ROOT = Path(CONF.model_volume_mountpoint) / "esmc600m"
runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .uv_pip_install("numpy==2.5.1", "scipy==1.18.1", "polars==1.44.2")
    .pipe(patch_image_for_helper, include_workflow_modules=True)
)
feature_base = tabpfn_app.base_image.uv_pip_install(
    "esm==3.4.1.post1",
    "transformers==4.57.6",
    "tokenizers==0.22.2",
    "einops==0.8.2",
)
setup_image = feature_base.pipe(patch_image_for_helper).add_local_python_source(
    "biomodals.workflow.protein_optimization",
    "biomodals.app.design.mutation_ridge",
    "biomodals.app.misc.tabpfn",
)
feature_image = (
    feature_base
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    .pipe(patch_image_for_helper)
    .add_local_python_source(
        "biomodals.workflow.protein_optimization",
        "biomodals.app.design.mutation_ridge",
        "biomodals.app.misc.tabpfn",
    )
)
app = include_dependency_apps(
    modal.App(CONF.name, image=runtime_image, tags=CONF.tags), CONF.depends_on_apps
)
EXECUTION_COORDINATOR_ENTRYPOINTS = frozenset({"submit_protein_optimization_workflow"})


@app.function(
    image=setup_image,
    cpu=2,
    memory=4096,
    timeout=CONF.timeout,
    max_containers=1,
    volumes=CONF.mounts(model_volume=True, model_ro=False, model_mount_subdir=False),
)
def prepare_protein_optimization_models(
    scientific_versions: dict[str, str],
) -> AppRunResult:
    """Provision and checksum immutable assets on CPU under normal Run ownership."""
    expected = {
        name: version
        for name, version in SCIENTIFIC_VERSIONS.items()
        if name != "ridge"
    }
    if scientific_versions != expected:
        raise ValueError("Target deployment changed Exploration scientific versions")
    MODEL_VOLUME.reload()
    provision_checkpoint(tabpfn_app.MODEL_ROOT)
    provision_encoder(ENCODER_ROOT)
    MODEL_VOLUME.commit()
    return AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="prepared_models",
                kind=ArtifactKind.REPORT,
                storage=InlineBytes(
                    data=b"Pinned ESMC600M and TabPFN assets verified",
                    filename="prepared-models.txt",
                    media_type="text/plain",
                ),
            )
        ],
    )


@app.function(
    image=feature_image,
    gpu=CONF.gpu,
    cpu=4,
    memory=(8192, 65536),
    timeout=CONF.timeout,
    volumes=CONF.mounts(model_volume=True, model_mount_subdir=False)
    | tabpfn_app.CONF.mounts(output_volume=True),
)
def extract_protein_optimization_features(
    design_json: str, output_key: str, scientific_versions: dict[str, str]
) -> AppRunResult:
    """Publish frozen feature inputs directly in the included table app's Volume."""
    MODEL_VOLUME.reload()
    tabpfn_app.CONF.output_volume.reload()
    result = feature_tables(
        design_json,
        output_key=output_key,
        scientific_versions=scientific_versions,
        volume_root=Path(tabpfn_app.CONF.output_volume_mountpoint),
        volume_name=tabpfn_app.CONF.output_volume_name,
        model_root=ENCODER_ROOT,
    )
    tabpfn_app.CONF.output_volume.commit()
    return result


def _check_external(artifact):
    roots = {}
    for config in (ridge_app.CONF, tabpfn_app.CONF):
        if artifact.storage.volume_name == config.output_volume_name:
            config.output_volume.reload()
            roots[config.output_volume_name] = config.output_volume_mountpoint
    return check_external_artifact_status(
        artifact, artifact_volume_name=OUT_VOLUME_NAME, volume_roots=roots
    )


@app.cls(
    cpu=(0.125, 4),
    memory=(512, 16384),
    timeout=CONF.timeout,
    max_containers=1,
    scaledown_window=COORDINATOR_SCALEDOWN_WINDOW_SECONDS,
    volumes={OUT_VOLUME_MOUNTPOINT: OUT_VOLUME}
    | ridge_app.CONF.mounts(output_volume=True)
    | tabpfn_app.CONF.mounts(output_volume=True),
)
@modal.concurrent(max_inputs=8)
class ExecutionCoordinator:
    """The established run-scoped lifecycle, with no workflow-local scheduler."""

    execution_run_id: str = modal.parameter()
    deployment_environment: str = modal.parameter()
    deployment_name: str = modal.parameter()
    deployment_version: int = modal.parameter()
    development: bool = modal.parameter()

    @modal.enter()
    def enter(self) -> None:
        """Refresh staged requests before creating the shared single writer."""
        initialize_execution_coordinator_host(self)
        execution_coordinator_identity(self)
        OUT_VOLUME.reload()

    def _adapter(
        self, *, development: bool | None = None
    ) -> OptimizationExecutionCoordinator:
        run_id, deployment = execution_coordinator_identity(self)
        return execution_coordinator_adapter(
            self,
            development=development,
            factory=lambda selected: OptimizationExecutionCoordinator(
                execution_run_id=run_id,
                deployment=deployment,
                volume_root=OUT_VOLUME_MOUNTPOINT,
                output_volume=OUT_VOLUME,
                output_volume_name=OUT_VOLUME_NAME,
                provider_driver=development_modal_call_driver(
                    {
                        "mutation_ridge_score": ridge_app.mutation_ridge_score,
                        "tabpfn_fit_predict": tabpfn_app.tabpfn_fit_predict,
                        FEATURE_OPERATION: extract_protein_optimization_features,
                        PREPARE_OPERATION: prepare_protein_optimization_models,
                    },
                    workload_name=CONF.name,
                )
                if selected
                else ModalCallDriver(),
                external_checker=_check_external,
            ),
        )

    @modal.method()
    def run(self, development: bool = False) -> ExecutionOverview:
        """Drive one immutable scientific Run."""
        return self._adapter(development=development).run()

    @modal.method()
    def resume(self) -> ExecutionOverview:
        """Reconcile existing ownership without retrying conclusive failures."""
        return self._adapter().resume()

    @modal.method()
    def status(self) -> ExecutionOverview:
        """Read the durable shared execution projection."""
        return self._adapter().status()

    @modal.method()
    def cancel(self) -> ExecutionOverview:
        """Cancel only this Run's owned provider calls."""
        return self._adapter().cancel()

    @modal.method()
    def result(self) -> AppRunResult:
        """Return the verified terminal CSV directory reference."""
        return self._adapter().result()

    @modal.method()
    def provider_calls(
        self,
        node_key: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        newest_first: bool = False,
    ) -> ProviderCallPage:
        """Page shared call/log identities without recreating timing logic."""
        return self._adapter().provider_calls(
            node_key=node_key,
            cursor=UUID(cursor) if cursor else None,
            limit=limit,
            newest_first=newest_first,
        )

    @modal.method()
    def provider_call(self, provider_call_id: str) -> ProviderCallDiagnostic | None:
        """Inspect one owned provider call."""
        return self._adapter().provider_call(UUID(provider_call_id))

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
        """Prepare an explicit compatible successor through the shared host."""
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
        """Drive a prepared successor, retaining exact prior call ownership."""
        return self._adapter().drive_prepared()

    @modal.method()
    def restart_from(self, predecessor_execution_run_id: str) -> ExecutionOverview:
        """Verify a CLI successor's supplied design matches the predecessor."""
        adapter = self._adapter()
        adapter.prepare_restart(
            predecessor_execution_run_id=UUID(predecessor_execution_run_id),
            predecessor_deployment=None,
            candidate_request=load_execution_request(
                OUT_VOLUME_MOUNTPOINT, UUID(self.execution_run_id)
            ),
        )
        return adapter.drive_prepared()

    @modal.exit()
    def exit(self) -> None:
        """Checkpoint best-effort without cancelling children on preemption."""
        adapter = getattr(self, "_coordinator_adapter", None)
        if adapter is not None:
            adapter.close()


@app.local_entrypoint()
def submit_protein_optimization_workflow(
    input_csv: str,
    parental_fasta: str | None = None,
    mode: str = "combination",
    settings_json: str | None = None,
    run_id: str | None = None,
    wait: bool = True,
    dry_run: bool = False,
    max_containers: int | None = None,
    max_gpu_containers: int | None = None,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str = CONF.name,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Review local inputs, run one selected mode and report the candidate CSV.

    Args:
        input_csv: mutations,label CSV with optional id and quoted mutation lists.
        parental_fasta: Full chain-ID FASTA, required only for Exploration.
        mode: combination or exploration, using that mode's complete defaults.
        settings_json: Optional complete settings JSON; its mode must match --mode.
        run_id: Display label, defaulting to the input CSV stem.
        wait: Wait for completion and report final volume-backed CSV location.
        dry_run: Validate and print the graph without remote calls or model downloads.
        max_containers: Global total provider-call ceiling.
        max_gpu_containers: Global GPU subset ceiling; Combination uses no GPU calls.
        use_deployed_coordinator: Use an exact deployed workflow for durable execution.
        deployment_environment: Modal environment containing the pinned workflow.
        deployment_name: Containing workflow deployment name.
        deployment_version: Exact numeric deployment version.
        restart_from: Compatible predecessor Run UUID for explicit recovery.
    """
    if mode not in mode_defaults():
        raise ValueError("Choose combination or exploration")
    settings = (
        mode_defaults()[mode]
        if settings_json is None
        else OptimizationSettings.model_validate_json(
            read_bounded_file_bytes(
                Path(settings_json).expanduser(),
                field_name="settings",
                max_bytes=1024 * 1024,
            )
        )
    )
    if settings.mode != mode:
        raise ValueError("Settings mode must match the requested mode")
    design = OptimizationDesign(
        measurements_csv=read_bounded_file_bytes(
            Path(input_csv).expanduser(),
            field_name="measurements",
            max_bytes=MAX_INPUT_BYTES,
        ).decode(),
        parental_fasta=(
            read_bounded_file_bytes(
                Path(parental_fasta).expanduser(),
                field_name="parental FASTA",
                max_bytes=MAX_INPUT_BYTES,
            ).decode()
            if parental_fasta is not None
            else None
        ),
        settings=settings,
    )
    if dry_run:
        print_workflow_dag(optimization_graph(design).validate())
        return
    total, gpu = resolve_provider_call_limits(
        max_containers=max_containers,
        max_gpu_containers=max_gpu_containers,
        default_max_containers=8,
        default_max_gpu_containers=1 if mode == "exploration" else 0,
    )
    predecessor = UUID(restart_from) if restart_from else None
    if predecessor is not None and not use_deployed_coordinator:
        raise ValueError("Restart requires a pinned deployed workflow")
    deployment = DeploymentIdentity(
        deployment_environment if use_deployed_coordinator else "development",
        deployment_name,
        deployment_version if use_deployed_coordinator else 1,
    )
    execution_run_id = uuid4()
    request = OptimizationExecutionRequest(
        run_name=run_id or Path(input_csv).stem,
        design=design,
        max_active_provider_calls=total,
        max_active_gpu_provider_calls=gpu,
    )
    REQUEST_FILE.stage(OUT_VOLUME, execution_run_id, request.to_bytes())
    stage_execution_launch(OUT_VOLUME, execution_run_id, predecessor)
    coordinator = execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
        local_coordinator=cast(Any, ExecutionCoordinator),
    )
    print(
        f"Deployment Identity: {deployment.environment}/{deployment.deployment_name}/v{deployment.deployment_version}\nExecution Run ID: {execution_run_id}",
        flush=True,
    )
    call = (
        coordinator.run.spawn(development=not use_deployed_coordinator)
        if predecessor is None
        else coordinator.restart_from.spawn(
            predecessor_execution_run_id=str(predecessor)
        )
    )
    print(f"Coordinator FunctionCall ID: {call.object_id}", flush=True)
    if wait:
        overview = call.get()
        print(f"Protein optimization: {overview.run.status.value}", flush=True)
        if overview.run.status.value == "succeeded":
            result = AppRunResult.model_validate(coordinator.result.remote())
            for output in result.outputs:
                if output.name == "optimization_results" and isinstance(
                    output.storage, VolumePath
                ):
                    print(
                        f"Candidates CSV: volume={output.storage.volume_name} path={output.storage.path}/candidates.csv",
                        flush=True,
                    )
