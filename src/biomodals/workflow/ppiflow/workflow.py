"""PPIFlow composition root built on the reusable workflow runtime."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID, uuid4

import modal
import orjson
import polars as pl
import yaml

from biomodals.app.bioinfo import rosetta_app
from biomodals.app.bioinfo.rosetta.execution_contracts import (
    RosettaTaskSpec,
    validate_task_publication_from_volume,
)
from biomodals.app.design import ligandmpnn_app, ppiflow_app
from biomodals.app.fold import alphafold3_app, flowpacker_app
from biomodals.app.fold.alphafold3.inference_inputs import (
    DECLARED_MODEL_IDENTITY,
)
from biomodals.app.score import af3score_app, dockq_app
from biomodals.execution import (
    AvailabilityStatus,
    CoordinatorNode,
    DeploymentIdentity,
    ExecutionGraph,
    NodeAggregationPolicy,
    NodeRunContext,
    ProviderCallSpec,
    ProviderNode,
    PullTaskProviderNode,
    PullWorkerCallSpec,
    TaskDefinition,
    TaskProviderNode,
    republish_execution_artifact,
)
from biomodals.execution.definition_plan import app_scientific_version
from biomodals.execution.modal import orchestrator, resolve_provider_call_limits
from biomodals.helper import patch_image_for_helper
from biomodals.helper.app_run import (
    volume_path_from_mount_path,
)
from biomodals.helper.catalog import include_dependency_apps
from biomodals.helper.constant import MAX_TIMEOUT
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppConfig,
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    ExecutionArtifact,
    InlineBytes,
    VolumePath,
)
from biomodals.workflow.display import print_workflow_dag
from biomodals.workflow.ppiflow import analysis_runtime, rosetta_runtime
from biomodals.workflow.ppiflow import manifests as ppiflow_manifests
from biomodals.workflow.ppiflow import staging as ppiflow_staging
from biomodals.workflow.ppiflow import tables as ppiflow_tables
from biomodals.workflow.ppiflow.af3score_runtime import (
    postprocess_ppiflow_af3score_stage as _postprocess_ppiflow_af3score_stage,
)
from biomodals.workflow.ppiflow.af3score_runtime import (
    prepare_ppiflow_af3score_stage as _prepare_ppiflow_af3score_stage,
)
from biomodals.workflow.ppiflow.af3score_runtime import (
    run_ppiflow_af3score_batch as _run_ppiflow_af3score_batch,
)
from biomodals.workflow.ppiflow.dockq_runtime import (
    run_ppiflow_dockq_stage as _run_ppiflow_dockq_stage,
)
from biomodals.workflow.ppiflow.flowpacker_runtime import (
    run_ppiflow_flowpacker_stage as _run_ppiflow_flowpacker_stage,
)
from biomodals.workflow.ppiflow.ligandmpnn_runtime import (
    run_ppiflow_ligandmpnn_candidate as _run_ppiflow_ligandmpnn_candidate,
)
from biomodals.workflow.ppiflow.model_validation import (
    validate_ppiflow_model as _validate_ppiflow_model,
)
from biomodals.workflow.ppiflow.ppiflow_runtime import (
    run_ppiflow_design_stage as _run_ppiflow_design_stage,
)
from biomodals.workflow.ppiflow.ppiflow_runtime import (
    run_ppiflow_partial_candidate as _run_ppiflow_partial_candidate,
)
from biomodals.workflow.ppiflow.ppiflow_runtime import (
    validated_partial_args as _validated_ppiflow_partial_args,
)
from biomodals.workflow.ppiflow.refold_runtime import (
    run_ppiflow_refold_candidate as _run_ppiflow_refold_candidate,
)
from biomodals.workflow.ppiflow.runtime_support import (
    file_sha256 as _file_sha256,
)
from biomodals.workflow.ppiflow.runtime_support import (
    optional_config_int as _optional_config_int,
)
from biomodals.workflow.ppiflow.runtime_support import (
    patterns_from_config as _patterns_from_config,
)
from biomodals.workflow.ppiflow.runtime_support import (
    result_with_output_kind as _result_with_output_kind,
)

PPI_FLOW_OUTPUT_STRUCTURE_PATTERNS = (
    "outputs/*.pdb",
    "outputs/**/*.pdb",
    "outputs/*.cif",
    "outputs/**/*.cif",
)
_SCIENTIFIC_SCHEMA_VERSION = "3"

DEPENDENCY_APPS = (
    "ppiflow",
    "rosetta",
    "flowpacker",
    "ligandmpnn",
    "dockq",
    "af3score",
    "alphafold3",
)
CONF = AppConfig(
    tags={"depends_on": "-".join(DEPENDENCY_APPS)},
    depends_on_apps=DEPENDENCY_APPS,
    name="PPIFlowWorkflow",
    package_name="biomodals-ppiflow-workflow",
    version="0.1.0",
    python_version="3.13",
    timeout=int(os.environ.get("TIMEOUT", str(MAX_TIMEOUT))),
)

runtime_image = (
    modal.Image
    .debian_slim(python_version=CONF.python_version)
    .env(CONF.default_env)
    .pipe(patch_image_for_helper, include_workflow_modules=True)
    .add_local_python_source(
        "biomodals.app.bioinfo.rosetta",
        "biomodals.workflow.ppiflow",
    )
)
ppiflow_task_image = ppiflow_app.runtime_image.add_local_python_source(
    "biomodals.app.design.ppiflow_app",
    "biomodals.workflow.ppiflow",
)
ligandmpnn_task_image = ligandmpnn_app.runtime_image.uv_pip_install(
    "polars==1.42.0"
).add_local_python_source(
    "biomodals.app.design.ligandmpnn_app",
    "biomodals.workflow.ppiflow",
)
flowpacker_task_image = flowpacker_app.runtime_image.add_local_python_source(
    "biomodals.app.fold.flowpacker_app",
    "biomodals.workflow.ppiflow",
)
rosetta_task_image = rosetta_app.runtime_image.add_local_python_source(
    "biomodals.workflow.ppiflow"
)
dockq_task_image = dockq_app.runtime_image.add_local_python_source(
    "biomodals.app.score.dockq_app",
    "biomodals.workflow.ppiflow",
)
af3score_task_image = af3score_app.runtime_image.add_local_python_source(
    "biomodals.app.fold.alphafold3.inference_inputs",
    "biomodals.app.fold.alphafold3.profiles",
    "biomodals.app.score.af3score_app",
    "biomodals.app.score.af3score_execution",
    "biomodals.app.score.af3score_publications",
    "biomodals.workflow.ppiflow",
)
alphafold3_task_image = alphafold3_app.runtime_image.add_local_python_source(
    "biomodals.app.fold.alphafold3",
    "biomodals.app.fold.alphafold3_app",
    "biomodals.workflow.ppiflow",
)
app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags).include(
    orchestrator.app, inherit_tags=True
)
app = include_dependency_apps(app, CONF.depends_on_apps)
validate_ppiflow_model = app.function(
    image=runtime_image,
    cpu=1.0,
    memory=1024,
    timeout=CONF.timeout,
    volumes=ppiflow_app.CONF.mounts(model_volume=True),
)(_validate_ppiflow_model)
PPI_FLOW_OUTPUT_VOLUME = ppiflow_app.CONF.output_volume
PPI_FLOW_OUTPUT_VOLUME_NAME = ppiflow_app.CONF.output_volume_name
PPI_FLOW_OUTPUT_MOUNTPOINT = ppiflow_app.CONF.output_volume_mountpoint
FLOWPACKER_OUTPUT_VOLUME = flowpacker_app.CONF.output_volume
FLOWPACKER_OUTPUT_VOLUME_NAME = flowpacker_app.CONF.output_volume_name
FLOWPACKER_OUTPUT_MOUNTPOINT = flowpacker_app.CONF.output_volume_mountpoint
AF3SCORE_OUTPUT_VOLUME = af3score_app.CONF.output_volume
AF3SCORE_OUTPUT_VOLUME_NAME = af3score_app.CONF.output_volume_name
AF3SCORE_OUTPUT_MOUNTPOINT = af3score_app.CONF.output_volume_mountpoint
ALPHAFOLD3_OUTPUT_VOLUME = alphafold3_app.CONF.output_volume
ALPHAFOLD3_OUTPUT_VOLUME_NAME = alphafold3_app.CONF.output_volume_name
ALPHAFOLD3_OUTPUT_MOUNTPOINT = alphafold3_app.CONF.output_volume_mountpoint
ROSETTA_OUTPUT_VOLUME = rosetta_app.CONF.output_volume
ROSETTA_OUTPUT_VOLUME_NAME = rosetta_app.CONF.output_volume_name
ROSETTA_OUTPUT_MOUNTPOINT = rosetta_app.CONF.output_volume_mountpoint
WORKFLOW_OUTPUT_VOLUME = orchestrator.OUT_VOLUME
WORKFLOW_OUTPUT_VOLUME_NAME = orchestrator.OUT_VOLUME_NAME
WORKFLOW_OUTPUT_MOUNTPOINT = orchestrator.CONF.output_volume_mountpoint
PPI_FLOW_SOURCE_VOLUME_ROOTS = {
    PPI_FLOW_OUTPUT_VOLUME_NAME: PPI_FLOW_OUTPUT_MOUNTPOINT,
    FLOWPACKER_OUTPUT_VOLUME_NAME: FLOWPACKER_OUTPUT_MOUNTPOINT,
    AF3SCORE_OUTPUT_VOLUME_NAME: AF3SCORE_OUTPUT_MOUNTPOINT,
    ALPHAFOLD3_OUTPUT_VOLUME_NAME: ALPHAFOLD3_OUTPUT_MOUNTPOINT,
    ROSETTA_OUTPUT_VOLUME_NAME: ROSETTA_OUTPUT_MOUNTPOINT,
    WORKFLOW_OUTPUT_VOLUME_NAME: WORKFLOW_OUTPUT_MOUNTPOINT,
}
PPI_FLOW_SOURCE_VOLUMES = {
    PPI_FLOW_OUTPUT_VOLUME_NAME: PPI_FLOW_OUTPUT_VOLUME,
    FLOWPACKER_OUTPUT_VOLUME_NAME: FLOWPACKER_OUTPUT_VOLUME,
    AF3SCORE_OUTPUT_VOLUME_NAME: AF3SCORE_OUTPUT_VOLUME,
    ALPHAFOLD3_OUTPUT_VOLUME_NAME: ALPHAFOLD3_OUTPUT_VOLUME,
    ROSETTA_OUTPUT_VOLUME_NAME: ROSETTA_OUTPUT_VOLUME,
    WORKFLOW_OUTPUT_VOLUME_NAME: WORKFLOW_OUTPUT_VOLUME,
}
PPI_FLOW_SOURCE_VOLUME_MOUNTS: dict[
    str | PurePosixPath, modal.Volume | modal.CloudBucketMount
] = {
    PPI_FLOW_OUTPUT_MOUNTPOINT: PPI_FLOW_OUTPUT_VOLUME,
    FLOWPACKER_OUTPUT_MOUNTPOINT: FLOWPACKER_OUTPUT_VOLUME,
    AF3SCORE_OUTPUT_MOUNTPOINT: AF3SCORE_OUTPUT_VOLUME,
    ALPHAFOLD3_OUTPUT_MOUNTPOINT: ALPHAFOLD3_OUTPUT_VOLUME,
    ROSETTA_OUTPUT_MOUNTPOINT: ROSETTA_OUTPUT_VOLUME,
    WORKFLOW_OUTPUT_MOUNTPOINT: WORKFLOW_OUTPUT_VOLUME,
}
PPI_FLOW_TASK_VOLUME_MOUNTS = {
    **PPI_FLOW_SOURCE_VOLUME_MOUNTS,
    **ppiflow_app.CONF.mounts(output_volume=True, model_volume=True),
}
LIGANDMPNN_TASK_VOLUME_MOUNTS = {
    **PPI_FLOW_SOURCE_VOLUME_MOUNTS,
    **ligandmpnn_app.CONF.mounts(model_volume=True),
}
FLOWPACKER_TASK_VOLUME_MOUNTS = {
    **PPI_FLOW_SOURCE_VOLUME_MOUNTS,
    **flowpacker_app.CONF.mounts(
        output_volume=True,
        model_volume=True,
        model_ro=False,
    ),
}
AF3SCORE_TASK_VOLUME_MOUNTS = {
    **PPI_FLOW_SOURCE_VOLUME_MOUNTS,
    **af3score_app.CONF.mounts(
        output_volume=True,
        model_volume=True,
        model_mount_subdir=False,
    ),
}
ALPHAFOLD3_TASK_VOLUME_MOUNTS = {
    **PPI_FLOW_SOURCE_VOLUME_MOUNTS,
    **alphafold3_app.CONF.mounts(
        output_volume=True,
        model_volume=True,
        model_ro=True,
    ),
    alphafold3_app.JAX_CACHE_MOUNTPOINT: alphafold3_app.JAX_CACHE_VOLUME,
}
ROSETTA_TASK_VOLUME_MOUNTS = rosetta_app.CONF.mounts(output_volume=True)


run_ppiflow_design_stage = app.function(
    image=ppiflow_task_image,
    gpu=ppiflow_app.CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_TASK_VOLUME_MOUNTS,
)(_run_ppiflow_design_stage)


run_ppiflow_flowpacker_stage = app.function(
    image=flowpacker_task_image,
    gpu=flowpacker_app.CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=FLOWPACKER_TASK_VOLUME_MOUNTS,
)(_run_ppiflow_flowpacker_stage)


run_ppiflow_dockq_stage = app.function(
    image=dockq_task_image,
    cpu=(0.125, 16.125),
    memory=(512, 16384),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)(_run_ppiflow_dockq_stage)


check_ppiflow_external_artifact = app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 4096),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)(analysis_runtime.check_ppiflow_external_artifact)


normalize_ppiflow_stage2_input = app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 4096),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)(analysis_runtime.normalize_ppiflow_stage2_input)


filter_ppiflow_artifacts = app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)(analysis_runtime.filter_ppiflow_artifacts)


derive_ppiflow_fixed_positions = app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)(analysis_runtime.derive_ppiflow_fixed_positions)


rank_ppiflow_artifacts = app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)(analysis_runtime.rank_ppiflow_artifacts)


prepare_ppiflow_rosetta_stage = app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)(rosetta_runtime.prepare_ppiflow_rosetta_stage)


def _read_af3score_plan_artifacts(
    artifacts: Sequence[ExecutionArtifact],
) -> dict[str, object]:
    """Load the single prepared AF3Score plan from a workflow artifact."""
    if len(artifacts) != 1:
        raise ValueError(f"Expected one AF3Score task plan, found {len(artifacts)}")
    path = ppiflow_staging.artifact_mount_path(
        artifacts[0],
        PPI_FLOW_SOURCE_VOLUME_ROOTS,
    )
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("AF3Score task plan must be a JSON object")
    return value


prepare_ppiflow_af3score_stage = app.function(
    image=af3score_task_image,
    cpu=(0.125, 16.125),
    memory=(1024, 32768),
    timeout=CONF.timeout,
    volumes=AF3SCORE_TASK_VOLUME_MOUNTS,
)(_prepare_ppiflow_af3score_stage)


run_ppiflow_af3score_batch = app.function(
    image=af3score_task_image,
    gpu=af3score_app.CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=AF3SCORE_TASK_VOLUME_MOUNTS,
)(_run_ppiflow_af3score_batch)


postprocess_ppiflow_af3score_stage = app.function(
    image=af3score_task_image,
    cpu=(0.125, 16.125),
    memory=(1024, 16384),
    timeout=CONF.timeout,
    volumes=AF3SCORE_TASK_VOLUME_MOUNTS,
)(_postprocess_ppiflow_af3score_stage)


run_ppiflow_ligandmpnn_candidate = app.function(
    image=ligandmpnn_task_image,
    gpu=ligandmpnn_app.CONF.gpu,
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=LIGANDMPNN_TASK_VOLUME_MOUNTS,
)(_run_ppiflow_ligandmpnn_candidate)


run_ppiflow_partial_candidate = app.function(
    image=ppiflow_task_image,
    gpu=ppiflow_app.CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 65536),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_TASK_VOLUME_MOUNTS,
)(_run_ppiflow_partial_candidate)


run_ppiflow_refold_candidate = app.function(
    image=alphafold3_task_image,
    gpu=alphafold3_app.CONF.gpu,
    cpu=(0.125, 16.125),
    memory=(1024, 131072),
    timeout=CONF.timeout,
    volumes=ALPHAFOLD3_TASK_VOLUME_MOUNTS,
)(_run_ppiflow_refold_candidate)


run_ppiflow_rosetta_worker = app.function(
    image=rosetta_task_image,
    cpu=(0.125, 30.125),
    memory=(1024, 43008),
    timeout=CONF.timeout,
    volumes=ROSETTA_TASK_VOLUME_MOUNTS,
)(rosetta_runtime.run_ppiflow_rosetta_worker)


finalize_ppiflow_rosetta_stage = app.function(
    image=runtime_image,
    cpu=0.125,
    memory=(512, 8192),
    timeout=CONF.timeout,
    volumes=PPI_FLOW_SOURCE_VOLUME_MOUNTS,
)(rosetta_runtime.finalize_ppiflow_rosetta_stage)


_OBSOLETE_CONFIG_KEYS = (
    "candidate_concurrency",
    "max_child_calls",
    "max_batches",
    "max_num_pods",
    "num_jobs",
)
_OPERATIONAL_CONFIG_KEYS = (
    "prepare_workers",
    "_max_containers",
    "_max_gpu_containers",
)
_INPUT_DIGESTS_KEY = "_biomodals_input_sha256"


@dataclass
class _ConfiguredAppStepNode(ProviderNode):
    """Base class for configured PPIFlow app-backed workflow nodes."""

    step_name: str
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def _run_name(self, context: NodeRunContext) -> str:
        run_name = sanitize_filename(
            str(
                self.config.get("run_name")
                or f"{context.workload_run_key}-{self.step_name}"
            )
        )
        return run_name

    def _structure_inputs(
        self,
        context: NodeRunContext,
    ) -> list[ExecutionArtifact]:
        artifacts = context.inputs.get("structures") or []
        if not artifacts:
            raise ValueError(
                f"PPIFlow workflow step {self.step_name!r} requires structure inputs"
            )
        return artifacts


@dataclass(frozen=True)
class PPIFlowModelValidationNode(ProviderNode):
    """Validate the selected mutable checkpoint before GPU admission."""

    model_name: str
    expected_sha256: str

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Prepare one CPU checkpoint hash for this Execution Run."""
        del context
        return ProviderCallSpec(
            function_name="validate_ppiflow_model",
            uses_gpu=False,
            kwargs={
                "model_path": str(
                    Path(ppiflow_app.CONF.model_volume_mountpoint) / self.model_name
                ),
                "expected_sha256": self.expected_sha256,
            },
            runtime_image_key="ppiflow-model-validation",
        )


@dataclass
class PPIFlowDesignNode(_ConfiguredAppStepNode):
    """Initial PPIFlow design step with candidate-manifest publication."""

    config: dict[str, Any] = field(default_factory=dict, metadata={"dag_hash": False})
    scientific_config: dict[str, Any] = field(default_factory=dict)

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Prepare one direct PPIFlow design call for kernel submission."""
        raw_args = self.config.get("args", self.config)
        if not isinstance(raw_args, dict):
            raise ValueError(f"PPIFlow step {self.step_name!r} args must be a mapping")
        return ProviderCallSpec(
            function_name="run_ppiflow_design_stage",
            uses_gpu=True,
            kwargs={
                "args": ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args}),
                "run_name": self._run_name(context),
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
                "step_name": self.step_name,
            },
            runtime_image_key="ppiflow",
        )


@dataclass
class PPIFlowPartialNode(_ConfiguredAppStepNode, TaskProviderNode):
    """PPIFlow partial design with one kernel Task per candidate."""

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        """Discover stable candidate Tasks from the upstream manifest."""
        return _candidate_remote_tasks(
            context,
            max_candidates=_optional_config_int(self.config, "max_structures"),
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        """Prepare one partial-design candidate for kernel submission."""
        candidate_id = _candidate_task_id(context, task, step_name=self.step_name)
        return ProviderCallSpec(
            function_name="run_ppiflow_partial_candidate",
            uses_gpu=True,
            kwargs={
                "artifacts": self._structure_inputs(context),
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "candidate_id": candidate_id,
                "config": self.config,
                "step_name": self.step_name,
                "run_name": self._run_name(context),
            },
            runtime_image_key="ppiflow",
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish deterministic partial-design candidate outcomes."""
        return _finalize_candidate_tasks(
            context,
            step_name=self.step_name,
            stage_role="partial_design",
            operation_mode="ppiflow_partial",
            results=results,
            errors=errors,
        )


@dataclass
class LigandMPNNNode(_ConfiguredAppStepNode, TaskProviderNode):
    """LigandMPNN design with one kernel Task per input candidate."""

    def _model_type(self) -> str:
        """Return the configured sequence-design model."""
        return str(
            self.config.get(
                "model_type",
                "abmpnn" if self.step_name.startswith("AbMPNN") else "protein_mpnn",
            )
        )

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        """Discover stable candidate Tasks from the upstream manifest."""
        return _candidate_remote_tasks(
            context,
            max_candidates=_optional_config_int(self.config, "max_structures"),
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        """Prepare one LigandMPNN candidate for kernel submission."""
        candidate_id = _candidate_task_id(context, task, step_name=self.step_name)
        script_mode = str(self.config.get("script_mode", "run"))
        model_type = self._model_type()
        cli_kwargs = _ligandmpnn_cli_kwargs(
            self.config,
            script_mode=script_mode,
            model_type=model_type,
        )
        return ProviderCallSpec(
            function_name="run_ppiflow_ligandmpnn_candidate",
            uses_gpu=True,
            kwargs={
                "artifacts": self._structure_inputs(context),
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "candidate_id": candidate_id,
                "config": self.config,
                "run_name": self._run_name(context),
                "script_mode": script_mode,
                "cli_args": ligandmpnn_app.build_ligandmpnn_cli_args(**cli_kwargs),
                "step_name": self.step_name,
            },
            runtime_image_key="ligandmpnn",
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish deterministic sequence-design candidate outcomes."""
        return _finalize_candidate_tasks(
            context,
            step_name=self.step_name,
            stage_role="sequence_design",
            operation_mode=self._model_type(),
            results=results,
            errors=errors,
        )


@dataclass
class FlowPackerNode(_ConfiguredAppStepNode):
    """FlowPacker side-chain packing step."""

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Prepare the FlowPacker app call."""
        return ProviderCallSpec(
            function_name="run_ppiflow_flowpacker_stage",
            uses_gpu=True,
            kwargs={
                "artifacts": self._structure_inputs(context),
                "config": self.config,
                "run_name": self._run_name(context),
            },
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Expose FlowPacker archives as structure artifacts."""
        result = AppRunResult.model_validate(result)
        return _result_with_output_kind(
            result,
            ArtifactKind.STRUCTURES,
            {"step_name": self.step_name} | dict(metadata),
        )


@dataclass
class AF3ScorePrepareNode(_ConfiguredAppStepNode):
    """Prepare finite AF3Score candidate Tasks without nested Modal calls."""

    def scientific_versions(self) -> dict[str, str]:
        """Bind workflow identity to the scoring code and model weights."""
        return {
            "af3score": (
                af3score_app.CONF.repo_commit_hash
                or af3score_app.CONF.version
                or "unknown"
            ),
            "alphafold3.model": DECLARED_MODEL_IDENTITY,
        }

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Prepare AF3Score inputs and its durable Task plan."""
        structures = context.inputs.get("structures") or []
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return ProviderCallSpec(
            function_name="prepare_ppiflow_af3score_stage",
            uses_gpu=False,
            kwargs={
                "artifacts": structures,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "config": self.config,
                "step_name": self.step_name,
                "execution_run_name": sanitize_filename(
                    f"ppiflow-af3score-{context.execution_run_id}-{context.node_id}"
                ),
            },
            runtime_image_key="af3score-cpu",
        )


@dataclass
class AF3ScoreBatchNode(_ConfiguredAppStepNode, TaskProviderNode):
    """Schedule candidate Tasks through AF3Score's prepared GPU batches."""

    aggregation_policy = NodeAggregationPolicy.ALLOW_PARTIAL

    def _plan(self, context: NodeRunContext) -> dict[str, object]:
        return _read_af3score_plan_artifacts(context.inputs.get("af3score_plan") or [])

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        """Discover one Task per candidate that still needs GPU scoring."""
        plan = self._plan(context)
        candidates = plan.get("candidates")
        if not isinstance(candidates, list):
            raise TypeError("AF3Score task plan candidates must be a list")
        run_name = str(plan["run_name"])
        tasks = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TypeError("AF3Score candidate plan entries must be objects")
            candidate_payload = cast(Mapping[str, object], candidate)
            chunk = candidate_payload.get("chunk")
            if chunk is None:
                continue
            if not isinstance(chunk, Mapping):
                raise TypeError("AF3Score candidate chunk must be an object")
            candidate_id = str(candidate_payload["candidate_id"])
            tasks.append(
                TaskDefinition(
                    task_key=candidate_id,
                    scientific_payload=candidate_payload["scientific_payload"],
                    execution_payload={
                        "candidate_id": candidate_id,
                        "chunk": dict(chunk),
                        "input_name": str(candidate_payload["input_name"]),
                        "run_name": run_name,
                    },
                )
            )
        return tuple(tasks)

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        """Prepare the single-Task tail of one AF3Score batch."""
        return self.prepare_remote_task_batch(context, (task,))

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[TaskDefinition, ...],
    ) -> ProviderCallSpec:
        """Prepare one direct AF3Score GPU call for compatible Tasks."""
        if not tasks:
            raise ValueError("AF3Score provider batch cannot be empty")
        plan = self._plan(context)
        input_digests = plan.get("input_digests")
        publication_key = plan.get("publication_key")
        if not isinstance(input_digests, Mapping) or not isinstance(
            publication_key, str
        ):
            raise TypeError("AF3Score task plan publication data is invalid")
        normalized_digests = {
            str(key): str(value) for key, value in input_digests.items()
        }
        payloads: list[Mapping[str, object]] = []
        chunks: list[Mapping[str, object]] = []
        for task in tasks:
            if not isinstance(task.execution_payload, Mapping):
                raise TypeError("AF3Score Task execution payload must be an object")
            payload = cast(Mapping[str, object], task.execution_payload)
            if str(payload["candidate_id"]) != task.task_key:
                raise ValueError("AF3Score Task identity does not match its payload")
            chunk = payload.get("chunk")
            if not isinstance(chunk, Mapping):
                raise TypeError("AF3Score Task chunk must be an object")
            payloads.append(payload)
            chunks.append(cast(Mapping[str, object], chunk))
        first = payloads[0]
        first_chunk = chunks[0]
        batch_name = str(first_chunk["batch_name"])
        run_name = str(first["run_name"])
        if any(
            str(payload["run_name"]) != run_name or chunk != first_chunk
            for payload, chunk in zip(payloads, chunks, strict=True)
        ):
            raise ValueError("AF3Score provider batch mixes incompatible chunks")
        task_count = first_chunk["task_count"]
        if not isinstance(task_count, int):
            raise TypeError("AF3Score batch Task count must be an integer")
        batch_input_ids = tuple(
            Path(str(payload["input_name"])).stem for payload in payloads
        )
        return ProviderCallSpec(
            function_name="run_ppiflow_af3score_batch",
            uses_gpu=True,
            kwargs={
                "run_name": run_name,
                "batch_name": batch_name,
                "batch_json_dir": str(first_chunk["batch_json_dir"]),
                "batch_pdb_dir": str(first_chunk["batch_pdb_dir"]),
                "task_keys": [task.task_key for task in tasks],
                "input_names": [str(payload["input_name"]) for payload in payloads],
                "input_digests": {
                    input_id: normalized_digests[input_id]
                    for input_id in batch_input_ids
                },
                "publication_key": publication_key,
            },
            runtime_image_key="af3score-gpu",
            compatibility_key=f"{run_name}:{batch_name}",
            max_tasks_per_call=task_count,
        )

    def process_remote_task_batch_result(
        self,
        task_keys: tuple[str, ...],
        result: Any,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, AppRunResult]:
        """Decode AF3Score's per-candidate outcomes from one GPU call."""
        del metadata
        if not isinstance(result, Mapping):
            raise TypeError("AF3Score batch result must be an object")
        result_by_task = cast(Mapping[str, object], result)
        return {
            task_key: AppRunResult.model_validate(result_by_task[task_key])
            for task_key in task_keys
        }

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Record the aggregate batch outcome; postprocessing publishes scores."""
        del context
        status = (
            AppRunStatus.PARTIAL
            if errors and results
            else AppRunStatus.FAILED
            if errors
            else AppRunStatus.SUCCEEDED
        )
        return AppRunResult(
            status=status,
            warnings=[f"{key}: {errors[key]}" for key in sorted(errors)],
            metrics={
                "failed_candidates": len(errors),
                "scored_candidates": len(results),
            },
        )


@dataclass
class AF3ScoreNode(_ConfiguredAppStepNode, TaskProviderNode):
    """Postprocess AF3Score candidates while preserving partial outcomes."""

    aggregation_policy = NodeAggregationPolicy.ALLOW_PARTIAL

    def _plan(self, context: NodeRunContext) -> dict[str, object]:
        return _read_af3score_plan_artifacts(context.inputs.get("af3score_plan") or [])

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        """Discover one postprocessing outcome per requested candidate."""
        candidates = self._plan(context).get("candidates")
        if not isinstance(candidates, list):
            raise TypeError("AF3Score task plan candidates must be a list")
        tasks = []
        for raw_candidate in candidates:
            if not isinstance(raw_candidate, Mapping):
                raise TypeError("AF3Score candidate plan entries must be objects")
            candidate = cast(Mapping[str, object], raw_candidate)
            tasks.append(
                TaskDefinition(
                    task_key=str(candidate["candidate_id"]),
                    scientific_payload=cast(
                        Mapping[str, object], candidate["scientific_payload"]
                    ),
                )
            )
        return tuple(tasks)

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        """Describe the shared CPU postprocessor for one candidate."""
        return self.prepare_remote_task_batch(context, (task,))

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[TaskDefinition, ...],
    ) -> ProviderCallSpec:
        """Prepare one CPU call that postprocesses every admitted candidate."""
        if not tasks:
            raise ValueError("AF3Score postprocess batch cannot be empty")
        plan_artifacts = context.inputs.get("af3score_plan") or []
        candidate_count = len(self.discover_remote_tasks(context))
        return ProviderCallSpec(
            function_name="postprocess_ppiflow_af3score_stage",
            uses_gpu=False,
            kwargs={
                "plan_artifacts": plan_artifacts,
                "task_keys": [task.task_key for task in tasks],
                "step_name": self.step_name,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
            },
            runtime_image_key="af3score-cpu",
            compatibility_key=context.node_id,
            max_tasks_per_call=candidate_count,
        )

    def process_remote_task_batch_result(
        self,
        task_keys: tuple[str, ...],
        result: Any,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, AppRunResult]:
        """Decode per-candidate postprocessing outcomes."""
        del metadata
        if not isinstance(result, Mapping):
            raise TypeError("AF3Score postprocess result must be an object")
        result_by_task = cast(Mapping[str, object], result)
        return {
            task_key: AppRunResult.model_validate(result_by_task[task_key])
            for task_key in task_keys
        }

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish the aggregate status after retaining usable task outputs."""
        del context
        return AppRunResult(
            status=(
                AppRunStatus.PARTIAL
                if errors and results
                else AppRunStatus.FAILED
                if errors
                else AppRunStatus.SUCCEEDED
            ),
            warnings=[f"{key}: {errors[key]}" for key in sorted(errors)],
        )


@dataclass
class RosettaPrepareNode(_ConfiguredAppStepNode):
    """Stage PPIFlow candidates and publish a finite Rosetta Task plan."""

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Prepare the Rosetta input and job manifests without nested calls."""
        structures = context.inputs.get("structures") or []
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return ProviderCallSpec(
            function_name="prepare_ppiflow_rosetta_stage",
            uses_gpu=False,
            kwargs={
                "artifacts": structures,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "config": self.config,
                "step_name": self.step_name,
                "run_name": self._run_name(context),
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
            },
            runtime_image_key="rosetta-stage-cpu",
        )


@dataclass
class RosettaWorkerNode(_ConfiguredAppStepNode, PullTaskProviderNode):
    """Execute staged Rosetta candidates through kernel-owned pull Tasks."""

    def _plan(self, context: NodeRunContext) -> dict[str, object]:
        artifacts = context.inputs.get("rosetta_plan") or []
        if len(artifacts) != 1:
            raise ValueError(f"Expected one Rosetta task plan, found {len(artifacts)}")
        return rosetta_runtime._load_rosetta_plan(
            context.resolve_artifact(artifacts[0])
        )

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        """Discover the complete staged Rosetta Task set in manifest order."""
        plan = self._plan(context)
        run_root = (
            Path(str(plan["run_root"]))
            .relative_to(ROSETTA_OUTPUT_MOUNTPOINT)
            .as_posix()
        )
        return tuple(
            TaskDefinition(
                task_key=task.task_key,
                scientific_payload=task.scientific_payload,
                execution_payload={
                    "task": task.to_dict(),
                    "run_root": run_root,
                },
            )
            for task in (
                RosettaTaskSpec.from_dict(value)
                for value in cast(list[object], plan["tasks"])
            )
        )

    def prepare_pull_worker(
        self,
        context: NodeRunContext,
    ) -> PullWorkerCallSpec:
        """Bind the derived worker pool to the workflow's Rosetta function."""
        plan = self._plan(context)
        worker_count, claim_capacity, max_parallel = (
            rosetta_runtime._rosetta_worker_policy(
                len(cast(list[object], plan["tasks"])),
                self.config,
            )
        )
        return PullWorkerCallSpec(
            function_name="run_ppiflow_rosetta_worker",
            uses_gpu=False,
            claim_capacity=claim_capacity,
            max_worker_calls=worker_count,
            kwargs={
                "run_name": str(plan["run_name"]),
                "run_id": str(plan["run_id"]),
                "claim_capacity": claim_capacity,
                "max_parallel": max_parallel,
            },
            runtime_image_key="rosetta-cpu",
            compatibility_key=f"{plan['run_name']}:{plan['run_id']}",
        )

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus:
        """Revalidate the fingerprint-bound Rosetta marker and required files."""
        del context, result, artifacts
        try:
            recovered = self._recover_task_result(task, expected_fingerprint)
        except Exception:  # noqa: BLE001
            return AvailabilityStatus.UNKNOWN
        return (
            AvailabilityStatus.AVAILABLE
            if recovered is not None
            else AvailabilityStatus.MISSING
        )

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        """Rebuild the canonical receipt after a lost completion callback."""
        del context
        return self._recover_task_result(task, expected_fingerprint)

    @staticmethod
    def _recover_task_result(
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        payload = task.execution_payload
        if not isinstance(payload, Mapping):
            raise TypeError("Rosetta Task execution payload must be an object")
        spec = RosettaTaskSpec.from_dict(payload.get("task"))
        run_root = payload.get("run_root")
        if not isinstance(run_root, str):
            raise TypeError("Rosetta Task run_root must be text")
        if not validate_task_publication_from_volume(
            ROSETTA_OUTPUT_VOLUME,
            run_root,
            spec,
            expected_fingerprint,
        ):
            return None
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[rosetta_runtime._rosetta_task_receipt(spec, expected_fingerprint)],
            metrics={"candidate_id": spec.candidate_id or spec.task_key},
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish a deterministic outcome summary for remote finalization."""
        del context
        status = (
            AppRunStatus.PARTIAL
            if results and errors
            else AppRunStatus.SUCCEEDED
            if not errors
            else AppRunStatus.FAILED
        )
        return AppRunResult(
            status=status,
            outputs=[rosetta_runtime._rosetta_task_outcomes_artifact(results, errors)],
            warnings=[f"{task_key}: {errors[task_key]}" for task_key in sorted(errors)],
            metrics={
                "successful_candidates": len(results),
                "failed_candidates": len(errors),
            },
        )


@dataclass
class _RosettaNode(_ConfiguredAppStepNode):
    """Finalize one PPIFlow Rosetta stage from durable Task outcomes."""

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Validate remote outputs and publish the established stage contract."""
        plan_artifacts = context.inputs.get("rosetta_plan") or []
        outcome_artifacts = context.inputs.get("rosetta_outcomes") or []
        if not plan_artifacts or not outcome_artifacts:
            raise ValueError(f"{self.step_name} requires Rosetta Task results")
        return ProviderCallSpec(
            function_name="finalize_ppiflow_rosetta_stage",
            uses_gpu=False,
            kwargs={
                "plan_artifacts": plan_artifacts,
                "outcome_artifacts": outcome_artifacts,
                "config": self.config,
                "step_name": self.step_name,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
            },
            runtime_image_key="rosetta-finalize-cpu",
        )


@dataclass
class RosettaFixNode(_RosettaNode):
    """Rosetta fixed-position analysis step."""


@dataclass
class RosettaRelaxNode(_RosettaNode):
    """Rosetta relaxation step."""


@dataclass
class ReFoldNode(_ConfiguredAppStepNode, TaskProviderNode):
    """AlphaFold3 refolding with one kernel Task per candidate."""

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        """Discover stable candidate Tasks from the upstream manifest."""
        return _candidate_remote_tasks(
            context,
            max_candidates=_optional_config_int(self.config, "max_structures"),
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        """Prepare one candidate wrapper for kernel submission."""
        candidate_id = _candidate_task_id(context, task, step_name=self.step_name)
        return ProviderCallSpec(
            function_name="run_ppiflow_refold_candidate",
            uses_gpu=True,
            kwargs={
                "artifacts": self._structure_inputs(context),
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "candidate_id": candidate_id,
                "config": self.config,
                "step_name": self.step_name,
                "run_name": self._run_name(context),
            },
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Publish deterministic candidate outcomes after all Tasks finish."""
        return _finalize_candidate_tasks(
            context,
            step_name=self.step_name,
            stage_role="refold",
            operation_mode="alphafold3",
            results=results,
            errors=errors,
        )


@dataclass
class DockQNode(_ConfiguredAppStepNode):
    """DockQ model/reference scoring step."""

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Prepare the DockQ app call."""
        model_artifacts = context.inputs.get("models") or []
        if not model_artifacts:
            raise ValueError(f"{self.step_name} requires model structure inputs")
        return ProviderCallSpec(
            function_name="run_ppiflow_dockq_stage",
            uses_gpu=False,
            kwargs={
                "reference_artifacts": self._structure_inputs(context),
                "model_artifacts": model_artifacts,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "config": self.config,
                "run_name": self._run_name(context),
            },
        )

    def process_remote_result(
        self, result: AppRunResult, metadata: Mapping[str, object]
    ) -> AppRunResult:
        """Attach DockQ workflow metadata to score outputs."""
        result = AppRunResult.model_validate(result)
        return _result_with_output_kind(
            result,
            ArtifactKind.SCORES,
            {"step_name": self.step_name} | dict(metadata),
        )


@dataclass
class ExistingStructuresNode(ProviderNode):
    """Reference existing structures for stage-2-only PPIFlow runs."""

    step_name: str
    storage: VolumePath
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Prepare Stage2Input normalization for kernel submission."""
        return ProviderCallSpec(
            function_name="normalize_ppiflow_stage2_input",
            uses_gpu=False,
            kwargs={
                "storage": self.storage,
                "config": self.config,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
                "step_name": self.step_name,
            },
        )


@dataclass
class FilterStructuresNode(ProviderNode):
    """Filter structures using score artifacts."""

    step_name: str
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Prepare score filtering for kernel submission."""
        structures = context.inputs.get("structures") or []
        scores = context.inputs.get("scores") or []
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        if not scores:
            raise ValueError(f"{self.step_name} requires score inputs")
        return ProviderCallSpec(
            function_name="filter_ppiflow_artifacts",
            uses_gpu=False,
            kwargs={
                "structures": structures,
                "scores": scores,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "config": self.config,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
                "step_name": self.step_name,
            },
        )


@dataclass
class FixedPositionsNode(ProviderNode):
    """Convert Rosetta residue energies into fixed-position constraints."""

    step_name: str
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Prepare fixed-position conversion for kernel submission."""
        artifacts = context.inputs.get("structures") or []
        if not artifacts:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return ProviderCallSpec(
            function_name="derive_ppiflow_fixed_positions",
            uses_gpu=False,
            kwargs={
                "artifacts": artifacts,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "config": self.config,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
                "step_name": self.step_name,
            },
        )


@dataclass
class RankNode(ProviderNode):
    """Rank final designs."""

    step_name: str
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Prepare score-aware ranking for kernel submission."""
        structures = context.inputs.get("structures") or []
        score_artifacts = [
            artifact
            for input_name, artifact_list in context.inputs.items()
            if input_name not in {"structures", "candidate_manifest"}
            for artifact in artifact_list
        ]
        if not structures:
            raise ValueError(f"{self.step_name} requires structure inputs")
        return ProviderCallSpec(
            function_name="rank_ppiflow_artifacts",
            uses_gpu=False,
            kwargs={
                "structures": structures,
                "candidate_manifests": (context.inputs.get("candidate_manifest") or []),
                "score_artifacts": score_artifacts,
                "config": self.config,
                "run_id": context.workload_run_key,
                "node_id": context.node_id,
                "step_name": self.step_name,
            },
        )


@dataclass
class ReportNode(CoordinatorNode):
    """Write the final design report."""

    step_name: str
    config: dict[str, Any] = field(
        default_factory=dict,
        metadata={"dag_hash_exclude_keys": _OPERATIONAL_CONFIG_KEYS},
    )

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Execute report generation logic."""
        artifacts = [
            artifact
            for artifact_list in context.inputs.values()
            for artifact in artifact_list
        ]
        ranked_rows: list[dict[str, object]] = []
        attrition_rows: list[dict[str, object]] = []
        for artifact in context.inputs.get("rank", []):
            path = ppiflow_staging.artifact_mount_path(
                artifact,
                PPI_FLOW_SOURCE_VOLUME_ROOTS,
            )
            if path.is_file() and path.suffix == ".csv":
                ranked_rows.extend(
                    pl.read_csv(path, infer_schema_length=0).iter_rows(named=True)
                )
        manifest_frames: list[tuple[str, pl.DataFrame]] = []
        audit_frames: dict[str, list[pl.DataFrame]] = {}
        for artifact in artifacts:
            path = ppiflow_staging.artifact_mount_path(
                artifact,
                PPI_FLOW_SOURCE_VOLUME_ROOTS,
            )
            if artifact.kind != ArtifactKind.TABLE or not path.is_file():
                continue
            if path.suffix == ".parquet":
                try:
                    manifest_frame = ppiflow_manifests.read_manifest(path)
                except ValueError:
                    continue
                stage_names = (
                    manifest_frame.get_column("stage_name").unique().to_list()
                    if manifest_frame.height
                    else ["unknown"]
                )
                manifest_frames.extend(
                    (
                        str(stage_name),
                        manifest_frame.filter(pl.col("stage_name") == stage_name),
                    )
                    for stage_name in stage_names
                )
            elif path.name == "filter_audit.csv":
                audit_frame = pl.read_csv(path, infer_schema_length=0)
                if "stage_name" not in audit_frame.columns:
                    continue
                for stage_name in audit_frame.get_column("stage_name").unique():
                    audit_frames.setdefault(str(stage_name), []).append(
                        audit_frame.filter(pl.col("stage_name") == stage_name)
                    )
        for stage_name, manifest_frame in manifest_frames:
            matching_audits = audit_frames.get(stage_name, [])
            audit_frame = (
                pl.concat(matching_audits, how="diagonal") if matching_audits else None
            )
            attrition_rows.extend(
                ppiflow_tables.candidate_attrition_rows(
                    stage_name=stage_name,
                    manifest_frame=manifest_frame,
                    audit_frame=audit_frame,
                )
            )
        markdown = ppiflow_tables.render_report_markdown(
            step_name=self.step_name,
            artifact_count=len(artifacts),
            ranked_rows=ranked_rows,
            attrition_rows=attrition_rows,
            max_rows=int(self.config.get("max_rows", 25)),
        )
        report_filename = str(
            self.config.get("report_filename") or "design_report.html"
        )
        scientific_artifacts = list(
            {artifact.artifact_id: artifact for artifact in artifacts}.values()
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="design_report",
                    kind=ArtifactKind.REPORT,
                    storage=InlineBytes(
                        data=markdown.encode("utf-8"),
                        filename="design_report.md",
                        media_type="text/markdown",
                    ),
                    metadata={"step_name": self.step_name},
                ),
                AppOutput(
                    name="design_report_html",
                    kind=ArtifactKind.REPORT,
                    storage=InlineBytes(
                        data=ppiflow_tables.render_report_html(markdown).encode(
                            "utf-8"
                        ),
                        filename=report_filename,
                        media_type="text/html",
                    ),
                    metadata={"step_name": self.step_name},
                ),
                *(
                    republish_execution_artifact(artifact)
                    for artifact in scientific_artifacts
                ),
            ],
        )


def _candidate_rows_for_task_discovery(
    context: NodeRunContext,
    *,
    max_candidates: int | None,
) -> list[dict[str, object]]:
    """Load active candidate rows from workflow-owned manifest artifacts."""
    artifacts = context.inputs.get("candidate_manifest") or []
    if not artifacts:
        raise ValueError(
            f"PPIFlow Node {context.node_id!r} requires a candidate manifest"
        )
    frames = [
        ppiflow_manifests.read_manifest(context.resolve_artifact(artifact))
        for artifact in artifacts
    ]
    frame = pl.concat(frames, how="diagonal") if len(frames) > 1 else frames[0]
    ppiflow_manifests.validate_manifest_frame(frame)
    active = frame.filter(pl.col("candidate_status") == AppRunStatus.SUCCEEDED.value)
    if max_candidates is not None:
        active = active.head(max_candidates)
    rows = active.to_dicts()
    for row in rows:
        files = row.get("files")
        has_structure_digest = isinstance(files, Sequence) and any(
            isinstance(file_record, Mapping)
            and file_record.get("role") in {"structure", "structures"}
            and bool(file_record.get("content_sha256"))
            for file_record in files
        )
        if not has_structure_digest:
            raise ValueError(
                "PPIFlow candidate manifest row "
                f"{row['candidate_id']!r} has no structure content digest"
            )
    return rows


def _candidate_remote_tasks(
    context: NodeRunContext,
    *,
    max_candidates: int | None,
) -> tuple[TaskDefinition, ...]:
    """Discover one stable kernel Task per active candidate row."""
    return tuple(
        TaskDefinition(
            task_key=str(row["candidate_id"]),
            scientific_payload=row,
            execution_payload={"candidate_id": str(row["candidate_id"])},
        )
        for row in _candidate_rows_for_task_discovery(
            context,
            max_candidates=max_candidates,
        )
    )


def _candidate_task_id(
    context: NodeRunContext,
    task: TaskDefinition,
    *,
    step_name: str,
) -> str:
    """Validate and return one persisted candidate Task identity."""
    if not isinstance(task.execution_payload, Mapping):
        raise TypeError(f"{step_name} Task execution payload must be a mapping")
    candidate_id = str(task.execution_payload["candidate_id"])
    if candidate_id != task.task_key or context.task_key != task.task_key:
        raise ValueError(f"{step_name} Task identity does not match its payload")
    return candidate_id


def _finalize_candidate_tasks(
    context: NodeRunContext,
    *,
    step_name: str,
    stage_role: str,
    operation_mode: str,
    results: Mapping[str, AppRunResult],
    errors: Mapping[str, str],
) -> AppRunResult:
    """Publish one deterministic manifest for terminal candidate Tasks."""
    candidate_ids = sorted((*results, *errors))
    rows = [
        ppiflow_manifests.candidate_manifest_row(
            candidate_id=candidate_id,
            stage_name=step_name,
            stage_role=stage_role,
            operation_mode=operation_mode,
            candidate_status=(
                AppRunStatus.SUCCEEDED.value
                if candidate_id in results
                else AppRunStatus.FAILED.value
            ),
            error=errors.get(candidate_id),
            files=(
                _task_result_structure_files(context, results[candidate_id])
                if candidate_id in results
                else ()
            ),
        )
        for candidate_id in candidate_ids
    ]
    status = (
        AppRunStatus.PARTIAL
        if results and errors
        else AppRunStatus.SUCCEEDED
        if not errors
        else AppRunStatus.FAILED
    )
    return AppRunResult(
        status=status,
        outputs=[_task_manifest_output(context, step_name, rows)],
        warnings=[
            f"{candidate_id}: {errors[candidate_id]}" for candidate_id in sorted(errors)
        ],
    )


def _task_manifest_output(
    context: NodeRunContext,
    step_name: str,
    rows: Sequence[Mapping[str, object]],
) -> AppOutput:
    """Write one task-aggregated manifest inside the workflow run Volume."""
    if context.volume_root is None or context.artifact_volume_name is None:
        raise RuntimeError("Workflow Volume context is unavailable")
    manifest_path = context.work_dir / ppiflow_manifests.MANIFEST_FILENAME
    ppiflow_manifests.write_manifest(rows, manifest_path)
    return ppiflow_manifests.manifest_artifact_output(
        manifest_path=manifest_path,
        mount_root=str(context.volume_root),
        volume_name=context.artifact_volume_name,
        stage_name=step_name,
        row_count=len(rows),
    )


def _task_result_structure_files(
    context: NodeRunContext,
    result: AppRunResult,
) -> list[dict[str, object]]:
    """Describe workflow-owned Task structures for downstream fingerprints."""
    if context.volume_root is None or context.artifact_volume_name is None:
        raise RuntimeError("Workflow Volume context is unavailable")
    records = []
    for output in result.outputs:
        candidate_files = output.metadata.get("candidate_files")
        if isinstance(candidate_files, Sequence):
            records.extend(
                dict(file_record)
                for file_record in candidate_files
                if isinstance(file_record, Mapping)
            )
            continue
        if (
            output.kind != ArtifactKind.STRUCTURES
            or not isinstance(output.storage, VolumePath)
            or output.storage.volume_name != context.artifact_volume_name
        ):
            continue
        root = output.storage.at_mountpoint(context.volume_root)
        paths = (
            [root]
            if root.is_file()
            else sorted(path for path in root.rglob("*") if path.is_file())
        )
        for path in paths:
            relative = path.relative_to(context.volume_root).as_posix()
            records.append(
                ppiflow_manifests.candidate_file_record(
                    role="structure",
                    workflow_path=relative,
                    volume_name=context.artifact_volume_name,
                    path=path.name
                    if root.is_file()
                    else path.relative_to(root).as_posix(),
                    media_type=output.storage.media_type,
                    size_bytes=path.stat().st_size,
                    content_sha256=_file_sha256(path),
                )
            )
    return records


def _inline_rosetta_config_files(steps_doc: dict[str, Any]) -> dict[str, Any]:
    staged_steps = deepcopy(steps_doc)
    for step_name in ("RosettaFixStep", "RosettaRelaxStep"):
        if step_name not in staged_steps:
            continue
        cfg = _step_cfg(staged_steps, step_name)
        for field_name in ("rosetta_script", "flags_file"):
            value = cfg.get(field_name)
            if isinstance(value, str):
                cfg[field_name] = rosetta_runtime._resolve_rosetta_config_text(
                    value, field_name
                )
    return staged_steps


def _ligandmpnn_cli_kwargs(
    config: Mapping[str, object],
    *,
    script_mode: str,
    model_type: str,
) -> dict[str, object]:
    excluded = {
        "run_name",
        "seeds",
        "script_mode",
        "structure_index",
        "max_structures",
        "structure_patterns",
        "patterns",
        "bias_aa_per_residue_bytes",
        "omit_aa_per_residue_bytes",
    }
    allowed = set(ligandmpnn_app.build_ligandmpnn_cli_args.__annotations__)
    allowed.discard("return")
    kwargs = {
        key: value
        for key, value in config.items()
        if key in allowed and key not in excluded
    }
    kwargs["script_mode"] = script_mode
    kwargs["model_type"] = model_type
    return kwargs


def build_ppiflow_workflow(
    *,
    task_yaml_bytes: bytes,
    steps_yaml_bytes: bytes,
    stage: int | None = None,
    max_containers: int | None = None,
    max_gpu_containers: int | None = None,
) -> ExecutionGraph:
    """Build a PPIFlow workflow DAG from upstream-style YAML files."""
    if stage not in {None, 1, 2}:
        raise ValueError("stage must be omitted, 1, or 2")
    task_doc = _load_yaml_bytes(task_yaml_bytes)
    steps_doc = _load_yaml_bytes(steps_yaml_bytes)
    max_containers = 16 if max_containers is None else max_containers
    max_gpu_containers = (
        max_containers if max_gpu_containers is None else max_gpu_containers
    )
    if max_containers < 1:
        raise ValueError("max_containers must be at least 1")
    if not 0 <= max_gpu_containers <= max_containers:
        raise ValueError("max_gpu_containers must be between zero and max_containers")
    task = _task_section(task_doc)
    enabled = _enabled_section(task_doc)
    steps_doc = _steps_doc_with_run_limits(
        steps_doc,
        enabled_steps=(name for name, is_enabled in enabled.items() if is_enabled),
        max_containers=max_containers,
        max_gpu_containers=max_gpu_containers,
    )
    gentype = str(task.get("gentype") or task.get("design_mode") or "binder")
    model_versions = _ppiflow_model_scientific_versions(
        enabled=enabled,
        gentype=gentype,
        stage=stage,
        steps=steps_doc,
    )
    workflow = ExecutionGraph(
        "ppiflow-v2",
        scientific_versions={
            "af3score": app_scientific_version(af3score_app.CONF),
            "alphafold3": app_scientific_version(alphafold3_app.CONF),
            "alphafold3.model": DECLARED_MODEL_IDENTITY,
            "biomodals.workflow.ppiflow": _SCIENTIFIC_SCHEMA_VERSION,
            "dockq": app_scientific_version(dockq_app.CONF),
            "flowpacker": app_scientific_version(flowpacker_app.CONF),
            "ligandmpnn": app_scientific_version(ligandmpnn_app.CONF),
            "ppiflow": app_scientific_version(ppiflow_app.CONF),
            **model_versions,
            "rosetta": app_scientific_version(rosetta_app.CONF),
        },
    )
    report_table_inputs: dict[str, Any] = {}
    report_partial_sources: list[Any] = []
    model_validation = None
    if model_versions:
        version_key, expected_sha256 = next(iter(model_versions.items()))
        model_name = version_key.removeprefix("ppiflow.model.")
        model_validation = workflow.add_node(
            PPIFlowModelValidationNode(model_name, expected_sha256),
            id="ppiflow-model-validation",
            reuse_predecessor_publication=False,
        )

    stage1_tail = None
    stage1_allows_partial = False
    if stage in {None, 1}:
        stage1_tail, stage1_allows_partial = _add_stage1_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            report_table_inputs=report_table_inputs,
            report_partial_sources=report_partial_sources,
            model_validation=model_validation,
        )

    if stage in {None, 2}:
        stage2_upstream = stage1_tail
        if stage == 2:
            stage2_upstream = workflow.add_node(
                _stage2_input_node(task, steps_doc),
                id="stage2-existing-input",
            )
        _add_stage2_nodes(
            workflow=workflow,
            enabled=enabled,
            steps=steps_doc,
            gentype=gentype,
            upstream=stage2_upstream,
            upstream_allows_partial=stage1_allows_partial,
            report_table_inputs=report_table_inputs,
            report_partial_sources=report_partial_sources,
            model_validation=model_validation,
        )

    return workflow


def _ppiflow_model_scientific_versions(
    *,
    enabled: dict[str, bool],
    gentype: str,
    stage: int | None,
    steps: dict[str, Any],
) -> dict[str, str]:
    active_steps: list[tuple[str, bool]] = []
    if stage in {None, 1} and _step_enabled(enabled, "PPIFlowStep"):
        active_steps.append(("PPIFlowStep", False))
    if stage in {None, 2} and _step_enabled(enabled, "PartialStep"):
        active_steps.append(("PartialStep", True))
    if not active_steps:
        return {}
    try:
        expected_model = f"{gentype}.ckpt"
        model_sha256 = ppiflow_app.PPI_FLOW_MODEL_SHA256[expected_model]
    except KeyError as error:
        raise ValueError(f"Unsupported PPIFlow gentype: {gentype!r}") from error
    for step_name, partial in active_steps:
        selected_model = _ppiflow_step_model_weights_name(
            _step_cfg(steps, step_name),
            partial=partial,
            fixed_positions_from_upstream=(
                partial and _step_enabled(enabled, "RosettaFixStep")
            ),
        )
        if selected_model != expected_model:
            raise ValueError(
                f"PPIFlow gentype {gentype!r} disagrees with enabled step "
                f"{step_name!r} model {selected_model!r}"
            )
    return {f"ppiflow.model.{expected_model}": model_sha256}


def _ppiflow_step_model_weights_name(
    config: dict[str, Any],
    *,
    partial: bool,
    fixed_positions_from_upstream: bool = False,
) -> str:
    raw_args = deepcopy(config.get("args", config))
    if not isinstance(raw_args, dict):
        raise ValueError("PPIFlow step args must be a mapping")
    if partial:
        app_args = _validated_ppiflow_partial_args(
            raw_args,
            structure_path="/workflow-input/candidate.pdb",
            fixed_positions=("A1" if fixed_positions_from_upstream else None),
        )
    else:
        app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
    expected_types = (
        (
            ppiflow_app.SampleAntibodyNanobodyPartialConfig,
            ppiflow_app.SampleBinderPartialConfig,
        )
        if partial
        else (
            ppiflow_app.SampleAntibodyNanobodyConfig,
            ppiflow_app.SampleBinderConfig,
        )
    )
    if not isinstance(app_args.args, expected_types):
        raise ValueError("PPIFlow step uses the wrong operation mode")
    return app_args.model_weights_name


def _add_stage1_nodes(
    *,
    workflow: ExecutionGraph,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
    report_table_inputs: dict[str, Any],
    report_partial_sources: list[Any],
    model_validation: Any,
):
    tail = None
    partial_tail = None
    if _step_enabled(enabled, "PPIFlowStep"):
        design_config = _step_cfg(steps, "PPIFlowStep")
        runtime_config, scientific_config = _ppiflow_design_configs(design_config)
        tail = workflow.add_node(
            PPIFlowDesignNode(
                "PPIFlowStep",
                runtime_config,
                scientific_config,
            ),
            id="stage1-ppiflow-design",
            depends_on=([model_validation] if model_validation is not None else None),
        )

    mpnn_step = None
    if gentype == "binder" and _step_enabled(enabled, "MPNNStep_stage1"):
        mpnn_step = ("stage1-ligandmpnn", "MPNNStep_stage1")
    elif gentype in {"antibody", "nanobody"} and _step_enabled(
        enabled, "AbMPNNStep_stage1"
    ):
        mpnn_step = ("stage1-abmpnn", "AbMPNNStep_stage1")
    if mpnn_step is not None:
        node_id, step_name = mpnn_step
        tail = workflow.add_node(
            LigandMPNNNode(
                step_name,
                _step_cfg(steps, step_name),
            ),
            id=node_id,
            inputs=_structure_inputs(tail),
            aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
        )
        partial_tail = tail
        report_table_inputs["stage1_mpnn_seqs"] = tail.outputs(kind=ArtifactKind.TABLE)
        report_partial_sources.append(tail)

    if _step_enabled(enabled, "FlowpackerStep_stage1"):
        tail = workflow.add_node(
            FlowPackerNode(
                "FlowpackerStep_stage1",
                _step_cfg(steps, "FlowpackerStep_stage1"),
            ),
            id="stage1-flowpacker",
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = None

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage1"):
        score = _add_af3score_nodes(
            workflow=workflow,
            node_id="stage1-af3score",
            step_name="AF3scoreStep_stage1",
            config=_step_cfg(steps, "AF3scoreStep_stage1"),
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )

    if _step_enabled(enabled, "FilterStep_stage1"):
        inputs = _structure_inputs(tail)
        if score is not None:
            inputs["scores"] = score.outputs(kind=ArtifactKind.SCORES)
        tail = workflow.add_node(
            FilterStructuresNode(
                "FilterStep_stage1",
                _step_cfg(steps, "FilterStep_stage1"),
            ),
            id="stage1-filter",
            inputs=inputs,
            accept_partial_from=_partial_sources(partial_tail, score),
        )
        partial_tail = None
        report_table_inputs["stage1_filter_tables"] = tail.outputs(
            kind=ArtifactKind.TABLE
        )
    return tail, partial_tail is not None


def _add_af3score_nodes(
    *,
    workflow: ExecutionGraph,
    node_id: str,
    step_name: str,
    config: dict[str, Any],
    inputs: dict[str, Any],
    accept_partial_from: list[Any] | None,
):
    """Add prepare, fixed-batch GPU, and postprocess Nodes for AF3Score."""
    prepare = workflow.add_node(
        AF3ScorePrepareNode(step_name, config),
        id=f"{node_id}-prepare",
        inputs=inputs,
        accept_partial_from=accept_partial_from,
    )
    batches = workflow.add_node(
        AF3ScoreBatchNode(step_name, config),
        id=f"{node_id}-batches",
        inputs={
            "af3score_plan": prepare.outputs(kind=ArtifactKind.TABLE),
        },
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
        allow_empty_result=True,
    )
    return workflow.add_node(
        AF3ScoreNode(step_name, config),
        id=node_id,
        inputs={
            "af3score_plan": prepare.outputs(kind=ArtifactKind.TABLE),
        },
        depends_on=[batches],
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
        accept_partial_from=[batches],
    )


def _add_rosetta_nodes(
    *,
    workflow: ExecutionGraph,
    node_id: str,
    step_name: str,
    finalizer_class: type[_RosettaNode],
    config: dict[str, Any],
    inputs: dict[str, Any],
    accept_partial_from: list[Any] | None,
):
    """Add prepare, pull-worker, and publication Nodes for Rosetta."""
    prepare = workflow.add_node(
        RosettaPrepareNode(step_name, config),
        id=f"{node_id}-prepare",
        inputs=inputs,
        accept_partial_from=accept_partial_from,
    )
    workers = workflow.add_node(
        RosettaWorkerNode(step_name, config),
        id=f"{node_id}-workers",
        inputs={
            "rosetta_plan": prepare.outputs(kind=ArtifactKind.TABLE),
        },
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )
    return workflow.add_node(
        finalizer_class(step_name, config),
        id=node_id,
        inputs={
            "rosetta_plan": prepare.outputs(kind=ArtifactKind.TABLE),
            "rosetta_outcomes": workers.outputs(kind=ArtifactKind.TABLE),
        },
        accept_partial_from=[workers],
    )


def _add_stage2_nodes(
    *,
    workflow: ExecutionGraph,
    enabled: dict[str, bool],
    steps: dict[str, Any],
    gentype: str,
    upstream,
    upstream_allows_partial: bool,
    report_table_inputs: dict[str, Any],
    report_partial_sources: list[Any],
    model_validation: Any,
) -> None:
    tail = upstream
    partial_tail = upstream if upstream_allows_partial else None
    if _step_enabled(enabled, "RosettaFixStep"):
        tail = _add_rosetta_nodes(
            workflow=workflow,
            node_id="stage2-rosetta-fix",
            step_name="RosettaFixStep",
            finalizer_class=RosettaFixNode,
            config=_step_cfg(steps, "RosettaFixStep"),
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = tail

    if _step_enabled(enabled, "RosettaFixStep") and _step_enabled(
        enabled, "PartialStep"
    ):
        tail = workflow.add_node(
            FixedPositionsNode(
                "FixedPositions",
                {"gentype": gentype} | _step_cfg(steps, "FixedPositions"),
            ),
            id="stage2-fixed-positions",
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = None

    if _step_enabled(enabled, "PartialStep"):
        tail = workflow.add_node(
            PPIFlowPartialNode(
                "PartialStep",
                _step_cfg(steps, "PartialStep"),
            ),
            id="stage2-partial-ppiflow",
            inputs=_structure_inputs(tail),
            aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
            accept_partial_from=_partial_sources(partial_tail),
            depends_on=([model_validation] if model_validation is not None else None),
        )
        partial_tail = tail

    mpnn_step = None
    if gentype == "binder" and _step_enabled(enabled, "MPNNStep_stage2"):
        mpnn_step = ("stage2-ligandmpnn", "MPNNStep_stage2")
    elif gentype in {"antibody", "nanobody"} and _step_enabled(
        enabled, "AbMPNNStep_stage2"
    ):
        mpnn_step = ("stage2-abmpnn", "AbMPNNStep_stage2")
    if mpnn_step is not None:
        node_id, step_name = mpnn_step
        tail = workflow.add_node(
            LigandMPNNNode(
                step_name,
                _step_cfg(steps, step_name),
            ),
            id=node_id,
            inputs=_structure_inputs(tail),
            aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = tail
        report_table_inputs["mpnn_seqs"] = tail.outputs(kind=ArtifactKind.TABLE)
        report_partial_sources.append(tail)

    if _step_enabled(enabled, "FlowpackerStep_stage2"):
        tail = workflow.add_node(
            FlowPackerNode(
                "FlowpackerStep_stage2",
                _step_cfg(steps, "FlowpackerStep_stage2"),
            ),
            id="stage2-flowpacker",
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )
        partial_tail = None

    score = None
    if _step_enabled(enabled, "AF3scoreStep_stage2"):
        score = _add_af3score_nodes(
            workflow=workflow,
            node_id="stage2-af3score",
            step_name="AF3scoreStep_stage2",
            config=_step_cfg(steps, "AF3scoreStep_stage2"),
            inputs=_structure_inputs(tail),
            accept_partial_from=_partial_sources(partial_tail),
        )

    filtered = tail
    if _step_enabled(enabled, "FilterStep_stage2"):
        inputs = _structure_inputs(tail)
        if score is not None:
            inputs["scores"] = score.outputs(kind=ArtifactKind.SCORES)
        filtered = workflow.add_node(
            FilterStructuresNode(
                "FilterStep_stage2",
                _step_cfg(steps, "FilterStep_stage2"),
            ),
            id="stage2-filter",
            inputs=inputs,
            accept_partial_from=_partial_sources(partial_tail, score),
        )
        partial_tail = None
        report_table_inputs["filter_tables"] = filtered.outputs(kind=ArtifactKind.TABLE)

    refold = None
    if _step_enabled(enabled, "ReFoldStep"):
        refold = workflow.add_node(
            ReFoldNode(
                "ReFoldStep",
                _step_cfg(steps, "ReFoldStep"),
            ),
            id="stage2-alphafold3-refold",
            inputs=_structure_inputs(filtered),
            aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
            accept_partial_from=_partial_sources(partial_tail),
        )
        report_table_inputs["refold_metrics"] = refold.outputs(kind=ArtifactKind.TABLE)

    dockq = None
    if _step_enabled(enabled, "DockQStep"):
        inputs = _structure_inputs(filtered)
        if refold is not None:
            inputs["models"] = refold.outputs(kind=ArtifactKind.STRUCTURES)
        dockq = workflow.add_node(
            DockQNode(
                "DockQStep",
                _step_cfg(steps, "DockQStep"),
            ),
            id="stage2-dockq",
            inputs=inputs,
            accept_partial_from=_partial_sources(partial_tail, refold),
        )

    relaxed = None
    if _step_enabled(enabled, "RosettaRelaxStep"):
        inputs = _structure_inputs(filtered)
        if dockq is not None:
            inputs["dockq"] = dockq.outputs(kind=ArtifactKind.SCORES)
        relaxed = _add_rosetta_nodes(
            workflow=workflow,
            node_id="stage2-rosetta-relax",
            step_name="RosettaRelaxStep",
            finalizer_class=RosettaRelaxNode,
            config=_step_cfg(steps, "RosettaRelaxStep"),
            inputs=inputs,
            accept_partial_from=_partial_sources(partial_tail),
        )

    rank = None
    if _step_enabled(enabled, "RankStep"):
        inputs = _structure_inputs(relaxed or filtered)
        if dockq is not None:
            inputs["dockq"] = dockq.outputs(kind=ArtifactKind.SCORES)
        if refold is not None:
            inputs["refold"] = refold.outputs(kind=ArtifactKind.STRUCTURES)
            inputs["refold_metrics"] = refold.outputs(kind=ArtifactKind.TABLE)
        if score is not None:
            inputs["af3scores"] = score.outputs(kind=ArtifactKind.SCORES)
        rank = workflow.add_node(
            RankNode(
                "RankStep",
                {"gentype": gentype} | _step_cfg(steps, "RankStep"),
            ),
            id="stage2-rank",
            inputs=inputs,
            accept_partial_from=_partial_sources(
                relaxed if relaxed is not None else partial_tail,
                refold,
                score,
            ),
        )

    if _step_enabled(enabled, "ReportStep"):
        final_structures = relaxed or filtered
        inputs = (
            {
                "structures": rank.outputs(kind=ArtifactKind.STRUCTURES),
                "rank": rank.outputs(kind=ArtifactKind.TABLE),
                "candidate_manifest": final_structures.outputs(
                    kind=ArtifactKind.TABLE,
                    role=ppiflow_manifests.MANIFEST_FILE_ROLE,
                ),
            }
            if rank is not None
            else _structure_inputs(final_structures)
        )
        inputs.update(report_table_inputs)
        workflow.add_node(
            ReportNode("ReportStep", _step_cfg(steps, "ReportStep")),
            id="stage2-report",
            inputs=inputs,
            accept_partial_from=_partial_sources(
                relaxed if relaxed is not None else partial_tail,
                refold,
                *report_partial_sources,
            ),
        )


def _structure_inputs(upstream) -> dict[str, Any]:
    if upstream is None:
        return {}
    return {
        "structures": upstream.outputs(kind=ArtifactKind.STRUCTURES),
        "candidate_manifest": upstream.outputs(
            kind=ArtifactKind.TABLE,
            role=ppiflow_manifests.MANIFEST_FILE_ROLE,
        ),
    }


def _partial_sources(*sources: Any) -> list[Any] | None:
    """Return the present partial-result dependencies for one Node."""
    present = [source for source in sources if source is not None]
    return present or None


def _stage2_input_node(
    task: Mapping[str, Any],
    steps: Mapping[str, Any],
) -> ExistingStructuresNode:
    raw_cfg = steps.get("Stage2Input") or task.get("stage2_input")
    if not isinstance(raw_cfg, Mapping):
        raise ValueError(
            "stage=2 PPIFlow runs require a Stage2Input step config or "
            "task.stage2_input mapping with an existing structure path"
        )
    raw_path = raw_cfg.get("path")
    if raw_path is None:
        raise ValueError("Stage2Input requires a 'path' value")
    volume_name = str(raw_cfg.get("volume_name", PPI_FLOW_OUTPUT_VOLUME_NAME))
    storage = analysis_runtime._volume_path_from_stage_config(
        str(raw_path), volume_name=volume_name
    )
    return ExistingStructuresNode(
        "Stage2Input",
        storage,
        dict(raw_cfg),
    )


_STAGE2_INPUT_SNAPSHOT_KEY = "_input_snapshot"


def _client_volume_file_record(storage: VolumePath) -> dict[str, object]:
    volume = PPI_FLOW_SOURCE_VOLUMES.get(storage.volume_name)
    if volume is None:
        raise ValueError(f"Unknown Stage2Input volume {storage.volume_name!r}")
    return analysis_runtime._streamed_volume_file_record(
        volume_name=storage.volume_name,
        path=storage.path,
        chunks=volume.read_file(storage.path),
    )


def _manifest_structure_storages(
    frame: pl.DataFrame,
) -> list[tuple[VolumePath, int, str]]:
    records: dict[tuple[str, str], tuple[VolumePath, int, str]] = {}
    for row in frame.iter_rows(named=True):
        for raw_file in row["files"]:
            if raw_file.get("role") != "structure":
                continue
            volume_name = raw_file.get("volume_name")
            app_volume_path = raw_file.get("app_volume_path")
            size_bytes = raw_file.get("size_bytes")
            content_sha256 = raw_file.get("content_sha256")
            if not volume_name or not app_volume_path:
                raise ValueError(
                    "Stage2Input manifest structure files require volume_name "
                    "and app_volume_path"
                )
            if size_bytes is None or not content_sha256:
                raise ValueError(
                    "Stage2Input manifest structure files require size_bytes "
                    "and content_sha256"
                )
            storage = VolumePath(
                volume_name=str(volume_name),
                path=str(app_volume_path),
            )
            key = (storage.volume_name, storage.path)
            bound = (storage, int(size_bytes), str(content_sha256))
            previous = records.setdefault(key, bound)
            if previous != bound:
                raise ValueError(
                    f"Stage2Input manifest has conflicting records for {storage}"
                )
    if not records:
        raise ValueError("Stage2Input manifest contains no structure files")
    return [records[key] for key in sorted(records)]


def _read_client_manifest(storage: VolumePath) -> tuple[bytes, pl.DataFrame]:
    volume = PPI_FLOW_SOURCE_VOLUMES.get(storage.volume_name)
    if volume is None:
        raise ValueError(f"Unknown Stage2Input volume {storage.volume_name!r}")
    data = b"".join(volume.read_file(storage.path))
    frame = pl.read_parquet(BytesIO(data))
    ppiflow_manifests.validate_manifest_frame(frame)
    return data, frame


def _client_stage2_structure_storages(
    storage: VolumePath,
    *,
    patterns: Sequence[str] | None,
) -> list[VolumePath]:
    volume = PPI_FLOW_SOURCE_VOLUMES.get(storage.volume_name)
    if volume is None:
        raise ValueError(f"Unknown Stage2Input volume {storage.volume_name!r}")
    entries = sorted(
        str(entry.path).lstrip("/")
        for entry in volume.iterdir(storage.path, recursive=True)
        if int(entry.type) == 1
    )
    root = PurePosixPath(storage.path)
    if entries == [storage.path]:
        return [storage]
    selected = []
    for path in entries:
        try:
            relative = PurePosixPath(path).relative_to(root).as_posix()
        except ValueError:
            continue
        if ppiflow_staging.matches_structure_pattern(relative, patterns):
            selected.append(VolumePath(volume_name=storage.volume_name, path=path))
    if not selected:
        raise FileNotFoundError(f"Stage2Input did not find structures under {storage}")
    return selected


def _bind_stage2_input_identity(
    *,
    task_doc: dict[str, Any],
    steps_doc: dict[str, Any],
    stage: int | None,
) -> dict[str, Any]:
    """Bind a stage-2-only workflow to the exact external input bytes."""
    if stage != 2:
        return steps_doc
    raw_cfg = steps_doc.get("Stage2Input") or _task_section(task_doc).get(
        "stage2_input"
    )
    if not isinstance(raw_cfg, Mapping):
        raise ValueError("stage=2 PPIFlow runs require a Stage2Input mapping")
    config = dict(raw_cfg)
    raw_path = config.get("path")
    if raw_path is None:
        raise ValueError("Stage2Input requires a 'path' value")
    storage = analysis_runtime._volume_path_from_stage_config(
        str(raw_path),
        volume_name=str(config.get("volume_name", PPI_FLOW_OUTPUT_VOLUME_NAME)),
    )
    manifest_storage = analysis_runtime._stage2_manifest_storage_from_config(
        config,
        default_volume_name=storage.volume_name,
    )
    snapshot: dict[str, object] = {}
    if manifest_storage is not None:
        manifest_bytes, frame = _read_client_manifest(manifest_storage)
        snapshot["manifest"] = analysis_runtime._streamed_volume_file_record(
            volume_name=manifest_storage.volume_name,
            path=manifest_storage.path,
            chunks=(manifest_bytes,),
        )
        manifest_structures = _manifest_structure_storages(frame)
        structure_storages = [item[0] for item in manifest_structures]
        expected = {
            (item[0].volume_name, item[0].path): (item[1], item[2])
            for item in manifest_structures
        }
    else:
        structure_storages = _client_stage2_structure_storages(
            storage,
            patterns=_patterns_from_config(config),
        )
        expected = {}
    structure_records = [
        _client_volume_file_record(item) for item in structure_storages
    ]
    for record in structure_records:
        key = (str(record["volume_name"]), str(record["path"]))
        if key in expected and expected[key] != (
            record["size_bytes"],
            record["content_sha256"],
        ):
            raise ValueError(
                "Stage2Input manifest digest does not match structure bytes: "
                f"{record['path']}"
            )
    snapshot["structures"] = structure_records
    config[_STAGE2_INPUT_SNAPSHOT_KEY] = snapshot
    bound = deepcopy(steps_doc)
    if "Stage2Input" in bound:
        bound["Stage2Input"] = config
    else:
        task = _task_section(task_doc)
        task["stage2_input"] = config
    return bound


def _load_yaml_bytes(data: bytes) -> dict[str, Any]:
    loaded = yaml.safe_load(data.decode("utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("YAML root must be a mapping")
    return loaded


def _task_section(task_doc: dict[str, Any]) -> dict[str, Any]:
    section = task_doc.get("task", task_doc)
    if not isinstance(section, dict):
        raise ValueError("task.yaml must contain a mapping under 'task'")
    return section


def _enabled_section(task_doc: dict[str, Any]) -> dict[str, bool]:
    enabled = task_doc.get("steps", {})
    if not isinstance(enabled, dict):
        raise ValueError("task.yaml 'steps' section must be a mapping")
    return {str(key): bool(value) for key, value in enabled.items()}


def _step_enabled(enabled: dict[str, bool], step_name: str) -> bool:
    return bool(enabled.get(step_name, False))


def _step_cfg(steps: dict[str, Any], step_name: str) -> dict[str, Any]:
    cfg = steps.get(step_name, {})
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"steps.yaml entry {step_name!r} must be a mapping")
    return cfg


def _ppiflow_design_configs(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate runtime paths from content-based scientific input identity."""
    runtime_config = deepcopy(config)
    raw_digests = runtime_config.pop(_INPUT_DIGESTS_KEY, {})
    if not isinstance(raw_digests, Mapping):
        raise ValueError(f"{_INPUT_DIGESTS_KEY} must be a mapping")
    input_digests = {str(key): str(value) for key, value in raw_digests.items()}

    scientific_config = deepcopy(runtime_config)
    for key in _OPERATIONAL_CONFIG_KEYS:
        scientific_config.pop(key, None)
    scientific_args = scientific_config.get("args", scientific_config)
    if not isinstance(scientific_args, dict):
        raise ValueError("PPIFlow step args must be a mapping")
    for field_name in input_digests:
        scientific_args.pop(field_name, None)
    if input_digests:
        scientific_config["input_sha256"] = dict(sorted(input_digests.items()))
    return runtime_config, scientific_config


def _steps_doc_with_run_limits(
    steps_doc: dict[str, Any],
    *,
    enabled_steps: Iterable[str],
    max_containers: int,
    max_gpu_containers: int,
) -> dict[str, Any]:
    configured = dict(steps_doc)
    for step_name in enabled_steps:
        if step_name == "Stage2Input":
            continue
        raw_cfg = steps_doc.get(step_name)
        if raw_cfg is None:
            raw_cfg = {}
        if not isinstance(raw_cfg, dict):
            configured[step_name] = raw_cfg
            continue
        cfg = {
            key: value
            for key, value in raw_cfg.items()
            if key not in _OBSOLETE_CONFIG_KEYS
        }
        cfg["_max_containers"] = max_containers
        cfg["_max_gpu_containers"] = max_gpu_containers
        configured[step_name] = cfg
    return configured


def _ppiflow_input_fields(args: object) -> tuple[str, ...]:
    if isinstance(args, ppiflow_app.SampleAntibodyNanobodyConfig):
        return ("antigen_pdb", "framework_pdb")
    if isinstance(args, ppiflow_app.SampleAntibodyNanobodyPartialConfig):
        return ("complex_pdb",)
    if isinstance(
        args,
        (ppiflow_app.SampleBinderConfig, ppiflow_app.SampleBinderPartialConfig),
    ):
        return ("input_pdb",)
    raise TypeError(f"Unsupported PPIFlow args type: {type(args).__name__}")


def _active_ppiflow_app_steps(
    task_doc: dict[str, Any], stage: int | None
) -> tuple[str, ...]:
    """Return PPIFlow app steps that should be staged for the selected run."""
    if stage not in {None, 1, 2}:
        raise ValueError("stage must be omitted, 1, or 2")
    enabled = _enabled_section(task_doc)
    active_steps: list[str] = []
    if stage in {None, 1} and _step_enabled(enabled, "PPIFlowStep"):
        active_steps.append("PPIFlowStep")
    return tuple(active_steps)


def _stage_ppiflow_app_inputs(
    *,
    steps_doc: dict[str, Any],
    run_id: str,
    app_steps: tuple[str, ...],
) -> dict[str, Any]:
    """Upload local PPIFlow app inputs and rewrite step args to mounted paths."""
    staged_steps = deepcopy(steps_doc)
    uploads: list[tuple[Path, str]] = []
    volume_root = Path(ppiflow_app.CONF.output_volume_mountpoint)

    for step_name in app_steps:
        if step_name not in staged_steps:
            continue
        cfg = _step_cfg(staged_steps, step_name)
        raw_args = cfg.get("args", cfg)
        if not isinstance(raw_args, dict):
            continue

        app_args = ppiflow_app.PPIFlowArgs.model_validate({"args": raw_args})
        input_digests: dict[str, str] = {}
        for field_name in _ppiflow_input_fields(app_args.args):
            current_value = getattr(app_args.args, field_name)
            current_path = Path(current_value)
            if current_path.is_absolute() and current_path.is_relative_to(volume_root):
                remote_storage = volume_path_from_mount_path(
                    str(current_path),
                    str(volume_root),
                    ppiflow_app.CONF.output_volume_name,
                )
                digest = hashlib.sha256()
                for chunk in ppiflow_app.CONF.output_volume.read_file(
                    remote_storage.path
                ):
                    digest.update(chunk)
                input_digests[field_name] = digest.hexdigest()
                continue

            local_path = current_path.expanduser().resolve()
            if not local_path.exists():
                raise FileNotFoundError(
                    f"PPIFlow {step_name} input {field_name!r} was not found "
                    f"locally or in the mounted output volume: {current_value}"
                )

            with local_path.open("rb") as input_file:
                content_sha256 = hashlib.file_digest(input_file, "sha256").hexdigest()
            input_digests[field_name] = content_sha256
            suffix = "".join(part.lower() for part in local_path.suffixes)
            remote_rel = (
                Path(run_id)
                / sanitize_filename(step_name)
                / sanitize_filename(field_name)
                / f"{content_sha256}{suffix}"
            )
            raw_args[field_name] = str(volume_root / remote_rel)
            uploads.append((local_path, remote_rel.as_posix()))
        if input_digests:
            cfg[_INPUT_DIGESTS_KEY] = input_digests

    if uploads:
        with ppiflow_app.CONF.output_volume.batch_upload(force=True) as batch:
            for local_path, remote_rel in uploads:
                remote_storage = volume_path_from_mount_path(
                    str(volume_root / remote_rel),
                    str(volume_root),
                    ppiflow_app.CONF.output_volume_name,
                )
                print(
                    f"Uploading PPIFlow input '{local_path}' to {remote_storage}",
                    flush=True,
                )
                batch.put_file(local_path, f"/{remote_storage.path}")
    return staged_steps


@app.local_entrypoint()
def submit_ppiflow_workflow(
    task_yaml: str,
    steps_yaml: str,
    run_id: str | None = None,
    stage: int | None = None,
    wait: bool = True,
    max_containers: int | None = None,
    max_gpu_containers: int | None = None,
    dry_run: bool = False,
    use_deployed_coordinator: bool = False,
    deployment_environment: str = "main",
    deployment_name: str | None = None,
    deployment_version: int = 1,
    restart_from: str | None = None,
) -> None:
    """Build and submit a PPIFlow workflow from task and step YAML files.

    Args:
        task_yaml: Path to the PPIFlow task YAML declaring enabled workflow
            steps and design mode.
        steps_yaml: Path to the YAML file containing per-step app arguments.
        run_id: Stable workflow run id for durable ledger state. Defaults to
            the task YAML filename stem.
        stage: Optional stage selector. Use 1 for stage 1 only, 2 for stage 2
            only, or omit to build both stages.
        wait: Wait locally for the remote workflow result. Disable to print the
            Modal function call id for asynchronous collection.
        max_containers: Maximum active workload containers for this Run.
        max_gpu_containers: Maximum active GPU workload containers within the
            total container limit.
        dry_run: Print the workflow DAG graph and skip orchestrator execution.
        use_deployed_coordinator: Submit through an exact named deployment.
        deployment_environment: Modal Environment containing the deployment.
        deployment_name: Modal app deployment name. Defaults to this workflow.
        deployment_version: Exact numeric Modal deployment version.
        restart_from: Optional predecessor Execution Run ID for a Successor Run.
    """
    predecessor_execution_run_id = None if restart_from is None else UUID(restart_from)
    if predecessor_execution_run_id is not None and not use_deployed_coordinator:
        raise ValueError("restart_from requires an exact deployed workflow coordinator")
    task_yaml_path = Path(task_yaml).expanduser().resolve()
    steps_yaml_path = Path(steps_yaml).expanduser().resolve()
    resolved_run_id = sanitize_filename(run_id or task_yaml_path.stem)
    task_yaml_bytes = task_yaml_path.read_bytes()
    steps_yaml_bytes = steps_yaml_path.read_bytes()
    task_doc = _load_yaml_bytes(task_yaml_bytes)
    total_limit, gpu_limit = resolve_provider_call_limits(
        default_max_containers=16,
        default_max_gpu_containers=16,
        max_containers=max_containers,
        max_gpu_containers=max_gpu_containers,
    )
    if dry_run:
        workflow = build_ppiflow_workflow(
            task_yaml_bytes=task_yaml_bytes,
            steps_yaml_bytes=steps_yaml_bytes,
            stage=stage,
            max_containers=total_limit,
            max_gpu_containers=gpu_limit,
        )
        print_workflow_dag(workflow.validate())
        return

    steps_doc = _stage_ppiflow_app_inputs(
        steps_doc=_load_yaml_bytes(steps_yaml_bytes),
        run_id=resolved_run_id,
        app_steps=_active_ppiflow_app_steps(task_doc, stage),
    )
    steps_doc = _inline_rosetta_config_files(steps_doc)
    steps_doc = _bind_stage2_input_identity(
        task_doc=task_doc,
        steps_doc=steps_doc,
        stage=stage,
    )
    workflow = build_ppiflow_workflow(
        task_yaml_bytes=yaml.safe_dump(task_doc).encode("utf-8"),
        steps_yaml_bytes=yaml.safe_dump(steps_doc).encode("utf-8"),
        stage=stage,
        max_containers=total_limit,
        max_gpu_containers=gpu_limit,
    )

    execution_run_id = uuid4()
    deployment = DeploymentIdentity(
        environment=(
            deployment_environment if use_deployed_coordinator else "development"
        ),
        deployment_name=(
            (deployment_name or CONF.name) if use_deployed_coordinator else CONF.name
        ),
        deployment_version=deployment_version if use_deployed_coordinator else 1,
    )
    coordinator = orchestrator.execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
    )
    orchestrator_kwargs = {
        "graph": workflow,
        "workload_run_key": resolved_run_id,
        "max_parallel_nodes": total_limit,
        "max_active_provider_calls": total_limit,
        "max_active_gpu_provider_calls": gpu_limit,
        "strict_external_artifact_checks": True,
        "external_artifact_checker_function_name": ("check_ppiflow_external_artifact"),
    }
    if not use_deployed_coordinator:
        orchestrator_kwargs["development_function_handles"] = {
            "validate_ppiflow_model": validate_ppiflow_model,
            "run_ppiflow_design_stage": run_ppiflow_design_stage,
            "run_ppiflow_partial_candidate": run_ppiflow_partial_candidate,
            "run_ppiflow_ligandmpnn_candidate": run_ppiflow_ligandmpnn_candidate,
            "run_ppiflow_flowpacker_stage": run_ppiflow_flowpacker_stage,
            "prepare_ppiflow_af3score_stage": prepare_ppiflow_af3score_stage,
            "run_ppiflow_af3score_batch": run_ppiflow_af3score_batch,
            "postprocess_ppiflow_af3score_stage": (postprocess_ppiflow_af3score_stage),
            "prepare_ppiflow_rosetta_stage": prepare_ppiflow_rosetta_stage,
            "run_ppiflow_rosetta_worker": run_ppiflow_rosetta_worker,
            "finalize_ppiflow_rosetta_stage": finalize_ppiflow_rosetta_stage,
            "run_ppiflow_refold_candidate": run_ppiflow_refold_candidate,
            "run_ppiflow_dockq_stage": run_ppiflow_dockq_stage,
            "filter_ppiflow_artifacts": filter_ppiflow_artifacts,
            "derive_ppiflow_fixed_positions": derive_ppiflow_fixed_positions,
            "rank_ppiflow_artifacts": rank_ppiflow_artifacts,
            "normalize_ppiflow_stage2_input": normalize_ppiflow_stage2_input,
            "check_ppiflow_external_artifact": check_ppiflow_external_artifact,
        }
    print(
        f"Submitting PPIFlow workflow '{resolved_run_id}' with "
        f"{len(workflow.validate().nodes)} node(s)",
        flush=True,
    )
    function_call = orchestrator.submit_workflow_run(
        coordinator,
        execution_run_id=execution_run_id,
        deployment=deployment,
        predecessor_execution_run_id=predecessor_execution_run_id,
        coordinator_kwargs=orchestrator_kwargs,
    )
    print(
        "Coordinator FunctionCall ID: "
        f"{getattr(function_call, 'object_id', function_call)}",
        flush=True,
    )
    if wait:
        result: AppRunResult | str = AppRunResult.model_validate(function_call.get())
    else:
        result = str(getattr(function_call, "object_id", function_call))
    if isinstance(result, AppRunResult):
        print(f"PPIFlow workflow run finished with status: {result.status}", flush=True)
    else:
        print(f"PPIFlow workflow run submitted. FunctionCall id: {result}", flush=True)
