"""AlphaFold3 Nodes for the shared execution graph."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from time import monotonic
from typing import Any, cast
from uuid import UUID

from biomodals.app.fold.alphafold3.environment import (
    EnvironmentAsset,
    EnvironmentRuntime,
    acquire_asset_claim,
    asset_ready,
    fail_asset_claim_if_current,
    required_environment_assets,
)
from biomodals.app.fold.alphafold3.execution_planning import (
    INFERENCE_SUMMARY,
    MSA_ASSEMBLIES,
    PREPARE_ENVIRONMENT,
    RAW_SEARCHES,
    REQUEST_PUBLICATION,
    SEED_PREDICTIONS,
    STAGE_INFERENCE,
    STAGE_REQUEST,
    TEMPLATE_SEARCHES,
    AlphaFold3ExecutionPlanning,
)
from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
)
from biomodals.app.fold.alphafold3.msa_search import SearchRuntime
from biomodals.app.fold.alphafold3.seed_predictions import InferenceRuntime
from biomodals.app.fold.alphafold3.template_search import TemplateRuntime
from biomodals.execution import (
    AvailabilityStatus,
    ExecutionArtifact,
    ExecutionGraph,
    ExecutionPlanMetadata,
    PreparedTaskBatch,
)
from biomodals.execution.nodes import (
    CoordinatorNode,
    NodeRunContext,
    ProviderCallSpec,
    ProviderNode,
    TaskDefinition,
    TaskProviderNode,
)
from biomodals.schema import AppRunResult, AppRunStatus

LOGGER = logging.getLogger(__name__)


@dataclass
class _StageRequestNode(CoordinatorNode):
    """Represent validation already performed by the request loader."""

    def run(self, context: NodeRunContext) -> AppRunResult:
        del context
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def recover_result_publication(
        self,
        context: NodeRunContext,
    ) -> AppRunResult | None:
        return self.run(context)


@dataclass
class _PrepareEnvironmentNode(TaskProviderNode):
    """Provision only the shared assets required by this request."""

    request: AlphaFold3ExecutionRequest = field(
        repr=False,
        metadata={"dag_hash": False},
    )
    runtime: EnvironmentRuntime = field(repr=False, metadata={"dag_hash": False})
    execution_run_id: UUID = field(metadata={"dag_hash": False})
    blocked_until: dict[str, float] = field(
        default_factory=dict,
        repr=False,
        metadata={"dag_hash": False},
    )

    def _assets(self) -> tuple[EnvironmentAsset, ...]:
        return required_environment_assets(
            self.request.config,
            search_msa=self.request.search_msa,
            search_protein_templates=self.request.search_protein_templates,
        )

    def _asset(self, task_key: str) -> EnvironmentAsset:
        try:
            return next(asset for asset in self._assets() if asset.key == task_key)
        except StopIteration as error:
            raise RuntimeError(
                "Persisted environment asset identity changed"
            ) from error

    def _generation_id(self, task_key: str) -> str:
        return sha256(
            f"{self.execution_run_id}:{PREPARE_ENVIRONMENT}:{task_key}".encode()
        ).hexdigest()

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        del context
        return tuple(
            TaskDefinition(
                task_key=asset.key,
                scientific_payload=asset.to_record(),
                execution_payload={"asset_kind": asset.kind},
            )
            for asset in self._assets()
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        del context
        asset = self._asset(task.task_key)
        return ProviderCallSpec(
            function_name="prepare_alphafold3_environment_asset",
            uses_gpu=False,
            kwargs={
                "asset_record": asset.to_record(),
                "generation_id": self._generation_id(asset.key),
            },
            runtime_image_key="alphafold3-environment-setup",
        )

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[TaskDefinition, ...],
    ) -> ProviderCallSpec | PreparedTaskBatch:
        if len(tasks) != 1:
            raise ValueError("Environment assets are submitted independently")
        task = tasks[0]
        asset = self._asset(task.task_key)
        if monotonic() < self.blocked_until.get(asset.key, 0):
            return PreparedTaskBatch(call=None)
        self.runtime.volume_for(asset).reload()
        if asset_ready(self.runtime, asset):
            return PreparedTaskBatch(
                call=None,
                completed={task.task_key: AppRunResult(status=AppRunStatus.SUCCEEDED)},
            )
        claim = acquire_asset_claim(
            self.runtime,
            asset,
            self._generation_id(asset.key),
        )
        if claim is None:
            self.blocked_until[asset.key] = monotonic() + 30
            return PreparedTaskBatch(call=None)
        self.blocked_until.pop(asset.key, None)
        return self.prepare_remote_task(context, task)

    def process_remote_task_result(
        self,
        task_key: str,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del metadata
        value = cast(dict[str, object], result)
        if value.get("asset_key") != task_key or value.get("status") not in {
            "published",
            "reused",
        }:
            raise RuntimeError(f"Invalid environment setup result: {result!r}")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del context, expected_fingerprint
        asset = self._asset(task.task_key)
        self.runtime.volume_for(asset).reload()
        if not asset_ready(self.runtime, asset):
            return None
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        del context, results
        for task_key, message in errors.items():
            asset = self._asset(task_key)
            fail_asset_claim_if_current(
                self.runtime,
                asset,
                self._generation_id(task_key),
                detail={
                    "asset_key": asset.key,
                    "error_type": "ProviderCallFailed",
                    "message": message,
                },
            )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED if not errors else AppRunStatus.FAILED,
            warnings=list(errors.values()),
        )

    def finalize_cancelled_remote_tasks(self, context: NodeRunContext) -> None:
        del context
        for asset in self._assets():
            try:
                fail_asset_claim_if_current(
                    self.runtime,
                    asset,
                    self._generation_id(asset.key),
                    detail={
                        "asset_key": asset.key,
                        "error_type": "ProviderCallCancelled",
                    },
                )
            except Exception:
                LOGGER.warning(
                    "Could not release cancelled AlphaFold3 setup claim %s",
                    asset.key,
                    exc_info=True,
                )


@dataclass
class _AlphaFold3TaskNode(TaskProviderNode):
    """Execute one deterministic CPU-backed AlphaFold3 Task collection."""

    stage: str
    planning: AlphaFold3ExecutionPlanning = field(
        repr=False,
        metadata={"dag_hash": False},
    )

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def refresh_result_storage(self) -> None:
        self.planning.refresh_result_storage(self.stage)

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        del context
        return tuple(
            TaskDefinition(
                task_key=item.plan.task_key,
                scientific_payload=item.plan.scientific_payload,
                execution_payload=item.plan.execution_payload,
            )
            for item in self.planning.planned_tasks(self.stage)
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        del context
        return self.planning.task_call(
            self.stage,
            self.planning.planned_task(self.stage, task.task_key),
        )

    def process_remote_task_result(
        self,
        task_key: str,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del metadata
        return self.planning.process_task_result(
            self.stage,
            self.planning.planned_task(self.stage, task_key),
            result,
        )

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del context
        item = self.planning.planned_task(self.stage, task.task_key)
        if self.planning.task_fingerprint(self.stage, item) != expected_fingerprint:
            raise RuntimeError("Persisted AlphaFold3 Task identity changed")
        return self.planning.recover_task_result(self.stage, item)

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del result, artifacts
        try:
            publication = self.recover_remote_task_result(
                context,
                task,
                expected_fingerprint,
            )
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return (
            AvailabilityStatus.AVAILABLE
            if publication is not None
            else AvailabilityStatus.MISSING
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        del context, results
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED if not errors else AppRunStatus.FAILED,
            warnings=list(errors.values()),
        )


@dataclass
class _SeedPredictionNode(_AlphaFold3TaskNode):
    """Batch and claim independent GPU seed predictions after preflight."""

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        del context
        return self.planning.seed_call_preview(
            self.planning.planned_task(self.stage, task.task_key)
        )

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[TaskDefinition, ...],
    ) -> ProviderCallSpec | PreparedTaskBatch:
        del context
        return self.planning.prepare_seed_batch(
            tuple(
                self.planning.planned_task(self.stage, task.task_key) for task in tasks
            )
        )

    def process_remote_task_batch_result(
        self,
        task_keys: tuple[str, ...],
        result: Any,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, AppRunResult]:
        del result, metadata
        self.planning.invalidate({SEED_PREDICTIONS})
        publications = {}
        for task_key in task_keys:
            publication = self.planning.recover_task_result(
                self.stage,
                self.planning.planned_task(self.stage, task_key),
            )
            publications[task_key] = (
                publication
                if publication is not None
                else AppRunResult(
                    status=AppRunStatus.FAILED,
                    warnings=[
                        f"AlphaFold3 seed publication is unavailable: {task_key}"
                    ],
                )
            )
        return publications


@dataclass
class _StageInferenceNode(CoordinatorNode):
    """Publish the immutable enriched input consumed by GPU workers."""

    planning: AlphaFold3ExecutionPlanning = field(
        repr=False,
        metadata={"dag_hash": False},
    )

    def run(self, context: NodeRunContext) -> AppRunResult:
        del context
        return self.planning.stage_inference()

    def recover_result_publication(
        self,
        context: NodeRunContext,
    ) -> AppRunResult | None:
        del context
        observation = self.planning.staged_inference_observation()
        if observation == AvailabilityStatus.UNKNOWN:
            raise OSError("Could not inspect the staged AlphaFold3 input")
        if observation == AvailabilityStatus.MISSING:
            return None
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, result, artifacts
        return self.planning.staged_inference_observation()


@dataclass
class _InferenceSummaryNode(ProviderNode):
    """Publish the accumulated summary after every requested seed."""

    planning: AlphaFold3ExecutionPlanning = field(
        repr=False,
        metadata={"dag_hash": False},
    )

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def refresh_result_storage(self) -> None:
        self.planning.invalidate({SEED_PREDICTIONS})

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        return self.planning.summary_call()

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del metadata
        publication = self.planning.summary_result(result)
        if publication is None:
            raise FileNotFoundError("AlphaFold3 inference summary is unavailable")
        return publication

    def recover_result_publication(
        self,
        context: NodeRunContext,
    ) -> AppRunResult | None:
        del context
        return self.planning.summary_result()

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, result, artifacts
        try:
            publication = self.planning.summary_result()
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return (
            AvailabilityStatus.AVAILABLE
            if publication is not None
            else AvailabilityStatus.MISSING
        )


@dataclass
class _RequestPublicationNode(ProviderNode):
    """Publish the request view and its terminal invocation receipt."""

    planning: AlphaFold3ExecutionPlanning = field(
        repr=False,
        metadata={"dag_hash": False},
    )

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        return self.planning.request_call()

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del metadata
        publication = self.planning.request_result(result)
        if publication is None:
            raise FileNotFoundError("AlphaFold3 request publication is unavailable")
        return publication

    def recover_result_publication(
        self,
        context: NodeRunContext,
    ) -> AppRunResult | None:
        del context
        return self.planning.request_result()

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, result, artifacts
        try:
            publication = self.planning.request_result()
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return (
            AvailabilityStatus.AVAILABLE
            if publication is not None
            else AvailabilityStatus.MISSING
        )


def alphafold3_execution_graph(
    request: AlphaFold3ExecutionRequest,
    *,
    execution_run_id: UUID,
    output_volume: Any,
    search_runtime: SearchRuntime,
    template_runtime: TemplateRuntime,
    inference_runtime: InferenceRuntime,
    environment_runtime: EnvironmentRuntime,
) -> ExecutionGraph:
    """Build AlphaFold3's staged search, inference, and publication graph."""
    planning = AlphaFold3ExecutionPlanning(
        request=request,
        execution_run_id=execution_run_id,
        output_volume=output_volume,
        search_runtime=search_runtime,
        template_runtime=template_runtime,
        inference_runtime=inference_runtime,
    )
    graph = ExecutionGraph(
        "alphafold3",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="alphafold3",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    previous = graph.add_node(_StageRequestNode(), id=STAGE_REQUEST)
    previous = graph.add_node(
        _PrepareEnvironmentNode(request, environment_runtime, execution_run_id),
        id=PREPARE_ENVIRONMENT,
        depends_on=[previous],
        allow_empty_result=True,
    )
    previous = graph.add_node(
        _AlphaFold3TaskNode(RAW_SEARCHES, planning),
        id=RAW_SEARCHES,
        depends_on=[previous],
        allow_empty_result=True,
    )
    previous = graph.add_node(
        _AlphaFold3TaskNode(MSA_ASSEMBLIES, planning),
        id=MSA_ASSEMBLIES,
        depends_on=[previous],
        allow_empty_result=True,
    )
    previous = graph.add_node(
        _AlphaFold3TaskNode(TEMPLATE_SEARCHES, planning),
        id=TEMPLATE_SEARCHES,
        depends_on=[previous],
        allow_empty_result=True,
    )
    previous = graph.add_node(
        _StageInferenceNode(planning),
        id=STAGE_INFERENCE,
        depends_on=[previous],
    )
    previous = graph.add_node(
        _SeedPredictionNode(SEED_PREDICTIONS, planning),
        id=SEED_PREDICTIONS,
        depends_on=[previous],
    )
    previous = graph.add_node(
        _InferenceSummaryNode(planning),
        id=INFERENCE_SUMMARY,
        depends_on=[previous],
    )
    graph.add_node(
        _RequestPublicationNode(planning),
        id=REQUEST_PUBLICATION,
        depends_on=[previous],
    )
    return graph


def _result_envelope(result: object) -> dict[str, object]:
    """Retain the bounded provider-result diagnostic used by app tests."""
    if isinstance(result, dict):
        envelope = cast(dict[str, object], result)
        if isinstance(envelope.get("execution_result"), dict):
            return envelope
    return {"invalid_result": repr(result)[:4096]}
