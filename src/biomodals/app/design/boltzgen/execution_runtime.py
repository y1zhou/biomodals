"""BoltzGen Nodes for the shared execution graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biomodals.app.design.boltzgen.execution_contracts import (
    boltzgen_run_root,
    is_boltzgen_run_complete,
    load_collection_publication,
)
from biomodals.app.design.boltzgen.execution_request import (
    COLLECT_RESULTS_NODE,
    DESIGN_RUNS_NODE,
    BoltzGenExecutionRequest,
)
from biomodals.execution import (
    AvailabilityStatus,
    ExecutionArtifact,
    ExecutionGraph,
    ExecutionPlanMetadata,
)
from biomodals.execution.nodes import (
    NodeRunContext,
    ProviderCallSpec,
    ProviderNode,
    TaskDefinition,
    TaskProviderNode,
)
from biomodals.schema import AppRunResult, AppRunStatus


@dataclass
class _DesignRunsNode(TaskProviderNode):
    request: BoltzGenExecutionRequest
    output_root: Path

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        del context
        return tuple(
            TaskDefinition(
                task_key=run_id,
                scientific_payload={"run_id": run_id},
                execution_payload={"run_id": run_id},
            )
            for run_id in self.request.run_ids
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        del context
        run_id = task.task_key
        return ProviderCallSpec(
            function_name="run_boltzgen_task",
            uses_gpu=True,
            runtime_image_key="boltzgen-gpu",
            kwargs={
                "out_dir": str(
                    boltzgen_run_root(
                        self.output_root,
                        self.request.run_name,
                        run_id,
                    )
                ),
                "input_yaml_path": str(
                    self.output_root.joinpath(*self.request.config_path.parts)
                ),
                "protocol": self.request.protocol,
                "num_designs": self.request.num_designs,
                "steps": self.request.steps,
                "extra_args": self.request.extra_args,
                "replace_claim_owner": dict(self.request.replace_claim_owners).get(
                    run_id
                ),
                "task_fingerprint": self.request.design_task_fingerprints[run_id],
            },
            metadata={"run_id": run_id},
            provider_call_id_kwarg="claim_owner",
        )

    def process_remote_task_result(
        self,
        task_key: str,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del result, metadata
        publication = self._publication(task_key)
        if publication is None:
            raise FileNotFoundError(
                f"BoltzGen design publication is unavailable: {task_key}"
            )
        return publication

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del context
        return self._publication(task.task_key, expected_fingerprint)

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, result, artifacts
        try:
            available = self._publication(task.task_key, expected_fingerprint)
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return (
            AvailabilityStatus.AVAILABLE
            if available is not None
            else AvailabilityStatus.MISSING
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        del context, results
        if errors:
            return AppRunResult(
                status=AppRunStatus.FAILED,
                warnings=list(errors.values()),
            )
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def _publication(
        self,
        run_id: str,
        task_fingerprint: str | None = None,
    ) -> AppRunResult | None:
        fingerprint = (
            self.request.design_task_fingerprints[run_id]
            if task_fingerprint is None
            else task_fingerprint
        )
        available = is_boltzgen_run_complete(
            boltzgen_run_root(
                self.output_root,
                self.request.run_name,
                run_id,
            ),
            task_fingerprint=fingerprint,
        )
        return AppRunResult(status=AppRunStatus.SUCCEEDED) if available else None


@dataclass
class _CollectResultsNode(ProviderNode):
    request: BoltzGenExecutionRequest
    output_root: Path

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        return ProviderCallSpec(
            function_name="collect_boltzgen_data",
            uses_gpu=False,
            runtime_image_key="boltzgen-cpu",
            kwargs={
                "run_name": self.request.run_name,
                "run_ids": list(self.request.run_ids),
                "task_fingerprints": self.request.design_task_fingerprints,
                "protocol": self.request.protocol,
                "num_designs": self.request.num_designs,
                "budget": self.request.budget,
                "steps": self.request.steps,
                "extra_args": self.request.extra_args,
                "filter_results": self.request.filter_results,
                "filter_rmsd_threshold": self.request.filter_rmsd_threshold,
                "publication_path": (
                    self.request.collection_publication_path.as_posix()
                ),
            },
        )

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del result, metadata
        publication = self._publication()
        if publication is None:
            raise FileNotFoundError("BoltzGen collection publication is unavailable")
        return publication

    def recover_result_publication(
        self,
        context: NodeRunContext,
    ) -> AppRunResult | None:
        del context
        return self._publication()

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, result, artifacts
        try:
            available = self._publication()
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return (
            AvailabilityStatus.AVAILABLE
            if available is not None
            else AvailabilityStatus.MISSING
        )

    def _publication(self) -> AppRunResult | None:
        publication = load_collection_publication(
            self.output_root,
            self.request.collection_publication_path,
            run_name=self.request.run_name,
            run_ids=self.request.run_ids,
            task_fingerprints=self.request.design_task_fingerprints,
        )
        return (
            AppRunResult(status=AppRunStatus.SUCCEEDED)
            if publication is not None
            else None
        )


def boltzgen_execution_graph(
    request: BoltzGenExecutionRequest,
    *,
    output_root: str | Path,
) -> ExecutionGraph:
    """Build BoltzGen's independent design fan-out and collection graph."""
    graph = ExecutionGraph(
        "boltzgen",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="boltzgen",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    designs = graph.add_node(
        _DesignRunsNode(request, Path(output_root)),
        id=DESIGN_RUNS_NODE,
    )
    graph.add_node(
        _CollectResultsNode(request, Path(output_root)),
        id=COLLECT_RESULTS_NODE,
        depends_on=[designs],
    )
    return graph
