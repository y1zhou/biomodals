"""Rosetta pull-worker Node for the shared execution graph."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from biomodals.app.bioinfo.rosetta.execution_contracts import (
    RosettaTaskSpec,
    validate_task_publication,
)
from biomodals.app.bioinfo.rosetta.execution_request import (
    ROSETTA_TASKS_NODE,
    RosettaExecutionRequest,
)
from biomodals.execution import (
    AvailabilityStatus,
    ExecutionArtifact,
    ExecutionGraph,
    ExecutionPlanMetadata,
)
from biomodals.execution.nodes import (
    NodeRunContext,
    PullTaskProviderNode,
    PullWorkerCallSpec,
    TaskDefinition,
)
from biomodals.schema import AppRunResult, AppRunStatus


@dataclass
class _RosettaTasksNode(PullTaskProviderNode):
    """Execute independent Rosetta commands through bounded pull workers."""

    request: RosettaExecutionRequest
    output_root: Path

    @property
    def run_root(self) -> Path:
        return self.output_root / self.request.workload_run_key

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        del context
        return tuple(
            TaskDefinition(
                task_key=task.task_key,
                scientific_payload=task.scientific_payload,
                execution_payload=task.to_dict(),
            )
            for task in self.request.tasks
        )

    def prepare_pull_worker(
        self,
        context: NodeRunContext,
    ) -> PullWorkerCallSpec:
        del context
        return PullWorkerCallSpec(
            function_name="run_rosetta_worker",
            uses_gpu=False,
            claim_capacity=self.request.claim_capacity,
            max_worker_calls=self.request.max_active_provider_calls,
            kwargs={
                "run_name": self.request.run_name,
                "run_id": self.request.run_id,
                "claim_capacity": self.request.claim_capacity,
                "max_parallel": self.request.max_parallel_per_worker,
            },
            runtime_image_key="rosetta-cpu",
            compatibility_key="rosetta-worker",
        )

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
            available = validate_task_publication(
                self.run_root,
                RosettaTaskSpec.from_dict(task.execution_payload),
                expected_fingerprint,
            )
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del context
        if not validate_task_publication(
            self.run_root,
            RosettaTaskSpec.from_dict(task.execution_payload),
            expected_fingerprint,
        ):
            return None
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        del context, results
        return AppRunResult(
            status=(AppRunStatus.SUCCEEDED if not errors else AppRunStatus.FAILED),
            warnings=list(errors.values()),
        )


def rosetta_execution_graph(
    request: RosettaExecutionRequest,
    *,
    output_root: str | Path,
) -> ExecutionGraph:
    """Build Rosetta's single pull-worker Execution Definition."""
    graph = ExecutionGraph(
        "rosetta",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="rosetta",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    graph.add_node(
        _RosettaTasksNode(request, Path(output_root)),
        id=ROSETTA_TASKS_NODE,
    )
    return graph
