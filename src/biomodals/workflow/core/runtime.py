"""Local workflow runtime scheduler."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import biomodals.workflow.core.display as workflow_display
from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    NodeStatus,
    RunStatus,
    WorkflowArtifact,
)
from biomodals.workflow.core._runtime import availability, bootstrap, scheduler
from biomodals.workflow.core._runtime.node_runner import NodeRunner
from biomodals.workflow.core._runtime.remote_calls import RemoteCallManager
from biomodals.workflow.core._runtime.services import RuntimeServices
from biomodals.workflow.core._runtime.volume_sync import (
    WorkflowVolume,
    WorkflowVolumeSync,
)
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.ledger import WorkflowLedger
from biomodals.workflow.core.nodes import (
    RemoteFunctionCall,
)

FunctionCallResolver = Callable[[str], RemoteFunctionCall]


class WorkflowRuntime:
    """Local runtime core for scheduling workflow nodes against a ledger."""

    def __init__(
        self,
        *,
        workflow: Workflow,
        volume_root: str | Path,
        workflow_volume_name: str,
        workflow_volume: WorkflowVolume | None = None,
        function_call_resolver: FunctionCallResolver | None = None,
        remote_call_poll_timeout: float | int = 0,
        max_ready_workers: int = 32,
    ):
        """Initialize a runtime for one workflow and ledger root."""
        self.workflow = workflow
        volume_root_path = Path(volume_root)
        self.ledger = WorkflowLedger(volume_root_path)
        # TODO: Replace this debug-only wave history with structured scheduling
        # diagnostics that can record ready/completed/blocked reasons and timing
        # without expanding the public workflow API.
        self.executed_waves: list[list[str]] = []
        self._volume_sync = WorkflowVolumeSync(
            workflow_volume=workflow_volume,
            ledger=self.ledger,
        )
        self._services = RuntimeServices(
            ledger=self.ledger,
            volume_root=volume_root_path,
            workflow_volume_name=workflow_volume_name,
            volume_sync=self._volume_sync,
        )
        self._remote_calls = RemoteCallManager(
            services=self._services,
            function_call_resolver=function_call_resolver,
            remote_call_poll_timeout=remote_call_poll_timeout,
        )
        self._node_runner = NodeRunner(
            services=self._services,
            remote_calls=self._remote_calls,
            node_is_complete=self._node_is_complete,
            max_ready_workers=max_ready_workers,
        )

    def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
        """Run the workflow until every node succeeds or no progress is possible."""
        definition = self.workflow.validate()
        workflow_display.print_workflow_message(
            f"[workflow] Starting workflow '{definition.name}' run '{run_id}' "
            f"with {len(definition.nodes)} node(s)",
            style="bold cyan",
        )
        workflow_display.print_workflow_dag(definition)
        bootstrap.start_run(
            definition,
            run_id=run_id,
            force=force,
            services=self._services,
        )

        while True:
            decision = scheduler.evaluate_progress(
                definition,
                ledger=self.ledger,
                node_is_complete=self._node_is_complete,
            )
            if decision.status == scheduler.SchedulerDecisionStatus.SUCCEEDED:
                self.ledger.mark_run_status(RunStatus.SUCCEEDED)
                self._volume_sync.commit()
                return AppRunResult(status=AppRunStatus.SUCCEEDED)

            if decision.status == scheduler.SchedulerDecisionStatus.BLOCKED_RUNNING:
                self._volume_sync.commit()
                return AppRunResult(
                    status=AppRunStatus.PARTIAL,
                    warnings=decision.warnings,
                )

            if decision.status == scheduler.SchedulerDecisionStatus.FAILED_NO_PROGRESS:
                self.ledger.mark_run_status(RunStatus.FAILED)
                self._volume_sync.commit()
                return AppRunResult(
                    status=AppRunStatus.FAILED,
                    warnings=decision.warnings,
                )

            self.executed_waves.append(decision.ready)
            for node_id, node_result in self._node_runner.run_ready_nodes(
                definition,
                decision.ready,
            ):
                if node_result.status in {AppRunStatus.FAILED, AppRunStatus.PARTIAL}:
                    error = self._node_error_message(node_result)
                    node_status = self.ledger.load_node_status(node_id)
                    if node_status.status != NodeStatus.FAILED or not node_status.error:
                        self.ledger.mark_node_failed(node_id, error)
                    self.ledger.mark_run_status(RunStatus.FAILED)
                    self._volume_sync.commit()
                    return AppRunResult(
                        status=node_result.status,
                        warnings=node_result.warnings or [error],
                    )

    def _node_is_complete(self, node_id: str) -> bool:
        if not self.ledger.node_is_complete(node_id):
            return False
        errors = [
            error
            for artifact in self.ledger.load_node_output_artifacts(node_id)
            for error in self._artifact_availability_errors(artifact)
        ]
        if not errors:
            return True
        workflow_display.print_workflow_message(
            "[workflow] Node output artifacts unavailable: "
            f"{node_id}: {'; '.join(errors)}",
            style="yellow",
        )
        return False

    def _artifact_availability_errors(self, artifact: WorkflowArtifact) -> list[str]:
        return availability.format_artifact_availability_errors(
            availability.artifact_availability_errors(
                artifact,
                workflow_volume_name=self._services.workflow_volume_name,
                volume_root=self._services.volume_root,
                run_root=self.ledger.run_root,
            )
        )

    def cancel_active_remote_calls(self, *, terminate_containers: bool = True) -> None:
        """Cancel Modal function calls spawned by this runtime instance."""
        self._remote_calls.cancel_active(terminate_containers=terminate_containers)

    @staticmethod
    def _node_error_message(result: AppRunResult) -> str:
        if result.warnings:
            return result.warnings[0]
        if result.status == AppRunStatus.PARTIAL:
            return "Node returned partial status"
        return "Node returned failed status"

    def close(self) -> None:
        """Close durable local resources owned by the runtime."""
        self.ledger.close()
