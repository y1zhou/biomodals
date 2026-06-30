"""Local workflow runtime scheduler."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
from biomodals.workflow.core._runtime.diagnostics import RuntimeDiagnostics
from biomodals.workflow.core._runtime.node_runner import NodeRunner
from biomodals.workflow.core._runtime.remote_calls import RemoteCallManager
from biomodals.workflow.core._runtime.volume_sync import (
    WorkflowVolume,
    WorkflowVolumeSync,
)
from biomodals.workflow.core.artifact_availability import (
    ExternalArtifactChecker,
    mounted_volume_checker,
)
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.ledger import WorkflowLedger
from biomodals.workflow.core.nodes import RemoteFunctionCall

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
        strict_external_artifact_checks: bool = False,
        external_artifact_checker: ExternalArtifactChecker | None = None,
        external_volume_roots: Mapping[str, str | Path] | None = None,
    ):
        """Initialize a runtime for one workflow and ledger root."""
        if strict_external_artifact_checks:
            if external_artifact_checker is None and external_volume_roots is None:
                raise ValueError(
                    "strict_external_artifact_checks requires "
                    "external_artifact_checker or external_volume_roots"
                )
            if external_artifact_checker is None and external_volume_roots is not None:
                external_artifact_checker = mounted_volume_checker(
                    workflow_volume_name=workflow_volume_name,
                    volume_roots=external_volume_roots,
                )
        else:
            external_artifact_checker = None
        self.workflow = workflow
        volume_root_path = Path(volume_root)
        self.volume_root = volume_root_path
        self.workflow_volume_name = workflow_volume_name
        self.external_artifact_checker = external_artifact_checker
        self.ledger = WorkflowLedger(volume_root_path)
        self.diagnostics = RuntimeDiagnostics()
        self._volume_sync = WorkflowVolumeSync(
            workflow_volume=workflow_volume,
            ledger=self.ledger,
        )
        self._remote_calls = RemoteCallManager(
            ledger=self.ledger,
            volume_sync=self._volume_sync,
            function_call_resolver=function_call_resolver,
            remote_call_poll_timeout=remote_call_poll_timeout,
        )
        self._node_runner = NodeRunner(
            ledger=self.ledger,
            volume_root=self.volume_root,
            workflow_volume_name=self.workflow_volume_name,
            volume_sync=self._volume_sync,
            external_artifact_checker=self.external_artifact_checker,
            remote_calls=self._remote_calls,
            node_is_complete=self._node_is_complete,
            max_ready_workers=max_ready_workers,
        )

    def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
        """Run the workflow until every node succeeds or no progress is possible."""
        self.diagnostics = RuntimeDiagnostics(run_id=run_id)
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
            ledger=self.ledger,
            volume_sync=self._volume_sync,
        )

        while True:
            decision = scheduler.evaluate_progress(
                definition,
                ledger=self.ledger,
                node_is_complete=self._node_is_complete,
            )
            self.diagnostics.record_scheduler_decision(decision)
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
            unknown_reasons = [
                reason
                for artifact in self.ledger.load_node_output_artifacts(node_id)
                for reason in self._artifact_availability_unknown_reasons(artifact)
            ]
            if unknown_reasons:
                workflow_display.print_workflow_message(
                    "[workflow] Node output artifact availability unknown: "
                    f"{node_id}: {'; '.join(unknown_reasons)}",
                    style="yellow",
                )
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
                workflow_volume_name=self.workflow_volume_name,
                volume_root=self.volume_root,
                run_root=self.ledger.run_root,
                external_artifact_checker=self.external_artifact_checker,
            )
        )

    def _artifact_availability_unknown_reasons(
        self, artifact: WorkflowArtifact
    ) -> list[str]:
        return availability.artifact_availability_unknown_reasons(
            artifact,
            workflow_volume_name=self.workflow_volume_name,
            volume_root=self.volume_root,
            external_artifact_checker=self.external_artifact_checker,
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
