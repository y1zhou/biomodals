"""Local workflow runtime scheduler."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Protocol

from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    ArtifactSelector,
    AttemptRecord,
    NodeExecutionPolicy,
    NodePlacement,
    RunStatus,
    WorkflowArtifact,
    WorkflowRun,
)
from biomodals.workflow.core.artifacts import materialize_app_run_result
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.ledger import WorkflowLedger
from biomodals.workflow.core.nodes import NodeRunContext, WorkflowNode

RemoteNodeRunner = Callable[[WorkflowNode, NodeRunContext], AppRunResult]


class WorkflowVolume(Protocol):
    """Minimal Modal Volume boundary used by the workflow runtime."""

    def commit(self) -> object:
        """Persist pending writes to the mounted volume."""

    def reload(self) -> object:
        """Refresh local view of writes made by other containers."""


class WorkflowRuntime:
    """Local runtime core for scheduling workflow nodes against a ledger."""

    def __init__(
        self,
        *,
        workflow: Workflow,
        volume_root: str | Path,
        workflow_volume_name: str,
        workflow_volume: WorkflowVolume | None = None,
        remote_node_runner: RemoteNodeRunner | None = None,
    ):
        """Initialize a runtime for one workflow and ledger root."""
        self.workflow = workflow
        self.volume_root = Path(volume_root)
        self.workflow_volume_name = workflow_volume_name
        self.workflow_volume = workflow_volume
        self.remote_node_runner = remote_node_runner
        self.ledger = WorkflowLedger(self.volume_root)
        self.executed_waves: list[list[str]] = []

    @classmethod
    def from_definition(
        cls,
        *,
        workflow_name: str,
        workflow_definition: Workflow | dict[str, object],
        volume_root: str | Path,
        workflow_volume_name: str | None = None,
        workflow_volume: WorkflowVolume | None = None,
        remote_node_runner: RemoteNodeRunner | None = None,
    ) -> WorkflowRuntime:
        """Create a runtime from a Python workflow definition."""
        if isinstance(workflow_definition, Workflow):
            if workflow_definition.name != workflow_name:
                raise ValueError(
                    "workflow_name must match the supplied Workflow definition"
                )
            return cls(
                workflow=workflow_definition,
                volume_root=volume_root,
                workflow_volume_name=(
                    workflow_volume_name or f"{workflow_name}-outputs"
                ),
                workflow_volume=workflow_volume,
                remote_node_runner=remote_node_runner,
            )
        raise NotImplementedError(
            "Serialized workflow definition dictionaries are deferred; pass a "
            "Python Workflow object to the first runtime"
        )

    def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
        """Run the workflow until every node succeeds or no progress is possible."""
        definition = self.workflow.validate()
        run_path = self.volume_root / definition.name / run_id / "run.json"
        self._reload_volume()
        if run_path.exists() and force:
            self.ledger.reset_run(definition.name, run_id)
            self._commit_volume()
            self.ledger.create_run(
                WorkflowRun(workflow_name=definition.name, run_id=run_id)
            )
            self._commit_volume()
        elif run_path.exists():
            self.ledger.load_run(definition.name, run_id)
        else:
            self.ledger.create_run(
                WorkflowRun(workflow_name=definition.name, run_id=run_id)
            )
            self._commit_volume()
        self.ledger.mark_run_status(RunStatus.RUNNING)
        self._commit_volume()

        while True:
            completed = self._completed_nodes(definition.nodes.keys())
            if len(completed) == len(definition.nodes):
                self.ledger.mark_run_status(RunStatus.SUCCEEDED)
                self._commit_volume()
                return AppRunResult(status=AppRunStatus.SUCCEEDED)

            ready = [
                node_id
                for node_id, dependencies in definition.dependencies.items()
                if node_id not in completed
                and dependencies.issubset(completed)
                and not self.ledger.node_is_complete(node_id)
            ]
            if not ready:
                self.ledger.mark_run_status(RunStatus.FAILED)
                self._commit_volume()
                return AppRunResult(
                    status=AppRunStatus.FAILED,
                    warnings=["No runnable workflow nodes remain"],
                )

            self.executed_waves.append(ready)
            for node_id, node_result in self._run_ready_nodes(ready):
                if node_result.status in {AppRunStatus.FAILED, AppRunStatus.PARTIAL}:
                    error = self._node_error_message(node_result)
                    self.ledger.mark_node_failed(node_id, error)
                    self.ledger.mark_run_status(RunStatus.FAILED)
                    self._commit_volume()
                    return AppRunResult(
                        status=node_result.status,
                        warnings=node_result.warnings or [error],
                    )

    def _completed_nodes(self, node_ids) -> set[str]:
        return {
            node_id for node_id in node_ids if self.ledger.node_is_complete(node_id)
        }

    def _run_ready_nodes(self, node_ids: list[str]) -> list[tuple[str, AppRunResult]]:
        results: list[tuple[str, AppRunResult]] = []
        with ThreadPoolExecutor(max_workers=len(node_ids)) as executor:
            futures = {
                executor.submit(self._run_node, node_id): node_id
                for node_id in node_ids
            }
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    results.append((node_id, future.result()))
                except Exception as exc:  # noqa: BLE001
                    self.ledger.mark_node_failed(node_id, str(exc))
                    self._commit_volume()
                    results.append((
                        node_id,
                        AppRunResult(
                            status=AppRunStatus.FAILED,
                            warnings=[str(exc)],
                        ),
                    ))
        return results

    def _run_node(self, node_id: str) -> AppRunResult:
        definition = self.workflow.validate()
        spec = definition.nodes[node_id]
        if (
            spec.node.execution_policy == NodeExecutionPolicy.RERUN
            and self.ledger.node_has_state(node_id)
            and not self.ledger.node_is_complete(node_id)
        ):
            self.ledger.reset_node(node_id)
            self._commit_volume()
        attempt_id = self._next_attempt_id(node_id)
        inputs = self._resolve_inputs(spec.inputs)
        input_artifact_ids = [
            artifact.artifact_id
            for artifacts in inputs.values()
            for artifact in artifacts
        ]
        self.ledger.mark_node_running(
            node_id,
            attempt_id,
            input_artifact_ids=input_artifact_ids,
            execution_policy=spec.node.execution_policy,
            placement=spec.node.placement,
        )
        attempt = self.ledger.record_attempt_started(node_id, attempt_id)
        attempt_dir = self._attempt_dir(attempt)
        cache_dir = self.ledger.run_root / "nodes" / node_id / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._commit_volume()

        result = self._dispatch_node(
            spec.node,
            NodeRunContext(
                run_id=self.ledger.run_id or "",
                node_id=node_id,
                attempt_id=attempt_id,
                cache_dir=cache_dir,
                inputs=inputs,
            ),
        )
        self.ledger.record_app_result(node_id, attempt_id, result)
        if result.status in {AppRunStatus.FAILED, AppRunStatus.PARTIAL}:
            if result.logs:
                log_artifacts = materialize_app_run_result(
                    result=AppRunResult(status=result.status, logs=result.logs),
                    workflow_volume_name=self.workflow_volume_name,
                    attempt_dir=attempt_dir,
                    artifact_dir=self.ledger.run_root / "artifacts",
                    producing_node_id=node_id,
                    volume_root=self.volume_root,
                )
                self.ledger.record_artifacts(log_artifacts)
            self._commit_volume()
            return result

        artifacts = materialize_app_run_result(
            result=result,
            workflow_volume_name=self.workflow_volume_name,
            attempt_dir=attempt_dir,
            artifact_dir=self.ledger.run_root / "artifacts",
            producing_node_id=node_id,
            volume_root=self.volume_root,
        )
        self.ledger.record_artifacts(artifacts)
        self.ledger.mark_node_succeeded(
            node_id,
            [artifact.artifact_id for artifact in artifacts],
        )
        self._commit_volume()
        return result

    def _resolve_inputs(
        self,
        selectors: dict[str, ArtifactSelector],
    ) -> dict[str, list[WorkflowArtifact]]:
        self._reload_volume()
        return {
            input_name: self.ledger.select_artifacts(selector)
            for input_name, selector in selectors.items()
        }

    def _dispatch_node(
        self, node: WorkflowNode, context: NodeRunContext
    ) -> AppRunResult:
        if (
            node.placement == NodePlacement.REMOTE
            and self.remote_node_runner is not None
        ):
            return self.remote_node_runner(node, context)
        return node.run(context)

    @staticmethod
    def _node_error_message(result: AppRunResult) -> str:
        if result.warnings:
            return result.warnings[0]
        if result.status == AppRunStatus.PARTIAL:
            return "Node returned partial status"
        return "Node returned failed status"

    def _next_attempt_id(self, node_id: str) -> str:
        attempts_dir = self.ledger.run_root / "nodes" / node_id / "attempts"
        if not attempts_dir.exists():
            return "attempt-1"
        attempts = sorted(path.name for path in attempts_dir.iterdir() if path.is_dir())
        return f"attempt-{len(attempts) + 1}"

    def _attempt_dir(self, attempt: AttemptRecord) -> Path:
        return (
            self.ledger.run_root
            / "nodes"
            / attempt.node_id
            / "attempts"
            / attempt.attempt_id
        )

    def _commit_volume(self) -> None:
        if self.workflow_volume is not None:
            self.workflow_volume.commit()

    def _reload_volume(self) -> None:
        if self.workflow_volume is not None:
            self.workflow_volume.reload()
