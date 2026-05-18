"""Local workflow runtime scheduler."""

from __future__ import annotations

from pathlib import Path

from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    ArtifactSelector,
    AttemptRecord,
    WorkflowArtifact,
    WorkflowRun,
)
from biomodals.workflow.core.artifacts import materialize_app_run_result
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.ledger import WorkflowLedger
from biomodals.workflow.core.nodes import NodeRunContext


class WorkflowRuntime:
    """Local runtime core for scheduling workflow nodes against a ledger."""

    def __init__(
        self,
        *,
        workflow: Workflow,
        volume_root: str | Path,
        workflow_volume_name: str,
    ):
        """Initialize a runtime for one workflow and ledger root."""
        self.workflow = workflow
        self.volume_root = Path(volume_root)
        self.workflow_volume_name = workflow_volume_name
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
            )
        raise NotImplementedError(
            "Serialized workflow definition dictionaries are deferred; pass a "
            "Python Workflow object to the first runtime"
        )

    def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
        """Run the workflow until every node succeeds or no progress is possible."""
        definition = self.workflow.validate()
        run_path = self.volume_root / definition.name / run_id / "run.json"
        if run_path.exists() and not force:
            self.ledger.load_run(definition.name, run_id)
        else:
            self.ledger.create_run(
                WorkflowRun(workflow_name=definition.name, run_id=run_id)
            )

        while True:
            completed = self._completed_nodes(definition.nodes.keys())
            if len(completed) == len(definition.nodes):
                return AppRunResult(status=AppRunStatus.SUCCEEDED)

            ready = [
                node_id
                for node_id, dependencies in definition.dependencies.items()
                if node_id not in completed
                and dependencies.issubset(completed)
                and not self.ledger.node_is_complete(node_id)
            ]
            if not ready:
                return AppRunResult(
                    status=AppRunStatus.FAILED,
                    warnings=["No runnable workflow nodes remain"],
                )

            self.executed_waves.append(ready)
            for node_id in ready:
                node_result = self._run_node(node_id)
                if node_result.status == AppRunStatus.FAILED:
                    self.ledger.mark_node_failed(node_id, "Node returned failed status")
                    return AppRunResult(status=AppRunStatus.FAILED)

    def _completed_nodes(self, node_ids) -> set[str]:
        return {
            node_id for node_id in node_ids if self.ledger.node_is_complete(node_id)
        }

    def _run_node(self, node_id: str) -> AppRunResult:
        definition = self.workflow.validate()
        spec = definition.nodes[node_id]
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
        if result.status == AppRunStatus.FAILED:
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
        return result

    def _resolve_inputs(
        self,
        selectors: dict[str, ArtifactSelector],
    ) -> dict[str, list[WorkflowArtifact]]:
        return {
            input_name: self.ledger.select_artifacts(selector)
            for input_name, selector in selectors.items()
        }

    @staticmethod
    def _dispatch_node(node, context: NodeRunContext) -> AppRunResult:
        return node.run(context)

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
