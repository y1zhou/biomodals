"""Workflow node attempt execution and finalization."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import biomodals.workflow.core.display as workflow_display
from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    ArtifactSelector,
    NodeExecutionPolicy,
    NodePlacement,
    NodeStatus,
    WorkflowArtifact,
)
from biomodals.workflow.core._runtime import availability
from biomodals.workflow.core._runtime.remote_calls import RemoteCallManager
from biomodals.workflow.core._runtime.services import RuntimeServices
from biomodals.workflow.core.artifacts import materialize_app_run_result
from biomodals.workflow.core.builder import WorkflowDefinition
from biomodals.workflow.core.nodes import NodeRunContext, WorkflowNode


class NodeRunner:
    """Run ready workflow nodes and reconcile their outputs into the ledger."""

    def __init__(
        self,
        *,
        services: RuntimeServices,
        remote_calls: RemoteCallManager,
        node_is_complete: Callable[[str], bool],
        max_ready_workers: int,
    ) -> None:
        self.ledger = services.ledger
        self.volume_root = services.volume_root
        self.workflow_volume_name = services.workflow_volume_name
        self.volume_sync = services.volume_sync
        self.remote_calls = remote_calls
        self.node_is_complete = node_is_complete
        self.max_ready_workers = max_ready_workers

    def run_ready_nodes(
        self, definition: WorkflowDefinition, node_ids: list[str]
    ) -> list[tuple[str, AppRunResult]]:
        """Run all currently ready nodes, using runtime-level concurrency limits."""
        results: list[tuple[str, AppRunResult]] = []
        max_workers = min(len(node_ids), max(1, self.max_ready_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.run_node, definition, node_id): node_id
                for node_id in node_ids
            }
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    results.append((node_id, future.result()))
                except Exception as exc:  # noqa: BLE001
                    error = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                    workflow_display.print_workflow_message(
                        f"[workflow] Node failed: {node_id}: {exc}",
                        style="red",
                    )
                    self.ledger.mark_node_failed(node_id, error)
                    self.volume_sync.commit()
                    results.append((
                        node_id,
                        AppRunResult(
                            status=AppRunStatus.FAILED,
                            warnings=[str(exc)],
                        ),
                    ))
        return results

    def run_node(self, definition: WorkflowDefinition, node_id: str) -> AppRunResult:
        """Run one workflow node attempt or recover a durable remote result."""
        spec = definition.nodes[node_id]
        if (
            spec.node.execution_policy == NodeExecutionPolicy.RERUN
            and self.ledger.node_has_state(node_id)
            and not self.ledger.node_is_running(node_id)
            and not self.node_is_complete(node_id)
        ):
            self._reset_node_for_rerun(node_id)
        recovered = self._recover_remote_node_if_possible(node_id, spec.node)
        if recovered is not None:
            return recovered
        if (
            spec.node.execution_policy == NodeExecutionPolicy.RERUN
            and self.ledger.node_has_state(node_id)
            and not self.node_is_complete(node_id)
        ):
            self._reset_node_for_rerun(node_id)
        attempt_id = self.ledger.next_attempt_id(node_id)
        inputs = self._resolve_inputs(spec.inputs)
        input_artifact_ids = [
            artifact.artifact_id
            for artifacts in inputs.values()
            for artifact in artifacts
        ]
        with self.volume_sync.lock:
            self.ledger.mark_node_running(
                node_id,
                attempt_id,
                input_artifact_ids=input_artifact_ids,
                execution_policy=spec.node.execution_policy,
                placement=spec.node.placement,
            )
            workflow_display.print_workflow_message(
                f"[workflow] Node started: {node_id} attempt={attempt_id} "
                f"placement={spec.node.placement.value}",
                style="yellow",
            )
            self.ledger.record_node_inputs(node_id, inputs)
            attempt = self.ledger.record_attempt_started(node_id, attempt_id)
            attempt_dir = (
                self.ledger.run_root
                / "nodes"
                / attempt.node_id
                / "attempts"
                / attempt.attempt_id
            )
            cache_dir = self.ledger.run_root / "nodes" / node_id / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.volume_sync.commit()

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
        return self._finalize_node_result(
            node_id=node_id,
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            result=result,
        )

    def _reset_node_for_rerun(self, node_id: str) -> None:
        with self.volume_sync.lock:
            self.ledger.reset_node(node_id)
            self.volume_sync.commit()

    def _finalize_node_result(
        self,
        *,
        node_id: str,
        attempt_id: str,
        attempt_dir: Path,
        result: AppRunResult,
    ) -> AppRunResult:
        with self.volume_sync.lock:
            # Materialization writes files under the mounted workflow volume. Keep
            # it serialized with reload/commit so scheduler workers cannot observe
            # a partially synchronized volume view.
            materialized = materialize_app_run_result(
                result=result,
                workflow_volume_name=self.workflow_volume_name,
                attempt_dir=attempt_dir,
                artifact_dir=self.ledger.run_root / "artifacts",
                producing_node_id=node_id,
                volume_root=self.volume_root,
            )
            persisted_result = materialized.result
            if result.status in {AppRunStatus.FAILED, AppRunStatus.PARTIAL}:
                workflow_display.print_workflow_message(
                    f"[workflow] Node failed: {node_id} attempt={attempt_id}: "
                    f"{self._node_error_message(result)}",
                    style="red",
                )
                self.ledger.record_artifacts(materialized.artifacts)
                self.ledger.record_attempt_completed(
                    node_id,
                    attempt_id,
                    NodeStatus.FAILED,
                    result=persisted_result,
                    error=self._node_error_message(result),
                )
                self.volume_sync.commit()
                return result

            artifacts = materialized.artifacts
            self.ledger.record_artifacts(artifacts)
            self.ledger.mark_node_succeeded(
                node_id,
                [artifact.artifact_id for artifact in artifacts],
            )
            self.ledger.record_attempt_completed(
                node_id,
                attempt_id,
                NodeStatus.SUCCEEDED,
                result=persisted_result,
            )
            self.volume_sync.commit()
        workflow_display.print_workflow_message(
            f"[workflow] Node succeeded: {node_id} attempt={attempt_id} "
            f"artifacts={len(artifacts)}",
            style="green",
        )
        return result

    def _resolve_inputs(
        self,
        selectors: dict[str, ArtifactSelector],
    ) -> dict[str, list[WorkflowArtifact]]:
        self.volume_sync.reload()
        inputs = {
            input_name: self.ledger.select_artifacts(selector)
            for input_name, selector in selectors.items()
        }
        errors = [
            error
            for artifacts in inputs.values()
            for artifact in artifacts
            for error in self._artifact_availability_errors(artifact)
        ]
        if errors:
            raise FileNotFoundError(
                "Workflow input artifacts are unavailable:\n"
                + "\n".join(f"- {error}" for error in errors)
            )
        return inputs

    def _artifact_availability_errors(self, artifact: WorkflowArtifact) -> list[str]:
        return availability.format_artifact_availability_errors(
            availability.artifact_availability_errors(
                artifact,
                workflow_volume_name=self.workflow_volume_name,
                volume_root=self.volume_root,
                run_root=self.ledger.run_root,
            )
        )

    def _dispatch_node(
        self, node: WorkflowNode, context: NodeRunContext
    ) -> AppRunResult:
        if node.placement == NodePlacement.REMOTE:
            return self.remote_calls.run_node(node, context)
        with self.volume_sync.lock:
            return node.run(context)

    def _recover_remote_node_if_possible(
        self, node_id: str, node: WorkflowNode
    ) -> AppRunResult | None:
        recovered = self.remote_calls.recover_node(node_id, node)
        if recovered is None:
            return None
        return self._finalize_node_result(
            node_id=node_id,
            attempt_id=recovered.attempt_id,
            attempt_dir=recovered.attempt_dir,
            result=recovered.result,
        )

    @staticmethod
    def _node_error_message(result: AppRunResult) -> str:
        if result.warnings:
            return result.warnings[0]
        if result.status == AppRunStatus.PARTIAL:
            return "Node returned partial status"
        return "Node returned failed status"
