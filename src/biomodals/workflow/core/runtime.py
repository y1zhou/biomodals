"""Local workflow runtime scheduler."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import RLock
from typing import Any

import orjson

import biomodals.workflow.core.display as workflow_display
from biomodals.schema import (
    AppRunResult,
    AppRunStatus,
    ArtifactSelector,
    AttemptRecord,
    NodeExecutionPolicy,
    NodePlacement,
    NodeStatus,
    RunStatus,
    WorkflowArtifact,
    WorkflowRun,
)
from biomodals.workflow.core._runtime import availability, hashing, scheduler
from biomodals.workflow.core._runtime.volume_sync import (
    WorkflowVolume,
    WorkflowVolumeSync,
)
from biomodals.workflow.core.artifacts import materialize_app_run_result
from biomodals.workflow.core.builder import Workflow
from biomodals.workflow.core.ledger import WorkflowLedger
from biomodals.workflow.core.nodes import (
    NodeRunContext,
    RemoteFunctionCall,
    RemoteNodeSubmission,
    WorkflowNode,
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
        self.volume_root = Path(volume_root)
        self.workflow_volume_name = workflow_volume_name
        self.workflow_volume = workflow_volume
        self.function_call_resolver = function_call_resolver
        self.remote_call_poll_timeout = remote_call_poll_timeout
        self.max_ready_workers = max_ready_workers
        self.ledger = WorkflowLedger(self.volume_root)
        # TODO: Replace this debug-only wave history with structured scheduling
        # diagnostics that can record ready/completed/blocked reasons and timing
        # without expanding the public workflow API.
        self.executed_waves: list[list[str]] = []
        self._active_remote_calls: dict[str, RemoteFunctionCall] = {}
        self._active_remote_calls_lock = RLock()
        self._volume_sync = WorkflowVolumeSync(
            workflow_volume=self.workflow_volume,
            ledger=self.ledger,
        )
        self._workflow_volume_access_lock = self._volume_sync.lock

    def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
        """Run the workflow until every node succeeds or no progress is possible."""
        definition = self.workflow.validate()
        dag_hash = hashing.dag_hash(definition)
        workflow_display.print_workflow_message(
            f"[workflow] Starting workflow '{definition.name}' run '{run_id}' "
            f"with {len(definition.nodes)} node(s)",
            style="bold cyan",
        )
        workflow_display.print_workflow_dag(definition)
        self._volume_sync.reload()
        run_exists = self.ledger.run_exists(definition.name, run_id)
        if run_exists and force:
            self.ledger.reset_run(definition.name, run_id)
            self._volume_sync.commit()
            self.ledger.create_run(
                WorkflowRun(
                    workflow_name=definition.name, run_id=run_id, dag_hash=dag_hash
                )
            )
            self._volume_sync.commit()
        elif run_exists:
            existing_run = self.ledger.load_run(definition.name, run_id)
            if existing_run.dag_hash is not None and existing_run.dag_hash != dag_hash:
                raise ValueError(
                    "DAG hash does not match existing workflow run; rerun with force"
                )
        else:
            self.ledger.create_run(
                WorkflowRun(
                    workflow_name=definition.name,
                    run_id=run_id,
                    dag_hash=dag_hash,
                )
            )
            self._volume_sync.commit()
        self.ledger.mark_run_status(RunStatus.RUNNING)
        self._volume_sync.commit()

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
            for node_id, node_result in self._run_ready_nodes(decision.ready):
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

    def _run_ready_nodes(self, node_ids: list[str]) -> list[tuple[str, AppRunResult]]:
        results: list[tuple[str, AppRunResult]] = []
        max_workers = min(len(node_ids), max(1, self.max_ready_workers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_node, node_id): node_id
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
                    self._volume_sync.commit()
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
            and not self.ledger.node_is_running(node_id)
            and not self._node_is_complete(node_id)
        ):
            self._reset_node_for_rerun(node_id)
        recovered = self._recover_remote_node_if_possible(node_id, spec.node)
        if recovered is not None:
            return recovered
        if (
            spec.node.execution_policy == NodeExecutionPolicy.RERUN
            and self.ledger.node_has_state(node_id)
            and not self._node_is_complete(node_id)
        ):
            self._reset_node_for_rerun(node_id)
        attempt_id = self._next_attempt_id(node_id)
        inputs = self._resolve_inputs(spec.inputs)
        input_artifact_ids = [
            artifact.artifact_id
            for artifacts in inputs.values()
            for artifact in artifacts
        ]
        with self._workflow_volume_access_lock:
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
            attempt_dir = self._attempt_dir(attempt)
            cache_dir = self.ledger.run_root / "nodes" / node_id / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._volume_sync.commit()

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
        with self._workflow_volume_access_lock:
            self.ledger.reset_node(node_id)
            self._volume_sync.commit()

    def _finalize_node_result(
        self,
        *,
        node_id: str,
        attempt_id: str,
        attempt_dir: Path,
        result: AppRunResult,
    ) -> AppRunResult:
        with self._workflow_volume_access_lock:
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
                self._volume_sync.commit()
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
            self._volume_sync.commit()
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
        self._volume_sync.reload()
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
                workflow_volume_name=self.workflow_volume_name,
                volume_root=self.volume_root,
                run_root=self.ledger.run_root,
            )
        )

    def _dispatch_node(
        self, node: WorkflowNode, context: NodeRunContext
    ) -> AppRunResult:
        if node.placement == NodePlacement.REMOTE:
            return self._run_remote_node(node, context)
        with self._workflow_volume_access_lock:
            return node.run(context)

    def _run_remote_node(
        self, node: WorkflowNode, context: NodeRunContext
    ) -> AppRunResult:
        remote_result = node.submit_remote(context)
        submission = self._normalize_remote_submission(node, remote_result)
        call_id = str(submission.function_call.object_id)
        with self._active_remote_calls_lock:
            self._active_remote_calls[call_id] = submission.function_call
        try:
            with self._workflow_volume_access_lock:
                self.ledger.record_remote_call(
                    call_id=call_id,
                    node_id=context.node_id,
                    attempt_id=context.attempt_id,
                    function_name=submission.function_name
                    or self._remote_function_name(node),
                    call_kind="node",
                    metadata=submission.metadata,
                )
                self._volume_sync.commit()
            raw_result = self._collect_remote_call(call_id, submission.function_call)
            self._volume_sync.reload()
            try:
                result = self._process_remote_node_result(
                    node,
                    raw_result,
                    submission.metadata,
                )
            except Exception as exc:
                self._record_remote_call_exception(call_id, exc)
                raise
            self._record_remote_call_success(call_id, result)
            return result
        finally:
            with self._active_remote_calls_lock:
                self._active_remote_calls.pop(call_id, None)

    def _normalize_remote_submission(
        self, node: WorkflowNode, remote_result: RemoteNodeSubmission
    ) -> RemoteNodeSubmission:
        if isinstance(remote_result, RemoteNodeSubmission):
            function_call = remote_result.function_call
            if not hasattr(function_call, "object_id") or not hasattr(
                function_call, "get"
            ):
                raise TypeError(
                    "Remote workflow node submission did not include a FunctionCall"
                )
            return remote_result
        raise TypeError("submit_remote(context) must return RemoteNodeSubmission")

    def _process_remote_node_result(
        self, node: WorkflowNode, result: object, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        return AppRunResult.model_validate(node.process_remote_result(result, metadata))

    def cancel_active_remote_calls(self, *, terminate_containers: bool = True) -> None:
        """Cancel Modal function calls spawned by this runtime instance."""
        with self._active_remote_calls_lock:
            active_remote_calls = dict(self._active_remote_calls)
        if not active_remote_calls:
            return

        workflow_display.print_workflow_message(
            "[workflow] Cancelling "
            f"{len(active_remote_calls)} in-flight remote call(s)",
            style="yellow",
        )
        for call_id, function_call in active_remote_calls.items():
            cancel = getattr(function_call, "cancel", None)
            if cancel is None:
                workflow_display.print_workflow_message(
                    f"[workflow] Remote call cannot be cancelled: {call_id}",
                    style="yellow",
                )
                continue
            try:
                cancel(terminate_containers=terminate_containers)
            except Exception as exc:  # noqa: BLE001
                workflow_display.print_workflow_message(
                    f"[workflow] Remote call cancellation failed: {call_id}: {exc}",
                    style="red",
                )
                continue

            workflow_display.print_workflow_message(
                f"[workflow] Remote call cancelled: {call_id}",
                style="yellow",
            )
            try:
                self.ledger.mark_remote_call_status(
                    call_id,
                    "cancelled",
                    completed=True,
                )
                self._volume_sync.commit()
            except Exception as exc:  # noqa: BLE001
                workflow_display.print_workflow_message(
                    "[workflow] Remote call cancellation status could not be "
                    f"recorded: {call_id}: {exc}",
                    style="red",
                )

    def _recover_remote_node_if_possible(
        self, node_id: str, node: WorkflowNode
    ) -> AppRunResult | None:
        succeeded_call = self.ledger.latest_remote_call(
            node_id,
            statuses=("succeeded",),
        )
        if succeeded_call is not None:
            result = self.ledger.load_attempt_app_result(
                node_id,
                str(succeeded_call["attempt_id"]),
            )
            if result is not None:
                self._volume_sync.reload()
                return self._finalize_node_result(
                    node_id=node_id,
                    attempt_id=str(succeeded_call["attempt_id"]),
                    attempt_dir=self.ledger.run_root
                    / "nodes"
                    / node_id
                    / "attempts"
                    / str(succeeded_call["attempt_id"]),
                    result=result,
                )

        remote_call = self.ledger.latest_remote_call(
            node_id,
            statuses=("submitted", "running"),
        )
        if remote_call is None:
            return None
        call_id = str(remote_call["call_id"])
        try:
            function_call = self._resolve_function_call(call_id)
            result = self._collect_remote_call(call_id, function_call)
        except _RemoteCallExpired:
            return None
        self._volume_sync.reload()
        result = self._process_remote_node_result(
            node,
            result,
            self._remote_call_metadata(remote_call),
        )
        self._record_remote_call_success(call_id, result)
        return self._finalize_node_result(
            node_id=node_id,
            attempt_id=str(remote_call["attempt_id"]),
            attempt_dir=self.ledger.run_root
            / "nodes"
            / node_id
            / "attempts"
            / str(remote_call["attempt_id"]),
            result=result,
        )

    def _collect_remote_call(
        self, call_id: str, function_call: RemoteFunctionCall
    ) -> object:
        try:
            try:
                raw_result = function_call.get(timeout=self.remote_call_poll_timeout)
            except TimeoutError:
                self.ledger.mark_remote_call_status(call_id, "running")
                self._volume_sync.commit()
                raw_result = function_call.get()
        except Exception as exc:
            self._record_remote_call_exception(call_id, exc)
            raise

        return raw_result

    def _record_remote_call_success(self, call_id: str, result: AppRunResult) -> None:
        remote_call = self.ledger.load_remote_call(call_id)
        remote_call_metadata = {}
        if remote_call is not None:
            remote_call_metadata = self._remote_call_metadata(remote_call)
        self.ledger.mark_remote_call_status(
            call_id,
            "succeeded",
            completed=True,
            metadata=remote_call_metadata | {"result_status": result.status.value},
        )
        self._volume_sync.commit()

    def _record_remote_call_exception(self, call_id: str, exc: Exception) -> None:
        if exc.__class__.__name__ == "OutputExpiredError":
            self.ledger.mark_remote_call_status(
                call_id,
                "expired",
                error=str(exc),
                completed=True,
            )
            self._volume_sync.commit()
            raise _RemoteCallExpired(str(exc)) from exc
        self.ledger.mark_remote_call_status(
            call_id,
            "failed",
            error=str(exc),
            completed=True,
        )
        self._volume_sync.commit()

    def _resolve_function_call(self, call_id: str) -> RemoteFunctionCall:
        if self.function_call_resolver is not None:
            return self.function_call_resolver(call_id)

        import modal

        return modal.FunctionCall.from_id(call_id)

    def _remote_function_name(self, node: WorkflowNode) -> str:
        node_name = f"{node.__class__.__module__}.{node.__class__.__qualname__}"
        return node_name

    @staticmethod
    def _remote_call_metadata(remote_call: Mapping[str, Any]) -> dict[str, Any]:
        raw_metadata = remote_call.get("metadata_json")
        if not raw_metadata:
            return {}
        return orjson.loads(raw_metadata)

    @staticmethod
    def _node_error_message(result: AppRunResult) -> str:
        if result.warnings:
            return result.warnings[0]
        if result.status == AppRunStatus.PARTIAL:
            return "Node returned partial status"
        return "Node returned failed status"

    def _next_attempt_id(self, node_id: str) -> str:
        return self.ledger.next_attempt_id(node_id)

    def _attempt_dir(self, attempt: AttemptRecord) -> Path:
        return (
            self.ledger.run_root
            / "nodes"
            / attempt.node_id
            / "attempts"
            / attempt.attempt_id
        )

    def close(self) -> None:
        """Close durable local resources owned by the runtime."""
        self.ledger.close()


class _RemoteCallExpired(RuntimeError):
    """Raised when Modal no longer has a recoverable function result."""
