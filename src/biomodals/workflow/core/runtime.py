"""Local workflow runtime scheduler."""

from __future__ import annotations

import hashlib
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Protocol

import orjson
from pydantic import BaseModel

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
from biomodals.workflow.core.artifacts import materialize_app_run_result
from biomodals.workflow.core.builder import Workflow, WorkflowDefinition
from biomodals.workflow.core.ledger import WorkflowLedger
from biomodals.workflow.core.nodes import NodeRunContext, WorkflowNode


class RemoteFunctionCall(Protocol):
    """Minimal Modal FunctionCall boundary used by remote workflow nodes."""

    object_id: str

    def get(self, timeout: float | int | None = None) -> object:
        """Return the remote function result or raise TimeoutError."""


RemoteNodeRunner = Callable[
    [WorkflowNode, NodeRunContext], AppRunResult | RemoteFunctionCall
]
FunctionCallResolver = Callable[[str], RemoteFunctionCall]


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
        remote_node_function_name: str | None = None,
        function_call_resolver: FunctionCallResolver | None = None,
        remote_call_poll_timeout: float | int = 0,
        max_ready_workers: int = 32,
    ):
        """Initialize a runtime for one workflow and ledger root."""
        self.workflow = workflow
        self.volume_root = Path(volume_root)
        self.workflow_volume_name = workflow_volume_name
        self.workflow_volume = workflow_volume
        self.remote_node_runner = remote_node_runner
        self.remote_node_function_name = remote_node_function_name
        self.function_call_resolver = function_call_resolver
        self.remote_call_poll_timeout = remote_call_poll_timeout
        self.max_ready_workers = max_ready_workers
        self.ledger = WorkflowLedger(self.volume_root)
        self.executed_waves: list[list[str]] = []
        self._active_remote_calls: dict[str, RemoteFunctionCall] = {}
        self._active_remote_calls_lock = RLock()

    def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
        """Run the workflow until every node succeeds or no progress is possible."""
        definition = self.workflow.validate()
        dag_hash = self._dag_hash(definition)
        print(
            f"[workflow] Starting workflow '{definition.name}' run '{run_id}' "
            f"with {len(definition.nodes)} node(s)",
            flush=True,
        )
        print(
            "[workflow] DAG graph: node_id [placement; class] <- dependency", flush=True
        )
        for node_id, spec in definition.nodes.items():
            dependencies = sorted(definition.dependencies[node_id])
            dependency_text = ", ".join(dependencies) if dependencies else "-"
            node_class = (
                f"{spec.node.__class__.__module__}.{spec.node.__class__.__qualname__}"
            )
            print(
                f"[workflow]   {node_id} "
                f"[{spec.node.placement.value}; {node_class}] <- {dependency_text}",
                flush=True,
            )
        self._reload_volume()
        run_exists = self.ledger.run_exists(definition.name, run_id)
        if run_exists and force:
            self.ledger.reset_run(definition.name, run_id)
            self._commit_volume()
            self.ledger.create_run(
                WorkflowRun(
                    workflow_name=definition.name,
                    run_id=run_id,
                    dag_hash=dag_hash,
                )
            )
            self._commit_volume()
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
                and self._node_can_make_progress(node_id)
            ]
            if not ready:
                running = [
                    node_id
                    for node_id, dependencies in definition.dependencies.items()
                    if node_id not in completed
                    and dependencies.issubset(completed)
                    and self.ledger.node_is_running(node_id)
                ]
                if running:
                    self._commit_volume()
                    return AppRunResult(
                        status=AppRunStatus.PARTIAL,
                        warnings=[
                            "Workflow has in-flight nodes without a recoverable "
                            f"remote call: {', '.join(sorted(running))}"
                        ],
                    )
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
                    node_status = self.ledger.load_node_status(node_id)
                    if node_status.status != NodeStatus.FAILED or not node_status.error:
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

    def _node_can_make_progress(self, node_id: str) -> bool:
        if self.ledger.node_is_complete(node_id):
            return False
        if not self.ledger.node_is_running(node_id):
            return True
        return (
            self.ledger.latest_remote_call(
                node_id,
                statuses=("submitted", "running", "succeeded"),
            )
            is not None
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
                    print(
                        f"[workflow] Node failed: {node_id}: {exc}",
                        flush=True,
                    )
                    self.ledger.mark_node_failed(node_id, error)
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
        recovered = self._recover_remote_node_if_possible(node_id)
        if recovered is not None:
            return recovered
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
        print(
            f"[workflow] Node started: {node_id} attempt={attempt_id} "
            f"placement={spec.node.placement.value}",
            flush=True,
        )
        self.ledger.record_node_inputs(node_id, inputs)
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
        return self._finalize_node_result(
            node_id=node_id,
            attempt_id=attempt_id,
            attempt_dir=attempt_dir,
            result=result,
        )

    def _finalize_node_result(
        self,
        *,
        node_id: str,
        attempt_id: str,
        attempt_dir: Path,
        result: AppRunResult,
    ) -> AppRunResult:
        self.ledger.record_app_result(node_id, attempt_id, result)
        if result.status in {AppRunStatus.FAILED, AppRunStatus.PARTIAL}:
            print(
                f"[workflow] Node failed: {node_id} attempt={attempt_id}: "
                f"{self._node_error_message(result)}",
                flush=True,
            )
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
            self.ledger.record_attempt_completed(
                node_id,
                attempt_id,
                NodeStatus.FAILED,
                result=result,
                error=self._node_error_message(result),
            )
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
        self.ledger.record_attempt_completed(
            node_id,
            attempt_id,
            NodeStatus.SUCCEEDED,
            result=result,
        )
        self._commit_volume()
        print(
            f"[workflow] Node succeeded: {node_id} attempt={attempt_id} "
            f"artifacts={len(artifacts)}",
            flush=True,
        )
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
            return self._run_remote_node(node, context)
        return node.run(context)

    def _run_remote_node(
        self,
        node: WorkflowNode,
        context: NodeRunContext,
    ) -> AppRunResult:
        if self.remote_node_runner is None:
            return node.run(context)

        remote_result = self.remote_node_runner(node, context)
        if isinstance(remote_result, AppRunResult):
            self._reload_volume()
            return remote_result

        call_id = str(remote_result.object_id)
        with self._active_remote_calls_lock:
            self._active_remote_calls[call_id] = remote_result
        try:
            self.ledger.record_remote_call(
                call_id=call_id,
                node_id=context.node_id,
                attempt_id=context.attempt_id,
                function_name=self._remote_function_name(node),
                call_kind="node",
            )
            self._commit_volume()
            result = self._collect_remote_call(call_id, remote_result)
            self._reload_volume()
            return result
        finally:
            with self._active_remote_calls_lock:
                self._active_remote_calls.pop(call_id, None)

    def cancel_active_remote_calls(self, *, terminate_containers: bool = True) -> None:
        """Cancel Modal function calls spawned by this runtime instance."""
        with self._active_remote_calls_lock:
            active_remote_calls = dict(self._active_remote_calls)
        if not active_remote_calls:
            return

        print(
            "[workflow] Cancelling "
            f"{len(active_remote_calls)} in-flight remote call(s)",
            flush=True,
        )
        for call_id, function_call in active_remote_calls.items():
            cancel = getattr(function_call, "cancel", None)
            if cancel is None:
                print(
                    f"[workflow] Remote call cannot be cancelled: {call_id}",
                    flush=True,
                )
                continue
            try:
                cancel(terminate_containers=terminate_containers)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[workflow] Remote call cancellation failed: {call_id}: {exc}",
                    flush=True,
                )
                continue

            print(f"[workflow] Remote call cancelled: {call_id}", flush=True)
            try:
                self.ledger.mark_remote_call_status(
                    call_id,
                    "cancelled",
                    completed=True,
                )
                self._commit_volume()
            except Exception as exc:  # noqa: BLE001
                print(
                    "[workflow] Remote call cancellation status could not be "
                    f"recorded: {call_id}: {exc}",
                    flush=True,
                )

    def _recover_remote_node_if_possible(self, node_id: str) -> AppRunResult | None:
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
                self._reload_volume()
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
        self._reload_volume()
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
        self,
        call_id: str,
        function_call: RemoteFunctionCall,
    ) -> AppRunResult:
        try:
            try:
                raw_result = function_call.get(timeout=self.remote_call_poll_timeout)
            except TimeoutError:
                self.ledger.mark_remote_call_status(call_id, "running")
                self._commit_volume()
                raw_result = function_call.get()
        except Exception as exc:
            self._record_remote_call_exception(call_id, exc)
            raise

        try:
            result = AppRunResult.model_validate(raw_result)
        except Exception as exc:
            self.ledger.mark_remote_call_status(
                call_id,
                "failed",
                error=str(exc),
                completed=True,
            )
            self._commit_volume()
            raise

        remote_call = self.ledger.load_remote_call(call_id)
        if remote_call is not None:
            self.ledger.record_app_result(
                str(remote_call["node_id"]),
                str(remote_call["attempt_id"]),
                result,
            )
            self._commit_volume()
        self.ledger.mark_remote_call_status(
            call_id,
            "succeeded",
            completed=True,
            metadata={"result_status": result.status.value},
        )
        self._commit_volume()
        return result

    def _record_remote_call_exception(self, call_id: str, exc: Exception) -> None:
        if exc.__class__.__name__ == "OutputExpiredError":
            self.ledger.mark_remote_call_status(
                call_id,
                "expired",
                error=str(exc),
                completed=True,
            )
            self._commit_volume()
            raise _RemoteCallExpired(str(exc)) from exc
        self.ledger.mark_remote_call_status(
            call_id,
            "failed",
            error=str(exc),
            completed=True,
        )
        self._commit_volume()

    def _resolve_function_call(self, call_id: str) -> RemoteFunctionCall:
        if self.function_call_resolver is not None:
            return self.function_call_resolver(call_id)

        import modal

        return modal.FunctionCall.from_id(call_id)

    def _remote_function_name(self, node: WorkflowNode) -> str:
        if self.remote_node_function_name is not None:
            return self.remote_node_function_name
        if self.remote_node_runner is not None:
            function_name = getattr(self.remote_node_runner, "function_name", None)
            if function_name is not None:
                return str(function_name)
        node_name = f"{node.__class__.__module__}.{node.__class__.__qualname__}"
        return node_name

    @staticmethod
    def _node_error_message(result: AppRunResult) -> str:
        if result.warnings:
            return result.warnings[0]
        if result.status == AppRunStatus.PARTIAL:
            return "Node returned partial status"
        return "Node returned failed status"

    @staticmethod
    def _dag_hash(definition: WorkflowDefinition) -> str:
        payload = {
            "name": definition.name,
            "nodes": {
                node_id: WorkflowRuntime._node_hash_payload(spec.node)
                | {
                    "inputs": {
                        input_name: selector.model_dump(mode="json")
                        for input_name, selector in sorted(spec.inputs.items())
                    },
                    "control_dependencies": sorted(spec.control_dependencies),
                    "dependencies": sorted(definition.dependencies[node_id]),
                }
                for node_id, spec in sorted(definition.nodes.items())
            },
        }
        encoded = orjson.dumps(
            payload,
            option=orjson.OPT_SORT_KEYS,
        )
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _node_hash_payload(node: WorkflowNode) -> dict[str, object]:
        payload: dict[str, object] = {
            "class": f"{node.__class__.__module__}.{node.__class__.__qualname__}",
            "execution_policy": node.execution_policy.value,
            "placement": node.placement.value,
        }
        if is_dataclass(node):
            payload["dataclass"] = WorkflowRuntime._stable_json_value(node)
        return payload

    @staticmethod
    def _stable_json_value(value: object) -> object:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", round_trip=True)
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, bytes):
            return {
                "bytes_sha256": hashlib.sha256(value).hexdigest(),
                "size_bytes": len(value),
            }
        if isinstance(value, Path):
            return value.as_posix()
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: WorkflowRuntime._stable_json_value(
                    getattr(value, field.name)
                )
                for field in fields(value)
                if field.metadata.get("dag_hash") is not False
            }
        if isinstance(value, Mapping):
            return {
                str(key): WorkflowRuntime._stable_json_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [WorkflowRuntime._stable_json_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            stable_items = [WorkflowRuntime._stable_json_value(item) for item in value]
            return sorted(
                stable_items,
                key=lambda item: orjson.dumps(item, option=orjson.OPT_SORT_KEYS),
            )
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(
            f"Unsupported DAG hash value type: {type(value).__module__}."
            f"{type(value).__qualname__}"
        )

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

    def _commit_volume(self) -> None:
        if self.workflow_volume is not None:
            with self.ledger.closed_for_volume_sync():
                self.workflow_volume.commit()

    def _reload_volume(self) -> None:
        if self.workflow_volume is not None:
            with self.ledger.closed_for_volume_sync():
                self.workflow_volume.reload()

    def close(self) -> None:
        """Close durable local resources owned by the runtime."""
        self.ledger.close()


class _RemoteCallExpired(RuntimeError):
    """Raised when Modal no longer has a recoverable function result."""
