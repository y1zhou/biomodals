"""Remote Modal function-call coordination for workflow nodes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import orjson

import biomodals.workflow.core.display as workflow_display
from biomodals.schema import AppRunResult
from biomodals.workflow.core._runtime.services import RuntimeServices
from biomodals.workflow.core.nodes import (
    NodeRunContext,
    RemoteFunctionCall,
    RemoteNodeSubmission,
    WorkflowNode,
)

FunctionCallResolver = Callable[[str], RemoteFunctionCall]


@dataclass(frozen=True)
class RecoveredRemoteResult:
    """Remote node result recovered from durable remote-call state."""

    attempt_id: str
    attempt_dir: Path
    result: AppRunResult


class RemoteCallManager:
    """Track, recover, and cancel remote Modal calls spawned by the runtime."""

    def __init__(
        self,
        *,
        services: RuntimeServices,
        function_call_resolver: FunctionCallResolver | None,
        remote_call_poll_timeout: float | int,
    ) -> None:
        self.ledger = services.ledger
        self.volume_sync = services.volume_sync
        self.function_call_resolver = function_call_resolver
        self.remote_call_poll_timeout = remote_call_poll_timeout
        self._active_remote_calls: dict[str, RemoteFunctionCall] = {}
        self._active_remote_calls_lock = RLock()

    def run_node(self, node: WorkflowNode, context: NodeRunContext) -> AppRunResult:
        """Submit, collect, and record one remote workflow node call."""
        remote_result = node.submit_remote(context)
        submission = self._normalize_remote_submission(remote_result)
        call_id = str(submission.function_call.object_id)
        with self._active_remote_calls_lock:
            self._active_remote_calls[call_id] = submission.function_call
        try:
            with self.volume_sync.lock:
                self.ledger.record_remote_call(
                    call_id=call_id,
                    node_id=context.node_id,
                    attempt_id=context.attempt_id,
                    function_name=submission.function_name
                    or self._remote_function_name(node),
                    call_kind="node",
                    metadata=submission.metadata,
                )
                self.volume_sync.commit()
            raw_result = self._collect_remote_call(call_id, submission.function_call)
            self.volume_sync.reload()
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

    def recover_node(
        self, node_id: str, node: WorkflowNode
    ) -> RecoveredRemoteResult | None:
        """Recover a remote node result from durable remote-call state if possible."""
        succeeded_call = self.ledger.latest_remote_call(
            node_id,
            statuses=("succeeded",),
        )
        if succeeded_call is not None:
            attempt_id = str(succeeded_call["attempt_id"])
            result = self.ledger.load_attempt_app_result(node_id, attempt_id)
            if result is not None:
                self.volume_sync.reload()
                return RecoveredRemoteResult(
                    attempt_id=attempt_id,
                    attempt_dir=self._attempt_dir(node_id, attempt_id),
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
            raw_result = self._collect_remote_call(call_id, function_call)
        except RemoteCallExpired:
            return None
        self.volume_sync.reload()
        result = self._process_remote_node_result(
            node,
            raw_result,
            self._remote_call_metadata(remote_call),
        )
        self._record_remote_call_success(call_id, result)
        attempt_id = str(remote_call["attempt_id"])
        return RecoveredRemoteResult(
            attempt_id=attempt_id,
            attempt_dir=self._attempt_dir(node_id, attempt_id),
            result=result,
        )

    def cancel_active(self, *, terminate_containers: bool = True) -> None:
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
                self.volume_sync.commit()
            except Exception as exc:  # noqa: BLE001
                workflow_display.print_workflow_message(
                    "[workflow] Remote call cancellation status could not be "
                    f"recorded: {call_id}: {exc}",
                    style="red",
                )

    @staticmethod
    def _normalize_remote_submission(
        remote_result: RemoteNodeSubmission,
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

    @staticmethod
    def _process_remote_node_result(
        node: WorkflowNode, result: object, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        return AppRunResult.model_validate(node.process_remote_result(result, metadata))

    def _collect_remote_call(
        self, call_id: str, function_call: RemoteFunctionCall
    ) -> object:
        try:
            try:
                raw_result = function_call.get(timeout=self.remote_call_poll_timeout)
            except TimeoutError:
                self.ledger.mark_remote_call_status(call_id, "running")
                self.volume_sync.commit()
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
        self.volume_sync.commit()

    def _record_remote_call_exception(self, call_id: str, exc: Exception) -> None:
        if exc.__class__.__name__ == "OutputExpiredError":
            self.ledger.mark_remote_call_status(
                call_id,
                "expired",
                error=str(exc),
                completed=True,
            )
            self.volume_sync.commit()
            raise RemoteCallExpired(str(exc)) from exc
        self.ledger.mark_remote_call_status(
            call_id,
            "failed",
            error=str(exc),
            completed=True,
        )
        self.volume_sync.commit()

    def _resolve_function_call(self, call_id: str) -> RemoteFunctionCall:
        if self.function_call_resolver is not None:
            return self.function_call_resolver(call_id)

        import modal

        return modal.FunctionCall.from_id(call_id)

    @staticmethod
    def _remote_function_name(node: WorkflowNode) -> str:
        return f"{node.__class__.__module__}.{node.__class__.__qualname__}"

    @staticmethod
    def _remote_call_metadata(remote_call: Mapping[str, Any]) -> dict[str, Any]:
        raw_metadata = remote_call.get("metadata_json")
        if not raw_metadata:
            return {}
        return orjson.loads(raw_metadata)

    def _attempt_dir(self, node_id: str, attempt_id: str) -> Path:
        return self.ledger.run_root / "nodes" / node_id / "attempts" / attempt_id


class RemoteCallExpired(RuntimeError):
    """Raised when Modal no longer has a recoverable function result."""
