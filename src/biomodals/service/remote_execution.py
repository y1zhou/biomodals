"""Shared service-side access to deployed Modal execution coordinators."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import modal

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionOverview,
    ProviderCallDiagnostic,
    ProviderCallPage,
)
from biomodals.execution.modal import deployed_execution_coordinator

CALL_GRAPH_TIMEOUT_SECONDS = 5.0
_CONCURRENT_STREAM_CLOSE = "aclose(): asynchronous generator is already running"


class RemoteDeploymentUnavailableError(RuntimeError):
    """The exact pinned coordinator cannot be resolved."""


class RemoteSubmissionOutcomeUnknownError(RuntimeError):
    """A coordinator spawn may have happened without returning its call ID."""


class RemoteExecutionIdentityMismatchError(RuntimeError):
    """A remote overview does not belong to the pinned service Job."""


@dataclass(frozen=True, slots=True)
class ExecutionLocator:
    """Exact address of one deployed Execution Run."""

    execution_run_id: UUID
    deployment: DeploymentIdentity


class RemoteExecutionClient:
    """Own Modal lifecycle, status, diagnostics, and log control-plane calls."""

    async def preflight(self, deployment: DeploymentIdentity) -> None:
        """Resolve the exact deployed coordinator without starting a container."""
        try:
            coordinator = await asyncio.to_thread(
                modal.Cls.from_name,
                deployment.deployment_name,
                "ExecutionCoordinator",
                environment_name=deployment.environment,
                version=deployment.deployment_version,
            )
            await coordinator.hydrate.aio()
            for method in (
                "run",
                "resume",
                "status",
                "cancel",
                "provider_calls",
                "provider_call",
            ):
                getattr(coordinator, method)
        except Exception as error:
            raise RemoteDeploymentUnavailableError(str(error)) from error

    async def launch(self, locator: ExecutionLocator) -> str:
        """Spawn one staged root Run and return durable launch evidence."""
        return await self._spawn_root(self._coordinator(locator).run)

    async def resume(self, locator: ExecutionLocator) -> str:
        """Spawn explicit recovery for one suspended or unknown Run."""
        return await self._spawn_root(self._coordinator(locator).resume)

    @staticmethod
    async def _spawn_root(method: Any) -> str:
        """Spawn one coordinator method and require its durable call handle."""
        try:
            call = await asyncio.to_thread(method.spawn)
            call_id = call.object_id
        except Exception as error:
            raise RemoteSubmissionOutcomeUnknownError(str(error)) from error
        if not call_id:
            raise RemoteSubmissionOutcomeUnknownError(
                "Modal did not return a coordinator Function Call ID"
            )
        return str(call_id)

    async def poll_root(
        self, locator: ExecutionLocator, function_call_id: str
    ) -> ExecutionOverview | None:
        """Poll one root call without invoking coordinator code."""
        call = modal.FunctionCall.from_id(function_call_id)
        try:
            overview = await asyncio.to_thread(call.get, timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            return None
        except modal.exception.NotFoundError as error:
            raise RemoteExecutionIdentityMismatchError(
                "Pinned root Function Call is unavailable"
            ) from error
        except modal.exception.RemoteError:
            # The root call can raise after durably suspending the Run. The
            # coordinator ledger, not Modal's call result, owns terminality.
            return await self.status(locator)
        return self._verified(locator, overview)

    async def status(self, locator: ExecutionLocator) -> ExecutionOverview:
        """Read one bounded remote execution overview."""
        try:
            overview = await asyncio.to_thread(self._coordinator(locator).status.remote)
        except modal.exception.NotFoundError as error:
            raise RemoteDeploymentUnavailableError(str(error)) from error
        return self._verified(locator, overview)

    async def cancel(self, locator: ExecutionLocator) -> ExecutionOverview:
        """Request durable cancellation from the execution authority."""
        try:
            overview = await asyncio.to_thread(self._coordinator(locator).cancel.remote)
        except modal.exception.NotFoundError as error:
            raise RemoteDeploymentUnavailableError(str(error)) from error
        return self._verified(locator, overview)

    async def queued_provider_call_handles(
        self,
        root_function_call_id: str | None,
        overview: ExecutionOverview,
    ) -> frozenset[str]:
        """Best-effort presentation hint for calls awaiting a Modal task."""
        if not root_function_call_id:
            return frozenset()
        active_handles = {
            call.provider_call_handle_id
            for call in overview.representative_provider_calls
            if call.provider_call_handle_id and not call.status.is_terminal
        }
        if not active_handles:
            return frozenset()
        try:
            call = modal.FunctionCall.from_id(root_function_call_id)
            graph = await asyncio.wait_for(
                call.get_call_graph.aio(),
                timeout=CALL_GRAPH_TIMEOUT_SECONDS,
            )
        except Exception:
            return frozenset()
        return frozenset(
            item.function_call_id
            for item in graph
            if item.function_call_id in active_handles
            and item.status == modal.types.InputStatus.PENDING
            and not item.task_id
        )

    async def provider_calls(
        self,
        locator: ExecutionLocator,
        *,
        node_key: str | None = None,
        cursor: UUID | None = None,
        limit: int = 50,
        newest_first: bool = False,
    ) -> ProviderCallPage:
        """Read one bounded page of Provider Call diagnostics."""
        try:
            return await asyncio.to_thread(
                self._coordinator(locator).provider_calls.remote,
                node_key,
                None if cursor is None else str(cursor),
                limit,
                newest_first,
            )
        except modal.exception.NotFoundError as error:
            raise RemoteDeploymentUnavailableError(str(error)) from error

    async def provider_call(
        self,
        locator: ExecutionLocator,
        provider_call_id: UUID,
        *,
        node_key: str | None = None,
    ) -> ProviderCallDiagnostic | None:
        """Resolve one opaque public selector with a bounded coordinator read."""
        try:
            call = await asyncio.to_thread(
                self._coordinator(locator).provider_call.remote,
                str(provider_call_id),
            )
        except modal.exception.NotFoundError as error:
            raise RemoteDeploymentUnavailableError(str(error)) from error
        if call is not None and node_key is not None and call.node_key != node_key:
            return None
        return call

    async def log_entries(
        self,
        function_call_id: str,
        *,
        live: bool,
        since: datetime | None = None,
        until: datetime | None = None,
        tail_entries: int = 200,
    ) -> AsyncIterator[Any]:
        """Yield SDK LogEntry objects without blocking the event loop."""
        call = modal.FunctionCall.from_id(function_call_id)
        if live:
            source = call.logs.stream.aio(timeout=None)
        elif since is None:
            source = call.logs.tail.aio(entries=tail_entries)
        else:
            source = call.logs.fetch.aio(since=since, until=until)
        try:
            async for item in source:
                yield item
        except RuntimeError as error:
            if str(error) != _CONCURRENT_STREAM_CLOSE:
                raise
        finally:
            try:
                await source.aclose()
            except RuntimeError as error:
                if str(error) != _CONCURRENT_STREAM_CLOSE:
                    raise

    @staticmethod
    def _verified(locator: ExecutionLocator, overview: object) -> ExecutionOverview:
        if not isinstance(overview, ExecutionOverview):
            raise RemoteExecutionIdentityMismatchError(
                "Remote result is not an execution overview"
            )
        run = overview.run
        if (
            run.execution_run_id != locator.execution_run_id
            or run.deployment != locator.deployment
        ):
            raise RemoteExecutionIdentityMismatchError(
                "Remote overview identity does not match the pinned Job locator"
            )
        return overview

    @staticmethod
    def _coordinator(locator: ExecutionLocator) -> Any:
        return deployed_execution_coordinator(
            execution_run_id=locator.execution_run_id,
            deployment=locator.deployment,
        )
