"""Shared service-side access to deployed Modal execution coordinators."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
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


class RemoteDeploymentUnavailableError(RuntimeError):
    """The exact pinned coordinator cannot be resolved."""


class RemoteSubmissionOutcomeUnknownError(RuntimeError):
    """A coordinator spawn may have happened without returning its call ID."""


class RemoteRootExecutionFailedError(RuntimeError):
    """The root coordinator call ended without returning an overview."""


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
            for method in ("run", "status", "cancel", "provider_calls"):
                getattr(coordinator, method)
        except Exception as error:
            raise RemoteDeploymentUnavailableError(str(error)) from error

    async def launch(self, locator: ExecutionLocator) -> str:
        """Spawn one staged root Run and return durable launch evidence."""
        coordinator = self._coordinator(locator)
        try:
            call = await asyncio.to_thread(coordinator.run.spawn)
            call_id = call.object_id
        except Exception as error:
            raise RemoteSubmissionOutcomeUnknownError(str(error)) from error
        if not call_id:
            raise RemoteSubmissionOutcomeUnknownError(
                "Modal did not return a coordinator Function Call ID"
            )
        return str(call_id)

    async def poll_root(self, function_call_id: str) -> ExecutionOverview | None:
        """Poll one root call without invoking coordinator code."""
        call = modal.FunctionCall.from_id(function_call_id)
        try:
            return await asyncio.to_thread(call.get, timeout=0)
        except modal.exception.TimeoutError:
            return None
        except modal.exception.RemoteError as error:
            raise RemoteRootExecutionFailedError(str(error)) from error

    async def status(self, locator: ExecutionLocator) -> ExecutionOverview:
        """Read one bounded remote execution overview."""
        try:
            return await asyncio.to_thread(self._coordinator(locator).status.remote)
        except modal.exception.NotFoundError as error:
            raise RemoteDeploymentUnavailableError(str(error)) from error

    async def cancel(self, locator: ExecutionLocator) -> ExecutionOverview:
        """Request durable cancellation from the execution authority."""
        try:
            return await asyncio.to_thread(self._coordinator(locator).cancel.remote)
        except modal.exception.NotFoundError as error:
            raise RemoteDeploymentUnavailableError(str(error)) from error

    async def provider_calls(
        self,
        locator: ExecutionLocator,
        *,
        node_key: str | None = None,
        cursor: UUID | None = None,
        limit: int = 50,
    ) -> ProviderCallPage:
        """Read one bounded page of Provider Call diagnostics."""
        try:
            return await asyncio.to_thread(
                self._coordinator(locator).provider_calls.remote,
                node_key,
                None if cursor is None else str(cursor),
                limit,
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
        """Resolve an opaque public selector through bounded remote pages."""
        cursor: UUID | None = None
        while True:
            page = await self.provider_calls(
                locator,
                node_key=node_key,
                cursor=cursor,
                limit=100,
            )
            match = next(
                (
                    call
                    for call in page.calls
                    if call.provider_call_id == provider_call_id
                ),
                None,
            )
            if match is not None or page.next_cursor is None:
                return match
            cursor = page.next_cursor

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
            source = call.logs.stream(timeout=30)
        elif since is None:
            source = call.logs.tail(entries=tail_entries)
        else:
            source = call.logs.fetch(since=since, until=until)
        iterator: Iterator[Any] = iter(source)
        while True:
            item = await asyncio.to_thread(next, iterator, None)
            if item is None:
                return
            yield item

    @staticmethod
    def _coordinator(locator: ExecutionLocator) -> Any:
        return deployed_execution_coordinator(
            execution_run_id=locator.execution_run_id,
            deployment=locator.deployment,
        )
