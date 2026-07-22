"""Stable GROMACS service facade composed from focused boundaries."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import modal

from biomodals.service.artifacts import ArtifactCache
from biomodals.service.gromacs.coordinator import GromacsReconciler
from biomodals.service.gromacs.provider import (
    ModalGromacsProvider,
    PollOutcome,
    SubmittedModalCall,
)
from biomodals.service.gromacs.results import (
    ArchiveNotReadyError,
    FinalArchive,
    GromacsResultInvalidError,
    ModalGromacsResults,
    ResultIdentityMismatchError,
)
from biomodals.service.gromacs.router import GromacsJobOptions
from biomodals.service.jobs import OperationLogRequest, operation_log_mode
from biomodals.service.modal_logs import ModalCLILogSource
from biomodals.service.runtime_config import ModalConfigurationSnapshot
from biomodals.service.store import JobOperationRecord, JobRecord


class ModalGromacsAdapter:
    """Expose compute and Result capabilities through one wiring facade."""

    def __init__(
        self,
        *,
        output_volume_name: str = "Gromacs-outputs",
        artifact_cache: ArtifactCache | None = None,
        call_resolver: Callable[[str], modal.FunctionCall] = modal.FunctionCall.from_id,
        function_resolver: Callable[..., modal.Function] | None = None,
        log_source: ModalCLILogSource | None = None,
    ) -> None:
        """Compose focused Modal compute and Result boundaries."""
        self.provider = ModalGromacsProvider(
            output_volume_name=output_volume_name,
            call_resolver=call_resolver,
            function_resolver=function_resolver,
        )
        self.results = ModalGromacsResults(
            output_volume_name=output_volume_name,
            artifact_cache=artifact_cache,
        )
        self.logs = log_source or ModalCLILogSource()

    async def preflight(
        self,
        app_name: str,
        environment_name: str,
        app_version: int,
    ) -> None:
        """Validate every required deployed Modal resource."""
        await self.provider.preflight(app_name, environment_name, app_version)

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
        *,
        run_name: str,
        modal_configuration: ModalConfigurationSnapshot,
    ) -> SubmittedModalCall:
        """Submit initial preparation through the compute boundary."""
        return await self.provider.submit(
            pdb_content,
            options,
            run_name=run_name,
            modal_configuration=modal_configuration,
        )

    async def submit_operation(
        self,
        job: JobRecord,
        operation: str,
    ) -> SubmittedModalCall:
        """Submit one dependency-ready successor operation."""
        return await self.provider.submit_operation(job, operation)

    async def poll(
        self,
        modal_call_id: str,
        *,
        operation: str | None = None,
    ) -> PollOutcome:
        """Poll one attached Modal call without blocking."""
        return await self.provider.poll(modal_call_id, operation=operation)

    async def cancel(self, modal_call_id: str) -> None:
        """Cancel one Modal call graph."""
        await self.provider.cancel(modal_call_id)

    async def open_operation_logs(
        self,
        job: JobRecord,
        operation: JobOperationRecord,
        selection: OperationLogRequest,
    ) -> AsyncIterable[bytes]:
        """Open live or historical logs for one attached Modal operation."""
        if operation.modal_call_id is None or operation.started_at is None:
            raise ValueError("GROMACS operation has no attached Modal call")
        mode = operation_log_mode(operation.state)
        if mode is None:
            raise ValueError("GROMACS operation does not retain inspectable logs")
        if selection.mode == "live" and mode != "live":
            raise ValueError("A terminal GROMACS operation cannot open live logs")
        live = selection.mode == "live"
        started_at = datetime.fromtimestamp(operation.started_at, UTC)
        ended_at = (
            datetime.fromtimestamp(operation.completed_at, UTC)
            if operation.completed_at is not None
            else None
        )
        return await self.logs.open(
            app_name=job.modal_app_name,
            environment_name=job.modal_environment,
            function_call_id=operation.modal_call_id,
            follow=live,
            since=(
                selection.since
                or (started_at - timedelta(seconds=1) if not live else None)
            ),
            until=(
                selection.until
                or (ended_at + timedelta(seconds=1) if ended_at is not None else None)
            ),
        )

    async def read_artifact(self, job: JobRecord) -> AsyncIterator[bytes]:
        """Read the authoritative published Result."""
        async for chunk in self.results.read_artifact(job):
            yield chunk

    async def cleanup_intermediates(self, job: JobRecord) -> None:
        """Remove rebuildable remote intermediate files."""
        await self.results.cleanup_intermediates(job)

    async def publish_archive(
        self,
        job: JobRecord,
        *,
        completed_at: int,
    ) -> FinalArchive:
        """Build and publish one immutable Result archive."""
        return await self.results.publish_archive(job, completed_at=completed_at)

    async def rebuild_artifact(self, job: JobRecord) -> AsyncIterator[bytes]:
        """Rebuild a published Result from authoritative remote files."""
        async for chunk in self.results.rebuild_artifact(job):
            yield chunk

    async def recover_archive(self, job: JobRecord) -> FinalArchive:
        """Recover a previously published immutable Result."""
        return await self.results.recover_archive(job)


__all__ = [
    "ArchiveNotReadyError",
    "FinalArchive",
    "GromacsReconciler",
    "GromacsResultInvalidError",
    "ModalGromacsAdapter",
    "PollOutcome",
    "ResultIdentityMismatchError",
    "SubmittedModalCall",
]
