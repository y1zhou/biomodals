"""Modal adapter for the provider-neutral GROMACS job API.

To run the control plane outside Modal after deploying ``GromacsAPI``, set
``BIOMODALS_API_KEY`` and use
``uv run --extra api uvicorn --factory biomodals.service.modal_gromacs:create_deployed_app``.
The factory fails closed when that bearer token is absent. Set
``BIOMODALS_GROMACS_APP`` when targeting a separately deployed GROMACS app.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from os import environ
from typing import Any, Protocol, cast
from uuid import uuid4

import modal
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, ValidationError

from biomodals.schema import AppRunResult
from biomodals.service.gromacs_api import (
    GromacsJobOptions,
    JobArtifact,
    JobArtifactFile,
    JobBackendUnavailableError,
    JobNotFoundError,
    JobResult,
    JobSnapshot,
    JobState,
    JobStatus,
    create_app,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_JOB_DICT = "biomodals-gromacs-api-jobs"
_MODAL_SERVICE_ERRORS = (
    modal.exception.AuthError,
    modal.exception.ConnectionError,
    modal.exception.InternalError,
    modal.exception.InvalidError,
    modal.exception.NotFoundError,
    modal.exception.PermissionDeniedError,
    modal.exception.ResourceExhaustedError,
    modal.exception.ServiceError,
)


class _AsyncGet(Protocol):
    def __call__(self, key: str, default: Any = None) -> Any: ...

    async def aio(self, key: str, default: Any = None) -> Any: ...


class _AsyncPut(Protocol):
    def __call__(self, key: str, value: Any) -> bool: ...

    async def aio(self, key: str, value: Any) -> bool: ...


class JobRegistry(Protocol):
    """Minimal distributed key-value boundary used by the Modal adapter."""

    get: _AsyncGet
    put: _AsyncPut


class ModalJobRecord(BaseModel):
    """Private lookup record; clients only receive its opaque public id."""

    model_config = ConfigDict(frozen=True)

    modal_call_id: str
    run_name: str


def create_job_registry() -> modal.Dict:
    """Look up the shared registry used to validate public API job ids."""
    return modal.Dict.from_name(
        environ.get("BIOMODALS_GROMACS_JOB_DICT", DEFAULT_JOB_DICT),
        create_if_missing=True,
    )


class ModalGromacsBackend:
    """Submit GROMACS jobs through one deployed Modal function."""

    def __init__(
        self,
        job_function: modal.Function,
        job_registry: modal.Dict | JobRegistry,
        *,
        call_resolver: Callable[[str], modal.FunctionCall] = modal.FunctionCall.from_id,
    ) -> None:
        """Bind the adapter to a GROMACS function and distributed job registry."""
        self._job_function = job_function
        self._job_registry = cast(JobRegistry, job_registry)
        self._call_resolver = call_resolver

    @staticmethod
    def _failed_snapshot(job_id: str, run_name: str) -> JobSnapshot:
        return JobSnapshot(
            job_id=job_id,
            status=JobState.FAILED,
            run_name=run_name,
            detail="GROMACS job failed; inspect Modal logs",
        )

    async def _resolve_job(
        self,
        job_id: str,
    ) -> tuple[ModalJobRecord, modal.FunctionCall]:
        if not job_id.startswith("job-"):
            raise JobNotFoundError(job_id)
        try:
            raw_record = await self._job_registry.get.aio(job_id)
        except _MODAL_SERVICE_ERRORS as exc:
            raise JobBackendUnavailableError from exc
        if raw_record is None:
            raise JobNotFoundError(job_id)
        try:
            if isinstance(raw_record, (bytes, str)):
                record = ModalJobRecord.model_validate_json(raw_record)
            else:
                record = ModalJobRecord.model_validate(raw_record)
        except ValidationError as exc:
            raise JobNotFoundError(job_id) from exc
        try:
            call = self._call_resolver(record.modal_call_id)
        except (modal.exception.InvalidError, modal.exception.NotFoundError) as exc:
            raise JobNotFoundError(job_id) from exc
        return record, call

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
    ) -> JobStatus:
        """Spawn a uniquely named Modal call and return immediately."""
        run_name = f"api-{uuid4().hex}"
        job_id = f"job-{uuid4().hex}"
        try:
            call = await self._job_function.spawn.aio(
                pdb_content=pdb_content,
                run_name=run_name,
                simulation_time_ns=options.simulation_time_ns,
                run_pdbfixer=options.run_pdbfixer,
                cpu_only=options.cpu_only,
            )
        except _MODAL_SERVICE_ERRORS as exc:
            raise JobBackendUnavailableError from exc

        record = ModalJobRecord(
            modal_call_id=call.object_id,
            run_name=run_name,
        )
        try:
            await self._job_registry.put.aio(job_id, record.model_dump_json())
        except Exception as exc:
            try:
                await call.cancel.aio()
            except (Exception, modal.exception.InputCancellation):
                LOGGER.exception("Could not cancel unregistered Modal call")
            raise JobBackendUnavailableError from exc
        return JobStatus(
            job_id=job_id,
            status=JobState.PENDING,
            run_name=run_name,
        )

    async def _timeout_was_poll(self, call: modal.FunctionCall) -> bool:
        get_call_graph = getattr(call, "get_call_graph", None)
        if get_call_graph is None:
            return True
        try:
            graph = await get_call_graph.aio()
        except _MODAL_SERVICE_ERRORS as exc:
            raise JobBackendUnavailableError from exc
        return not graph or graph[0].status == modal.call_graph.InputStatus.PENDING

    async def inspect(self, job_id: str) -> JobSnapshot:
        """Poll a registered Modal call and translate its completed result."""
        record, call = await self._resolve_job(job_id)
        try:
            raw_result = await call.get.aio(timeout=0)
        except modal.exception.OutputExpiredError:
            return JobSnapshot(
                job_id=job_id,
                status=JobState.EXPIRED,
                run_name=record.run_name,
                detail="Modal call output expired; Volume artifacts remain",
            )
        except (modal.exception.InvalidError, modal.exception.NotFoundError) as exc:
            raise JobNotFoundError(job_id) from exc
        except modal.exception.InputCancellation:
            return JobSnapshot(
                job_id=job_id,
                status=JobState.CANCELLED,
                run_name=record.run_name,
            )
        except _MODAL_SERVICE_ERRORS as exc:
            raise JobBackendUnavailableError from exc
        except TimeoutError:
            if await self._timeout_was_poll(call):
                return JobSnapshot(
                    job_id=job_id,
                    status=JobState.PENDING,
                    run_name=record.run_name,
                )
            LOGGER.exception("GROMACS job %s raised TimeoutError", job_id)
            return self._failed_snapshot(job_id, record.run_name)
        except Exception:
            LOGGER.exception("GROMACS job %s failed", job_id)
            return self._failed_snapshot(job_id, record.run_name)

        try:
            result = AppRunResult.model_validate(raw_result)
            artifacts = []
            for output in result.outputs:
                output_run_name = output.metadata.get("run_name")
                if output_run_name is not None and output_run_name != record.run_name:
                    raise ValueError("GROMACS result run name does not match its job")
                raw_files = output.metadata.get("files")
                files = (
                    [
                        JobArtifactFile.model_validate(file)
                        for file in raw_files
                        if isinstance(file, Mapping)
                    ]
                    if isinstance(raw_files, list)
                    else []
                )
                artifacts.append(
                    JobArtifact(
                        name=output.name,
                        kind=str(output.kind),
                        files=files,
                    )
                )
        except Exception:
            LOGGER.exception("GROMACS job %s returned an invalid result", job_id)
            return self._failed_snapshot(job_id, record.run_name)
        return JobSnapshot(
            job_id=job_id,
            status=JobState(result.status),
            run_name=record.run_name,
            result=JobResult(run_name=record.run_name, artifacts=artifacts),
        )

    async def cancel(self, job_id: str) -> JobStatus:
        """Cancel a registered Modal call without blocking the ASGI event loop."""
        record, call = await self._resolve_job(job_id)
        try:
            await call.cancel.aio()
        except (modal.exception.InvalidError, modal.exception.NotFoundError) as exc:
            raise JobNotFoundError(job_id) from exc
        except _MODAL_SERVICE_ERRORS as exc:
            raise JobBackendUnavailableError from exc
        return JobStatus(
            job_id=job_id,
            status=JobState.CANCELLED,
            run_name=record.run_name,
        )


def create_deployed_app() -> FastAPI:
    """Create an authenticated Uvicorn app backed by deployed Modal objects."""
    api_key = environ.get("BIOMODALS_API_KEY")
    if not api_key:
        raise RuntimeError("BIOMODALS_API_KEY is required for external serving")
    job_function = modal.Function.from_name(
        environ.get("BIOMODALS_GROMACS_APP", "GromacsAPI"),
        "run_gromacs_job",
    )
    return create_app(
        ModalGromacsBackend(job_function, create_job_registry()),
        api_key=api_key,
    )
