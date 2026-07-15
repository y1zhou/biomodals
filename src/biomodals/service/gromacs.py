"""HTTP routes and Modal adapter for asynchronous GROMACS jobs.

Run an external control plane with
``uv run --extra api uvicorn --factory biomodals.service.gromacs:create_deployed_app``.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Mapping
from os import environ
from typing import Annotated, Any, Protocol, cast
from uuid import uuid4

import modal
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from biomodals.helper.pdb import validate_pdb_content
from biomodals.schema import AppRunResult
from biomodals.service.api import (
    MAX_MULTIPART_OVERHEAD_BYTES,
    ErrorResponse,
    create_api,
    model_response,
)

if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum
else:
    from backports.strenum import StrEnum  # noqa: UP035,I001

MAX_PDB_BYTES = 10 * 1024 * 1024


class JobNotFoundError(LookupError):
    """Raised when the GROMACS backend cannot resolve a public job id."""

    def __init__(self, job_id: str) -> None:
        """Remember the missing id for the HTTP error response."""
        self.job_id = job_id
        super().__init__(f"Job '{job_id}' was not found")


class JobBackendUnavailableError(RuntimeError):
    """Raised when the GROMACS backend cannot currently answer a request."""


class JobState(StrEnum):
    """Provider-neutral states exposed by the GROMACS job API."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class GromacsJobOptions(BaseModel):
    """Bounded user settings accepted by the public API."""

    model_config = ConfigDict(frozen=True)

    simulation_time_ns: int = Field(default=5, ge=1, le=100)
    run_pdbfixer: bool = False
    cpu_only: bool = False


class JobStatus(BaseModel):
    """Current state of one submitted job."""

    job_id: str
    status: JobState
    run_name: str | None = None
    detail: str | None = None


class JobArtifactFile(BaseModel):
    """One provider-neutral file in an output artifact."""

    path: str
    role: str | None = None
    media_type: str | None = None


class JobArtifact(BaseModel):
    """One named output and its file manifest."""

    name: str
    kind: str
    files: list[JobArtifactFile] = Field(default_factory=list)


class JobResult(BaseModel):
    """Portable result returned after a GROMACS job completes."""

    run_name: str
    artifacts: list[JobArtifact] = Field(default_factory=list)


class JobSnapshot(JobStatus):
    """Backend snapshot used by status and result routes."""

    result: JobResult | None = None


class JobBackend(Protocol):
    """Compute boundary used by the HTTP layer."""

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
    ) -> JobStatus:
        """Submit a GROMACS job without waiting for its result."""
        ...

    async def inspect(self, job_id: str) -> JobSnapshot:
        """Poll one job without waiting for completion."""
        ...

    async def cancel(self, job_id: str) -> JobStatus:
        """Request cancellation of one job."""
        ...


async def _read_pdb(upload: UploadFile, *, max_bytes: int) -> bytes:
    content = bytearray()
    while chunk := await upload.read(min(1024 * 1024, max_bytes + 1)):
        content.extend(chunk)
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"PDB upload exceeds the {max_bytes}-byte limit",
            )
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="PDB upload is empty",
        )
    try:
        validate_pdb_content(bytes(content), max_bytes=max_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return bytes(content)


def create_app(
    backend: JobBackend,
    *,
    max_pdb_bytes: int = MAX_PDB_BYTES,
    api_key: str | None = None,
    trusted_proxy_auth: bool = False,
) -> FastAPI:
    """Create a FastAPI app around a pluggable GROMACS job backend."""
    web_app = create_api(
        title="Biomodals GROMACS API",
        version="1.0.0",
        max_body_bytes=max_pdb_bytes + MAX_MULTIPART_OVERHEAD_BYTES,
        api_key=api_key,
        trusted_proxy_auth=trusted_proxy_auth,
    )

    @web_app.exception_handler(JobNotFoundError)
    async def job_not_found(
        _request: Request,
        exc: JobNotFoundError,
    ) -> Response:
        return model_response(
            ErrorResponse(detail=str(exc)),
            status_code=status.HTTP_404_NOT_FOUND,
        )

    @web_app.exception_handler(JobBackendUnavailableError)
    async def backend_unavailable(
        _request: Request,
        _exc: JobBackendUnavailableError,
    ) -> Response:
        return model_response(
            ErrorResponse(detail="Job backend is temporarily unavailable"),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @web_app.post(
        "/jobs",
        response_model=JobStatus,
        response_model_exclude_none=True,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_job(
        pdb: Annotated[UploadFile, File(description="Input structure in PDB format")],
        simulation_time_ns: Annotated[int, Form(ge=1, le=100)] = 5,
        run_pdbfixer: Annotated[bool, Form()] = False,
        cpu_only: Annotated[bool, Form()] = False,
    ) -> JobStatus:
        try:
            pdb_content = await _read_pdb(pdb, max_bytes=max_pdb_bytes)
        finally:
            await pdb.close()
        return await backend.submit(
            pdb_content,
            GromacsJobOptions(
                simulation_time_ns=simulation_time_ns,
                run_pdbfixer=run_pdbfixer,
                cpu_only=cpu_only,
            ),
        )

    @web_app.get(
        "/jobs/{job_id}",
        response_model=JobStatus,
        response_model_exclude_none=True,
    )
    async def inspect_job(job_id: str) -> JobStatus:
        return await backend.inspect(job_id)

    @web_app.get(
        "/jobs/{job_id}/result",
        response_model=JobResult,
        response_model_exclude_none=True,
        responses={status.HTTP_202_ACCEPTED: {"model": JobStatus}},
    )
    async def get_job_result(job_id: str) -> JobResult | Response:
        snapshot = await backend.inspect(job_id)
        if snapshot.status == JobState.PENDING:
            return model_response(
                JobStatus.model_validate(snapshot),
                status_code=status.HTTP_202_ACCEPTED,
            )
        if snapshot.result is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Job is {snapshot.status}",
            )
        return snapshot.result

    @web_app.delete(
        "/jobs/{job_id}",
        response_model=JobStatus,
        response_model_exclude_none=True,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def cancel_job(job_id: str) -> JobStatus:
        return await backend.cancel(job_id)

    return web_app


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
