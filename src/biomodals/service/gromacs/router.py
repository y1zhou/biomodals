"""Workload-specific HTTP submission for GROMACS jobs."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal, Protocol
from uuid import UUID, uuid4

import orjson
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field

from biomodals.helper.pdb import validate_pdb_content
from biomodals.service.api import (
    CodedAPIError,
    CodedErrorResponse,
    ErrorResponse,
    MutationForbiddenResponse,
    require_unsafe_session,
)
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.jobs import JobView, Reconciler, WorkloadRegistration
from biomodals.service.runtime_config import (
    ModalConfigurationSnapshot,
    RuntimeConfiguration,
)
from biomodals.service.store import (
    IdempotencyConflictError,
    JobLimitExceededError,
    JobRecord,
    JobState,
    ServiceStore,
)

MAX_PDB_BYTES = 10 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_SIMULATION_TIME_NS = 200
LOGGER = logging.getLogger(__name__)


class GromacsJobOptions(BaseModel):
    """Bounded GROMACS settings accepted from the browser."""

    model_config = ConfigDict(frozen=True)

    simulation_time_ns: int = Field(default=5, ge=1, le=MAX_SIMULATION_TIME_NS)
    run_pdbfixer: bool = False
    cpu_only: bool = False


class PdbInvalidResponse(CodedErrorResponse):
    """Semantic PDB validation failure."""

    code: Literal["pdb_invalid"]


class SubmissionConflictResponse(CodedErrorResponse):
    """Submission conflicts with idempotency or active-job state."""

    code: Literal["idempotency_conflict", "active_job_limit_reached"]


class PayloadTooLargeResponse(CodedErrorResponse):
    """Submission exceeded the configured request boundary."""

    code: Literal["payload_too_large"]


class ComputeUnavailableResponse(CodedErrorResponse):
    """Remote compute could not accept the durable Job."""

    code: Literal["compute_unavailable"]


class SubmittedCall(Protocol):
    """Detached provider call returned after submission."""

    modal_call_id: str
    run_name: str
    provider_operation: str


class GromacsAdapter(Protocol):
    """Narrow Modal boundary used by this workload router."""

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
        *,
        run_name: str,
        modal_configuration: ModalConfigurationSnapshot,
    ) -> SubmittedCall:
        """Spawn one detached scientific job."""
        ...

    async def cancel(self, modal_call_id: str) -> None:
        """Request cancellation of one provider call graph."""
        ...


async def _read_pdb(upload: UploadFile, *, max_bytes: int) -> bytes:
    content = bytearray()
    try:
        while chunk := await upload.read(min(1024 * 1024, max_bytes + 1)):
            content.extend(chunk)
            if len(content) > max_bytes:
                raise CodedAPIError(
                    413,
                    "payload_too_large",
                    f"PDB upload exceeds {max_bytes} bytes",
                )
    finally:
        await upload.close()
    try:
        validate_pdb_content(bytes(content), max_bytes=max_bytes)
    except ValueError as exc:
        raise CodedAPIError(400, "pdb_invalid", str(exc)) from exc
    return bytes(content)


def _display_name(filename: str | None, supplied: str | None) -> str:
    if supplied is not None:
        return supplied
    safe_filename = (filename or "gromacs").replace("\\", "/")
    stem = PurePosixPath(safe_filename).stem.strip() or "gromacs"
    stem = re.sub(r"\s+", " ", stem)[:100]
    return f"{stem} · {datetime.now(UTC):%Y-%m-%d}"


def _request_identity(
    pdb_content: bytes,
    *,
    supplied_display_name: str | None,
    options: GromacsJobOptions,
) -> tuple[str, str]:
    encoded = orjson.dumps(options.model_dump(), option=orjson.OPT_SORT_KEYS)
    display_identity = orjson.dumps({"display_name": supplied_display_name})
    digest = hashlib.sha256()
    digest.update(len(pdb_content).to_bytes(8, "big"))
    digest.update(pdb_content)
    digest.update(encoded)
    digest.update(display_identity)
    return digest.hexdigest(), encoded.decode()


def create_router(
    adapter: GromacsAdapter,
    *,
    max_pdb_bytes: int = MAX_PDB_BYTES,
) -> APIRouter:
    """Create the GROMACS router around an injectable compute adapter."""
    router = APIRouter(prefix="/api/v1/gromacs", tags=["gromacs"])

    @router.post(
        "/jobs",
        response_model=JobView,
        response_model_exclude_none=True,
        status_code=202,
        responses={
            400: {"model": PdbInvalidResponse},
            401: {"model": ErrorResponse},
            403: {"model": MutationForbiddenResponse},
            409: {"model": SubmissionConflictResponse},
            413: {"model": PayloadTooLargeResponse},
            503: {"model": ComputeUnavailableResponse},
        },
    )
    async def submit_job(
        request: Request,
        session: Annotated[AuthenticatedSession, Depends(require_unsafe_session)],
        idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
        pdb: Annotated[UploadFile, File(description="Input PDB structure")],
        display_name: Annotated[str | None, Form(max_length=120)] = None,
        simulation_time_ns: Annotated[int, Form(ge=1, le=MAX_SIMULATION_TIME_NS)] = 5,
        run_pdbfixer: Annotated[bool, Form()] = False,
        cpu_only: Annotated[bool, Form()] = False,
    ) -> JobView:
        pdb_content = await _read_pdb(pdb, max_bytes=max_pdb_bytes)
        options = GromacsJobOptions(
            simulation_time_ns=simulation_time_ns,
            run_pdbfixer=run_pdbfixer,
            cpu_only=cpu_only,
        )
        normalized_supplied_name = (
            re.sub(r"\s+", " ", display_name).strip()
            if display_name is not None and display_name.strip()
            else None
        )
        normalized_name = _display_name(pdb.filename, normalized_supplied_name)
        request_hash, parameters_json = _request_identity(
            pdb_content,
            supplied_display_name=normalized_supplied_name,
            options=options,
        )
        store: ServiceStore = request.app.state.store
        configuration: RuntimeConfiguration = request.app.state.configuration
        admission_configuration = configuration.admission_configuration("gromacs")
        now = int(time.time())
        try:
            admission = store.admit_job(
                owner_user_id=session.principal.user_id,
                display_name=normalized_name,
                idempotency_key=str(idempotency_key),
                request_hash=request_hash,
                parameters_json=parameters_json,
                configuration=admission_configuration,
                now=now,
            )
        except IdempotencyConflictError as exc:
            raise CodedAPIError(409, "idempotency_conflict", str(exc)) from exc
        except JobLimitExceededError as exc:
            raise CodedAPIError(
                409,
                "active_job_limit_reached",
                str(exc),
            ) from exc

        if (
            admission.job.modal_call_id is not None
            or admission.job.state != JobState.QUEUED
        ):
            return JobView.from_record(admission.job)

        run_name = admission.job.run_name or f"api-{admission.job.job_id.hex}"
        submission_token = uuid4().hex
        claimed = store.claim_submission(
            admission.job.job_id,
            run_name=run_name,
            submission_token=submission_token,
            now=now,
        )
        if claimed is None:
            current = store.get_job(
                session.principal.user_id,
                admission.job.job_id,
            )
            if current is None:  # pragma: no cover - admission owns the row
                raise HTTPException(404, "Job not found")
            return JobView.from_record(current)

        try:
            submitted = await adapter.submit(
                pdb_content,
                options,
                run_name=run_name,
                modal_configuration=claimed.modal_configuration,
            )
            if submitted.run_name != run_name:
                raise RuntimeError("Compute returned the wrong GROMACS run name")
            job = store.mark_submitted(
                admission.job.job_id,
                modal_call_id=submitted.modal_call_id,
                provider_operation=submitted.provider_operation,
                run_name=submitted.run_name,
                submission_token=submission_token,
            )
        except Exception as exc:
            LOGGER.exception("Could not submit GROMACS job %s", admission.job.job_id)
            store.release_submission(
                admission.job.job_id,
                submission_token=submission_token,
                now=int(time.time()),
            )
            raise CodedAPIError(
                503,
                "compute_unavailable",
                "GROMACS compute is temporarily unavailable",
            ) from exc
        return JobView.from_record(job)

    return router


def create_registration(
    adapter: GromacsAdapter,
    *,
    reconciler: Reconciler | None = None,
    max_pdb_bytes: int = MAX_PDB_BYTES,
) -> WorkloadRegistration:
    """Explicitly register GROMACS routes and lifecycle hooks."""

    async def cancel(job: JobRecord) -> None:
        if job.modal_call_id is not None:
            await adapter.cancel(job.modal_call_id)

    read_artifact = getattr(adapter, "read_artifact", None)
    return WorkloadRegistration(
        name="gromacs",
        router=create_router(
            adapter,
            max_pdb_bytes=max_pdb_bytes,
        ),
        reconciler=reconciler,
        cancel=cancel,
        read_artifact=read_artifact,
        max_body_bytes=max_pdb_bytes + MAX_MULTIPART_OVERHEAD_BYTES,
    )
