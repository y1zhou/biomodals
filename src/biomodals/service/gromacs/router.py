"""Workload-specific HTTP submission for GROMACS jobs."""

from __future__ import annotations

import hashlib
import logging
import re
import time
import unicodedata
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
    Request,
    UploadFile,
)
from pydantic import BaseModel, ConfigDict, Field

from biomodals.helper.pdb import validate_pdb_content
from biomodals.service.api import (
    CodedAPIError,
    CodedErrorResponse,
    ErrorResponse,
    PayloadTooLargeResponse,
    request_id_from,
    require_unsafe_session,
)
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.gromacs.plan import prepare_operation
from biomodals.service.jobs import (
    JobLifecycleLocks,
    JobView,
    PreflightWorkload,
    ReadArtifact,
    Reconciler,
    WorkloadRegistration,
)
from biomodals.service.runtime_config import (
    ModalConfigurationSnapshot,
    RuntimeConfiguration,
)
from biomodals.service.store import (
    IdempotencyConflictError,
    JobLimitExceededError,
    JobOperationState,
    JobRecord,
    ServiceStore,
    UserNotFoundError,
)
from biomodals.service.submission import (
    ModalJobSubmitter,
    SubmittedModalOperation,
)
from biomodals.service.workloads import GROMACS_WORKLOAD

MAX_PDB_BYTES = 10 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_SIMULATION_TIME_NS = 200
LOGGER = logging.getLogger(__name__)
_RUN_NAME_SEPARATOR = re.compile(r"[^a-z0-9]+")
_RUN_NAME_SUFFIX = re.compile(r"[0-9a-f]{32}")
_MAX_RUN_NAME_SLUG_LENGTH = 64


def gromacs_run_name(display_name: str, job_id: UUID) -> str:
    """Build a readable, path-safe name with collision-proof job identity."""
    ascii_name = (
        unicodedata
        .normalize("NFKD", display_name)
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    slug = _RUN_NAME_SEPARATOR.sub("-", ascii_name).strip("-")
    slug = slug[:_MAX_RUN_NAME_SLUG_LENGTH].rstrip("-")
    return f"{slug or 'gromacs-simulation'}-{job_id.hex}"


def is_gromacs_run_name(value: str) -> bool:
    """Return whether a service-generated GROMACS run name is path-safe."""
    slug, separator, suffix = value.rpartition("-")
    return bool(
        separator
        and 1 <= len(slug) <= _MAX_RUN_NAME_SLUG_LENGTH
        and _RUN_NAME_SEPARATOR.sub("-", slug) == slug
        and not slug.startswith("-")
        and not slug.endswith("-")
        and _RUN_NAME_SUFFIX.fullmatch(suffix)
    )


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


class ComputeUnavailableResponse(CodedErrorResponse):
    """Remote compute could not accept the durable Job."""

    code: Literal["compute_unavailable"]


class SubmissionForbiddenResponse(CodedErrorResponse):
    """Account state changed after browser Session authentication."""

    code: Literal["account_disabled", "csrf_invalid", "origin_not_allowed"]


class GromacsAdapter(Protocol):
    """Narrow Modal boundary used by this workload router."""

    async def submit(
        self,
        pdb_content: bytes,
        options: GromacsJobOptions,
        *,
        run_name: str,
        modal_configuration: ModalConfigurationSnapshot,
    ) -> SubmittedModalOperation:
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


def _filename_display_identity(filename: str | None) -> str:
    """Return the stable filename-derived portion of a generated display name."""
    safe_filename = (filename or "gromacs").replace("\\", "/")
    stem = PurePosixPath(safe_filename).stem.strip() or "gromacs"
    return re.sub(r"\s+", " ", stem)[:100]


def _display_name(filename: str | None, supplied: str | None) -> str:
    if supplied is not None:
        return supplied
    return f"{_filename_display_identity(filename)} · {datetime.now(UTC):%Y-%m-%d}"


def _request_identity(
    pdb_content: bytes,
    *,
    display_identity: str,
    options: GromacsJobOptions,
) -> tuple[str, str]:
    encoded = orjson.dumps(options.model_dump(), option=orjson.OPT_SORT_KEYS)
    encoded_display_identity = orjson.dumps({"display_name": display_identity})
    digest = hashlib.sha256()
    digest.update(len(pdb_content).to_bytes(8, "big"))
    digest.update(pdb_content)
    digest.update(encoded)
    digest.update(encoded_display_identity)
    return digest.hexdigest(), encoded.decode()


def create_router(
    adapter: GromacsAdapter,
    *,
    lifecycle_locks: JobLifecycleLocks,
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
            403: {"model": SubmissionForbiddenResponse},
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
            display_identity=(
                normalized_supplied_name or _filename_display_identity(pdb.filename)
            ),
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
        except UserNotFoundError as exc:
            raise CodedAPIError(
                403,
                "account_disabled",
                "This account cannot submit new jobs",
            ) from exc

        LOGGER.info(
            "event=job_admission job_id=%s workload=gromacs replay=%s "
            "stage=prepare_simulation request_id=%s",
            admission.job.job_id,
            not admission.created,
            request_id_from(request),
        )

        run_name = admission.job.run_name or gromacs_run_name(
            admission.job.display_name,
            admission.job.job_id,
        )
        operation = prepare_operation(cpu_only=options.cpu_only)
        submitter = ModalJobSubmitter(store, lifecycle_locks)

        async def spawn(claimed_job: JobRecord) -> SubmittedModalOperation:
            return await adapter.submit(
                pdb_content,
                options,
                run_name=run_name,
                modal_configuration=claimed_job.modal_configuration,
            )

        try:
            result = await submitter.submit(
                admission.job,
                operation=operation,
                run_name=run_name,
                submission_token=uuid4().hex,
                spawn=spawn,
                cancel=adapter.cancel,
            )
        except Exception as exc:
            LOGGER.exception(
                "Could not submit GROMACS job %s request_id=%s",
                admission.job.job_id,
                request_id_from(request),
            )
            raise CodedAPIError(
                503,
                "compute_unavailable",
                "GROMACS compute is temporarily unavailable",
            ) from exc

        job = result.job
        if result.attached:
            stage = JobView.from_record(job).stage
            LOGGER.info(
                "event=stage_attached job_id=%s workload=gromacs stage=%s "
                "function=%s request_id=%s",
                job.job_id,
                stage.code if stage is not None else "none",
                operation.partition(":")[0],
                request_id_from(request),
            )
        return JobView.from_record(job)

    return router


def create_registration(
    adapter: GromacsAdapter,
    *,
    reconciler: Reconciler | None = None,
    lifecycle_locks: JobLifecycleLocks | None = None,
    read_artifact: ReadArtifact | None = None,
    rebuild_artifact: ReadArtifact | None = None,
    preflight: PreflightWorkload | None = None,
    max_pdb_bytes: int = MAX_PDB_BYTES,
) -> WorkloadRegistration:
    """Explicitly register GROMACS routes and lifecycle hooks."""

    async def cancel(store: ServiceStore, job: JobRecord) -> None:
        first_error: Exception | None = None
        for call in store.list_operations(job.job_id):
            if call.state != JobOperationState.RUNNING or call.modal_call_id is None:
                continue
            try:
                await adapter.cancel(call.modal_call_id)
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error

    if reconciler is not None and lifecycle_locks is None:
        raise ValueError("A reconciler must share the route lifecycle locks")
    lifecycle_locks = lifecycle_locks or JobLifecycleLocks()
    return WorkloadRegistration(
        definition=GROMACS_WORKLOAD,
        router=create_router(
            adapter,
            lifecycle_locks=lifecycle_locks,
            max_pdb_bytes=max_pdb_bytes,
        ),
        lifecycle_locks=lifecycle_locks,
        reconciler=reconciler,
        cancel=cancel,
        read_artifact=read_artifact,
        rebuild_artifact=rebuild_artifact,
        preflight=preflight,
        max_body_bytes=max_pdb_bytes + MAX_MULTIPART_OVERHEAD_BYTES,
    )
