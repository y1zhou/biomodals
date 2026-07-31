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
    Request,
    UploadFile,
)

from biomodals.app.bioinfo.gromacs_execution import execution_plan
from biomodals.execution.modal import ModalSubmissionOutcomeUnknownError
from biomodals.helper.pdb import validate_pdb_content
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.gromacs.contracts import (
    MAX_SIMULATION_TIME_NS,
    GromacsJobOptions,
    artifact_request_sha256,
    gromacs_run_name,
)
from biomodals.service.gromacs.execution import (
    GromacsExecutionAdapter,
    GromacsExecutionCoordinator,
)
from biomodals.service.http_contract import (
    CodedAPIError,
    CodedErrorResponse,
    ErrorResponse,
    PayloadTooLargeResponse,
    request_id_from,
    require_unsafe_session,
)
from biomodals.service.jobs import (
    JobLifecycleLocks,
    JobView,
    OpenOperationLogs,
    PreflightWorkload,
    ReadArtifact,
    Reconciler,
    WorkloadRegistration,
    can_view_job_logs,
)
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import (
    IdempotencyConflictError,
    JobLimitExceededError,
    JobRecord,
    ServiceStore,
    UserNotFoundError,
)
from biomodals.service.workloads import GROMACS_WORKLOAD

MAX_PDB_BYTES = 10 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
LOGGER = logging.getLogger(__name__)


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


class GromacsAdapter(GromacsExecutionAdapter, Protocol):
    """Complete GROMACS service boundary used by routes and coordination."""


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
) -> tuple[str, str, str]:
    parameters_json = options.model_dump_json()
    artifact_digest = artifact_request_sha256(pdb_content, parameters_json)
    encoded_display_identity = orjson.dumps({"display_name": display_identity})
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(artifact_digest))
    digest.update(encoded_display_identity)
    return (
        digest.hexdigest(),
        parameters_json,
        artifact_digest,
    )


def create_router(
    adapter: GromacsAdapter,
    *,
    lifecycle_locks: JobLifecycleLocks,
    job_logs_supported: bool,
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
        request_hash, parameters_json, artifact_digest = _request_identity(
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
        new_job_id = uuid4()
        execution_run_id = uuid4()
        run_name = gromacs_run_name(normalized_name, new_job_id)
        plan = execution_plan(
            cpu_only=options.cpu_only,
            workload_run_key=run_name,
            pdb_sha256=hashlib.sha256(pdb_content).hexdigest(),
            simulation_time_ns=options.simulation_time_ns,
            run_pdbfixer=options.run_pdbfixer,
        )
        try:
            admission = store.admit_job(
                owner_user_id=session.principal.user_id,
                display_name=normalized_name,
                idempotency_key=str(idempotency_key),
                request_hash=request_hash,
                parameters_json=parameters_json,
                artifact_request_sha256=artifact_digest,
                configuration=admission_configuration,
                now=now,
                new_job_id=new_job_id,
                execution_plan=plan,
                execution_run_id=execution_run_id,
                max_active_provider_calls=3,
                max_active_gpu_provider_calls=1,
                input_content=pdb_content,
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

        try:
            coordinator = GromacsExecutionCoordinator(
                store,
                adapter,
                lifecycle_locks=lifecycle_locks,
            )
            await coordinator.advance(admission.job.job_id)
        except ModalSubmissionOutcomeUnknownError:
            LOGGER.warning(
                "event=submission_outcome_unknown job_id=%s workload=gromacs "
                "request_id=%s",
                admission.job.job_id,
                request_id_from(request),
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

        job = store.get_job_by_id(admission.job.job_id)
        if job is None:  # pragma: no cover - admission owns the row
            raise RuntimeError("Admitted GROMACS Job disappeared")
        return JobView.from_record(
            job,
            definition=GROMACS_WORKLOAD,
            can_view_logs=can_view_job_logs(
                is_admin=session.principal.is_admin,
                owner_visibility_enabled=configuration.workload(
                    "gromacs"
                ).job_logs_visible_to_owner.value,
                logs_supported=job_logs_supported,
            ),
        )

    return router


def create_registration(
    adapter: GromacsAdapter,
    *,
    reconciler: Reconciler | None = None,
    lifecycle_locks: JobLifecycleLocks | None = None,
    read_artifact: ReadArtifact | None = None,
    rebuild_artifact: ReadArtifact | None = None,
    open_operation_logs: OpenOperationLogs | None = None,
    preflight: PreflightWorkload | None = None,
    max_pdb_bytes: int = MAX_PDB_BYTES,
) -> WorkloadRegistration:
    """Explicitly register GROMACS routes and lifecycle hooks."""

    async def cancel(store: ServiceStore, job: JobRecord) -> None:
        await GromacsExecutionCoordinator(
            store,
            adapter,
            lifecycle_locks=lifecycle_locks,
        ).cancel_job(job.job_id)

    if reconciler is not None and lifecycle_locks is None:
        raise ValueError("A reconciler must share the route lifecycle locks")
    lifecycle_locks = lifecycle_locks or JobLifecycleLocks()
    return WorkloadRegistration(
        definition=GROMACS_WORKLOAD,
        router=create_router(
            adapter,
            lifecycle_locks=lifecycle_locks,
            job_logs_supported=open_operation_logs is not None,
            max_pdb_bytes=max_pdb_bytes,
        ),
        lifecycle_locks=lifecycle_locks,
        reconciler=reconciler,
        cancel=cancel,
        read_artifact=read_artifact,
        rebuild_artifact=rebuild_artifact,
        open_operation_logs=open_operation_logs,
        preflight=preflight,
        max_body_bytes=max_pdb_bytes + MAX_MULTIPART_OVERHEAD_BYTES,
    )
