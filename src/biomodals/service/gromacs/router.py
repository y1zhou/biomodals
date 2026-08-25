"""Typed GROMACS service submission."""

from __future__ import annotations

import hashlib
import re
import time
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated
from uuid import UUID, uuid4

import orjson
from fastapi import APIRouter, Depends, File, Form, Header, Request, UploadFile

from biomodals.app.bioinfo.gromacs_execution import concrete_gromacs_seed
from biomodals.app.bioinfo.gromacs_execution_runtime import GromacsExecutionRequest
from biomodals.execution import DeploymentIdentity
from biomodals.helper.pdb import validate_pdb_content
from biomodals.service.auth import AuthenticatedSession
from biomodals.service.gromacs.contracts import (
    MAX_SIMULATION_TIME_NS,
    GromacsJobOptions,
    gromacs_run_name,
)
from biomodals.service.http_contract import CodedAPIError, require_unsafe_session
from biomodals.service.jobs import JobView
from biomodals.service.pending import PendingRequestStore
from biomodals.service.remote_execution import RemoteExecutionClient
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import (
    IdempotencyConflictError,
    JobLimitExceededError,
    ServiceStore,
    UserNotFoundError,
)
from biomodals.service.tool_runtime import JobLifecycle

MAX_PDB_BYTES = 10 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024


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
    except ValueError as error:
        raise CodedAPIError(400, "pdb_invalid", str(error)) from error
    return bytes(content)


def create_router(
    *,
    store: ServiceStore,
    configuration: RuntimeConfiguration,
    pending: PendingRequestStore,
    remote: RemoteExecutionClient,
    lifecycle: JobLifecycle,
    max_pdb_bytes: int = MAX_PDB_BYTES,
) -> APIRouter:
    """Create the GROMACS endpoint around shared admission and lifecycle."""
    router = APIRouter(prefix="/api/v1/gromacs", tags=["gromacs"])

    @router.post("/jobs", response_model=JobView, status_code=202)
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
        normalized_name = _display_name(pdb.filename, display_name)
        digest = _submission_digest(pdb_content, normalized_name, options)
        replay = store.find_idempotent_job(
            session.principal.user_id,
            tool="gromacs",
            idempotency_key=str(idempotency_key),
        )
        if replay is not None:
            if replay.request_digest != digest:
                raise CodedAPIError(
                    409,
                    "idempotency_conflict",
                    "Idempotency key was already used for another request",
                )
            return _view(replay, session, configuration)

        effective = configuration.tool("gromacs")
        deployment = DeploymentIdentity(
            configuration.modal_environment().value,
            effective.modal_app_name.value,
            effective.modal_app_version.value,
        )
        await remote.preflight(deployment)
        job_id = uuid4()
        run_name = gromacs_run_name(normalized_name, job_id)
        execution_request = GromacsExecutionRequest(
            run_name=run_name,
            pdb_content=pdb_content,
            simulation_time_ns=options.simulation_time_ns,
            run_pdbfixer=options.run_pdbfixer,
            cpu_only=options.cpu_only,
            num_threads=16,
            use_openmp_threads=False,
            ld_seed=concrete_gromacs_seed(
                -1,
                run_identity=str(job_id),
                purpose="ld-seed",
            ),
            gen_seed=concrete_gromacs_seed(
                -1,
                run_identity=str(job_id),
                purpose="gen-seed",
            ),
            genion_seed=concrete_gromacs_seed(
                0,
                run_identity=str(job_id),
                purpose="genion-seed",
                random_sentinel=0,
            ),
            max_active_provider_calls=effective.max_active_provider_calls,
            max_active_gpu_provider_calls=effective.max_active_gpu_provider_calls,
        )
        pending.put(job_id, execution_request.to_bytes())
        try:
            admission = store.admit_job(
                owner_user_id=session.principal.user_id,
                tool="gromacs",
                display_name=normalized_name,
                idempotency_key=str(idempotency_key),
                request_digest=digest,
                modal_environment=deployment.environment,
                modal_app_name=deployment.deployment_name,
                modal_app_version=deployment.deployment_version,
                tool_active_job_limit=effective.active_job_limit.value,
                global_active_job_limit=configuration.global_active_job_limit().value,
                max_active_provider_calls=execution_request.max_active_provider_calls,
                max_active_gpu_provider_calls=(
                    execution_request.max_active_gpu_provider_calls
                ),
                now=int(time.time()),
                new_job_id=job_id,
            )
        except (IdempotencyConflictError, JobLimitExceededError) as error:
            pending.delete(job_id)
            raise CodedAPIError(409, "job_conflict", str(error)) from error
        except UserNotFoundError as error:
            pending.delete(job_id)
            raise CodedAPIError(403, "account_disabled", str(error)) from error
        try:
            job = await lifecycle.advance(admission.job.job_id)
        except Exception:
            job = admission.job
        return _view(job, session, configuration)

    return router


def _submission_digest(
    content: bytes,
    display_name: str,
    options: GromacsJobOptions,
) -> str:
    digest = hashlib.sha256(content)
    digest.update(options.model_dump_json().encode())
    digest.update(orjson.dumps({"display_name": display_name}))
    return digest.hexdigest()


def _display_name(filename: str | None, supplied: str | None) -> str:
    if supplied is not None and supplied.strip():
        return re.sub(r"\s+", " ", supplied).strip()
    safe_filename = (filename or "gromacs").replace("\\", "/")
    stem = PurePosixPath(safe_filename).stem.strip() or "gromacs"
    return f"{re.sub(r'\s+', ' ', stem)[:100]} · {datetime.now(UTC):%Y-%m-%d}"


def _view(
    job,
    session: AuthenticatedSession,
    configuration: RuntimeConfiguration,
) -> JobView:
    return JobView.from_record(
        job,
        can_view_logs=(
            session.principal.is_admin
            or configuration.tool("gromacs").job_logs_visible_to_owner.value
        ),
    )
