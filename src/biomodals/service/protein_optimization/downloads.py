"""Short-lived selected-ID tickets without duplicating candidate CSVs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from biomodals.helper.artifacts import read_bounded_file_bytes, replace_bytes_atomic
from biomodals.service.protein_optimization.contracts import (
    MAX_REQUEST_BYTES,
    OptimizationDownloadTicket,
    OptimizationSelectedCandidates,
)
from biomodals.service.protein_optimization.results import selected_csv

TICKET_SECONDS = 300


class SelectedDownload(OptimizationSelectedCandidates):
    """Local-only intent, bound to owner, Job and exact cached scientific result."""

    owner_user_id: UUID
    job_id: UUID
    token: UUID
    expires_at: int
    csv_sha256: str


def prepare_selection(
    path: Path,
    index: Path,
    csv_sha256: str,
    ids: list[str],
    *,
    owner_user_id: UUID,
    job_id: UUID,
    now: int,
) -> OptimizationDownloadTicket:
    """Validate all IDs before replacing the single outstanding ticket for this Job."""
    stream = selected_csv(index, csv_sha256, ids)
    try:
        next(stream)
    finally:
        stream.close()
    retained = SelectedDownload(
        owner_user_id=owner_user_id,
        job_id=job_id,
        token=uuid4(),
        expires_at=now + TICKET_SECONDS,
        csv_sha256=csv_sha256,
        ids=ids,
    )
    replace_bytes_atomic(path, retained.model_dump_json().encode())
    return OptimizationDownloadTicket(
        download_url=f"/api/v1/protein-optimization/jobs/{job_id}/candidates.csv?ticket={retained.token}",
        expires_at=datetime.fromtimestamp(retained.expires_at, UTC),
    )


def load_selection(
    path: Path,
    *,
    token: UUID,
    owner_user_id: UUID,
    job_id: UUID,
    csv_sha256: str,
    now: int,
) -> list[str]:
    """A URL is not a bearer capability; exact owner access is checked again."""
    retained = SelectedDownload.model_validate_json(
        read_bounded_file_bytes(
            path, field_name="Selected download", max_bytes=MAX_REQUEST_BYTES
        )
    )
    if (
        retained.token != token
        or retained.owner_user_id != owner_user_id
        or retained.job_id != job_id
        or retained.csv_sha256 != csv_sha256
        or retained.expires_at <= now
    ):
        raise ValueError(
            "Selected download has expired or was replaced; prepare it again"
        )
    return retained.ids
