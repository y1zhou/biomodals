"""Immutable source evidence for checkpoint-based production extensions."""

from __future__ import annotations

import asyncio
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from biomodals.app.bioinfo.gromacs.execution import GROMACS_SCIENTIFIC_VERSION
from biomodals.helper.io import require_safe_filename_component
from biomodals.helper.modal_volume import read_modal_volume_file

if TYPE_CHECKING:
    from biomodals.app.bioinfo.gromacs.execution_runtime import GromacsExecutionRequest

MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


class ContinuationSource(BaseModel):
    """Bind a new plan to one completed source and its native checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    execution_run_id: UUID
    run_name: str
    file_stem: str
    simulation_time_ns: int = Field(ge=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("run_name", "file_stem")
    @classmethod
    def safe_component(cls, value: str) -> str:
        """Keep saved source paths within a single app-owned directory."""
        require_safe_filename_component(value, field_name="continuation source")
        return value


async def read_continuation_source(
    volume: Any, execution_run_id: UUID
) -> tuple[ContinuationSource, GromacsExecutionRequest]:
    """Inspect retained files through Volume reads, without launching compute.

    CPU preparation subsequently validates all content and native endpoint data.
    A final publication is required even for historical pre-continuation runs.
    """
    from biomodals.app.bioinfo.gromacs.execution import (
        PREPARE_CONTINUATION,
        PREPARE_RESULT,
    )
    from biomodals.app.bioinfo.gromacs.execution_runtime import (
        gromacs_node_paths,
        gromacs_publication_path,
        load_execution_request_from_volume,
        parse_gromacs_publication,
    )

    request = await asyncio.to_thread(
        load_execution_request_from_volume, volume, execution_run_id
    )
    if request.gromacs_version != GROMACS_SCIENTIFIC_VERSION:
        raise ValueError("Source uses an incompatible GROMACS version")
    marker_path = gromacs_publication_path(request, PREPARE_RESULT).as_posix()
    marker = await read_modal_volume_file(volume, marker_path, max_bytes=1024 * 1024)
    if parse_gromacs_publication(request, PREPARE_RESULT, marker) is None:
        raise ValueError("Source has no valid completed scientific publication")
    production = "production_run_cpu" if request.cpu_only else "production_run_gpu"
    native_marker = await read_modal_volume_file(
        volume,
        gromacs_publication_path(request, production).as_posix(),
        max_bytes=1024 * 1024,
    )
    native_files = parse_gromacs_publication(request, production, native_marker)
    if native_files is None:
        raise ValueError("Source native production publication is unavailable")
    prefix = f"{request.run_name}/production_{request.file_stem}"
    entries = {
        entry.path.lstrip("/"): entry
        for entry in await volume.listdir.aio(request.run_name)
    }
    for suffix in (".tpr", ".xtc", ".edr", ".log", ".cpt"):
        entry = entries.get(prefix + suffix)
        if entry is None or entry.size <= 0:
            raise FileNotFoundError(
                "Source checkpoint or native append outputs are missing"
            )
    inherited = set(gromacs_node_paths(request, PREPARE_CONTINUATION)) - {
        "continuation.json",
        "source.tpr",
    }
    if any(f"{request.run_name}/{name}" not in entries for name in inherited):
        raise FileNotFoundError("Source inherited inputs or analyses are missing")
    checkpoint = await read_modal_volume_file(
        volume, prefix + ".cpt", max_bytes=MAX_CHECKPOINT_BYTES
    )
    if not checkpoint:
        raise ValueError("Source checkpoint is empty")
    checkpoint_digest = sha256(checkpoint).hexdigest()
    published_checkpoint = next(
        (file for file in native_files if file.path.endswith(".cpt")), None
    )
    if (
        published_checkpoint
        and published_checkpoint.content_sha256 != checkpoint_digest
    ):
        raise ValueError("Source checkpoint changed after publication")
    return ContinuationSource(
        execution_run_id=execution_run_id,
        run_name=request.run_name,
        file_stem=request.file_stem,
        simulation_time_ns=request.simulation_time_ns,
        request_sha256=sha256(request.to_bytes()).hexdigest(),
        publication_sha256=sha256(marker).hexdigest(),
        checkpoint_sha256=checkpoint_digest,
    ), request
