"""Pure request and naming contracts shared across the GROMACS service."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

MAX_SIMULATION_TIME_NS = 200
_RUN_NAME_SEPARATOR = re.compile(r"[^a-z0-9]+")
_RUN_NAME_SUFFIX = re.compile(r"[0-9a-f]{32}")
_MAX_RUN_NAME_SLUG_LENGTH = 64


class GromacsJobOptions(BaseModel):
    """Bounded GROMACS settings accepted from the browser."""

    model_config = ConfigDict(frozen=True)

    simulation_time_ns: int = Field(default=5, ge=1, le=MAX_SIMULATION_TIME_NS)
    run_pdbfixer: bool = False
    cpu_only: bool = False


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


def artifact_request_sha256(pdb_content: bytes, parameters_json: str) -> str:
    """Identify the exact input and normalized parameters placed in a Result."""
    digest = hashlib.sha256()
    digest.update(len(pdb_content).to_bytes(8, byteorder="big"))
    digest.update(pdb_content)
    digest.update(parameters_json.encode())
    return digest.hexdigest()
