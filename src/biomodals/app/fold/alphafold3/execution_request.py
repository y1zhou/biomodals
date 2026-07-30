"""Trusted request state for one remotely coordinated AlphaFold3 App Run."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID

import orjson
from uniaf3.schema.alphafold3 import AF3Config

from biomodals.app.fold.alphafold3.artifacts import (
    read_bounded_file_bytes,
    read_volume_bytes,
    write_bytes_atomic,
)
from biomodals.app.fold.alphafold3.execution_plan import (
    build_alphafold3_execution_plan,
)
from biomodals.app.fold.alphafold3.inference_inputs import (
    serialize_af3_input,
    validate_inference_parameters,
    validate_inference_worker_budget,
    validate_inference_workload,
    validate_submitted_af3_input,
)
from biomodals.app.fold.alphafold3.invocation_cache import (
    PreparedInvocation,
    prepare_invocation,
)
from biomodals.app.fold.alphafold3.search_pipeline import (
    validate_search_worker_budget,
)
from biomodals.execution import ExecutionPlan

EXECUTION_REQUEST_SCHEMA_VERSION = 1
EXECUTION_REQUEST_FILENAME = "alphafold3-request.json"
MAX_EXECUTION_REQUEST_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AlphaFold3ExecutionRequest:
    """Immutable scientific input and operational limits for one App Run."""

    config: AF3Config
    invocation: PreparedInvocation
    search_msa: bool
    search_protein_templates: bool
    max_parallel_search_workers: int
    max_num_gpus: int
    recycle: int
    sample: int

    @classmethod
    def prepare(
        cls,
        config: AF3Config,
        *,
        search_msa: bool,
        search_protein_templates: bool,
        max_parallel_search_workers: int,
        max_num_gpus: int,
        recycle: int,
        sample: int,
    ) -> AlphaFold3ExecutionRequest:
        """Validate one local request and bind its existing invocation identity."""
        validated = validate_submitted_af3_input(config)
        if not isinstance(search_msa, bool):
            raise TypeError("search_msa must be a boolean")
        if not isinstance(search_protein_templates, bool):
            raise TypeError("search_protein_templates must be a boolean")
        validate_search_worker_budget(max_parallel_search_workers)
        validate_inference_worker_budget(max_num_gpus)
        validate_inference_parameters(recycle, sample)
        validate_inference_workload(validated.modelSeeds, sample)
        invocation = prepare_invocation(
            validated,
            search_msa=search_msa,
            search_protein_templates=search_protein_templates,
            recycle=recycle,
            sample=sample,
        )
        return cls(
            config=validated,
            invocation=invocation,
            search_msa=search_msa,
            search_protein_templates=search_protein_templates,
            max_parallel_search_workers=max_parallel_search_workers,
            max_num_gpus=max_num_gpus,
            recycle=recycle,
            sample=sample,
        )

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Return the immutable scientific DAG persisted by the kernel."""
        return build_alphafold3_execution_plan(self.invocation)

    @property
    def max_active_provider_calls(self) -> int:
        """Return the total call ceiling for the sequential CPU/GPU phases."""
        return max(self.max_parallel_search_workers, self.max_num_gpus)

    def to_bytes(self) -> bytes:
        """Serialize trusted state without paths or local Python objects."""
        content = orjson.dumps(
            {
                "schema_version": EXECUTION_REQUEST_SCHEMA_VERSION,
                "config": orjson.loads(serialize_af3_input(self.config)),
                "invocation": {
                    "invocation_id": self.invocation.invocation_id,
                    "identity": self.invocation.identity,
                },
                "search_msa": self.search_msa,
                "search_protein_templates": self.search_protein_templates,
                "max_parallel_search_workers": self.max_parallel_search_workers,
                "max_num_gpus": self.max_num_gpus,
                "recycle": self.recycle,
                "sample": self.sample,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if not 0 < len(content) <= MAX_EXECUTION_REQUEST_BYTES:
            raise ValueError("AlphaFold3 execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> AlphaFold3ExecutionRequest:
        """Decode and revalidate one request staged by the thin local client."""
        if not 0 < len(content) <= MAX_EXECUTION_REQUEST_BYTES:
            raise ValueError("AlphaFold3 execution request exceeds its byte limit")
        try:
            value = orjson.loads(content)
        except orjson.JSONDecodeError as error:
            raise ValueError(
                "AlphaFold3 execution request is not valid JSON"
            ) from error
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != EXECUTION_REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported AlphaFold3 execution request schema")
        config = AF3Config.model_validate(value.get("config"))
        request = cls.prepare(
            config,
            search_msa=_required_bool(value, "search_msa"),
            search_protein_templates=_required_bool(
                value,
                "search_protein_templates",
            ),
            max_parallel_search_workers=_required_int(
                value,
                "max_parallel_search_workers",
            ),
            max_num_gpus=_required_int(value, "max_num_gpus"),
            recycle=_required_int(value, "recycle"),
            sample=_required_int(value, "sample"),
        )
        raw_invocation = value.get("invocation")
        if (
            not isinstance(raw_invocation, dict)
            or raw_invocation.get("invocation_id") != request.invocation.invocation_id
            or raw_invocation.get("identity") != request.invocation.identity
        ):
            raise ValueError(
                "Staged AlphaFold3 invocation does not match its normalized input"
            )
        return request


def execution_request_path(execution_run_id: UUID) -> PurePosixPath:
    """Return the Volume-relative immutable request path for one App Run."""
    return (
        PurePosixPath(".biomodals")
        / "execution"
        / "runs"
        / str(execution_run_id)
        / EXECUTION_REQUEST_FILENAME
    )


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: AlphaFold3ExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage one immutable request from the thin local client."""
    path = execution_request_path(execution_run_id)
    content = request.to_bytes()
    existing = read_volume_bytes(
        output_volume,
        path.as_posix(),
        max_bytes=MAX_EXECUTION_REQUEST_BYTES,
    )
    if existing is not None:
        if existing != content:
            raise RuntimeError(
                "Existing AlphaFold3 execution request conflicts with this run"
            )
        return path
    with output_volume.batch_upload(force=True) as batch:
        batch.put_file(BytesIO(content), f"/{path.as_posix()}")
    return path


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> AlphaFold3ExecutionRequest:
    """Load and revalidate one request from a coordinator's mounted Volume."""
    relative = execution_request_path(execution_run_id)
    content = read_bounded_file_bytes(
        Path(volume_root).joinpath(*relative.parts),
        field_name="AlphaFold3 execution request",
        max_bytes=MAX_EXECUTION_REQUEST_BYTES,
    )
    return AlphaFold3ExecutionRequest.from_bytes(content)


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: AlphaFold3ExecutionRequest,
) -> PurePosixPath:
    """Idempotently persist a request from inside a mounted coordinator."""
    relative = execution_request_path(execution_run_id)
    path = Path(volume_root).joinpath(*relative.parts)
    content = request.to_bytes()
    if path.exists():
        existing = read_bounded_file_bytes(
            path,
            field_name="AlphaFold3 execution request",
            max_bytes=MAX_EXECUTION_REQUEST_BYTES,
        )
        if existing != content:
            raise RuntimeError(
                "Existing AlphaFold3 execution request conflicts with this run"
            )
        return relative
    write_bytes_atomic(path, content)
    return relative


def _required_bool(value: dict[object, object], key: str) -> bool:
    selected = value.get(key)
    if not isinstance(selected, bool):
        raise TypeError(f"{key} must be a boolean")
    return selected


def _required_int(value: dict[object, object], key: str) -> int:
    selected = value.get(key)
    if isinstance(selected, bool) or not isinstance(selected, int):
        raise TypeError(f"{key} must be an integer")
    return cast(int, selected)
