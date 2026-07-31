"""Immutable request for one remotely coordinated direct Rosetta App Run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

import orjson

from biomodals.app.bioinfo.rosetta.execution_contracts import RosettaTaskSpec
from biomodals.execution import ExecutionPlan, NodeAggregationPolicy, NodePlan
from biomodals.helper.app_execution import ExecutionRequestFile
from biomodals.helper.shell import sanitize_filename

REQUEST_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 16 * 1024 * 1024
ROSETTA_TASKS_NODE = "rosetta-tasks"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "Rosetta execution request",
)


@dataclass(frozen=True)
class RosettaExecutionRequest:
    """Persisted Task collection plus operational worker limits."""

    run_name: str
    run_id: str
    tasks: tuple[RosettaTaskSpec, ...]
    app_version: str
    max_active_provider_calls: int
    claim_capacity: int
    max_parallel_per_worker: int

    def __post_init__(self) -> None:
        """Reject an unusable worker policy or duplicate Task identity."""
        if not self.run_name or not self.run_id or not self.tasks:
            raise ValueError("Rosetta run identity and Tasks cannot be empty")
        _require_safe_filename_component("run_name", self.run_name)
        _require_safe_filename_component("run_id", self.run_id)
        keys = tuple(task.task_key for task in self.tasks)
        if len(keys) != len(set(keys)):
            raise ValueError("Rosetta Task keys must be unique")
        if self.max_active_provider_calls < 1:
            raise ValueError("max_active_provider_calls must be positive")
        if self.claim_capacity < 1:
            raise ValueError("claim_capacity must be positive")
        if self.max_parallel_per_worker < 1:
            raise ValueError("max_parallel_per_worker must be positive")

    @property
    def workload_run_key(self) -> str:
        """Return the existing user-visible Rosetta run directory name."""
        return f"{self.run_name}-{self.run_id}"

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Build the single semantic pull-worker Node."""
        return ExecutionPlan(
            workload_name="rosetta",
            workload_run_key=self.workload_run_key,
            nodes=(
                NodePlan(
                    node_key=ROSETTA_TASKS_NODE,
                    aggregation_policy=NodeAggregationPolicy.COLLECT_ALL,
                ),
            ),
            scientific_payload={
                "run_name": self.run_name,
                "run_id": self.run_id,
                "tasks": [task.scientific_payload for task in self.tasks],
            },
            scientific_versions={
                "rosetta": self.app_version,
                "biomodals.rosetta.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
        )

    def to_bytes(self) -> bytes:
        """Encode the bounded request without trusting Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "run_id": self.run_id,
                "tasks": [task.to_dict() for task in self.tasks],
                "app_version": self.app_version,
                "max_active_provider_calls": self.max_active_provider_calls,
                "claim_capacity": self.claim_capacity,
                "max_parallel_per_worker": self.max_parallel_per_worker,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("Rosetta execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> RosettaExecutionRequest:
        """Decode and revalidate a staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("Rosetta execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("Rosetta execution request schema is unsupported")
        tasks = value.pop("tasks", None)
        if not isinstance(tasks, list):
            raise TypeError("Rosetta execution request Tasks must be a list")
        return cls(
            tasks=tuple(RosettaTaskSpec.from_dict(task) for task in tasks),
            **value,
        )


def request_relative_path(execution_run_id: UUID) -> PurePosixPath:
    """Return this App Run's reserved immutable request path."""
    return _REQUEST_FILE.path(execution_run_id)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: RosettaExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before detached coordinator launch."""
    return _REQUEST_FILE.stage(
        output_volume,
        execution_run_id,
        request.to_bytes(),
    )


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: RosettaExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request atomically."""
    return _REQUEST_FILE.persist(
        volume_root,
        execution_run_id,
        request.to_bytes(),
    )


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> RosettaExecutionRequest:
    """Load one request from the coordinator's mounted output Volume."""
    content = _REQUEST_FILE.load(volume_root, execution_run_id)
    return RosettaExecutionRequest.from_bytes(content)


def load_execution_request_from_volume(
    output_volume: Any,
    execution_run_id: UUID,
) -> RosettaExecutionRequest:
    """Load a request through Modal's local chunked Volume API."""
    content = _REQUEST_FILE.load_from_volume(output_volume, execution_run_id)
    return RosettaExecutionRequest.from_bytes(content)


def _require_safe_filename_component(field_name: str, value: str) -> None:
    """Reject workload identities that can select another path."""
    try:
        safe_value = sanitize_filename(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a safe filename component") from error
    if safe_value != value:
        raise ValueError(f"{field_name} must be a safe filename component")
