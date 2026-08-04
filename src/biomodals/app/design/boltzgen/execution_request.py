"""Immutable staged request for one remotely coordinated BoltzGen App Run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

import orjson

from biomodals.app.design.boltzgen.execution_contracts import (
    collection_publication_path,
)
from biomodals.execution import ExecutionPlan, NodeDependency, NodePlan
from biomodals.helper.app_execution import ExecutionRequestFile
from biomodals.helper.shell import sanitize_filename

REQUEST_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 8 * 1024 * 1024
DESIGN_RUNS_NODE = "design-runs"
COLLECT_RESULTS_NODE = "collect-results"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "BoltzGen execution request",
)


@dataclass(frozen=True)
class BoltzGenExecutionRequest:
    """Persisted scientific inputs plus operational call ceilings."""

    run_name: str
    run_ids: tuple[str, ...]
    input_sha256: str
    referenced_file_sha256: tuple[tuple[str, str], ...]
    protocol: str
    num_designs: int
    budget: int
    steps: str | None
    extra_args: str | None
    filter_results: bool
    filter_rmsd_threshold: float
    app_version: str
    repo_commit_hash: str
    max_active_provider_calls: int
    max_active_gpu_provider_calls: int
    replace_claim_owners: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Reject incomplete or ambiguous staged requests."""
        if not self.run_name:
            raise ValueError("run_name cannot be empty")
        if not self.run_ids or len(self.run_ids) != len(set(self.run_ids)):
            raise ValueError("run_ids must be a non-empty unique collection")
        _require_safe_filename_component("run_name", self.run_name)
        for run_id in self.run_ids:
            _require_safe_filename_component("run_id", run_id)
        if self.num_designs < 1:
            raise ValueError("num_designs must be positive")
        if self.max_active_provider_calls < 1:
            raise ValueError("max_active_provider_calls must be positive")
        if not (
            0 <= self.max_active_gpu_provider_calls <= self.max_active_provider_calls
        ):
            raise ValueError(
                "max_active_gpu_provider_calls must be between zero and "
                "max_active_provider_calls"
            )
        if len(dict(self.replace_claim_owners)) != len(self.replace_claim_owners):
            raise ValueError("replace_claim_owners contains duplicate run IDs")
        if not set(dict(self.replace_claim_owners)).issubset(self.run_ids):
            raise ValueError("replace_claim_owners names an unknown run ID")

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Build the two-stage direct-fan-out DAG."""
        return ExecutionPlan(
            workload_name="boltzgen",
            workload_run_key=self.run_name,
            nodes=(
                NodePlan(node_key=DESIGN_RUNS_NODE),
                NodePlan(
                    node_key=COLLECT_RESULTS_NODE,
                    dependencies=(NodeDependency(DESIGN_RUNS_NODE),),
                ),
            ),
            scientific_payload={
                "run_name": self.run_name,
                "run_ids": list(self.run_ids),
                "input_sha256": self.input_sha256,
                "referenced_file_sha256": [
                    list(item) for item in self.referenced_file_sha256
                ],
                "protocol": self.protocol,
                "num_designs": self.num_designs,
                "budget": self.budget,
                "steps": self.steps,
                "extra_args": self.extra_args,
                "filter_results": self.filter_results,
                "filter_rmsd_threshold": self.filter_rmsd_threshold,
            },
            scientific_versions={
                "boltzgen": self.app_version,
                "boltzgen_repository": self.repo_commit_hash,
                "biomodals.boltzgen.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
        )

    @property
    def collection_publication_path(self) -> PurePosixPath:
        """Return the terminal publication marker for this exact plan."""
        return collection_publication_path(
            run_name=self.run_name,
            workload_plan_fingerprint=(self.execution_plan.workload_plan_fingerprint),
        )

    @property
    def config_path(self) -> PurePosixPath:
        """Return the staged YAML path inside the app output Volume."""
        return (
            PurePosixPath(self.run_name) / "inputs" / "config" / f"{self.run_name}.yaml"
        )

    def to_bytes(self) -> bytes:
        """Encode the bounded immutable request."""
        content = orjson.dumps(
            {"schema_version": REQUEST_SCHEMA_VERSION, **asdict(self)},
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("BoltzGen execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> BoltzGenExecutionRequest:
        """Decode and validate one staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("BoltzGen execution request has an invalid size")
        value: Any = orjson.loads(content)
        if not isinstance(value, dict) or value.pop("schema_version", None) != (
            REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("BoltzGen execution request schema is unsupported")
        for field_name in (
            "run_ids",
            "referenced_file_sha256",
            "replace_claim_owners",
        ):
            value[field_name] = tuple(
                tuple(item) if isinstance(item, list) else item
                for item in value[field_name]
            )
        request = cls(**value)
        _ = request.execution_plan
        return request


def prepare_execution_request(
    *,
    run_name: str,
    run_ids: tuple[str, ...],
    yaml_content: bytes,
    additional_files: dict[str, bytes],
    protocol: str,
    num_designs: int,
    budget: int,
    steps: str | None,
    extra_args: str | None,
    filter_results: bool,
    filter_rmsd_threshold: float,
    app_version: str,
    repo_commit_hash: str,
    max_active_provider_calls: int,
    max_active_gpu_provider_calls: int | None = None,
) -> BoltzGenExecutionRequest:
    """Build a request whose fingerprint excludes only operational limits."""
    return BoltzGenExecutionRequest(
        run_name=run_name,
        run_ids=run_ids,
        input_sha256=sha256(yaml_content).hexdigest(),
        referenced_file_sha256=tuple(
            (path, sha256(content).hexdigest())
            for path, content in sorted(additional_files.items())
        ),
        protocol=protocol,
        num_designs=num_designs,
        budget=budget,
        steps=steps,
        extra_args=extra_args,
        filter_results=filter_results,
        filter_rmsd_threshold=filter_rmsd_threshold,
        app_version=app_version,
        repo_commit_hash=repo_commit_hash,
        max_active_provider_calls=max_active_provider_calls,
        max_active_gpu_provider_calls=(
            max_active_provider_calls
            if max_active_gpu_provider_calls is None
            else max_active_gpu_provider_calls
        ),
    )


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: BoltzGenExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage one immutable request before coordinator launch."""
    return _REQUEST_FILE.stage(
        output_volume,
        execution_run_id,
        request.to_bytes(),
    )


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: BoltzGenExecutionRequest,
) -> PurePosixPath:
    """Atomically persist a coordinator-generated Successor request."""
    return _REQUEST_FILE.persist(
        volume_root,
        execution_run_id,
        request.to_bytes(),
    )


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> BoltzGenExecutionRequest:
    """Load a request from a coordinator-mounted Volume."""
    content = _REQUEST_FILE.load(volume_root, execution_run_id)
    return BoltzGenExecutionRequest.from_bytes(content)


def load_execution_request_from_volume(
    output_volume: Any,
    execution_run_id: UUID,
) -> BoltzGenExecutionRequest:
    """Load a completed run's request through the client-side Volume API."""
    content = _REQUEST_FILE.load_from_volume(output_volume, execution_run_id)
    return BoltzGenExecutionRequest.from_bytes(content)


def _require_safe_filename_component(field_name: str, value: str) -> None:
    """Reject workload identities that can select another path."""
    try:
        safe_value = sanitize_filename(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be a safe filename component") from error
    if safe_value != value:
        raise ValueError(f"{field_name} must be a safe filename component")
