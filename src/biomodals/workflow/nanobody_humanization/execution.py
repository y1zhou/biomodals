"""Immutable prepared-parent requests hosted by the shared execution lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

import orjson

from biomodals.execution import DeploymentIdentity, ExecutionGraph, ExecutionPlan
from biomodals.execution.definition_plan import execution_plan
from biomodals.execution.modal import (
    ExecutionDefinitionCoordinatorLifecycle,
    ExecutionRequestFile,
    resolve_provider_call_limits,
)
from biomodals.schema import AppRunResult
from biomodals.workflow.nanobody_humanization.preparation import PreparedVH
from biomodals.workflow.nanobody_humanization.settings import NanobodySettings

MAX_REQUEST_BYTES = 4 * 1024 * 1024
_REQUEST_FILE = ExecutionRequestFile(
    "nanobody-request.json",
    MAX_REQUEST_BYTES,
    "Nanobody humanization execution request",
)


def _scientific_versions() -> dict[str, str]:
    from biomodals.workflow.nanobody_humanization.workflow import SCIENTIFIC_VERSIONS

    return dict(SCIENTIFIC_VERSIONS)


@dataclass(frozen=True)
class NanobodyExecutionRequest:
    """Frozen prepared parents, masks and native controls; no re-imputation on read."""

    run_name: str
    parents: tuple[PreparedVH, ...]
    settings: NanobodySettings = field(default_factory=NanobodySettings)
    max_active_provider_calls: int = 8
    max_active_gpu_provider_calls: int = 2
    scientific_versions: dict[str, str] = field(default_factory=_scientific_versions)

    def __post_init__(self) -> None:
        """Validate bounded admission and parent identities before staging bytes."""
        if not self.run_name or not 1 <= len(self.parents) <= 200:
            raise ValueError(
                "Nanobody run name and 1–200 prepared parents are required"
            )
        if len({parent.id for parent in self.parents}) != len(self.parents):
            raise ValueError("Duplicate prepared parent IDs")
        self.settings.validate_budget(len(self.parents))
        resolve_provider_call_limits(
            max_containers=self.max_active_provider_calls,
            max_gpu_containers=self.max_active_gpu_provider_calls,
            default_max_containers=8,
            default_max_gpu_containers=2,
        )

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Use the same immutable scientific definition for service and CLI."""
        return execution_plan(
            nanobody_execution_graph(self).validate(), workload_run_key=self.run_name
        )

    def to_bytes(self) -> bytes:
        """Serialize primitive scientific evidence, never executable pickle."""
        content = orjson.dumps(
            {
                "schema_version": 1,
                "run_name": self.run_name,
                "parents": [parent.model_dump() for parent in self.parents],
                "settings": self.settings.model_dump(),
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": self.max_active_gpu_provider_calls,
                "scientific_versions": self.scientific_versions,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("Nanobody execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> NanobodyExecutionRequest:
        """Restore exactly the saved domain and policy, not today's preparation."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("Nanobody execution request has an invalid size")
        value = orjson.loads(content)
        if not isinstance(value, dict) or value.pop("schema_version", None) != 1:
            raise ValueError("Unsupported nanobody execution request schema")
        value["parents"] = tuple(
            PreparedVH.model_validate(parent) for parent in value["parents"]
        )
        value["settings"] = NanobodySettings.model_validate(value["settings"])
        return cls(**value)


def nanobody_execution_graph(request: NanobodyExecutionRequest) -> ExecutionGraph:
    """Reject API/deployment scientific drift before any model operation."""
    from biomodals.workflow.nanobody_humanization.workflow import build_nanobody_graph

    if request.scientific_versions != _scientific_versions():
        raise ValueError(
            "Target deployment changed nanobody scientific versions. Update the API and "
            "containing workflow together, pin the matching deployment and submit a new Job."
        )
    return build_nanobody_graph(request.parents, request.settings)


def stage_execution_request(
    output_volume: Any, execution_run_id: UUID, request: NanobodyExecutionRequest
) -> PurePosixPath:
    """Stage immutable request bytes using the existing idempotent Volume helper."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path, execution_run_id: UUID, request: NanobodyExecutionRequest
) -> PurePosixPath:
    """Persist a compatible successor request inside the coordinator mount."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path, execution_run_id: UUID
) -> NanobodyExecutionRequest:
    """Read the frozen input without invoking numbering or scientific providers."""
    return NanobodyExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


def load_execution_request_from_volume(
    output_volume: Any, execution_run_id: UUID
) -> NanobodyExecutionRequest:
    """Read original/prepared inputs through the client Volume API."""
    return NanobodyExecutionRequest.from_bytes(
        _REQUEST_FILE.load_from_volume(output_volume, execution_run_id)
    )


def result_directory(execution_run_id: UUID) -> PurePosixPath:
    """Return the root Run's consolidated scientific publication directory."""
    return (
        PurePosixPath("workflow-runs")
        / str(execution_run_id)
        / "nodes/evaluate/result/nanobody_humanization"
    )


class NanobodyExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Bind a single-domain graph to the established coordinator and recovery rules."""

    _request_loader = staticmethod(load_execution_request)
    _request_persister = staticmethod(persist_execution_request)

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        output_volume_name: str,
        provider_driver: Any,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Keep scheduling, cancellation, ownership and recovery in the shared host."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            output_volume=output_volume,
            artifact_volume_name=output_volume_name,
            provider_driver=provider_driver,
            graph_builder=lambda request, predecessor: nanobody_execution_graph(
                request
            ),
            target_scientific_versions=_scientific_versions(),
            poll_interval_seconds=poll_interval_seconds,
        )

    def result(self) -> AppRunResult:
        """Return terminal references including validated predecessor publications."""
        with self._volume_io_lock, self._writer_lock:
            store = (
                self._runtime.store if self._runtime is not None else self._run_store()
            )
            try:
                overview = store.execution.overview(self.execution_run_id)
                self._verify_overview(overview)
                result = store.artifacts.load_node_result("publish")
                if not overview.run.status.is_terminal or result is None:
                    raise LookupError("Nanobody results are not available")
                return result
            finally:
                if self._runtime is None:
                    store.close()
