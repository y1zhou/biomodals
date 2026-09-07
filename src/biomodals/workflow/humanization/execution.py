"""Staged humanization requests hosted by the shared execution lifecycle."""

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
from biomodals.workflow.humanization.contracts import AntibodyPair
from biomodals.workflow.humanization.settings import HumanizationSettings

MAX_REQUEST_BYTES = 4 * 1024 * 1024
_REQUEST_FILE = ExecutionRequestFile(
    "humanization-request.json", MAX_REQUEST_BYTES, "Humanization execution request"
)


def _scientific_versions() -> dict[str, str]:
    from biomodals.workflow.humanization.workflow import SCIENTIFIC_VERSIONS

    return dict(SCIENTIFIC_VERSIONS)


@dataclass(frozen=True)
class HumanizationExecutionRequest:
    """Immutable paired input and native controls, separate from admission limits."""

    run_name: str
    pairs: tuple[AntibodyPair, ...]
    settings: HumanizationSettings = field(default_factory=HumanizationSettings)
    max_active_provider_calls: int = 4
    max_active_gpu_provider_calls: int = 2
    scientific_versions: dict[str, str] = field(default_factory=_scientific_versions)

    def __post_init__(self) -> None:
        """Reject invalid scientific controls and provider ceilings before staging."""
        from biomodals.workflow.humanization.workflow import (
            validate_humanization_settings,
        )

        if not self.run_name or not self.pairs:
            raise ValueError("Humanization run name and pairs are required")
        if len({pair.id for pair in self.pairs}) != len(self.pairs):
            raise ValueError("Duplicate parent IDs")
        resolve_provider_call_limits(
            max_containers=self.max_active_provider_calls,
            max_gpu_containers=self.max_active_gpu_provider_calls,
            default_max_containers=4,
            default_max_gpu_containers=2,
        )
        validate_humanization_settings(self.settings, pair_count=len(self.pairs))

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Use the same scientific graph identity as direct workflow submission."""
        return execution_plan(
            humanization_execution_graph(self).validate(),
            workload_run_key=self.run_name,
        )

    def to_bytes(self) -> bytes:
        """Encode primitive staged data, never an executable pickle."""
        content = orjson.dumps(
            {
                "schema_version": 1,
                "run_name": self.run_name,
                "pairs": [pair.model_dump() for pair in self.pairs],
                "settings": self.settings.model_dump(),
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": self.max_active_gpu_provider_calls,
                "scientific_versions": self.scientific_versions,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("Humanization execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> HumanizationExecutionRequest:
        """Validate the durable request before reconstructing its scientific graph."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("Humanization execution request has an invalid size")
        value = orjson.loads(content)
        if not isinstance(value, dict) or value.pop("schema_version", None) != 1:
            raise ValueError("Unsupported humanization execution request schema")
        value["pairs"] = tuple(
            AntibodyPair.model_validate(pair) for pair in value["pairs"]
        )
        value["settings"] = HumanizationSettings.model_validate(value["settings"])
        return cls(**value)


def humanization_execution_graph(
    request: HumanizationExecutionRequest,
) -> ExecutionGraph:
    """Reconstruct the existing generation/union/evaluation graph."""
    from biomodals.workflow.humanization.workflow import build_humanization_graph

    if request.scientific_versions != _scientific_versions():
        raise ValueError("Target deployment changed humanization scientific versions")
    return build_humanization_graph(request.pairs, request.settings)


def stage_execution_request(
    output_volume: Any, execution_run_id: UUID, request: HumanizationExecutionRequest
) -> PurePosixPath:
    """Idempotently stage immutable bytes through the client Volume API."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: HumanizationExecutionRequest,
) -> PurePosixPath:
    """Persist a shared-host successor request inside its mounted Volume."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path, execution_run_id: UUID
) -> HumanizationExecutionRequest:
    """Load one coordinator-mounted request."""
    return HumanizationExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


def load_execution_request_from_volume(
    output_volume: Any, execution_run_id: UUID
) -> HumanizationExecutionRequest:
    """Load one request through the client Volume API for service recovery."""
    return HumanizationExecutionRequest.from_bytes(
        _REQUEST_FILE.load_from_volume(output_volume, execution_run_id)
    )


def result_directory(execution_run_id: UUID) -> PurePosixPath:
    """Return the workflow-owned terminal scientific publication path."""
    return (
        PurePosixPath("workflow-runs")
        / str(execution_run_id)
        / "nodes/evaluate/result/humanization"
    )


class HumanizationExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Bind the workflow to the existing run/resume/status/cancel implementation."""

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
        """Bind workflow resources without adding a second scheduling lifecycle."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            output_volume=output_volume,
            artifact_volume_name=output_volume_name,
            provider_driver=provider_driver,
            graph_builder=lambda request, predecessor: humanization_execution_graph(
                request
            ),
            target_scientific_versions=_scientific_versions(),
            poll_interval_seconds=poll_interval_seconds,
        )

    def result(self) -> AppRunResult:
        """Read the final publication, including reused predecessor locations."""
        with self._volume_io_lock, self._writer_lock:
            store = (
                self._runtime.store if self._runtime is not None else self._run_store()
            )
            try:
                overview = store.execution.overview(self.execution_run_id)
                self._verify_overview(overview)
                result = store.artifacts.load_node_result("evaluate")
                if not overview.run.status.is_terminal or result is None:
                    raise LookupError("Humanization results are not available")
                return result
            finally:
                if self._runtime is None:
                    store.close()
