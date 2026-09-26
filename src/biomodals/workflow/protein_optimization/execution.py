"""Immutable requests and thin binding to the existing staged execution host."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import orjson

from biomodals.execution import DeploymentIdentity, ExecutionPlan
from biomodals.execution.definition_plan import execution_plan
from biomodals.execution.modal import (
    ExecutionDefinitionCoordinatorLifecycle,
    ExecutionRequestFile,
    resolve_provider_call_limits,
)
from biomodals.schema import AppRunResult
from biomodals.workflow.protein_optimization.design import (
    SCIENTIFIC_VERSIONS,
    OptimizationDesign,
)
from biomodals.workflow.protein_optimization.nodes import optimization_graph

MAX_REQUEST_BYTES = 32 * 1024 * 1024
REQUEST_FILE = ExecutionRequestFile(
    "protein-optimization-request.json",
    MAX_REQUEST_BYTES,
    "Protein optimization execution request",
)


@dataclass(frozen=True)
class OptimizationExecutionRequest:
    """Frozen reviewed design and the existing per-Run resource snapshot."""

    run_name: str
    design: OptimizationDesign
    max_active_provider_calls: int = 8
    max_active_gpu_provider_calls: int = 1
    scientific_versions: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Reject invalid resource limits and freeze selected scientific dependencies."""
        if not self.run_name:
            raise ValueError("Run name is required")
        resolve_provider_call_limits(
            max_containers=self.max_active_provider_calls,
            max_gpu_containers=self.max_active_gpu_provider_calls,
            default_max_containers=8,
            default_max_gpu_containers=1,
        )
        if (
            self.design.settings.mode == "exploration"
            and self.max_active_gpu_provider_calls < 1
        ):
            raise ValueError("Exploration requires at least one GPU provider-call slot")
        if self.scientific_versions is None:
            object.__setattr__(
                self, "scientific_versions", self.design.scientific_versions()
            )

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Validate deployment compatibility before building the same scientific graph."""
        if self.scientific_versions != self.design.scientific_versions():
            raise ValueError(
                "Target deployment changed protein optimization scientific versions; update API and containing workflow together"
            )
        return execution_plan(
            optimization_graph(self.design).validate(), workload_run_key=self.run_name
        )

    def to_bytes(self) -> bytes:
        """Persist bounded JSON, never user-supplied executable model objects."""
        content = orjson.dumps({
            "schema_version": 1,
            "run_name": self.run_name,
            "design": self.design.model_dump(mode="json"),
            "max_active_provider_calls": self.max_active_provider_calls,
            "max_active_gpu_provider_calls": self.max_active_gpu_provider_calls,
            "scientific_versions": self.scientific_versions,
        })
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("Protein optimization request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> OptimizationExecutionRequest:
        """Restore the exact retained design, not current frontend defaults."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("Invalid protein optimization request size")
        data = orjson.loads(content)
        if not isinstance(data, dict) or data.pop("schema_version", None) != 1:
            raise ValueError("Unsupported protein optimization request schema")
        if (
            not isinstance(data.get("scientific_versions"), dict)
            or not data["scientific_versions"]
        ):
            raise ValueError("Retained scientific versions are required")
        data["design"] = OptimizationDesign.model_validate(data["design"])
        return cls(**data)


def load_execution_request(
    volume_root: str | Path, run_id: UUID
) -> OptimizationExecutionRequest:
    """Read a staged immutable request without any provider work."""
    return OptimizationExecutionRequest.from_bytes(
        REQUEST_FILE.load(volume_root, run_id)
    )


def persist_execution_request(
    volume_root: str | Path, run_id: UUID, request: OptimizationExecutionRequest
):
    """Persist compatible successor request bytes through the shared helper."""
    return REQUEST_FILE.persist(volume_root, run_id, request.to_bytes())


class OptimizationExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Keep ownership, cancellation, recovery and persistence in the shared host."""

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
        external_checker,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Bind workload graph and actual mounted app-volume integrity checks."""
        self.external_checker = external_checker
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            output_volume=output_volume,
            artifact_volume_name=output_volume_name,
            provider_driver=provider_driver,
            graph_builder=lambda request, predecessor: optimization_graph(
                request.design
            ),
            target_scientific_versions=SCIENTIFIC_VERSIONS,
            poll_interval_seconds=poll_interval_seconds,
        )

    def _create_runtime(self, request, *, predecessor_execution_run_id=None):
        runtime = super()._create_runtime(
            request, predecessor_execution_run_id=predecessor_execution_run_id
        )
        runtime.configure_provider_boundary(
            provider_driver=self.provider_driver,
            external_artifact_checker=self.external_checker,
        )
        return runtime

    def result(self) -> AppRunResult:
        """Return terminal references, including validated successor reuse."""
        with self._volume_io_lock, self._writer_lock:
            store = (
                self._runtime.store if self._runtime is not None else self._run_store()
            )
            try:
                overview = store.execution.overview(self.execution_run_id)
                self._verify_overview(overview)
                result = store.artifacts.load_node_result("publish")
                if not overview.run.status.is_terminal or result is None:
                    raise LookupError("Protein optimization result is not available")
                return result
            finally:
                if self._runtime is None:
                    store.close()
