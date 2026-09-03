"""Single-node execution definition for the p-AbNatiV2 app."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

import orjson

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionGraph,
    ExecutionOverview,
    ExecutionPlan,
    ExecutionPlanMetadata,
    NodePlan,
    ProviderCallStatus,
)
from biomodals.execution.modal import (
    ExecutionDefinitionCoordinatorLifecycle,
    ExecutionRequestFile,
    load_execution_provider_result,
)
from biomodals.execution.nodes import NodeRunContext, ProviderCallSpec, ProviderNode
from biomodals.schema import AppRunResult

REQUEST_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = 512 * 1024 * 1024
HUMANIZE_NODE = "humanize"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "p-AbNatiV2 execution request",
)


@dataclass(frozen=True)
class PAbNatiV2ExecutionRequest:
    """Immutable paired input batch and scientific controls."""

    run_name: str
    csv_bytes: bytes
    mutate_cdrs: bool
    fixed_vh_positions: str
    fixed_vl_positions: str
    residue_score_threshold: float
    rasa_threshold: float
    max_relative_pairing_score_decrease: float
    forbidden_residues: str
    seed: int
    source_commit: str
    paired_model_md5: str
    structure_archive_md5: str
    runtime_identity: str
    max_active_provider_calls: int = 1
    max_active_gpu_provider_calls: int = 0

    def __post_init__(self) -> None:
        """Reject malformed staged requests before provider dispatch."""
        if not isinstance(self.run_name, str) or not self.run_name:
            raise ValueError("p-AbNatiV2 run name is required")
        if not isinstance(self.csv_bytes, bytes) or not self.csv_bytes:
            raise ValueError("p-AbNatiV2 CSV input is required")
        if type(self.mutate_cdrs) is not bool:
            raise ValueError("p-AbNatiV2 mutate_cdrs must be a boolean")
        if not all(
            isinstance(value, str)
            for value in (
                self.fixed_vh_positions,
                self.fixed_vl_positions,
                self.forbidden_residues,
            )
        ):
            raise ValueError("p-AbNatiV2 position and residue controls must be text")
        for name, value in (
            ("residue_score_threshold", self.residue_score_threshold),
            ("rasa_threshold", self.rasa_threshold),
            (
                "max_relative_pairing_score_decrease",
                self.max_relative_pairing_score_decrease,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"p-AbNatiV2 {name} must be numeric")
            if not 0 <= float(value) <= 1:
                raise ValueError(f"p-AbNatiV2 {name} must be between 0 and 1")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("p-AbNatiV2 seed must be an unsigned 32-bit integer")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.source_commit,
                self.paired_model_md5,
                self.structure_archive_md5,
                self.runtime_identity,
            )
        ):
            raise ValueError("p-AbNatiV2 scientific identities are required")
        if (
            type(self.max_active_provider_calls) is not int
            or type(self.max_active_gpu_provider_calls) is not int
            or self.max_active_provider_calls != 1
            or self.max_active_gpu_provider_calls != 0
        ):
            raise ValueError("p-AbNatiV2 currently uses one CPU provider call")

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Describe the initial single-container scientific run."""
        return ExecutionPlan(
            workload_name="pabnativ2",
            workload_run_key=self.run_name,
            nodes=(NodePlan(HUMANIZE_NODE),),
            scientific_payload={
                "input_csv_sha256": sha256(self.csv_bytes).hexdigest(),
                "mutate_cdrs": self.mutate_cdrs,
                "fixed_vh_positions": self.fixed_vh_positions,
                "fixed_vl_positions": self.fixed_vl_positions,
                "residue_score_threshold": self.residue_score_threshold,
                "rasa_threshold": self.rasa_threshold,
                "max_relative_pairing_score_decrease": (
                    self.max_relative_pairing_score_decrease
                ),
                "forbidden_residues": self.forbidden_residues,
                "seed": self.seed,
            },
            scientific_versions={
                "pabnativ2.source": self.source_commit,
                "pabnativ2.model.paired": self.paired_model_md5,
                "pabnativ2.model.structure": self.structure_archive_md5,
                "pabnativ2.runtime": self.runtime_identity,
                "biomodals.pabnativ2.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
        )

    def to_bytes(self) -> bytes:
        """Encode a bounded request without Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "csv_bytes": b64encode(self.csv_bytes).decode("ascii"),
                "mutate_cdrs": self.mutate_cdrs,
                "fixed_vh_positions": self.fixed_vh_positions,
                "fixed_vl_positions": self.fixed_vl_positions,
                "residue_score_threshold": self.residue_score_threshold,
                "rasa_threshold": self.rasa_threshold,
                "max_relative_pairing_score_decrease": (
                    self.max_relative_pairing_score_decrease
                ),
                "forbidden_residues": self.forbidden_residues,
                "seed": self.seed,
                "source_commit": self.source_commit,
                "paired_model_md5": self.paired_model_md5,
                "structure_archive_md5": self.structure_archive_md5,
                "runtime_identity": self.runtime_identity,
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": (self.max_active_gpu_provider_calls),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("p-AbNatiV2 execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> PAbNatiV2ExecutionRequest:
        """Decode and revalidate one staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("p-AbNatiV2 execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("p-AbNatiV2 execution request schema is unsupported")
        encoded_csv = value.pop("csv_bytes", None)
        if not isinstance(encoded_csv, str):
            raise TypeError("p-AbNatiV2 CSV content must be base64 text")
        value["csv_bytes"] = b64decode(encoded_csv, validate=True)
        return cls(**value)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: PAbNatiV2ExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: PAbNatiV2ExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> PAbNatiV2ExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return PAbNatiV2ExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


@dataclass
class _PAbNatiV2HumanizeNode(ProviderNode):
    request: PAbNatiV2ExecutionRequest

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Send the validated payload to the initial CPU worker."""
        del context
        return ProviderCallSpec(
            function_name="pabnativ2_humanize",
            uses_gpu=True,
            runtime_image_key="pabnativ2-cpu",
            kwargs={
                "run_name": self.request.run_name,
                "csv_bytes": self.request.csv_bytes,
                "mutate_cdrs": self.request.mutate_cdrs,
                "fixed_vh_positions": self.request.fixed_vh_positions,
                "fixed_vl_positions": self.request.fixed_vl_positions,
                "residue_score_threshold": self.request.residue_score_threshold,
                "rasa_threshold": self.request.rasa_threshold,
                "max_relative_pairing_score_decrease": (
                    self.request.max_relative_pairing_score_decrease
                ),
                "forbidden_residues": self.request.forbidden_residues,
                "seed": self.request.seed,
            },
        )


def pabnativ2_execution_graph(
    request: PAbNatiV2ExecutionRequest,
) -> ExecutionGraph:
    """Build the one-node p-AbNatiV2 execution graph."""
    graph = ExecutionGraph(
        "pabnativ2",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="pabnativ2",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    graph.add_node(_PAbNatiV2HumanizeNode(request), id=HUMANIZE_NODE)
    return graph


class PAbNatiV2ExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Bind one p-AbNatiV2 request to the shared execution runtime."""

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
        source_commit: str,
        paired_model_md5: str,
        structure_archive_md5: str,
        runtime_identity: str,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only the resources used by this adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            artifact_volume_name=output_volume_name,
            output_volume=output_volume,
            provider_driver=provider_driver,
            graph_builder=self._graph,
            target_scientific_versions={
                "pabnativ2.source": source_commit,
                "pabnativ2.model.paired": paired_model_md5,
                "pabnativ2.model.structure": structure_archive_md5,
                "pabnativ2.runtime": runtime_identity,
                "biomodals.pabnativ2.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
            max_parallel_nodes=1,
            poll_interval_seconds=poll_interval_seconds,
        )

    @staticmethod
    def _graph(
        request: PAbNatiV2ExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return pabnativ2_execution_graph(request)


def result_from_overview(
    overview: ExecutionOverview,
    output_volume: Any,
) -> AppRunResult:
    """Load the validated result from a successful provider call."""
    for call in overview.representative_provider_calls:
        if (
            call.node_key == HUMANIZE_NODE
            and call.status == ProviderCallStatus.SUCCEEDED
        ):
            value = load_execution_provider_result(
                output_volume,
                execution_run_id=overview.run.execution_run_id,
                envelope=call.result_envelope,
                max_bytes=MAX_RESULT_BYTES,
            )
            return AppRunResult.model_validate(value)
    raise LookupError("p-AbNatiV2 humanization result is unavailable")
