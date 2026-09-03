"""Single-node execution definition for the Humatch app."""

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
MAX_RESULT_BYTES = 128 * 1024 * 1024
HUMANIZE_NODE = "humanize"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "Humatch execution request",
)
_VH_FAMILIES = frozenset({"auto", *(f"hv{i}" for i in range(1, 8))})
_VL_FAMILIES = frozenset({
    "auto",
    *(f"lv{i}" for i in range(1, 11)),
    *(f"kv{i}" for i in range(1, 8)),
})


@dataclass(frozen=True)
class HumatchExecutionRequest:
    """Immutable paired batch and scientific humanization parameters."""

    run_name: str
    csv_bytes: bytes
    vh_target_family: str
    vl_target_family: str
    germline_likeness_target: float
    vh_classifier_target: float
    vl_classifier_target: float
    pair_classifier_target: float
    max_edits: int
    mutate_cdrs: bool
    fixed_vh_positions: str
    fixed_vl_positions: str
    app_version: str
    asset_record: str
    runtime_identity: str
    max_active_provider_calls: int = 1
    max_active_gpu_provider_calls: int = 0

    def __post_init__(self) -> None:
        """Reject malformed requests before remote staging."""
        if (
            not isinstance(self.run_name, str)
            or not isinstance(self.csv_bytes, bytes)
            or not self.run_name
            or not self.csv_bytes
        ):
            raise ValueError("Humatch run name and CSV input are required")
        if self.vh_target_family not in _VH_FAMILIES:
            raise ValueError("Unsupported Humatch VH target family")
        if self.vl_target_family not in _VL_FAMILIES:
            raise ValueError("Unsupported Humatch VL target family")
        for name, value in (
            ("germline_likeness_target", self.germline_likeness_target),
            ("vh_classifier_target", self.vh_classifier_target),
            ("vl_classifier_target", self.vl_classifier_target),
            ("pair_classifier_target", self.pair_classifier_target),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"Humatch {name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Humatch {name} must be between 0 and 1")
        if type(self.max_edits) is not int or not 0 <= self.max_edits <= 400:
            raise ValueError("Humatch max_edits must be between 0 and 400")
        if type(self.mutate_cdrs) is not bool:
            raise ValueError("Humatch mutate_cdrs must be a boolean")
        if not all(
            isinstance(value, str)
            for value in (self.fixed_vh_positions, self.fixed_vl_positions)
        ):
            raise ValueError("Humatch fixed positions must be comma-separated text")
        if (
            type(self.max_active_provider_calls) is not int
            or type(self.max_active_gpu_provider_calls) is not int
            or self.max_active_provider_calls != 1
            or self.max_active_gpu_provider_calls != 0
        ):
            raise ValueError("Humatch uses exactly one CPU provider call")
        if not all(
            isinstance(value, str) and value
            for value in (self.app_version, self.asset_record, self.runtime_identity)
        ):
            raise ValueError(
                "Humatch source, asset, and runtime identities are required"
            )

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Describe the single CPU inference node and scientific identity."""
        return ExecutionPlan(
            workload_name="humatch",
            workload_run_key=self.run_name,
            nodes=(NodePlan(HUMANIZE_NODE),),
            scientific_payload={
                "input_csv_sha256": sha256(self.csv_bytes).hexdigest(),
                "vh_target_family": self.vh_target_family,
                "vl_target_family": self.vl_target_family,
                "germline_likeness_target": self.germline_likeness_target,
                "vh_classifier_target": self.vh_classifier_target,
                "vl_classifier_target": self.vl_classifier_target,
                "pair_classifier_target": self.pair_classifier_target,
                "max_edits": self.max_edits,
                "mutate_cdrs": self.mutate_cdrs,
                "fixed_vh_positions": self.fixed_vh_positions,
                "fixed_vl_positions": self.fixed_vl_positions,
            },
            scientific_versions={
                "humatch": self.app_version,
                "humatch.assets": self.asset_record,
                "humatch.runtime": self.runtime_identity,
                "biomodals.humatch.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
        )

    def to_bytes(self) -> bytes:
        """Encode a bounded request without Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "csv_bytes": b64encode(self.csv_bytes).decode("ascii"),
                "vh_target_family": self.vh_target_family,
                "vl_target_family": self.vl_target_family,
                "germline_likeness_target": self.germline_likeness_target,
                "vh_classifier_target": self.vh_classifier_target,
                "vl_classifier_target": self.vl_classifier_target,
                "pair_classifier_target": self.pair_classifier_target,
                "max_edits": self.max_edits,
                "mutate_cdrs": self.mutate_cdrs,
                "fixed_vh_positions": self.fixed_vh_positions,
                "fixed_vl_positions": self.fixed_vl_positions,
                "app_version": self.app_version,
                "asset_record": self.asset_record,
                "runtime_identity": self.runtime_identity,
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": self.max_active_gpu_provider_calls,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("Humatch execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> HumatchExecutionRequest:
        """Decode and revalidate a staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("Humatch execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("Humatch execution request schema is unsupported")
        encoded_csv = value.pop("csv_bytes", None)
        if not isinstance(encoded_csv, str):
            raise TypeError("Humatch CSV content must be base64 text")
        value["csv_bytes"] = b64decode(encoded_csv, validate=True)
        return cls(**value)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: HumatchExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: HumatchExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> HumatchExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return HumatchExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


@dataclass
class _HumatchHumanizeNode(ProviderNode):
    request: HumatchExecutionRequest

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Send the validated batch to one CPU worker."""
        del context
        return ProviderCallSpec(
            function_name="humatch_humanize",
            uses_gpu=False,
            runtime_image_key="humatch-cpu",
            kwargs={
                name: getattr(self.request, name)
                for name in (
                    "run_name",
                    "csv_bytes",
                    "vh_target_family",
                    "vl_target_family",
                    "germline_likeness_target",
                    "vh_classifier_target",
                    "vl_classifier_target",
                    "pair_classifier_target",
                    "max_edits",
                    "mutate_cdrs",
                    "fixed_vh_positions",
                    "fixed_vl_positions",
                )
            },
        )


def humatch_execution_graph(request: HumatchExecutionRequest) -> ExecutionGraph:
    """Build the one-node Humatch execution graph."""
    graph = ExecutionGraph(
        "humatch",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="humatch",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    graph.add_node(_HumatchHumanizeNode(request), id=HUMANIZE_NODE)
    return graph


class HumatchExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Bind a Humatch request to the shared execution runtime."""

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
        app_version: str,
        asset_record: str,
        runtime_identity: str,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only resources used by this adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            artifact_volume_name=output_volume_name,
            output_volume=output_volume,
            provider_driver=provider_driver,
            graph_builder=self._graph,
            target_scientific_versions={
                "humatch": app_version,
                "humatch.assets": asset_record,
                "humatch.runtime": runtime_identity,
                "biomodals.humatch.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
            max_parallel_nodes=1,
            poll_interval_seconds=poll_interval_seconds,
        )

    @staticmethod
    def _graph(
        request: HumatchExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return humatch_execution_graph(request)


def result_from_overview(
    overview: ExecutionOverview,
    output_volume: Any,
) -> AppRunResult:
    """Load the validated result from a successful Humatch call."""
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
    raise LookupError("Humatch humanization result is unavailable")
