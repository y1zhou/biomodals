"""Single-node execution definition for Sapiens humanization."""

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
    "Sapiens execution request",
)


@dataclass(frozen=True)
class SapiensExecutionRequest:
    """Immutable input batch and scientific humanization parameters."""

    run_name: str
    csv_bytes: bytes
    iterations: int
    numbering_scheme: str
    cdr_definition: str
    mutate_cdrs: bool
    app_version: str
    model_vh_revision: str
    model_vl_revision: str
    tokenizer_revision: str
    runtime_identity: str
    max_active_provider_calls: int = 1
    max_active_gpu_provider_calls: int = 0

    def __post_init__(self) -> None:
        """Reject malformed requests before remote staging."""
        if (
            not isinstance(self.run_name, str)
            or not isinstance(self.csv_bytes, bytes)
            or not isinstance(self.app_version, str)
            or not self.run_name
            or not self.csv_bytes
            or not self.app_version
        ):
            raise ValueError("Sapiens run name, CSV input, and version are required")
        if type(self.iterations) is not int or not 1 <= self.iterations <= 5:
            raise ValueError("Sapiens iterations must be between 1 and 5")
        if not isinstance(self.numbering_scheme, str) or (
            self.numbering_scheme not in {"kabat", "chothia", "imgt"}
        ):
            raise ValueError("Unsupported antibody numbering scheme")
        if not isinstance(self.cdr_definition, str) or (
            self.cdr_definition not in {"kabat", "chothia", "imgt", "north"}
        ):
            raise ValueError("Unsupported CDR definition")
        if type(self.mutate_cdrs) is not bool:
            raise ValueError("Sapiens mutate_cdrs must be a boolean")
        if (
            type(self.max_active_provider_calls) is not int
            or type(self.max_active_gpu_provider_calls) is not int
            or self.max_active_provider_calls != 1
            or self.max_active_gpu_provider_calls != 0
        ):
            raise ValueError("Sapiens uses exactly one CPU provider call")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.model_vh_revision,
                self.model_vl_revision,
                self.tokenizer_revision,
                self.runtime_identity,
            )
        ):
            raise ValueError(
                "Sapiens model, tokenizer, and runtime identities are required"
            )

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Describe the single CPU inference node and scientific identity."""
        return ExecutionPlan(
            workload_name="sapiens",
            workload_run_key=self.run_name,
            nodes=(NodePlan(HUMANIZE_NODE),),
            scientific_payload={
                "input_csv_sha256": sha256(self.csv_bytes).hexdigest(),
                "iterations": self.iterations,
                "numbering_scheme": self.numbering_scheme,
                "cdr_definition": self.cdr_definition,
                "mutate_cdrs": self.mutate_cdrs,
            },
            scientific_versions={
                "sapiens": self.app_version,
                "sapiens.model.vh": self.model_vh_revision,
                "sapiens.model.vl": self.model_vl_revision,
                "sapiens.tokenizer": self.tokenizer_revision,
                "sapiens.runtime": self.runtime_identity,
                "biomodals.sapiens.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
        )

    def to_bytes(self) -> bytes:
        """Encode a bounded request without Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "csv_bytes": b64encode(self.csv_bytes).decode("ascii"),
                "iterations": self.iterations,
                "numbering_scheme": self.numbering_scheme,
                "cdr_definition": self.cdr_definition,
                "mutate_cdrs": self.mutate_cdrs,
                "app_version": self.app_version,
                "model_vh_revision": self.model_vh_revision,
                "model_vl_revision": self.model_vl_revision,
                "tokenizer_revision": self.tokenizer_revision,
                "runtime_identity": self.runtime_identity,
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": self.max_active_gpu_provider_calls,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("Sapiens execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> SapiensExecutionRequest:
        """Decode and revalidate a staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("Sapiens execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("Sapiens execution request schema is unsupported")
        encoded_csv = value.pop("csv_bytes", None)
        if not isinstance(encoded_csv, str):
            raise TypeError("Sapiens CSV content must be base64 text")
        value["csv_bytes"] = b64decode(encoded_csv, validate=True)
        return cls(**value)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: SapiensExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: SapiensExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> SapiensExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return SapiensExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


@dataclass
class _SapiensHumanizeNode(ProviderNode):
    request: SapiensExecutionRequest

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Send the validated scientific payload to one CPU worker."""
        del context
        return ProviderCallSpec(
            function_name="sapiens_humanize",
            uses_gpu=False,
            runtime_image_key="sapiens-cpu",
            kwargs={
                "run_name": self.request.run_name,
                "csv_bytes": self.request.csv_bytes,
                "iterations": self.request.iterations,
                "numbering_scheme": self.request.numbering_scheme,
                "cdr_definition": self.request.cdr_definition,
                "mutate_cdrs": self.request.mutate_cdrs,
            },
        )


def sapiens_execution_graph(request: SapiensExecutionRequest) -> ExecutionGraph:
    """Build the one-node Sapiens execution graph."""
    graph = ExecutionGraph(
        "sapiens",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="sapiens",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    graph.add_node(_SapiensHumanizeNode(request), id=HUMANIZE_NODE)
    return graph


class SapiensExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Bind a Sapiens request to the shared execution runtime."""

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
        model_vh_revision: str,
        model_vl_revision: str,
        tokenizer_revision: str,
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
                "sapiens": app_version,
                "sapiens.model.vh": model_vh_revision,
                "sapiens.model.vl": model_vl_revision,
                "sapiens.tokenizer": tokenizer_revision,
                "sapiens.runtime": runtime_identity,
                "biomodals.sapiens.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
            max_parallel_nodes=1,
            poll_interval_seconds=poll_interval_seconds,
        )

    @staticmethod
    def _graph(
        request: SapiensExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return sapiens_execution_graph(request)


def result_from_overview(
    overview: ExecutionOverview,
    output_volume: Any,
) -> AppRunResult:
    """Load the validated workflow result from a successful Sapiens call."""
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
    raise LookupError("Sapiens humanization result is unavailable")
