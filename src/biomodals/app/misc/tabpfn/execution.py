"""Content-bound tabular inputs and one reusable fit/predict execution Node."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from biomodals.app.misc.tabpfn.models import RUNTIME_IDENTITY
from biomodals.app.misc.tabpfn.tables import MAX_PARQUET_BYTES, TableSchema
from biomodals.execution import ExecutionGraph, ExecutionPlanMetadata
from biomodals.execution.nodes import NodeRunContext, ProviderCallSpec, ProviderNode
from biomodals.helper.artifacts import file_matches_sha256

OPERATION = "tabpfn_fit_predict"
PREPARE_OPERATION = "prepare_tabpfn_models"


class TableFile(BaseModel):
    """An immutable input in this app's Volume, not a client-selected host path."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    path: str = Field(
        pattern=r"^inputs/[0-9a-f]{32}/(training|inference)\.(csv|parquet)$"
    )
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0, le=MAX_PARQUET_BYTES)

    def resolve(self, root: Path) -> Path:
        """Reject drift and symlinks escaping the mounted input namespace."""
        path = (root / self.path).resolve()
        path.relative_to(root.resolve())
        if not file_matches_sha256(path, self.size_bytes, self.sha256):
            raise ValueError("Staged table failed content verification")
        return path


class TabPFNRequest(BaseModel):
    """Same-invocation fit and inference; foundation weights are never user input."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    training: TableFile
    inference: TableFile
    table_schema: TableSchema
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    n_estimators: int = Field(default=8, ge=1, le=32)
    batch_size: int = Field(default=256, ge=1, le=4096)

    def scientific_identity(self) -> dict:
        """Staging location is transport, not model or data identity."""
        value = self.model_dump(mode="json")
        for key in ("training", "inference"):
            value[key].pop("path")
        return value


@dataclass
class TabPFNNode(ProviderNode):
    """One app operation in its caller's Run, with no nested coordinator."""

    request: TabPFNRequest
    runtime_identity: str = RUNTIME_IDENTITY

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Pass descriptors, never large tables or fitted Python objects."""
        return ProviderCallSpec(
            function_name=OPERATION,
            uses_gpu=True,
            runtime_image_key="tabpfn-gpu",
            kwargs={
                "request_json": self.request.model_dump_json(),
                "runtime_identity": self.runtime_identity,
                "output_key": sha256(
                    f"{context.execution_run_id}/{context.node_id}/{context.task_key}".encode()
                ).hexdigest(),
            },
        )


@dataclass
class PrepareTabPFNNode(ProviderNode):
    """Explicit tracked CPU provisioning before read-only GPU inference."""

    runtime_identity: str = RUNTIME_IDENTITY

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        """Keep model downloads separate from native fit/predict."""
        return ProviderCallSpec(
            function_name=PREPARE_OPERATION,
            uses_gpu=False,
            kwargs={"runtime_identity": self.runtime_identity},
        )


def tabpfn_graph(request: TabPFNRequest) -> ExecutionGraph:
    """Keep standalone and composed scientific fitting contracts identical."""
    graph = ExecutionGraph(
        "tabpfn",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="tabpfn",
            scientific_payload=request.scientific_identity(),
            scientific_versions={"tabpfn": RUNTIME_IDENTITY},
        ),
    )
    prepare = graph.add_node(
        PrepareTabPFNNode(), id="prepare_models", reuse_predecessor_publication=False
    )
    graph.add_node(TabPFNNode(request), id="fit_predict", depends_on=[prepare])
    return graph
