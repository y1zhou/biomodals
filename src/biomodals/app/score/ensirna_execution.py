"""Direct ENsiRNA adaptation of the shared execution kernel."""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID

import orjson

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionArtifact,
    ExecutionGraph,
    ExecutionPlan,
    ExecutionPlanMetadata,
    NodeDependency,
    NodePlan,
    inline_json_result,
    republish_execution_artifact,
)
from biomodals.execution.modal import (
    ExecutionRequestFile,
    OutputClaimExecutionDefinitionCoordinatorLifecycle,
)
from biomodals.execution.nodes import (
    NodeRunContext,
    ProviderCallSpec,
    ProviderNode,
    TaskDefinition,
    TaskProviderNode,
)
from biomodals.helper.artifacts import sha256_file
from biomodals.helper.output_claim import acquire_output_claim
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    VolumePath,
)

REQUEST_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 16 * 1024 * 1024
DOWNLOAD_MODELS_NODE = "download-models"
PREPARE_NODE = "prepare-inputs"
CHUNKS_NODE = "prepare-pdb-chunks"
FINALIZE_NODE = "finalize-inputs"
PREPROCESS_NODE = "preprocess-dataset"
INFERENCE_NODE = "run-inference"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "ENsiRNA execution request",
)


@dataclass(frozen=True, slots=True)
class EnsirnaPdbChunkSpec:
    """One CPU Rosetta PDB preparation chunk."""

    chunk_name: str
    csv_path: str
    json_path: str
    pdb_dir: str


@dataclass(frozen=True, slots=True)
class EnsirnaPreparationPlan:
    """Volume-backed prepared-input contract for ENsiRNA inference."""

    cache_key: str
    prepared_dir: str
    json_path: str
    processed_dir: str
    candidate_count: int
    chunk_count: int
    chunks: list[EnsirnaPdbChunkSpec]
    cached: bool


@dataclass(frozen=True)
class EnsirnaExecutionRequest:
    """Immutable FASTA input plus operational preparation settings."""

    run_name: str
    fasta_content: bytes
    prepare_workers: int
    pdb_cores: int
    preprocess_shard_size: int
    force_generation: str | None
    app_version: str
    max_active_provider_calls: int | None = None
    max_active_gpu_provider_calls: int = 1
    replace_claim_owner: str | None = None

    def __post_init__(self) -> None:
        """Reject empty scientific inputs and unusable worker limits."""
        if not self.run_name or not self.fasta_content:
            raise ValueError("ENsiRNA run name and FASTA cannot be empty")
        if (
            self.prepare_workers < 1
            or self.pdb_cores < 1
            or self.preprocess_shard_size < 1
        ):
            raise ValueError("ENsiRNA worker settings must be positive")
        if self.prepare_workers * self.pdb_cores > 64:
            raise ValueError("prepare_workers * pdb_cores must not exceed 64")
        total_limit = self.max_active_provider_calls
        if total_limit is None:
            total_limit = self.prepare_workers
            object.__setattr__(
                self,
                "max_active_provider_calls",
                total_limit,
            )
        if (
            total_limit < 1
            or self.max_active_gpu_provider_calls < 0
            or self.max_active_gpu_provider_calls > total_limit
        ):
            raise ValueError("ENsiRNA provider-call limits are invalid")
        if not self.app_version:
            raise ValueError("ENsiRNA app version cannot be empty")

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Build model setup, CPU fan-out, preprocessing, and inference Nodes."""
        return ExecutionPlan(
            workload_name="ensirna",
            workload_run_key=self.run_name,
            nodes=(
                NodePlan(DOWNLOAD_MODELS_NODE),
                NodePlan(PREPARE_NODE),
                NodePlan(
                    CHUNKS_NODE,
                    dependencies=(NodeDependency(PREPARE_NODE),),
                    allow_empty_result=True,
                ),
                NodePlan(
                    FINALIZE_NODE,
                    dependencies=(NodeDependency(CHUNKS_NODE),),
                ),
                NodePlan(
                    PREPROCESS_NODE,
                    dependencies=(
                        NodeDependency(DOWNLOAD_MODELS_NODE),
                        NodeDependency(FINALIZE_NODE),
                    ),
                ),
                NodePlan(
                    INFERENCE_NODE,
                    dependencies=(NodeDependency(PREPROCESS_NODE),),
                ),
            ),
            scientific_payload={
                "fasta_sha256": sha256(self.fasta_content).hexdigest(),
                "force_generation": self.force_generation,
            },
            scientific_versions={
                "ensirna": self.app_version,
                "biomodals.ensirna.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
        )

    def to_bytes(self) -> bytes:
        """Encode the bounded request without Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "fasta_content": b64encode(self.fasta_content).decode("ascii"),
                "prepare_workers": self.prepare_workers,
                "pdb_cores": self.pdb_cores,
                "preprocess_shard_size": self.preprocess_shard_size,
                "force_generation": self.force_generation,
                "app_version": self.app_version,
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": (self.max_active_gpu_provider_calls),
                "replace_claim_owner": self.replace_claim_owner,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("ENsiRNA execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> EnsirnaExecutionRequest:
        """Decode and revalidate a staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("ENsiRNA execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("ENsiRNA execution request schema is unsupported")
        encoded_fasta = value.pop("fasta_content", None)
        if not isinstance(encoded_fasta, str):
            raise TypeError("ENsiRNA FASTA content must be base64 text")
        value["fasta_content"] = b64decode(encoded_fasta, validate=True)
        return cls(**value)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: EnsirnaExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: EnsirnaExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> EnsirnaExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return EnsirnaExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


class EnsirnaPublications:
    """Own ENsiRNA cache claims and publication reconstruction."""

    def __init__(
        self,
        *,
        request: EnsirnaExecutionRequest,
        execution_run_id: UUID,
        output_root: str | Path,
        output_volume_name: str,
        output_claims: Any,
    ) -> None:
        """Bind one Run to its content-addressed ENsiRNA cache."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.output_root = Path(output_root)
        self.output_volume_name = output_volume_name
        self.output_claims = output_claims

    @property
    def cache_key(self) -> str:
        """Return the established scientific cache key."""
        return _workload_module()._cache_key_for_fasta(
            self.request.fasta_content,
            force_generation=self.request.force_generation,
        )

    @property
    def layout(self):
        """Return the established cache directory layout."""
        return _workload_module()._layout_for_cache_key(self.cache_key)

    def claim(self, publication: str) -> None:
        """Claim one cache publication before provider admission."""
        acquire_output_claim(
            self.output_claims,
            claim_key=f"ensirna-{publication}:{self.cache_key}",
            owner=str(self.execution_run_id),
            replace_owner=self.request.replace_claim_owner,
        )

    @staticmethod
    def plan_result(plan: EnsirnaPreparationPlan) -> AppRunResult:
        """Publish one small preparation plan through execution storage."""
        return inline_json_result(
            name="preparation-plan",
            value=asdict(plan),
            filename="preparation-plan.json",
        )

    def cached_plan(self) -> EnsirnaPreparationPlan | None:
        """Recover a complete preprocessed plan when present."""
        return _workload_module()._cached_preparation_plan(
            cache_key=self.cache_key,
            layout=self.layout,
        )

    def result(self) -> AppRunResult | None:
        """Reconstruct the exact inference result when present."""
        app = _workload_module()
        if not app._result_ready(self.layout, self.cache_key):
            return None
        path = self.layout.outputs_dir / f"{app.APP_INFO.input_stem}_result.xlsx"
        marker = orjson.loads(app._result_marker_path(self.layout).read_bytes())
        if not isinstance(marker, dict):
            return None
        file = ArtifactFile(
            path=path.name,
            size_bytes=marker.get("size"),
            content_sha256=marker.get("sha256"),
        )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="ensirna-result",
                    kind=ArtifactKind.TABLE,
                    storage=VolumePath(
                        volume_name=self.output_volume_name,
                        path=path.relative_to(self.output_root).as_posix(),
                    ),
                    metadata={
                        "files": [file.model_dump(mode="json", exclude_none=True)]
                    },
                )
            ],
        )


@dataclass
class _EnsirnaDownloadNode(ProviderNode):
    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        return ProviderCallSpec(
            function_name="download_ensirna_models",
            uses_gpu=False,
            runtime_image_key="ensirna-cpu",
            kwargs={"force": False},
        )

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del metadata
        if result is not None:
            raise ValueError("ENsiRNA model download returned an unexpected value")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)


@dataclass
class _EnsirnaPrepareNode(ProviderNode):
    request: EnsirnaExecutionRequest
    publications: EnsirnaPublications

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        self.publications.claim("prepared")
        return ProviderCallSpec(
            function_name="ensirna_prepare_inputs",
            uses_gpu=False,
            runtime_image_key="ensirna-cpu",
            kwargs={
                "mrna_fasta_bytes": self.request.fasta_content,
                "max_prepare_jobs": self.request.prepare_workers,
                "force_generation": self.request.force_generation,
            },
        )

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del metadata
        return self.publications.plan_result(_plan_from_value(result))


@dataclass
class _EnsirnaChunkNode(TaskProviderNode):
    request: EnsirnaExecutionRequest
    publications: EnsirnaPublications

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def discover_remote_tasks(
        self,
        context: NodeRunContext,
    ) -> tuple[TaskDefinition, ...]:
        plan = _plan_from_context(context)
        return tuple(
            TaskDefinition(
                task_key=chunk.chunk_name,
                scientific_payload={"csv_sha256": sha256_file(Path(chunk.csv_path))},
                execution_payload={"chunk": asdict(chunk)},
            )
            for chunk in plan.chunks
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        del context
        self.publications.claim("prepared")
        chunk = _chunk_from_task(task)
        return ProviderCallSpec(
            function_name="ensirna_prepare_pdb_chunk",
            uses_gpu=False,
            runtime_image_key="ensirna-cpu",
            kwargs={"chunk": chunk, "pdb_cores": self.request.pdb_cores},
            metadata={"chunk": asdict(chunk)},
        )

    def process_remote_task_result(
        self,
        task_key: str,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        if not isinstance(result, dict) or result.get("chunk_name") != task_key:
            raise ValueError("ENsiRNA chunk worker returned invalid metadata")
        if not _workload_module()._chunk_artifacts_valid(
            _chunk_from_value(metadata.get("chunk"))
        ):
            raise FileNotFoundError("ENsiRNA chunk publication is unavailable")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del context, expected_fingerprint
        return (
            AppRunResult(status=AppRunStatus.SUCCEEDED)
            if _workload_module()._chunk_artifacts_valid(_chunk_from_task(task))
            else None
        )

    def observe_remote_task_publication(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del result, artifacts
        try:
            available = (
                self.recover_remote_task_result(context, task, expected_fingerprint)
                is not None
            )
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        del results
        if errors:
            return AppRunResult(
                status=AppRunStatus.FAILED,
                warnings=[
                    "; ".join(f"{key}: {value}" for key, value in errors.items())
                ],
            )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[republish_execution_artifact(context.single_input("plan"))],
        )


@dataclass
class _EnsirnaPlanNode(ProviderNode):
    stage: str
    request: EnsirnaExecutionRequest
    publications: EnsirnaPublications

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        plan = _plan_from_context(context)
        self.publications.claim("prepared")
        if self.stage == "finalize":
            return ProviderCallSpec(
                function_name="ensirna_finalize_prepared_inputs",
                uses_gpu=False,
                runtime_image_key="ensirna-cpu",
                kwargs={"plan": plan},
            )
        if self.stage == "preprocess":
            return ProviderCallSpec(
                function_name="ensirna_preprocess_dataset",
                uses_gpu=True,
                runtime_image_key="ensirna-gpu",
                kwargs={
                    "plan": plan,
                    "preprocess_shard_size": self.request.preprocess_shard_size,
                },
            )
        raise ValueError(f"Unknown ENsiRNA plan stage {self.stage!r}")

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del metadata
        plan = _plan_from_value(result)
        app = _workload_module()
        available = (
            len(app._json_records(Path(plan.json_path))) == plan.candidate_count
            if self.stage == "finalize"
            else app._is_prepared(self.publications.layout)
        )
        if not available:
            raise FileNotFoundError(f"ENsiRNA {self.stage} publication is unavailable")
        return self.publications.plan_result(plan)

    def recover_result_publication(
        self,
        context: NodeRunContext,
    ) -> AppRunResult | None:
        del context
        if self.stage != "preprocess":
            return None
        plan = self.publications.cached_plan()
        return None if plan is None else self.publications.plan_result(plan)


@dataclass
class _EnsirnaInferenceNode(ProviderNode):
    publications: EnsirnaPublications

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        self.publications.claim("result")
        return ProviderCallSpec(
            function_name="run_ensirna_inference",
            uses_gpu=True,
            runtime_image_key="ensirna-gpu",
            kwargs={
                "prepared_dir": _plan_from_context(context).prepared_dir,
                "force": False,
            },
        )

    def process_remote_result(
        self,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del metadata
        if not isinstance(result, bytes):
            raise ValueError("ENsiRNA inference returned invalid result bytes")
        publication = self.publications.result()
        if publication is None:
            raise FileNotFoundError("ENsiRNA result publication is unavailable")
        return publication

    def recover_result_publication(
        self,
        context: NodeRunContext,
    ) -> AppRunResult | None:
        del context
        return self.publications.result()

    def observe_result_publication(
        self,
        context: NodeRunContext,
        result: AppRunResult,
        artifacts: tuple[ExecutionArtifact, ...],
    ) -> AvailabilityStatus | None:
        del context, result, artifacts
        try:
            available = self.publications.result() is not None
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING


def _chunk_from_value(value: object) -> EnsirnaPdbChunkSpec:
    if not isinstance(value, Mapping):
        raise TypeError("ENsiRNA chunk payload is invalid")
    value = cast(Mapping[str, object], value)
    fields = {
        name: value.get(name)
        for name in ("chunk_name", "csv_path", "json_path", "pdb_dir")
    }
    if not all(isinstance(field, str) for field in fields.values()):
        raise TypeError("ENsiRNA chunk payload is invalid")
    return EnsirnaPdbChunkSpec(
        chunk_name=cast(str, fields["chunk_name"]),
        csv_path=cast(str, fields["csv_path"]),
        json_path=cast(str, fields["json_path"]),
        pdb_dir=cast(str, fields["pdb_dir"]),
    )


def _chunk_from_task(task: TaskDefinition) -> EnsirnaPdbChunkSpec:
    if not isinstance(task.execution_payload, Mapping):
        raise TypeError("ENsiRNA Task payload is invalid")
    return _chunk_from_value(task.execution_payload.get("chunk"))


def _plan_from_value(value: object) -> EnsirnaPreparationPlan:
    if not isinstance(value, Mapping):
        raise TypeError("ENsiRNA plan payload is invalid")
    value = cast(Mapping[str, object], value)
    chunks = value.get("chunks")
    cache_key = value.get("cache_key")
    prepared_dir = value.get("prepared_dir")
    json_path = value.get("json_path")
    processed_dir = value.get("processed_dir")
    candidate_count = value.get("candidate_count")
    chunk_count = value.get("chunk_count")
    cached = value.get("cached")
    if (
        not isinstance(cache_key, str)
        or not isinstance(prepared_dir, str)
        or not isinstance(json_path, str)
        or not isinstance(processed_dir, str)
        or type(candidate_count) is not int
        or type(chunk_count) is not int
        or not isinstance(cached, bool)
        or not isinstance(chunks, list)
    ):
        raise TypeError("ENsiRNA plan payload has invalid fields")
    parsed_chunks = [_chunk_from_value(chunk) for chunk in chunks]
    if len(parsed_chunks) != chunk_count:
        raise TypeError("ENsiRNA chunk payload is invalid")
    plan = EnsirnaPreparationPlan(
        cache_key=cache_key,
        prepared_dir=prepared_dir,
        json_path=json_path,
        processed_dir=processed_dir,
        candidate_count=candidate_count,
        chunk_count=chunk_count,
        chunks=parsed_chunks,
        cached=cached,
    )
    _workload_module()._validate_preparation_plan(plan)
    return plan


def _plan_from_context(context: NodeRunContext) -> EnsirnaPreparationPlan:
    return _plan_from_value(orjson.loads(context.read_input_bytes("plan")))


def _workload_module():
    """Import workload-owned validators after the Modal app finishes loading."""
    from biomodals.app.score import ensirna_app

    return ensirna_app


def ensirna_execution_graph(
    request: EnsirnaExecutionRequest,
    publications: EnsirnaPublications,
) -> ExecutionGraph:
    """Build ENsiRNA's established preparation and inference pipeline."""
    graph = ExecutionGraph(
        "ensirna",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="ensirna",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    download = graph.add_node(_EnsirnaDownloadNode(), id=DOWNLOAD_MODELS_NODE)
    prepare = graph.add_node(
        _EnsirnaPrepareNode(request, publications),
        id=PREPARE_NODE,
    )
    chunks = graph.add_node(
        _EnsirnaChunkNode(request, publications),
        id=CHUNKS_NODE,
        inputs={"plan": prepare.outputs(kind=ArtifactKind.TABLE)},
        allow_empty_result=True,
    )
    finalize = graph.add_node(
        _EnsirnaPlanNode("finalize", request, publications),
        id=FINALIZE_NODE,
        inputs={"plan": chunks.outputs(kind=ArtifactKind.TABLE)},
    )
    preprocess = graph.add_node(
        _EnsirnaPlanNode("preprocess", request, publications),
        id=PREPROCESS_NODE,
        inputs={"plan": finalize.outputs(kind=ArtifactKind.TABLE)},
        depends_on=[download],
    )
    graph.add_node(
        _EnsirnaInferenceNode(publications),
        id=INFERENCE_NODE,
        inputs={"plan": preprocess.outputs(kind=ArtifactKind.TABLE)},
    )
    return graph


class EnsirnaExecutionCoordinator(OutputClaimExecutionDefinitionCoordinatorLifecycle):
    """Bind one run-scoped writer to ENsiRNA publications."""

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
        output_claims: Any,
        provider_driver: Any,
        app_version: str,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only the deployment resources used by this adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            artifact_volume_name=output_volume_name,
            output_volume=output_volume,
            output_claims=output_claims,
            provider_driver=provider_driver,
            graph_builder=self._graph,
            target_scientific_versions={"ensirna": app_version},
            poll_interval_seconds=poll_interval_seconds,
        )
        self.output_volume_name = output_volume_name
        self.output_claims = output_claims

    def _graph(
        self,
        request: EnsirnaExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return ensirna_execution_graph(
            request=request,
            publications=EnsirnaPublications(
                request=request,
                execution_run_id=self.execution_run_id,
                output_root=self.volume_root,
                output_volume_name=self.output_volume_name,
                output_claims=self.output_claims,
            ),
        )
