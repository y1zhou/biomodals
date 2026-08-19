"""Direct Protenix adaptation of the shared execution kernel."""

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
from biomodals.helper.io import require_safe_filename_component
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
PROTENIX_DATA_RELEASE = "v1.0.0"
MAX_REQUEST_BYTES = 16 * 1024 * 1024
DOWNLOAD_NODE = "download-model-data"
PLAN_NODE = "plan-preprocessing"
MSA_NODE = "search-msa"
FINALIZE_NODE = "finalize-preprocessing"
INFERENCE_NODE = "run-protenix"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "Protenix execution request",
)


@dataclass(frozen=True, slots=True)
class ProtenixMsaTaskSpec:
    """One content-addressed MSA/template search."""

    task_key: str
    input_name: str
    query_command: str
    input_json_path: str
    output_dir: str
    msa_server_mode: str
    expected_json_path: str
    publication_key: str


@dataclass(frozen=True, slots=True)
class ProtenixPreparationPlan:
    """Prepared-input fan-out plus its stable final publication."""

    preparation_key: str
    prepared_json_path: str
    tasks: tuple[ProtenixMsaTaskSpec, ...]


@dataclass(frozen=True)
class ProtenixExecutionRequest:
    """Immutable input and scientific flags plus coordinator capacity."""

    run_name: str
    input_content: bytes
    model_name: str
    seeds: str
    cycle: int
    step: int
    sample: int
    dtype: str
    use_msa: bool
    msa_server_mode: str
    use_template: bool
    use_rna_msa: bool
    use_tfg_guidance: bool
    use_fast_layernorm: bool
    force_redownload: bool
    extra_args: str | None
    score_only: bool
    max_active_provider_calls: int
    app_version: str
    max_active_gpu_provider_calls: int = 1
    replace_claim_owner: str | None = None

    def __post_init__(self) -> None:
        """Reject empty inputs and unusable coordinator capacity."""
        if not self.run_name or not self.input_content or not self.model_name:
            raise ValueError("Protenix run name, input, and model cannot be empty")
        require_safe_filename_component(
            self.run_name,
            field_name="Protenix run_name",
        )
        if (
            self.max_active_provider_calls < 1
            or self.max_active_gpu_provider_calls < 0
            or self.max_active_gpu_provider_calls > self.max_active_provider_calls
        ):
            raise ValueError("Protenix provider-call limits are invalid")
        if not self.app_version:
            raise ValueError("Protenix app version cannot be empty")

    @property
    def requires_preprocessing(self) -> bool:
        """Return whether prediction needs MSA/template preparation."""
        return not self.score_only and (
            self.use_msa or self.use_template or self.use_rna_msa
        )

    @property
    def result_key(self) -> str:
        """Return the scientific identity used by the result publication."""
        return self.execution_plan.workload_plan_fingerprint

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Build download, optional CPU fan-out, and GPU inference Nodes."""
        nodes = [NodePlan(DOWNLOAD_NODE)]
        inference_dependencies = [NodeDependency(DOWNLOAD_NODE)]
        if self.requires_preprocessing:
            nodes.extend((
                NodePlan(PLAN_NODE),
                NodePlan(
                    MSA_NODE,
                    dependencies=(NodeDependency(PLAN_NODE),),
                    allow_empty_result=True,
                ),
                NodePlan(
                    FINALIZE_NODE,
                    dependencies=(NodeDependency(MSA_NODE),),
                ),
            ))
            inference_dependencies.append(NodeDependency(FINALIZE_NODE))
        nodes.append(
            NodePlan(INFERENCE_NODE, dependencies=tuple(inference_dependencies))
        )
        return ExecutionPlan(
            workload_name="protenix",
            workload_run_key=self.run_name,
            nodes=tuple(nodes),
            scientific_payload={
                "input_sha256": sha256(self.input_content).hexdigest(),
                "model_name": self.model_name,
                "seeds": self.seeds,
                "cycle": self.cycle,
                "step": self.step,
                "sample": self.sample,
                "dtype": self.dtype,
                "use_msa": self.use_msa,
                "msa_server_mode": self.msa_server_mode,
                "use_template": self.use_template,
                "use_rna_msa": self.use_rna_msa,
                "use_tfg_guidance": self.use_tfg_guidance,
                "use_fast_layernorm": self.use_fast_layernorm,
                "extra_args": self.extra_args,
                "score_only": self.score_only,
            },
            scientific_versions={
                "protenix": self.app_version,
                "protenix.model": self.model_name,
                "protenix.reference_data": PROTENIX_DATA_RELEASE,
                "biomodals.protenix.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
        )

    def to_bytes(self) -> bytes:
        """Encode the bounded request without Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "input_content": b64encode(self.input_content).decode("ascii"),
                "model_name": self.model_name,
                "seeds": self.seeds,
                "cycle": self.cycle,
                "step": self.step,
                "sample": self.sample,
                "dtype": self.dtype,
                "use_msa": self.use_msa,
                "msa_server_mode": self.msa_server_mode,
                "use_template": self.use_template,
                "use_rna_msa": self.use_rna_msa,
                "use_tfg_guidance": self.use_tfg_guidance,
                "use_fast_layernorm": self.use_fast_layernorm,
                "force_redownload": self.force_redownload,
                "extra_args": self.extra_args,
                "score_only": self.score_only,
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": (self.max_active_gpu_provider_calls),
                "app_version": self.app_version,
                "replace_claim_owner": self.replace_claim_owner,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("Protenix execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> ProtenixExecutionRequest:
        """Decode and revalidate a staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("Protenix execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("Protenix execution request schema is unsupported")
        encoded_input = value.pop("input_content", None)
        if not isinstance(encoded_input, str):
            raise TypeError("Protenix input content must be base64 text")
        value["input_content"] = b64decode(encoded_input, validate=True)
        return cls(**value)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: ProtenixExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: ProtenixExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> ProtenixExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return ProtenixExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


class ProtenixPublications:
    """Own Protenix cache refresh, claims, and result reconstruction."""

    def __init__(
        self,
        *,
        request: ProtenixExecutionRequest,
        execution_run_id: UUID,
        output_root: str | Path,
        output_volume_name: str,
        msa_cache_volume: Any,
        output_claims: Any,
    ) -> None:
        """Bind one Run to its result and MSA publication stores."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.output_root = Path(output_root)
        self.output_volume_name = output_volume_name
        self.msa_cache_volume = msa_cache_volume
        self.output_claims = output_claims

    def refresh_msa(self) -> None:
        """Refresh provider-published preparation cache state once per batch."""
        self.msa_cache_volume.reload()

    def claim(self, key: str) -> None:
        """Claim one content-addressed publication before provider admission."""
        acquire_output_claim(
            self.output_claims,
            claim_key=key,
            owner=str(self.execution_run_id),
            replace_owner=self.request.replace_claim_owner,
        )

    @staticmethod
    def plan_result(plan: ProtenixPreparationPlan) -> AppRunResult:
        """Publish a bounded preparation plan through execution storage."""
        return inline_json_result(
            name="preparation-plan",
            value=asdict(plan),
            filename="preparation-plan.json",
        )

    def result(self) -> AppRunResult | None:
        """Reconstruct the exact Protenix result archive when present."""
        app = _workload_module()
        if not app._result_ready(self.request.result_key, self.request.run_name):
            return None
        path = app._result_path(self.request.result_key, self.request.run_name)
        marker = orjson.loads(
            path.with_suffix(f"{path.suffix}.complete.json").read_bytes()
        )
        if not isinstance(marker, dict):
            return None
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="protenix-result",
                    kind=ArtifactKind.ARCHIVE,
                    storage=VolumePath(
                        volume_name=self.output_volume_name,
                        path=path.relative_to(self.output_root).as_posix(),
                    ),
                    metadata={
                        "files": [
                            ArtifactFile(
                                path=path.name,
                                size_bytes=marker.get("size"),
                                content_sha256=marker.get("sha256"),
                            ).model_dump(mode="json", exclude_none=True)
                        ]
                    },
                )
            ],
        )


@dataclass
class _ProtenixDownloadNode(ProviderNode):
    request: ProtenixExecutionRequest

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        return ProviderCallSpec(
            function_name="download_protenix_data",
            uses_gpu=False,
            runtime_image_key="protenix-cpu",
            kwargs={
                "model_name": self.request.model_name,
                "force": self.request.force_redownload,
                "include_templates": self.request.use_template,
            },
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del metadata
        if result is not None:
            raise ValueError("Protenix data download returned an unexpected value")
        return AppRunResult(status=AppRunStatus.SUCCEEDED)


@dataclass
class _ProtenixPlanNode(ProviderNode):
    request: ProtenixExecutionRequest
    publications: ProtenixPublications

    def refresh_result_storage(self) -> None:
        self.publications.refresh_msa()

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        return ProviderCallSpec(
            function_name="plan_protenix_inputs",
            uses_gpu=False,
            runtime_image_key="protenix-cpu",
            kwargs={
                "input_bytes": self.request.input_content,
                "msa_server_mode": self.request.msa_server_mode,
                "use_template": self.request.use_template,
                "use_rna_msa": self.request.use_rna_msa,
            },
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del metadata
        return self.publications.plan_result(_preparation_plan_from_value(result))


@dataclass
class _ProtenixMsaNode(TaskProviderNode):
    publications: ProtenixPublications

    def refresh_result_storage(self) -> None:
        self.publications.refresh_msa()

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        return tuple(
            TaskDefinition(
                task_key=task.task_key,
                scientific_payload={"publication_key": task.publication_key},
                execution_payload={"task": asdict(task)},
            )
            for task in _preparation_plan_from_context(context).tasks
        )

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        del context
        spec = _msa_task_from_definition(task)
        self.publications.claim(f"protenix-msa:{spec.publication_key}")
        return ProviderCallSpec(
            function_name="query_protenix_msa_server",
            uses_gpu=False,
            runtime_image_key="protenix-cpu",
            kwargs={"task": spec},
            metadata={"task": asdict(spec)},
        )

    def process_remote_task_result(
        self,
        task_key: str,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        del task_key
        if result is not None or not _workload_module()._msa_task_ready(
            _msa_task_from_value(metadata.get("task"))
        ):
            raise FileNotFoundError("Protenix MSA publication is unavailable")
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
            if _workload_module()._msa_task_ready(_msa_task_from_definition(task))
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
                status=AppRunStatus.FAILED, warnings=list(errors.values())
            )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[republish_execution_artifact(context.single_input("plan"))],
        )


@dataclass
class _ProtenixFinalizeNode(ProviderNode):
    request: ProtenixExecutionRequest
    publications: ProtenixPublications

    def refresh_result_storage(self) -> None:
        self.publications.refresh_msa()

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        plan = _preparation_plan_from_context(context)
        return ProviderCallSpec(
            function_name="finalize_protenix_inputs",
            uses_gpu=False,
            runtime_image_key="protenix-cpu",
            kwargs={
                "input_bytes": self.request.input_content,
                "plan": plan,
            },
            metadata={"plan": asdict(plan)},
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        if not isinstance(result, dict):
            raise ValueError("Protenix finalizer returned invalid metadata")
        plan = _preparation_plan_from_value(metadata.get("plan"))
        if not _workload_module()._prepared_ready(plan):
            raise FileNotFoundError("Protenix prepared input is unavailable")
        return self.publications.plan_result(plan)

    def recover_result_publication(
        self, context: NodeRunContext
    ) -> AppRunResult | None:
        if not context.inputs.get("plan"):
            return None
        plan = _preparation_plan_from_context(context)
        return (
            self.publications.plan_result(plan)
            if _workload_module()._prepared_ready(plan)
            else None
        )


@dataclass
class _ProtenixInferenceNode(ProviderNode):
    request: ProtenixExecutionRequest
    publications: ProtenixPublications

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        self.publications.claim(f"protenix-result:{self.request.result_key}")
        prepared_path = (
            _preparation_plan_from_context(context).prepared_json_path
            if self.request.requires_preprocessing
            else None
        )
        return ProviderCallSpec(
            function_name="run_protenix",
            uses_gpu=True,
            runtime_image_key="protenix-gpu",
            kwargs={
                "input_bytes": None if prepared_path else self.request.input_content,
                "prepared_input_path": prepared_path,
                "run_name": self.request.run_name,
                "result_key": self.request.result_key,
                "model_name": self.request.model_name,
                "seeds": self.request.seeds,
                "cycle": self.request.cycle,
                "step": self.request.step,
                "sample": self.request.sample,
                "dtype": self.request.dtype,
                "use_msa": self.request.use_msa,
                "msa_server_mode": (
                    "colabfold"
                    if self.request.score_only
                    else self.request.msa_server_mode
                ),
                "use_template": self.request.use_template,
                "use_rna_msa": self.request.use_rna_msa,
                "use_tfg_guidance": self.request.use_tfg_guidance,
                "use_fast_layernorm": self.request.use_fast_layernorm,
                "extra_args": self.request.extra_args,
                "score_only": self.request.score_only,
            },
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del metadata
        if not isinstance(result, dict) or "result_path" not in result:
            raise ValueError("Protenix inference returned invalid metadata")
        publication = self.publications.result()
        if publication is None:
            raise FileNotFoundError("Protenix result publication is unavailable")
        return publication

    def recover_result_publication(
        self, context: NodeRunContext
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


def _preparation_plan_from_value(value: object) -> ProtenixPreparationPlan:
    if not isinstance(value, Mapping):
        raise TypeError("Protenix preparation plan is invalid")
    value = cast(Mapping[str, object], value)
    preparation_key = value.get("preparation_key")
    prepared_json_path = value.get("prepared_json_path")
    tasks = value.get("tasks")
    if (
        not isinstance(preparation_key, str)
        or not isinstance(prepared_json_path, str)
        or not isinstance(tasks, list)
    ):
        raise TypeError("Protenix preparation plan has invalid fields")
    parsed_tasks = [_msa_task_from_value(task) for task in tasks]
    return ProtenixPreparationPlan(
        preparation_key=preparation_key,
        prepared_json_path=prepared_json_path,
        tasks=tuple(parsed_tasks),
    )


def _preparation_plan_from_context(context: NodeRunContext) -> ProtenixPreparationPlan:
    return _preparation_plan_from_value(orjson.loads(context.read_input_bytes("plan")))


def _msa_task_from_value(value: object) -> ProtenixMsaTaskSpec:
    if not isinstance(value, Mapping):
        raise TypeError("Protenix MSA Task is invalid")
    value = cast(Mapping[str, object], value)
    fields = tuple(
        value.get(name)
        for name in (
            "task_key",
            "input_name",
            "query_command",
            "input_json_path",
            "output_dir",
            "msa_server_mode",
            "expected_json_path",
            "publication_key",
        )
    )
    if not all(isinstance(field, str) for field in fields):
        raise TypeError("Protenix MSA Task has invalid fields")
    return ProtenixMsaTaskSpec(*cast(tuple[str, ...], fields))


def _msa_task_from_definition(task: TaskDefinition) -> ProtenixMsaTaskSpec:
    if not isinstance(task.execution_payload, Mapping):
        raise TypeError("Protenix Task payload is invalid")
    return _msa_task_from_value(task.execution_payload.get("task"))


def _workload_module():
    """Import workload-owned publication probes after Modal app loading."""
    from biomodals.app.fold import protenix_app

    return protenix_app


def protenix_execution_graph(
    request: ProtenixExecutionRequest,
    publications: ProtenixPublications,
) -> ExecutionGraph:
    """Build Protenix's download, preprocessing, and inference graph."""
    graph = ExecutionGraph(
        "protenix",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="protenix",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    graph.add_node(_ProtenixDownloadNode(request), id=DOWNLOAD_NODE)
    inference_inputs = {}
    if request.requires_preprocessing:
        plan = graph.add_node(
            _ProtenixPlanNode(request, publications),
            id=PLAN_NODE,
        )
        msa = graph.add_node(
            _ProtenixMsaNode(publications),
            id=MSA_NODE,
            inputs={"plan": plan.outputs(kind=ArtifactKind.TABLE)},
            allow_empty_result=True,
        )
        finalized = graph.add_node(
            _ProtenixFinalizeNode(request, publications),
            id=FINALIZE_NODE,
            inputs={"plan": msa.outputs(kind=ArtifactKind.TABLE)},
        )
        inference_inputs["plan"] = finalized.outputs(kind=ArtifactKind.TABLE)
    graph.add_node(
        _ProtenixInferenceNode(request, publications),
        id=INFERENCE_NODE,
        inputs=inference_inputs,
        depends_on=[DOWNLOAD_NODE],
    )
    return graph


class ProtenixExecutionCoordinator(OutputClaimExecutionDefinitionCoordinatorLifecycle):
    """Bind one run-scoped writer to Protenix publications."""

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
        msa_cache_volume: Any,
        output_claims: Any,
        provider_driver: Any,
        app_version: str,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only resources used by this workload adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            artifact_volume_name=output_volume_name,
            output_volume=output_volume,
            output_claims=output_claims,
            provider_driver=provider_driver,
            graph_builder=self._graph,
            target_scientific_versions={"protenix": app_version},
            poll_interval_seconds=poll_interval_seconds,
        )
        self.output_volume_name = output_volume_name
        self.msa_cache_volume = msa_cache_volume
        self.output_claims = output_claims

    def _graph(
        self,
        request: ProtenixExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return protenix_execution_graph(
            request=request,
            publications=ProtenixPublications(
                request=request,
                execution_run_id=self.execution_run_id,
                output_root=self.volume_root,
                output_volume_name=self.output_volume_name,
                msa_cache_volume=self.msa_cache_volume,
                output_claims=self.output_claims,
            ),
        )
