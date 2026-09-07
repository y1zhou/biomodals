"""Single-node execution definition for the HuDiff-Ab app."""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID

import orjson

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionGraph,
    ExecutionOverview,
    ExecutionPlan,
    ExecutionPlanMetadata,
    NodeDependency,
    NodePlan,
    RunStatus,
)
from biomodals.execution.modal import (
    ExecutionDefinitionCoordinatorLifecycle,
    ExecutionRequestFile,
)
from biomodals.execution.nodes import (
    CoordinatorNode,
    NodeRunContext,
    ProviderCallSpec,
    TaskDefinition,
    TaskProviderNode,
)
from biomodals.helper.artifacts import read_volume_file_exact, replace_bytes_atomic
from biomodals.helper.shell import sanitize_filename
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    ExecutionArtifact,
    InlineBytes,
    VolumePath,
)
from biomodals.schema.storage import ZSTD_MEDIA_TYPE

REQUEST_SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_PAIR_RESULT_BYTES = 4 * 1024 * 1024
MAX_RESULT_DESCRIPTOR_BYTES = 64 * 1024
PAIRS_PER_GPU_CALL = 2
HUMANIZE_NODE = "humanize"
COLLECT_NODE = "collect"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "HuDiff-Ab execution request",
)
_RESULT_FILE = ExecutionRequestFile(
    "result.json",
    MAX_RESULT_DESCRIPTOR_BYTES,
    "HuDiff-Ab result descriptor",
)


@dataclass(frozen=True)
class HuDiffAbExecutionRequest:
    """Immutable paired input batch and scientific controls."""

    run_name: str
    csv_bytes: bytes
    candidate_count: int
    seed: int
    sampling_order: str
    upstream_inference_dropout: bool
    source_commit: str
    checkpoint_sha256: str
    patch_identity: str
    runtime_identity: str
    max_active_provider_calls: int = 1
    max_active_gpu_provider_calls: int = 1

    def __post_init__(self) -> None:
        """Reject malformed staged requests before provider dispatch."""
        if not isinstance(self.run_name, str) or not self.run_name:
            raise ValueError("HuDiff-Ab run name is required")
        if (
            sanitize_filename(self.run_name) != self.run_name
            or len(self.run_name.encode("utf-8")) > 200
        ):
            raise ValueError("HuDiff-Ab run name must be a safe short filename")
        if not isinstance(self.csv_bytes, bytes) or not self.csv_bytes:
            raise ValueError("HuDiff-Ab CSV input is required")
        if type(self.candidate_count) is not int or not 1 <= self.candidate_count <= 25:
            raise ValueError("HuDiff-Ab candidate_count must be between 1 and 25")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("HuDiff-Ab seed must be an unsigned 32-bit integer")
        if self.sampling_order not in {"shuffle", "left_to_right"}:
            raise ValueError("HuDiff-Ab sampling_order is unsupported")
        if type(self.upstream_inference_dropout) is not bool:
            raise ValueError("HuDiff-Ab upstream_inference_dropout must be boolean")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.source_commit,
                self.checkpoint_sha256,
                self.patch_identity,
                self.runtime_identity,
            )
        ):
            raise ValueError("HuDiff-Ab scientific identities are required")
        if (
            type(self.max_active_provider_calls) is not int
            or type(self.max_active_gpu_provider_calls) is not int
            or self.max_active_provider_calls < 1
            or self.max_active_gpu_provider_calls < 1
            or self.max_active_gpu_provider_calls > self.max_active_provider_calls
        ):
            raise ValueError("HuDiff-Ab requires positive GPU provider-call capacity")

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Describe the paired humanization run."""
        return ExecutionPlan(
            workload_name="hudiff_ab",
            workload_run_key=self.run_name,
            nodes=(
                NodePlan(HUMANIZE_NODE),
                NodePlan(
                    COLLECT_NODE,
                    dependencies=(NodeDependency(HUMANIZE_NODE),),
                ),
            ),
            scientific_payload={
                "input_csv_sha256": sha256(self.csv_bytes).hexdigest(),
                "candidate_count": self.candidate_count,
                "seed": self.seed,
                "sampling_order": self.sampling_order,
                "upstream_inference_dropout": self.upstream_inference_dropout,
            },
            scientific_versions={
                "hudiff_ab.source": self.source_commit,
                "hudiff_ab.model": self.checkpoint_sha256,
                "hudiff_ab.patch": self.patch_identity,
                "hudiff_ab.runtime": self.runtime_identity,
                "biomodals.hudiff_ab.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
        )

    def to_bytes(self) -> bytes:
        """Encode a bounded request without Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "csv_bytes": b64encode(self.csv_bytes).decode("ascii"),
                "candidate_count": self.candidate_count,
                "seed": self.seed,
                "sampling_order": self.sampling_order,
                "upstream_inference_dropout": self.upstream_inference_dropout,
                "source_commit": self.source_commit,
                "checkpoint_sha256": self.checkpoint_sha256,
                "patch_identity": self.patch_identity,
                "runtime_identity": self.runtime_identity,
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": (self.max_active_gpu_provider_calls),
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("HuDiff-Ab execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> HuDiffAbExecutionRequest:
        """Decode and revalidate one staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("HuDiff-Ab execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("HuDiff-Ab execution request schema is unsupported")
        encoded_csv = value.pop("csv_bytes", None)
        if not isinstance(encoded_csv, str):
            raise TypeError("HuDiff-Ab CSV content must be base64 text")
        value["csv_bytes"] = b64decode(encoded_csv, validate=True)
        return cls(**value)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: HuDiffAbExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: HuDiffAbExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> HuDiffAbExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return HuDiffAbExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


def _pair_records(csv_bytes: bytes) -> tuple[dict[str, str], ...]:
    from biomodals.app.design.hudiff_ab.validation import (
        parse_hudiff_ab_csv,
        validate_antibody_chains,
    )

    frame = parse_hudiff_ab_csv(csv_bytes)
    validate_antibody_chains(frame)
    return tuple(frame.iter_rows(named=True))


def _worker_kwargs(request: HuDiffAbExecutionRequest) -> dict[str, Any]:
    return {
        name: getattr(request, name)
        for name in (
            "candidate_count",
            "seed",
            "sampling_order",
            "upstream_inference_dropout",
        )
    }


@dataclass
class _HuDiffAbHumanizeNode(TaskProviderNode):
    request: HuDiffAbExecutionRequest

    @cached_property
    def records(self) -> tuple[dict[str, str], ...]:
        """Validate the batch once and retain its stable input order."""
        return _pair_records(self.request.csv_bytes)

    @staticmethod
    def _pair(task: TaskDefinition) -> dict[str, str]:
        payload = task.execution_payload
        if not isinstance(payload, Mapping):
            raise TypeError("HuDiff-Ab pair execution payload must be an object")
        pair = {name: payload.get(name) for name in ("id", "vh", "vl")}
        if not all(isinstance(value, str) for value in pair.values()):
            raise TypeError("HuDiff-Ab pair execution payload is invalid")
        typed_pair = cast(dict[str, str], pair)
        if typed_pair["id"] != task.task_key:
            raise ValueError("HuDiff-Ab Task identity does not match its pair ID")
        return typed_pair

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        """Validate the complete batch and create one durable Task per pair."""
        del context
        return tuple(
            TaskDefinition(
                task_key=record["id"],
                scientific_payload={
                    "id": record["id"],
                    "vh_sha256": sha256(record["vh"].encode()).hexdigest(),
                    "vl_sha256": sha256(record["vl"].encode()).hexdigest(),
                },
                execution_payload=record,
            )
            for record in self.records
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        """Preserve the direct worker for a one-pair input."""
        if len(self.records) > 1:
            return self.prepare_remote_task_batch(context, (task,))
        del context
        pair = self._pair(task)
        return ProviderCallSpec(
            function_name="hudiff_ab_humanize_pair",
            uses_gpu=True,
            runtime_image_key="hudiff_ab-a10g",
            kwargs={"pair": pair, **_worker_kwargs(self.request)},
        )

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[TaskDefinition, ...],
    ) -> ProviderCallSpec:
        """Send up to two ordered pairs to one concurrent A10G worker."""
        if len(self.records) == 1:
            if len(tasks) != 1:
                raise ValueError("Single-pair HuDiff-Ab input cannot be batched")
            return self.prepare_remote_task(context, tasks[0])
        del context
        if not 1 <= len(tasks) <= PAIRS_PER_GPU_CALL:
            raise ValueError("HuDiff-Ab GPU batches must contain one or two pairs")
        return ProviderCallSpec(
            function_name="hudiff_ab_humanize_batch",
            uses_gpu=True,
            runtime_image_key="hudiff_ab-a10g-batch",
            compatibility_key="hudiff_ab-two-pair-batch",
            max_tasks_per_call=PAIRS_PER_GPU_CALL,
            kwargs={
                "pairs": [self._pair(task) for task in tasks],
                **_worker_kwargs(self.request),
            },
        )

    def process_remote_task_batch_result(
        self,
        task_keys: tuple[str, ...],
        result: Any,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, AppRunResult]:
        """Restore one independently publishable result per pair Task."""
        if len(self.records) == 1:
            return super().process_remote_task_batch_result(task_keys, result, metadata)
        if not isinstance(result, Mapping) or set(result) != set(task_keys):
            raise ValueError("HuDiff-Ab batch result does not match its pair Tasks")
        return {
            task_key: AppRunResult.model_validate(result[task_key])
            for task_key in task_keys
        }

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        """Expose pair publications to the coordinator-local collector."""
        del context
        if errors:
            return AppRunResult(
                status=AppRunStatus.FAILED,
                warnings=[f"{key}: {errors[key]}" for key in sorted(errors)],
                metrics={"completed_pairs": len(results), "failed_pairs": len(errors)},
            )
        return AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            metrics={"completed_pairs": len(results), "failed_pairs": 0},
        )


def _result_path(execution_run_id: UUID, run_name: str) -> PurePosixPath:
    return (
        PurePosixPath("workflow-runs")
        / str(execution_run_id)
        / "hudiff_ab"
        / f"{run_name}_hudiff_ab.tar.zst"
    )


def _read_pair_result(
    context: NodeRunContext, artifact: ExecutionArtifact
) -> Mapping[str, Any]:
    content = context.resolve_artifact(artifact).read_bytes()
    if not 0 < len(content) <= MAX_PAIR_RESULT_BYTES:
        raise ValueError("HuDiff-Ab pair publication has an invalid size")
    value = orjson.loads(content)
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("HuDiff-Ab pair publication schema is invalid")
    pair_result = value.get("pair_result")
    if not isinstance(pair_result, Mapping):
        raise ValueError("HuDiff-Ab pair publication is incomplete")
    return pair_result


@dataclass
class _CollectHuDiffAbResultsNode(CoordinatorNode):
    request: HuDiffAbExecutionRequest

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Assemble ordered pair results without starting another container."""
        if context.volume_root is None or context.artifact_volume_name is None:
            raise RuntimeError("HuDiff-Ab collection requires execution storage")
        pair_results: dict[str, Mapping[str, Any]] = {}
        for artifact in context.inputs.get("pair-results", []):
            result = _read_pair_result(context, artifact)
            if not isinstance(result.get("id"), str):
                raise ValueError("HuDiff-Ab pair publication has no pair ID")
            pair_id = cast(str, result["id"])
            if pair_id in pair_results:
                raise ValueError(f"Duplicate HuDiff-Ab pair publication: {pair_id}")
            pair_results[pair_id] = result

        records = _pair_records(self.request.csv_bytes)
        ordered_ids = tuple(record["id"] for record in records)
        if set(pair_results) != set(ordered_ids):
            raise ValueError("HuDiff-Ab pair publications do not match the input")
        from biomodals.app.design.hudiff_ab.app import _aggregate_hudiff_ab_results

        archive, metrics = _aggregate_hudiff_ab_results(
            run_name=self.request.run_name,
            csv_bytes=self.request.csv_bytes,
            pair_results=[pair_results[pair_id] for pair_id in ordered_ids],
            parameters=_worker_kwargs(self.request),
        )
        relative_path = _result_path(context.execution_run_id, self.request.run_name)
        output_path = context.volume_root.joinpath(*relative_path.parts)
        replace_bytes_atomic(output_path, archive)
        digest = sha256(archive).hexdigest()
        result = AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="hudiff_ab_humanization",
                    kind=ArtifactKind.ARCHIVE,
                    storage=VolumePath(
                        volume_name=context.artifact_volume_name,
                        path=relative_path.as_posix(),
                        media_type=ZSTD_MEDIA_TYPE,
                    ),
                    metadata={
                        "archive_format": "tar.zst",
                        "run_name": self.request.run_name,
                        "pair_count": len(ordered_ids),
                        "candidate_count": self.request.candidate_count,
                        "files": [
                            {
                                "path": relative_path.name,
                                "size_bytes": len(archive),
                                "content_sha256": digest,
                            }
                        ],
                    },
                )
            ],
            metrics=metrics,
        )
        _RESULT_FILE.persist(
            context.volume_root,
            context.execution_run_id,
            result.model_dump_json().encode(),
        )
        return result

    def recover_result_publication(
        self, context: NodeRunContext
    ) -> AppRunResult | None:
        """Recover a same-Run archive published before coordinator interruption."""
        if context.volume_root is None:
            return None
        path = context.volume_root.joinpath(
            *_RESULT_FILE.path(context.execution_run_id).parts
        )
        if not path.is_file():
            return None
        return AppRunResult.model_validate_json(
            _RESULT_FILE.load(context.volume_root, context.execution_run_id)
        )


def hudiff_ab_execution_graph(
    request: HuDiffAbExecutionRequest,
) -> ExecutionGraph:
    """Build fixed two-pair GPU batches followed by local collection."""
    graph = ExecutionGraph(
        "hudiff_ab",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="hudiff_ab",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    humanize = graph.add_node(
        _HuDiffAbHumanizeNode(request),
        id=HUMANIZE_NODE,
        reuse_predecessor_publication=False,
    )
    graph.add_node(
        _CollectHuDiffAbResultsNode(request),
        id=COLLECT_NODE,
        inputs={"pair-results": humanize.outputs(kind=ArtifactKind.REPORT)},
        reuse_predecessor_publication=False,
    )
    return graph


class HuDiffAbExecutionCoordinator(ExecutionDefinitionCoordinatorLifecycle):
    """Bind one HuDiff-Ab request to the shared execution runtime."""

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
        checkpoint_sha256: str,
        patch_identity: str,
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
                "hudiff_ab.source": source_commit,
                "hudiff_ab.model": checkpoint_sha256,
                "hudiff_ab.patch": patch_identity,
                "hudiff_ab.runtime": runtime_identity,
                "biomodals.hudiff_ab.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
            max_parallel_nodes=1,
            poll_interval_seconds=poll_interval_seconds,
        )

    @staticmethod
    def _graph(
        request: HuDiffAbExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return hudiff_ab_execution_graph(request)


def result_from_overview(
    overview: ExecutionOverview,
    output_volume: Any,
) -> AppRunResult:
    """Load and content-verify the terminal HuDiff-Ab archive."""
    if overview.run.status != RunStatus.SUCCEEDED:
        raise LookupError("HuDiff-Ab humanization result is unavailable")
    result = AppRunResult.model_validate_json(
        _RESULT_FILE.load_from_volume(output_volume, overview.run.execution_run_id)
    )
    outputs: list[AppOutput] = []
    for output in result.outputs:
        if not isinstance(output.storage, VolumePath):
            outputs.append(output)
            continue
        files = output.metadata.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("HuDiff-Ab archive manifest is invalid")
        file = files[0]
        if not isinstance(file, Mapping):
            raise ValueError("HuDiff-Ab archive manifest is invalid")
        data = read_volume_file_exact(
            output_volume,
            output.storage.path,
            size_bytes=file.get("size_bytes"),
            content_sha256=file.get("content_sha256"),
        )
        if len(data) > MAX_RESULT_BYTES:
            raise ValueError("HuDiff-Ab result archive exceeds the byte limit")
        outputs.append(
            output.model_copy(
                update={
                    "storage": InlineBytes(
                        data=data,
                        filename=PurePosixPath(output.storage.path).name,
                        media_type=output.storage.media_type,
                    )
                }
            )
        )
    return result.model_copy(update={"outputs": outputs})
