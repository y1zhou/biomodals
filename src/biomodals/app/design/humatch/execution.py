"""Fixed-batch execution definition for the Humatch app."""

from __future__ import annotations

from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import dataclass
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

REQUEST_SCHEMA_VERSION = 2
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = 128 * 1024 * 1024
MAX_PAIR_RESULT_BYTES = 512 * 1024
MAX_RESULT_DESCRIPTOR_BYTES = 64 * 1024
HUMATCH_BATCH_SIZE = 6
HUMANIZE_NODE = "humanize"
COLLECT_NODE = "collect"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "Humatch execution request",
)
_RESULT_FILE = ExecutionRequestFile(
    "result.json",
    MAX_RESULT_DESCRIPTOR_BYTES,
    "Humatch result descriptor",
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
        if sanitize_filename(self.run_name) != self.run_name:
            raise ValueError("Humatch run name must be a safe filename component")
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
            or self.max_active_provider_calls < 1
            or self.max_active_gpu_provider_calls != 0
        ):
            raise ValueError("Humatch requires a positive CPU provider-call limit")
        if not all(
            isinstance(value, str) and value
            for value in (self.app_version, self.asset_record, self.runtime_identity)
        ):
            raise ValueError(
                "Humatch source, asset, and runtime identities are required"
            )

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Describe paired inference and local collection scientific identity."""
        return ExecutionPlan(
            workload_name="humatch",
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


def _pair_records(csv_bytes: bytes) -> tuple[dict[str, str], ...]:
    from biomodals.app.design.humatch.app import parse_humatch_csv

    return tuple(parse_humatch_csv(csv_bytes).iter_rows(named=True))


def _worker_kwargs(request: HumatchExecutionRequest) -> dict[str, Any]:
    return {
        name: getattr(request, name)
        for name in (
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
    }


@dataclass
class _HumatchHumanizeNode(TaskProviderNode):
    request: HumatchExecutionRequest

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        """Create one durable Task per complete VH-VL pair."""
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
            for record in _pair_records(self.request.csv_bytes)
        )

    def prepare_remote_task(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
    ) -> ProviderCallSpec:
        """Prepare one pair with the same fixed-batch compatibility policy."""
        return self.prepare_remote_task_batch(context, (task,))

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[TaskDefinition, ...],
    ) -> ProviderCallSpec:
        """Send at most six ordered pairs to one CPU worker."""
        del context
        if not 1 <= len(tasks) <= HUMATCH_BATCH_SIZE:
            raise ValueError("Humatch provider batches must contain one to six pairs")
        records: list[dict[str, str]] = []
        for task in tasks:
            payload = task.execution_payload
            if not isinstance(payload, Mapping):
                raise TypeError("Humatch pair execution payload must be an object")
            record = {name: payload.get(name) for name in ("id", "vh", "vl")}
            if not all(isinstance(value, str) for value in record.values()):
                raise TypeError("Humatch pair execution payload is invalid")
            typed_record = cast(dict[str, str], record)
            if typed_record["id"] != task.task_key:
                raise ValueError("Humatch Task identity does not match its pair ID")
            records.append(typed_record)
        return ProviderCallSpec(
            function_name="humatch_humanize_batch",
            uses_gpu=False,
            runtime_image_key="humatch-cpu",
            compatibility_key="humatch-six-pair-cpu",
            max_tasks_per_call=HUMATCH_BATCH_SIZE,
            kwargs={"pairs": records, **_worker_kwargs(self.request)},
        )

    def process_remote_task_batch_result(
        self,
        task_keys: tuple[str, ...],
        result: Any,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, AppRunResult]:
        """Decode one worker return into independent durable pair results."""
        del metadata
        if not isinstance(result, Mapping) or result.get("schema_version") != 1:
            raise TypeError("Humatch batch result is invalid")
        raw_results = result.get("pair_results")
        metrics = result.get("metrics")
        if not isinstance(raw_results, list) or not isinstance(metrics, Mapping):
            raise TypeError("Humatch batch result is incomplete")
        result_ids: list[object] = []
        for pair_result in raw_results:
            if not isinstance(pair_result, Mapping):
                raise TypeError("Humatch pair result must be an object")
            humanized = pair_result.get("humanized")
            result_ids.append(
                humanized.get("id") if isinstance(humanized, Mapping) else None
            )
        if tuple(result_ids) != task_keys:
            raise ValueError("Humatch batch result order does not match its Tasks")

        decoded: dict[str, AppRunResult] = {}
        for task_key, pair_result in zip(task_keys, raw_results, strict=True):
            content = orjson.dumps({
                "schema_version": 1,
                "batch_task_keys": list(task_keys),
                "batch_metrics": dict(metrics),
                "pair_result": pair_result,
            })
            if len(content) > MAX_PAIR_RESULT_BYTES:
                raise ValueError(f"Humatch pair result is too large: {task_key}")
            decoded[task_key] = AppRunResult(
                status=AppRunStatus.SUCCEEDED,
                outputs=[
                    AppOutput(
                        name="humatch_pair_result",
                        kind=ArtifactKind.REPORT,
                        storage=InlineBytes(
                            data=content,
                            filename="humatch-pair.json",
                            media_type="application/json",
                        ),
                        metadata={"pair_id": task_key},
                    )
                ],
                metrics={"pair_count": 1},
            )
        return decoded

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


def _humatch_result_path(execution_run_id: UUID, run_name: str) -> PurePosixPath:
    return (
        PurePosixPath("workflow-runs")
        / str(execution_run_id)
        / "humatch"
        / f"{sanitize_filename(run_name)}_humatch.tar.zst"
    )


def _read_pair_result(
    context: NodeRunContext, artifact: ExecutionArtifact
) -> Mapping[str, Any]:
    content = context.resolve_artifact(artifact).read_bytes()
    if not 0 < len(content) <= MAX_PAIR_RESULT_BYTES:
        raise ValueError("Humatch pair publication has an invalid size")
    value = orjson.loads(content)
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("Humatch pair publication schema is invalid")
    return value


@dataclass
class _CollectHumatchResultsNode(CoordinatorNode):
    request: HumatchExecutionRequest

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Assemble ordered pair results without starting another container."""
        if context.volume_root is None or context.artifact_volume_name is None:
            raise RuntimeError("Humatch collection requires execution storage")
        publications = [
            _read_pair_result(context, artifact)
            for artifact in context.inputs.get("pair-results", [])
        ]
        pair_results: dict[str, Mapping[str, Any]] = {}
        batch_metrics: dict[tuple[str, ...], Mapping[str, Any]] = {}
        for publication in publications:
            pair_result = publication.get("pair_result")
            raw_batch_keys = publication.get("batch_task_keys")
            metrics = publication.get("batch_metrics")
            if (
                not isinstance(pair_result, Mapping)
                or not isinstance(raw_batch_keys, list)
                or not all(isinstance(key, str) for key in raw_batch_keys)
                or not isinstance(metrics, Mapping)
            ):
                raise ValueError("Humatch pair publication is incomplete")
            humanized = pair_result.get("humanized")
            if not isinstance(humanized, Mapping) or not isinstance(
                humanized.get("id"), str
            ):
                raise ValueError("Humatch pair publication has no pair ID")
            pair_id = cast(str, humanized["id"])
            if pair_id in pair_results:
                raise ValueError(f"Duplicate Humatch pair publication: {pair_id}")
            pair_results[pair_id] = pair_result
            batch_metrics[tuple(cast(list[str], raw_batch_keys))] = metrics

        records = _pair_records(self.request.csv_bytes)
        ordered_ids = tuple(record["id"] for record in records)
        if set(pair_results) != set(ordered_ids):
            raise ValueError("Humatch pair publications do not match the input")
        from biomodals.app.design.humatch.app import _aggregate_humatch_results

        archive, metrics = _aggregate_humatch_results(
            run_name=self.request.run_name,
            csv_bytes=self.request.csv_bytes,
            pair_results=[pair_results[pair_id] for pair_id in ordered_ids],
            batch_metrics=list(batch_metrics.values()),
            parameters=_worker_kwargs(self.request),
        )
        relative_path = _humatch_result_path(
            context.execution_run_id, self.request.run_name
        )
        output_path = context.volume_root.joinpath(*relative_path.parts)
        replace_bytes_atomic(output_path, archive)
        digest = sha256(archive).hexdigest()
        result = AppRunResult(
            status=AppRunStatus.SUCCEEDED,
            outputs=[
                AppOutput(
                    name="humatch_humanization",
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
                        "classifier_endpoints": ["input", "final"],
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


def humatch_execution_graph(request: HumatchExecutionRequest) -> ExecutionGraph:
    """Build per-pair fixed batches followed by local archive collection."""
    graph = ExecutionGraph(
        "humatch",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="humatch",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    humanize = graph.add_node(
        _HumatchHumanizeNode(request),
        id=HUMANIZE_NODE,
        reuse_predecessor_publication=False,
    )
    graph.add_node(
        _CollectHumatchResultsNode(request),
        id=COLLECT_NODE,
        inputs={"pair-results": humanize.outputs(kind=ArtifactKind.REPORT)},
        reuse_predecessor_publication=False,
    )
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
    """Load and content-verify the terminal Humatch archive."""
    if overview.run.status != RunStatus.SUCCEEDED:
        raise LookupError("Humatch humanization result is unavailable")
    result = AppRunResult.model_validate_json(
        _RESULT_FILE.load_from_volume(
            output_volume,
            overview.run.execution_run_id,
        )
    )
    outputs: list[AppOutput] = []
    for output in result.outputs:
        if not isinstance(output.storage, VolumePath):
            outputs.append(output)
            continue
        files = output.metadata.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("Humatch archive manifest is invalid")
        file = files[0]
        if not isinstance(file, Mapping):
            raise ValueError("Humatch archive manifest is invalid")
        data = read_volume_file_exact(
            output_volume,
            output.storage.path,
            size_bytes=file.get("size_bytes"),
            content_sha256=file.get("content_sha256"),
        )
        if len(data) > MAX_RESULT_BYTES:
            raise ValueError("Humatch result archive exceeds the byte limit")
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
