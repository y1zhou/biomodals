"""Single-node execution definition for the p-AbNatiV2 app."""

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

REQUEST_SCHEMA_VERSION = 2
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESULT_BYTES = 512 * 1024 * 1024
MAX_PAIR_RESULT_BYTES = 4 * 1024 * 1024
PAIRS_PER_GPU_CALL = 4
MAX_RESULT_DESCRIPTOR_BYTES = 64 * 1024
HUMANIZE_NODE = "humanize"
COLLECT_NODE = "collect"
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "p-AbNatiV2 execution request",
)
_RESULT_FILE = ExecutionRequestFile(
    "result.json",
    MAX_RESULT_DESCRIPTOR_BYTES,
    "p-AbNatiV2 result descriptor",
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
    max_active_gpu_provider_calls: int = 1

    def __post_init__(self) -> None:
        """Reject malformed staged requests before provider dispatch."""
        if not isinstance(self.run_name, str) or not self.run_name:
            raise ValueError("p-AbNatiV2 run name is required")
        if (
            sanitize_filename(self.run_name) != self.run_name
            or len(self.run_name.encode("utf-8")) > 200
        ):
            raise ValueError("p-AbNatiV2 run name must be a safe short filename")
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
            or self.max_active_provider_calls < 1
            or self.max_active_gpu_provider_calls < 1
            or self.max_active_gpu_provider_calls > self.max_active_provider_calls
        ):
            raise ValueError("p-AbNatiV2 requires positive GPU provider-call capacity")

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Describe the initial single-container scientific run."""
        return ExecutionPlan(
            workload_name="pabnativ2",
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


def _pair_records(csv_bytes: bytes) -> tuple[dict[str, str], ...]:
    from biomodals.app.design.pabnativ2.app import (
        _validate_antibody_chains,
        parse_pabnativ2_csv,
    )

    frame = parse_pabnativ2_csv(csv_bytes)
    _validate_antibody_chains(frame)
    return tuple(frame.iter_rows(named=True))


def _worker_kwargs(request: PAbNatiV2ExecutionRequest) -> dict[str, Any]:
    return {
        name: getattr(request, name)
        for name in (
            "mutate_cdrs",
            "fixed_vh_positions",
            "fixed_vl_positions",
            "residue_score_threshold",
            "rasa_threshold",
            "max_relative_pairing_score_decrease",
            "forbidden_residues",
            "seed",
        )
    }


@dataclass
class _PAbNatiV2HumanizeNode(TaskProviderNode):
    request: PAbNatiV2ExecutionRequest

    @cached_property
    def records(self) -> tuple[dict[str, str], ...]:
        """Validate the batch once and retain its stable input order."""
        return _pair_records(self.request.csv_bytes)

    @staticmethod
    def _pair(task: TaskDefinition) -> dict[str, str]:
        payload = task.execution_payload
        if not isinstance(payload, Mapping):
            raise TypeError("p-AbNatiV2 pair execution payload must be an object")
        pair = {name: payload.get(name) for name in ("id", "vh", "vl")}
        if not all(isinstance(value, str) for value in pair.values()):
            raise TypeError("p-AbNatiV2 pair execution payload is invalid")
        typed_pair = cast(dict[str, str], pair)
        if typed_pair["id"] != task.task_key:
            raise ValueError("p-AbNatiV2 Task identity does not match its pair ID")
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
        del context
        pair = self._pair(task)
        if len(self.records) > 1:
            return ProviderCallSpec(
                function_name="pabnativ2_humanize_batch",
                uses_gpu=True,
                runtime_image_key="pabnativ2-a10g-batch",
                compatibility_key="pabnativ2-four-pair-batch",
                max_tasks_per_call=PAIRS_PER_GPU_CALL,
                kwargs={"pairs": [pair], **_worker_kwargs(self.request)},
            )
        return ProviderCallSpec(
            function_name="pabnativ2_humanize_pair",
            uses_gpu=True,
            runtime_image_key="pabnativ2-a10g",
            kwargs={"pair": pair, **_worker_kwargs(self.request)},
        )

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[TaskDefinition, ...],
    ) -> ProviderCallSpec:
        """Send up to four ordered pairs to one multiprocessing worker."""
        if len(self.records) == 1:
            if len(tasks) != 1:
                raise ValueError("Single-pair p-AbNatiV2 input cannot be batched")
            return self.prepare_remote_task(context, tasks[0])
        del context
        if not 1 <= len(tasks) <= PAIRS_PER_GPU_CALL:
            raise ValueError("p-AbNatiV2 GPU batches must contain one to four pairs")
        return ProviderCallSpec(
            function_name="pabnativ2_humanize_batch",
            uses_gpu=True,
            runtime_image_key="pabnativ2-a10g-batch",
            compatibility_key="pabnativ2-four-pair-batch",
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
            raise ValueError("p-AbNatiV2 batch result does not match its pair Tasks")
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
        / "pabnativ2"
        / f"{run_name}_pabnativ2.tar.zst"
    )


def _read_pair_result(
    context: NodeRunContext, artifact: ExecutionArtifact
) -> Mapping[str, Any]:
    content = context.resolve_artifact(artifact).read_bytes()
    if not 0 < len(content) <= MAX_PAIR_RESULT_BYTES:
        raise ValueError("p-AbNatiV2 pair publication has an invalid size")
    value = orjson.loads(content)
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("p-AbNatiV2 pair publication schema is invalid")
    pair_result = value.get("pair_result")
    if not isinstance(pair_result, Mapping):
        raise ValueError("p-AbNatiV2 pair publication is incomplete")
    return pair_result


@dataclass
class _CollectPAbNatiV2ResultsNode(CoordinatorNode):
    request: PAbNatiV2ExecutionRequest

    def run(self, context: NodeRunContext) -> AppRunResult:
        """Assemble ordered pair results without starting another container."""
        if context.volume_root is None or context.artifact_volume_name is None:
            raise RuntimeError("p-AbNatiV2 collection requires execution storage")
        pair_results: dict[str, Mapping[str, Any]] = {}
        for artifact in context.inputs.get("pair-results", []):
            result = _read_pair_result(context, artifact)
            humanized = result.get("humanized")
            if not isinstance(humanized, Mapping) or not isinstance(
                humanized.get("id"), str
            ):
                raise ValueError("p-AbNatiV2 pair publication has no pair ID")
            pair_id = cast(str, humanized["id"])
            if pair_id in pair_results:
                raise ValueError(f"Duplicate p-AbNatiV2 pair publication: {pair_id}")
            pair_results[pair_id] = result

        records = _pair_records(self.request.csv_bytes)
        ordered_ids = tuple(record["id"] for record in records)
        if set(pair_results) != set(ordered_ids):
            raise ValueError("p-AbNatiV2 pair publications do not match the input")
        from biomodals.app.design.pabnativ2.app import _aggregate_pabnativ2_results

        archive, metrics = _aggregate_pabnativ2_results(
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
                    name="pabnativ2_humanization",
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
                        "score_endpoints": ["input", "final"],
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


def pabnativ2_execution_graph(
    request: PAbNatiV2ExecutionRequest,
) -> ExecutionGraph:
    """Build fixed GPU batches of durable pairs followed by local collection."""
    graph = ExecutionGraph(
        "pabnativ2",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="pabnativ2",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    humanize = graph.add_node(
        _PAbNatiV2HumanizeNode(request),
        id=HUMANIZE_NODE,
        reuse_predecessor_publication=False,
    )
    graph.add_node(
        _CollectPAbNatiV2ResultsNode(request),
        id=COLLECT_NODE,
        inputs={"pair-results": humanize.outputs(kind=ArtifactKind.REPORT)},
        reuse_predecessor_publication=False,
    )
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
    """Load and content-verify the terminal p-AbNatiV2 archive."""
    if overview.run.status != RunStatus.SUCCEEDED:
        raise LookupError("p-AbNatiV2 humanization result is unavailable")
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
            raise ValueError("p-AbNatiV2 archive manifest is invalid")
        file = files[0]
        if not isinstance(file, Mapping):
            raise ValueError("p-AbNatiV2 archive manifest is invalid")
        data = read_volume_file_exact(
            output_volume,
            output.storage.path,
            size_bytes=file.get("size_bytes"),
            content_sha256=file.get("content_sha256"),
        )
        if len(data) > MAX_RESULT_BYTES:
            raise ValueError("p-AbNatiV2 result archive exceeds the byte limit")
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
