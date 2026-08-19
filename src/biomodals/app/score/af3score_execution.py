"""Direct AF3Score adaptation of the shared execution kernel."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID

import orjson

from biomodals.app.score.af3score_publications import (
    COMPLETION_SAMPLE_SUBDIR,
    METRICS_FILENAME,
    _input_publication_ready,
    _input_publication_record,
    _metrics_publication_path,
    _metrics_publication_ready,
)
from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionArtifact,
    ExecutionGraph,
    ExecutionPlan,
    ExecutionPlanMetadata,
    NodeAggregationPolicy,
    NodeDependency,
    NodePlan,
    inline_json_result,
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
from biomodals.helper.app_run import AppRunLayout
from biomodals.helper.artifacts import replace_bytes_atomic, sha256_file
from biomodals.helper.io import require_safe_filename_component
from biomodals.helper.output_claim import acquire_output_claim
from biomodals.helper.task_budget import bounded_map
from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactFile,
    ArtifactKind,
    VolumePath,
)

REQUEST_SCHEMA_VERSION = 4
MAX_REQUEST_BYTES = 4 * 1024 * 1024
AF3SCORE_MODEL_IDENTITY = "AlphaFold3/af3.bin:v1"
PREPARE_NODE = "prepare"
BATCHES_NODE = "score-batches"
POSTPROCESS_NODE = "postprocess"
_CACHE_VALIDATION_WORKERS = 8
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "AF3Score execution request",
)
_STAGED_INPUT_ROOT = PurePosixPath(".biomodals/af3score/staged-inputs")


@dataclass(frozen=True)
class ChunkSpec:
    """One prepared AF3Score GPU batch."""

    batch_name: str
    batch_json_dir: str
    batch_pdb_dir: str


@dataclass(frozen=True)
class TaskSpec:
    """Preparation result used to discover the finite GPU Task set."""

    total: int
    pending: int
    skipped: int
    input_files: list[str]
    chunk_specs: list[ChunkSpec]
    output_dir: str
    failed_dir: str


@dataclass(frozen=True)
class AF3ScoreExecutionRequest:
    """Immutable staged inputs plus operational fan-out settings."""

    run_name: str
    inputs: tuple[tuple[str, str], ...]
    staged_input_key: str
    prepare_workers: int
    app_version: str
    model_identity: str = AF3SCORE_MODEL_IDENTITY
    max_active_provider_calls: int | None = None
    max_active_gpu_provider_calls: int | None = None
    replace_claim_owner: str | None = None

    def __post_init__(self) -> None:
        """Reject unsafe paths, duplicate inputs, and unusable limits."""
        require_safe_filename_component(self.run_name, field_name="run_name")
        if not self.inputs:
            raise ValueError("AF3Score inputs cannot be empty")
        names = tuple(name for name, _digest in self.inputs)
        if len(names) != len(set(names)):
            raise ValueError("AF3Score input names must be unique")
        for name, digest in self.inputs:
            if (
                Path(name).name != name
                or not name.endswith(".pdb")
                or Path(name).stem in {".", ".."}
            ):
                raise ValueError("AF3Score input names must be PDB filenames")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("AF3Score input digests must be lowercase SHA-256")
        if self.staged_input_key != af3score_staged_input_key(self.inputs):
            raise ValueError("AF3Score staged input key does not match its inputs")
        if self.prepare_workers < 1:
            raise ValueError("AF3Score preparation workers must be positive")
        total_limit = self.max_active_provider_calls
        if total_limit is None:
            total_limit = 10
            object.__setattr__(self, "max_active_provider_calls", total_limit)
        gpu_limit = self.max_active_gpu_provider_calls
        if gpu_limit is None:
            gpu_limit = min(10, total_limit)
            object.__setattr__(
                self,
                "max_active_gpu_provider_calls",
                gpu_limit,
            )
        if total_limit < 1 or gpu_limit < 1 or gpu_limit > total_limit:
            raise ValueError("AF3Score provider-call limits are invalid")
        if not self.app_version or not self.model_identity:
            raise ValueError("AF3Score scientific versions cannot be empty")

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return staged input names in deterministic encounter order."""
        return tuple(name for name, _digest in self.inputs)

    @property
    def input_digests(self) -> dict[str, str]:
        """Return input digests keyed by the output directory identifier."""
        return {Path(name).stem: digest for name, digest in self.inputs}

    @property
    def execution_plan(self) -> ExecutionPlan:
        """Build preparation, GPU fan-out, and postprocessing Nodes."""
        return ExecutionPlan(
            workload_name="af3score",
            workload_run_key=self.run_name,
            nodes=(
                NodePlan(PREPARE_NODE),
                NodePlan(
                    BATCHES_NODE,
                    dependencies=(NodeDependency(PREPARE_NODE),),
                    aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
                ),
                NodePlan(
                    POSTPROCESS_NODE,
                    dependencies=(NodeDependency(BATCHES_NODE, accept_partial=True),),
                    aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
                ),
            ),
            scientific_payload={
                "inputs": [
                    {"name": name, "sha256": digest} for name, digest in self.inputs
                ],
            },
            scientific_versions={
                "af3score": self.app_version,
                "alphafold3.model": self.model_identity,
                "biomodals.af3score.execution_request": str(REQUEST_SCHEMA_VERSION),
            },
        )

    def to_bytes(self) -> bytes:
        """Encode the bounded request without Python pickles."""
        content = orjson.dumps(
            {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "run_name": self.run_name,
                "inputs": [list(item) for item in self.inputs],
                "staged_input_key": self.staged_input_key,
                "prepare_workers": self.prepare_workers,
                "app_version": self.app_version,
                "model_identity": self.model_identity,
                "max_active_provider_calls": self.max_active_provider_calls,
                "max_active_gpu_provider_calls": (self.max_active_gpu_provider_calls),
                "replace_claim_owner": self.replace_claim_owner,
            },
            option=orjson.OPT_SORT_KEYS,
        )
        if len(content) > MAX_REQUEST_BYTES:
            raise ValueError("AF3Score execution request exceeds its byte limit")
        return content

    @classmethod
    def from_bytes(cls, content: bytes) -> AF3ScoreExecutionRequest:
        """Decode and revalidate a staged request."""
        if not 0 < len(content) <= MAX_REQUEST_BYTES:
            raise ValueError("AF3Score execution request has an invalid size")
        value: Any = orjson.loads(content)
        if (
            not isinstance(value, dict)
            or value.pop("schema_version", None) != REQUEST_SCHEMA_VERSION
        ):
            raise ValueError("AF3Score execution request schema is unsupported")
        inputs = value.pop("inputs", None)
        if not isinstance(inputs, list):
            raise TypeError("AF3Score inputs must be a list")
        value["inputs"] = tuple(tuple(item) for item in inputs)
        return cls(**value)


def stage_execution_request(
    output_volume: Any,
    execution_run_id: UUID,
    request: AF3ScoreExecutionRequest,
) -> PurePosixPath:
    """Idempotently stage a request before coordinator launch."""
    return _REQUEST_FILE.stage(output_volume, execution_run_id, request.to_bytes())


def af3score_staged_input_key(inputs: tuple[tuple[str, str], ...]) -> str:
    """Return the content address for one immutable AF3Score input set."""
    return sha256(
        orjson.dumps(
            sorted(inputs),
            option=orjson.OPT_SORT_KEYS,
        )
    ).hexdigest()


def af3score_staged_input_directory(staged_input_key: str) -> PurePosixPath:
    """Return the app-owned Volume directory for a staged input set."""
    if len(staged_input_key) != 64 or any(
        character not in "0123456789abcdef" for character in staged_input_key
    ):
        raise ValueError("AF3Score staged input key must be a lowercase SHA-256")
    return _STAGED_INPUT_ROOT / staged_input_key


def materialize_af3score_staged_inputs(
    volume_root: str | Path,
    staged_input_key: str,
    inputs: tuple[tuple[str, bytes, str], ...],
) -> Path:
    """Materialize immutable staged inputs from an already-mounted Volume."""
    directory = Path(volume_root).joinpath(
        *af3score_staged_input_directory(staged_input_key).parts
    )
    directory.mkdir(parents=True, exist_ok=True)
    for name, content, expected_digest in inputs:
        require_safe_filename_component(name, field_name="AF3Score input name")
        if sha256(content).hexdigest() != expected_digest:
            raise ValueError(f"AF3Score staged input digest is invalid: {name}")
        target = directory / name
        if target.is_file() and sha256_file(target) == expected_digest:
            continue
        replace_bytes_atomic(target, content)
    return directory


def stage_execution_inputs(
    output_volume: Any,
    staged_input_key: str,
    input_files: tuple[Path, ...],
) -> PurePosixPath:
    """Upload one immutable content-addressed AF3Score input set."""
    directory = af3score_staged_input_directory(staged_input_key)
    with output_volume.batch_upload(force=True) as batch:
        for path in input_files:
            batch.put_file(path, f"/{directory}/{path.name}")
    return directory


def persist_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
    request: AF3ScoreExecutionRequest,
) -> PurePosixPath:
    """Persist a coordinator-generated successor request."""
    return _REQUEST_FILE.persist(volume_root, execution_run_id, request.to_bytes())


def load_execution_request(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> AF3ScoreExecutionRequest:
    """Load one request inside the mounted coordinator."""
    return AF3ScoreExecutionRequest.from_bytes(
        _REQUEST_FILE.load(volume_root, execution_run_id)
    )


class AF3ScorePublications:
    """Own AF3Score cache validation, claims, and artifact reconstruction."""

    def __init__(
        self,
        *,
        request: AF3ScoreExecutionRequest,
        execution_run_id: UUID,
        output_claims: Any,
        output_root: str | Path,
        output_volume_name: str,
    ) -> None:
        """Bind one Run to its existing app-owned output directory."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.output_claims = output_claims
        self.output_root = Path(output_root)
        self.output_volume_name = output_volume_name

    @property
    def layout(self) -> AppRunLayout:
        """Return the established app-owned run layout."""
        return AppRunLayout.from_run_root(self.output_root / self.request.run_name)

    @property
    def publication_key(self) -> str:
        """Return the scientific publication identity for this request."""
        return self.request.execution_plan.workload_plan_fingerprint

    def output_complete(self, input_id: str) -> bool:
        """Validate one scored input publication."""
        digest = self.request.input_digests.get(input_id)
        if digest is None:
            return False
        return _input_publication_ready(
            self.layout.outputs_dir,
            input_id,
            publication_key=self.publication_key,
            input_sha256=digest,
        )

    def outputs_complete(self) -> bool:
        """Validate all requested per-input score publications in parallel."""
        return all(
            bounded_map(
                self.request.input_names,
                lambda name: self.output_complete(Path(name).stem),
                max_parallel=_CACHE_VALIDATION_WORKERS,
            )
        )

    def claim(self) -> None:
        """Claim the shared run-name output namespace before provider writes."""
        acquire_output_claim(
            self.output_claims,
            claim_key=(
                "af3score-output:" + sha256(self.request.run_name.encode()).hexdigest()
            ),
            owner=str(self.execution_run_id),
            replace_owner=self.request.replace_claim_owner,
        )

    @staticmethod
    def preparation_result(spec: TaskSpec) -> AppRunResult:
        """Publish bounded preparation metadata through execution storage."""
        return inline_json_result(
            name="task-spec",
            value=asdict(spec),
            filename="task-spec.json",
        )

    def result(self, input_ids: tuple[str, ...], *, metrics: bool) -> AppRunResult:
        """Publish exact per-input outputs and optional aggregate metrics."""
        outputs: list[AppOutput] = []
        for input_id in input_ids:
            digest = self.request.input_digests[input_id]
            marker = _input_publication_record(
                self.layout.outputs_dir,
                input_id,
                publication_key=self.publication_key,
                input_sha256=digest,
            )
            if marker is None or not self.output_complete(input_id):
                raise FileNotFoundError(f"AF3Score output is unavailable: {input_id}")
            records = cast(dict[str, dict[str, object]], marker["outputs"])
            sample = self.layout.outputs_dir / input_id / COMPLETION_SAMPLE_SUBDIR
            outputs.append(
                AppOutput(
                    name=f"score-{input_id}",
                    kind=ArtifactKind.DIRECTORY,
                    storage=VolumePath(
                        volume_name=self.output_volume_name,
                        path=sample.relative_to(self.output_root).as_posix(),
                    ),
                    metadata={
                        "input_id": input_id,
                        "files": [
                            ArtifactFile(
                                path=name,
                                size_bytes=cast(int, record["size"]),
                                content_sha256=cast(str, record["sha256"]),
                            ).model_dump(mode="json")
                            for name, record in sorted(records.items())
                        ],
                    },
                )
            )
        if metrics:
            marker = orjson.loads(
                _metrics_publication_path(self.layout.run_root).read_bytes()
            )
            path = self.layout.run_root / METRICS_FILENAME
            outputs.append(
                AppOutput(
                    name="af3score-metrics",
                    kind=ArtifactKind.TABLE,
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
            )
        return AppRunResult(status=AppRunStatus.SUCCEEDED, outputs=outputs)

    def terminal_result(self) -> AppRunResult | None:
        """Recover a complete terminal metrics publication."""
        if (
            not _metrics_publication_ready(self.layout.run_root, self.publication_key)
            or not self.outputs_complete()
        ):
            return None
        return self.result(tuple(self.request.input_digests), metrics=True)


def _task_spec_from_value(
    value: object, publications: AF3ScorePublications
) -> TaskSpec:
    if not isinstance(value, Mapping):
        raise TypeError("AF3Score preparation payload is invalid")
    value = cast(Mapping[str, object], value)
    chunks = value.get("chunk_specs")
    if not isinstance(chunks, list):
        raise TypeError("AF3Score chunk specifications are invalid")
    total = value.get("total")
    pending = value.get("pending")
    skipped = value.get("skipped")
    input_files = value.get("input_files")
    output_dir = value.get("output_dir")
    failed_dir = value.get("failed_dir")
    if (
        type(total) is not int
        or type(pending) is not int
        or type(skipped) is not int
        or not isinstance(input_files, list)
        or not all(isinstance(item, str) for item in input_files)
        or not isinstance(output_dir, str)
        or not isinstance(failed_dir, str)
    ):
        raise TypeError("AF3Score preparation payload has invalid fields")
    parsed_chunks = []
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            raise TypeError("AF3Score chunk specification is invalid")
        chunk = cast(Mapping[str, object], chunk)
        fields = tuple(
            chunk.get(name)
            for name in ("batch_name", "batch_json_dir", "batch_pdb_dir")
        )
        if not all(isinstance(field, str) for field in fields):
            raise TypeError("AF3Score chunk specification is invalid")
        parsed_chunks.append(ChunkSpec(*cast(tuple[str, str, str], fields)))
    spec = TaskSpec(
        total=total,
        pending=pending,
        skipped=skipped,
        input_files=cast(list[str], input_files),
        chunk_specs=parsed_chunks,
        output_dir=output_dir,
        failed_dir=failed_dir,
    )
    _validate_task_spec(spec, publications)
    return spec


def _validate_task_spec(spec: TaskSpec, publications: AF3ScorePublications) -> None:
    request = publications.request
    if (
        spec.total != len(request.inputs)
        or spec.pending + spec.skipped != spec.total
        or tuple(spec.input_files) != request.input_names
    ):
        raise ValueError("AF3Score preparation counts do not match the request")
    root = publications.layout.run_root.resolve()
    chunk_names: set[str] = set()
    task_ids: list[str] = []
    for chunk in spec.chunk_specs:
        if not chunk.batch_name or chunk.batch_name in chunk_names:
            raise ValueError("AF3Score chunk names must be unique")
        chunk_names.add(chunk.batch_name)
        for path in (Path(chunk.batch_json_dir), Path(chunk.batch_pdb_dir)):
            path.resolve().relative_to(root)
            if not path.is_dir():
                raise OSError(f"AF3Score prepared directory is missing: {path}")
        task_ids.extend(_chunk_input_ids(chunk))
    requested_ids = {Path(name).stem for name in request.input_names}
    if (
        len(task_ids) != spec.pending
        or len(task_ids) != len(set(task_ids))
        or not set(task_ids).issubset(requested_ids)
    ):
        raise ValueError("AF3Score prepared Tasks do not match pending inputs")


def _chunk_input_ids(chunk: ChunkSpec) -> tuple[str, ...]:
    return tuple(
        path.stem
        for path in sorted(Path(chunk.batch_json_dir).glob("*.json"))
        if path.is_file()
    )


def _task_spec_from_context(
    context: NodeRunContext,
    publications: AF3ScorePublications,
) -> TaskSpec:
    return _task_spec_from_value(
        orjson.loads(context.read_input_bytes("task-spec")), publications
    )


@dataclass
class _AF3ScorePrepareNode(ProviderNode):
    request: AF3ScoreExecutionRequest
    publications: AF3ScorePublications

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def prepare_remote(self, context: NodeRunContext) -> ProviderCallSpec:
        del context
        self.publications.claim()
        return ProviderCallSpec(
            function_name="af3score_prepare",
            uses_gpu=False,
            runtime_image_key="af3score-cpu",
            kwargs={
                "run_name": self.request.run_name,
                "staged_input_key": self.request.staged_input_key,
                "input_files": list(self.request.input_names),
                "input_digests": self.request.input_digests,
                "publication_key": self.publications.publication_key,
                "num_jobs": self.request.max_active_gpu_provider_calls,
                "prepare_workers": self.request.prepare_workers,
            },
        )

    def process_remote_result(
        self, result: Any, metadata: Mapping[str, Any]
    ) -> AppRunResult:
        del metadata
        return self.publications.preparation_result(
            _task_spec_from_value(result, self.publications)
        )


@dataclass
class _AF3ScoreBatchNode(TaskProviderNode):
    request: AF3ScoreExecutionRequest
    publications: AF3ScorePublications

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        spec = _task_spec_from_context(context, self.publications)
        tasks = []
        for chunk in spec.chunk_specs:
            input_ids = _chunk_input_ids(chunk)
            for input_id in input_ids:
                tasks.append(
                    TaskDefinition(
                        task_key=input_id,
                        scientific_payload={
                            "input_id": input_id,
                            "sha256": self.request.input_digests[input_id],
                        },
                        execution_payload={
                            "chunk": asdict(chunk),
                            "task_count": len(input_ids),
                        },
                    )
                )
        return tuple(tasks)

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        del context
        payload = cast(Mapping[str, object], task.execution_payload)
        chunk = ChunkSpec(**cast(dict[str, str], payload["chunk"]))
        return self._call_spec(chunk, cast(int, payload["task_count"]))

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[TaskDefinition, ...],
    ) -> ProviderCallSpec:
        del context
        payload = cast(Mapping[str, object], tasks[0].execution_payload)
        chunk = ChunkSpec(**cast(dict[str, str], payload["chunk"]))
        expected = _chunk_input_ids(chunk)
        if tuple(task.task_key for task in tasks) != expected:
            raise ValueError("AF3Score batch Tasks do not match their prepared chunk")
        return self._call_spec(chunk, len(expected))

    def _call_spec(self, chunk: ChunkSpec, task_count: int) -> ProviderCallSpec:
        input_ids = _chunk_input_ids(chunk)
        return ProviderCallSpec(
            function_name="af3score_run",
            uses_gpu=True,
            runtime_image_key="af3score-gpu",
            compatibility_key=chunk.batch_name,
            max_tasks_per_call=task_count,
            kwargs={
                "run_name": self.request.run_name,
                "batch_name": chunk.batch_name,
                "batch_json_dir": chunk.batch_json_dir,
                "batch_pdb_dir": chunk.batch_pdb_dir,
                "input_digests": {
                    input_id: self.request.input_digests[input_id]
                    for input_id in input_ids
                },
                "publication_key": self.publications.publication_key,
            },
        )

    def process_remote_task_batch_result(
        self,
        task_keys: tuple[str, ...],
        result: Any,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, AppRunResult]:
        del metadata
        if result is not None:
            raise ValueError("AF3Score GPU batch returned an unexpected value")
        return {
            task_key: (
                self.publications.result((task_key,), metrics=False)
                if self.publications.output_complete(task_key)
                else AppRunResult(
                    status=AppRunStatus.FAILED,
                    warnings=[f"AF3Score output is unavailable: {task_key}"],
                )
            )
            for task_key in task_keys
        }

    def process_remote_task_result(
        self,
        task_key: str,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        return self.process_remote_task_batch_result((task_key,), result, metadata)[
            task_key
        ]

    def recover_remote_task_result(
        self,
        context: NodeRunContext,
        task: TaskDefinition,
        expected_fingerprint: str,
    ) -> AppRunResult | None:
        del context, expected_fingerprint
        return (
            self.publications.result((task.task_key,), metrics=False)
            if self.publications.output_complete(task.task_key)
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
        del context, expected_fingerprint, result, artifacts
        return (
            AvailabilityStatus.AVAILABLE
            if self.publications.output_complete(task.task_key)
            else AvailabilityStatus.MISSING
        )

    def recover_result_publication(
        self, context: NodeRunContext
    ) -> AppRunResult | None:
        del context
        if not self.publications.outputs_complete():
            return None
        return self.publications.result(
            tuple(self.request.input_digests), metrics=False
        )

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        del context
        status = (
            AppRunStatus.PARTIAL
            if results and errors
            else AppRunStatus.SUCCEEDED
            if not errors
            else AppRunStatus.FAILED
        )
        return AppRunResult(
            status=status,
            outputs=[
                output for result in results.values() for output in result.outputs
            ],
            warnings=list(errors.values()),
        )


@dataclass
class _AF3ScorePostprocessNode(TaskProviderNode):
    request: AF3ScoreExecutionRequest
    publications: AF3ScorePublications

    def refresh_artifact_storage_before_result(self) -> bool:
        return True

    def discover_remote_tasks(
        self, context: NodeRunContext
    ) -> tuple[TaskDefinition, ...]:
        del context
        return tuple(
            TaskDefinition(
                task_key=Path(name).stem,
                scientific_payload={"input_id": Path(name).stem, "sha256": digest},
            )
            for name, digest in self.request.inputs
        )

    def prepare_remote_task(
        self, context: NodeRunContext, task: TaskDefinition
    ) -> ProviderCallSpec:
        del context, task
        return self._call_spec()

    def prepare_remote_task_batch(
        self,
        context: NodeRunContext,
        tasks: tuple[TaskDefinition, ...],
    ) -> ProviderCallSpec:
        del context, tasks
        return self._call_spec()

    def _call_spec(self) -> ProviderCallSpec:
        completed = tuple(
            input_id
            for input_id in self.request.input_digests
            if self.publications.output_complete(input_id)
        )
        return ProviderCallSpec(
            function_name="af3score_postprocess",
            uses_gpu=False,
            runtime_image_key="af3score-cpu",
            max_tasks_per_call=len(self.request.inputs),
            kwargs={
                "run_name": self.request.run_name,
                "staged_input_key": self.request.staged_input_key,
                "input_files": list(self.request.input_names),
                "input_digests": self.request.input_digests,
                "completed_input_ids": list(completed),
                "publication_key": self.publications.publication_key,
            },
        )

    def process_remote_task_batch_result(
        self,
        task_keys: tuple[str, ...],
        result: Any,
        metadata: Mapping[str, Any],
    ) -> Mapping[str, AppRunResult]:
        del metadata
        if not isinstance(result, Mapping) or not _metrics_publication_ready(
            self.publications.layout.run_root, self.publications.publication_key
        ):
            raise FileNotFoundError("AF3Score metrics publication is unavailable")
        return {
            task_key: (
                AppRunResult(status=AppRunStatus.SUCCEEDED)
                if self.publications.output_complete(task_key)
                else AppRunResult(
                    status=AppRunStatus.FAILED,
                    warnings=[f"AF3Score output is unavailable: {task_key}"],
                )
            )
            for task_key in task_keys
        }

    def process_remote_task_result(
        self,
        task_key: str,
        result: Any,
        metadata: Mapping[str, Any],
    ) -> AppRunResult:
        return self.process_remote_task_batch_result((task_key,), result, metadata)[
            task_key
        ]

    def recover_result_publication(
        self, context: NodeRunContext
    ) -> AppRunResult | None:
        del context
        return self.publications.terminal_result()

    def finalize_remote_tasks(
        self,
        context: NodeRunContext,
        results: Mapping[str, AppRunResult],
        errors: Mapping[str, str],
    ) -> AppRunResult:
        del context
        successful = tuple(results)
        result = self.publications.result(successful, metrics=True)
        return result.model_copy(
            update={
                "status": AppRunStatus.PARTIAL if errors else AppRunStatus.SUCCEEDED,
                "warnings": list(errors.values()),
            }
        )


def af3score_execution_graph(
    request: AF3ScoreExecutionRequest,
    publications: AF3ScorePublications,
) -> ExecutionGraph:
    """Build AF3Score's prepared, length-balanced GPU batch graph."""
    graph = ExecutionGraph(
        "af3score",
        plan_metadata=ExecutionPlanMetadata(
            workload_name="af3score",
            scientific_payload=request.execution_plan.scientific_payload,
            scientific_versions=dict(request.execution_plan.scientific_versions),
        ),
    )
    prepare = graph.add_node(
        _AF3ScorePrepareNode(request, publications), id=PREPARE_NODE
    )
    batches = graph.add_node(
        _AF3ScoreBatchNode(request, publications),
        id=BATCHES_NODE,
        inputs={"task-spec": prepare.outputs(kind=ArtifactKind.TABLE)},
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )
    graph.add_node(
        _AF3ScorePostprocessNode(request, publications),
        id=POSTPROCESS_NODE,
        depends_on=[batches],
        accept_partial_from=[batches],
        aggregation_policy=NodeAggregationPolicy.ALLOW_PARTIAL,
    )
    return graph


class AF3ScoreExecutionCoordinator(OutputClaimExecutionDefinitionCoordinatorLifecycle):
    """Bind one run-scoped writer to AF3Score publications."""

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
            target_scientific_versions={
                "af3score": app_version,
                "alphafold3.model": AF3SCORE_MODEL_IDENTITY,
            },
            poll_interval_seconds=poll_interval_seconds,
        )
        self.output_volume_name = output_volume_name
        self.output_claims = output_claims

    def _graph(
        self,
        request: AF3ScoreExecutionRequest,
        predecessor_execution_run_id: UUID | None,
    ) -> ExecutionGraph:
        del predecessor_execution_run_id
        return af3score_execution_graph(
            request=request,
            publications=AF3ScorePublications(
                request=request,
                execution_run_id=self.execution_run_id,
                output_claims=self.output_claims,
                output_root=self.volume_root,
                output_volume_name=self.output_volume_name,
            ),
        )
