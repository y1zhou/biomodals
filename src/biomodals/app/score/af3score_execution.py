"""Direct AF3Score adaptation of the shared execution kernel."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID

import orjson

from biomodals.app.fold.alphafold3.inference_inputs import DECLARED_MODEL_IDENTITY
from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    ExecutionRuntime,
    ExecutionSnapshot,
    NodeDependency,
    NodePlan,
    ProviderBinding,
    ProviderCallStatus,
    TaskPlan,
)
from biomodals.execution.scheduler import TaskDispatchDescriptor
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
    ExecutionRequestFile,
    ExecutionRunStore,
    ExecutionRuntimeLifecycle,
    ExecutionVolumeSync,
    persist_execution_launch,
)
from biomodals.helper.app_run import AppRunLayout
from biomodals.helper.output_claim import (
    acquire_output_claim,
    register_output_claim_successor,
)
from biomodals.helper.shell import sanitize_filename

REQUEST_SCHEMA_VERSION = 3
MAX_REQUEST_BYTES = 4 * 1024 * 1024
PREPARE_NODE = "prepare"
BATCHES_NODE = "score-batches"
POSTPROCESS_NODE = "postprocess"
METRICS_FILENAME = "af3score_metrics.csv"
COMPLETION_SAMPLE_SUBDIR = "seed-10_sample-0"
COMPLETION_REQUIRED_FILES = (
    "summary_confidences.json",
    "confidences.json",
)
_REQUEST_FILE = ExecutionRequestFile(
    "request.json",
    MAX_REQUEST_BYTES,
    "AF3Score execution request",
)


def _workload_module():
    from biomodals.app.score import af3score_app

    return af3score_app


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
    staged_input_execution_run_id: str
    prepare_workers: int
    max_batches: int
    app_version: str
    model_identity: str = DECLARED_MODEL_IDENTITY
    replace_claim_owner: str | None = None

    def __post_init__(self) -> None:
        """Reject unsafe paths, duplicate inputs, and unusable limits."""
        if not self.run_name or sanitize_filename(self.run_name) != self.run_name:
            raise ValueError("run_name must be a safe filename component")
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
        try:
            staged_run_id = UUID(self.staged_input_execution_run_id)
        except (TypeError, ValueError) as error:
            raise ValueError("AF3Score staged input Run ID must be a UUID") from error
        if str(staged_run_id) != self.staged_input_execution_run_id:
            raise ValueError("AF3Score staged input Run ID must be canonical")
        if self.prepare_workers < 1 or self.max_batches < 1:
            raise ValueError("AF3Score worker limits must be positive")
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
                ),
                NodePlan(
                    POSTPROCESS_NODE,
                    dependencies=(NodeDependency(BATCHES_NODE),),
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
                "staged_input_execution_run_id": self.staged_input_execution_run_id,
                "prepare_workers": self.prepare_workers,
                "max_batches": self.max_batches,
                "app_version": self.app_version,
                "model_identity": self.model_identity,
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


def stage_execution_inputs(
    output_volume: Any,
    execution_run_id: UUID,
    input_files: tuple[Path, ...],
) -> PurePosixPath:
    """Stage root inputs below the immutable Execution Run namespace."""
    directory = _REQUEST_FILE.path(execution_run_id).parent / "inputs"
    with output_volume.batch_upload(force=False) as batch:
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


def load_execution_request_from_volume(
    output_volume: Any,
    execution_run_id: UUID,
) -> AF3ScoreExecutionRequest:
    """Load one request through Modal's Volume API."""
    return AF3ScoreExecutionRequest.from_bytes(
        _REQUEST_FILE.load_from_volume(output_volume, execution_run_id)
    )


class AF3ScoreExecutionRuntime(ExecutionRuntimeLifecycle):
    """Drive one direct AF3Score request through durable fixed batches."""

    def __init__(
        self,
        *,
        request: AF3ScoreExecutionRequest,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: ExecutionRunStore,
        modal_driver: Any,
        output_volume: Any,
        output_claims: Any,
        output_root: str | Path,
        predecessor_execution_run_id: UUID | None = None,
        poll_interval_seconds: float = 1.0,
        now: Callable[[], int] | None = None,
    ) -> None:
        """Bind the kernel writer to AF3Score's existing publications."""
        self.request = request
        self._plan = request.execution_plan
        self._input_digests = request.input_digests
        self._publication_key = self._plan.workload_plan_fingerprint
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.store = store
        self.output_volume = output_volume
        self.output_claims = output_claims
        self.output_root = Path(output_root)
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._claim_acquired = False
        self._volume_sync = ExecutionVolumeSync(volume=output_volume, store=store)
        self._provider = ExecutionRuntime(
            store.execution,
            modal_driver=modal_driver,
            checkpoint=self._checkpoint,
            commit_local=store.commit,
            transaction=store.transaction,
        )

    @property
    def layout(self) -> AppRunLayout:
        """Return the established app-owned run layout."""
        return AppRunLayout.from_run_root(self.output_root / self.request.run_name)

    def advance_once(self) -> None:
        """Apply one publication, recovery, and admission cycle."""
        self._provider.repository = self.store.execution
        self._provider.advance_once(
            self.execution_run_id,
            recover_publications=self._recover_publications,
            reconcile_provider_calls=self._reconcile_provider_calls,
            decode_completed_calls=self._decode_completed_calls,
            start_ready_nodes=self._start_ready_nodes,
            admit_remote_tasks=self._admit_remote_tasks,
            now=self._now,
        )

    def _initialize(self):
        self._reload_output()
        self._provider.create_or_verify_run(
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=self.predecessor_execution_run_id,
            plan=self._plan,
            deployment=self.deployment,
            max_active_provider_calls=self.request.max_batches,
            max_active_gpu_provider_calls=self.request.max_batches,
            now=self._now(),
        )
        return self.store.execution

    def _recover_publications(self) -> None:
        self._provider.repository = self.store.execution
        self._provider.recover_publications(
            self.execution_run_id,
            observe_node=self._node_observation,
            observe_task=lambda node_key, task: (
                None
                if node_key == PREPARE_NODE
                else self._task_observation(node_key, task.task_key)
            ),
            now=self._now(),
        )

    def _node_observation(self, node_key: str) -> AvailabilityStatus:
        try:
            if node_key == PREPARE_NODE:
                available = False
            elif node_key == BATCHES_NODE:
                available = all(
                    self._output_complete(Path(name).stem)
                    for name in self.request.input_names
                )
            elif node_key == POSTPROCESS_NODE:
                available = _workload_module()._metrics_publication_ready(
                    self.layout.run_root,
                    self._publication_key,
                )
            else:
                raise ValueError(f"Unknown AF3Score Node {node_key!r}")
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def _task_observation(
        self,
        node_key: str,
        task_key: str,
    ) -> AvailabilityStatus:
        if node_key == BATCHES_NODE:
            try:
                available = self._output_complete(task_key)
            except OSError:
                return AvailabilityStatus.UNKNOWN
            return (
                AvailabilityStatus.AVAILABLE
                if available
                else AvailabilityStatus.MISSING
            )
        return self._node_observation(node_key)

    def _output_complete(self, input_id: str) -> bool:
        digest = self._input_digests.get(input_id)
        if digest is None:
            return False
        return _workload_module()._input_publication_ready(
            self.layout.outputs_dir,
            input_id,
            publication_key=self._publication_key,
            input_sha256=digest,
        )

    def _reconcile_provider_calls(self, required: set[str]) -> None:
        self._provider.repository = self.store.execution
        reconciled = self._provider.reconcile_provider_calls(
            self.execution_run_id,
            required_node_keys=required,
            encode_result=_result_envelope,
            now=self._now(),
        )
        if any(
            not original.status.is_terminal and updated.status.is_terminal
            for original, updated in reconciled
        ):
            self._reload_output()

    def _decode_completed_calls(self) -> None:
        self._provider.repository = self.store.execution
        self._provider.decode_completed_calls(
            self.execution_run_id,
            observe_task=lambda node_key, task, envelope: (
                self._completed_task_observation(
                    node_key,
                    task.task_key,
                    envelope,
                )
            ),
            missing_message="AF3Score returned without a valid publication",
            now=self._now(),
        )

    def _completed_task_observation(
        self,
        node_key: str,
        task_key: str,
        envelope: object,
    ) -> AvailabilityStatus:
        if not isinstance(envelope, dict):
            return AvailabilityStatus.MISSING
        if node_key == PREPARE_NODE:
            try:
                self._task_spec(envelope)
            except (TypeError, ValueError, OSError):
                return AvailabilityStatus.MISSING
            return AvailabilityStatus.AVAILABLE
        expected_kind = "batch" if node_key == BATCHES_NODE else "postprocess"
        if envelope.get("kind") != expected_kind:
            return AvailabilityStatus.MISSING
        return self._task_observation(node_key, task_key)

    def _start_ready_nodes(self, required: set[str]) -> None:
        self._provider.repository = self.store.execution
        self._provider.start_ready_nodes(
            self.execution_run_id,
            required_node_keys=required,
            task_plans=self._task_plans,
            observe_task=lambda node_key, task: (
                AvailabilityStatus.MISSING
                if node_key == PREPARE_NODE
                else self._task_observation(node_key, task.task_key)
            ),
            now=self._now(),
        )

    def _task_plans(self, node_key: str) -> tuple[TaskPlan, ...]:
        if node_key == PREPARE_NODE:
            return (
                TaskPlan(
                    task_key="prepare",
                    scientific_payload={"inputs": list(self.request.input_names)},
                ),
            )
        if node_key == POSTPROCESS_NODE:
            return (
                TaskPlan(
                    task_key="postprocess",
                    scientific_payload={"inputs": list(self.request.input_names)},
                ),
            )
        if node_key != BATCHES_NODE:
            raise ValueError(f"Unknown AF3Score Node {node_key!r}")
        spec = self._prepare_task_spec()
        digests = {Path(name).stem: digest for name, digest in self.request.inputs}
        plans = []
        for chunk in spec.chunk_specs:
            for input_id in self._chunk_input_ids(chunk):
                plans.append(
                    TaskPlan(
                        task_key=input_id,
                        scientific_payload={
                            "input_id": input_id,
                            "sha256": digests[input_id],
                        },
                        execution_payload={"chunk": asdict(chunk)},
                    )
                )
        return tuple(plans)

    def _prepare_task_spec(self) -> TaskSpec:
        repository = self.store.execution
        for call in repository.list_provider_calls(self.execution_run_id):
            if (
                call.node_key == PREPARE_NODE
                and call.status == ProviderCallStatus.SUCCEEDED
            ):
                return self._task_spec(call.result_envelope)
        raise RuntimeError("AF3Score preparation result is unavailable")

    def _task_spec(self, envelope: object) -> TaskSpec:
        if not isinstance(envelope, dict) or envelope.get("kind") != "prepare":
            raise TypeError("AF3Score preparation envelope is invalid")
        value = envelope.get("task_spec")
        if not isinstance(value, dict):
            raise TypeError("AF3Score preparation payload is invalid")
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
            if not isinstance(chunk, dict):
                raise TypeError("AF3Score chunk specification is invalid")
            batch_name = chunk.get("batch_name")
            batch_json_dir = chunk.get("batch_json_dir")
            batch_pdb_dir = chunk.get("batch_pdb_dir")
            if (
                not isinstance(batch_name, str)
                or not isinstance(batch_json_dir, str)
                or not isinstance(batch_pdb_dir, str)
            ):
                raise TypeError("AF3Score chunk specification is invalid")
            parsed_chunks.append(
                ChunkSpec(
                    batch_name=batch_name,
                    batch_json_dir=batch_json_dir,
                    batch_pdb_dir=batch_pdb_dir,
                )
            )
        spec = TaskSpec(
            total=total,
            pending=pending,
            skipped=skipped,
            input_files=cast(list[str], input_files),
            chunk_specs=parsed_chunks,
            output_dir=output_dir,
            failed_dir=failed_dir,
        )
        self._validate_task_spec(spec)
        return spec

    def _validate_task_spec(self, spec: TaskSpec) -> None:
        if (
            spec.total != len(self.request.inputs)
            or spec.pending + spec.skipped != spec.total
            or tuple(spec.input_files) != self.request.input_names
        ):
            raise ValueError("AF3Score preparation counts do not match the request")
        root = self.layout.run_root.resolve()
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
            task_ids.extend(self._chunk_input_ids(chunk))
        requested_ids = {Path(name).stem for name in self.request.input_names}
        if (
            len(task_ids) != spec.pending
            or len(task_ids) != len(set(task_ids))
            or not set(task_ids).issubset(requested_ids)
        ):
            raise ValueError("AF3Score prepared Tasks do not match pending inputs")

    def _chunk_input_ids(self, chunk: ChunkSpec) -> tuple[str, ...]:
        return tuple(
            path.stem
            for path in sorted(Path(chunk.batch_json_dir).glob("*.json"))
            if path.is_file()
        )

    def _admit_remote_tasks(self, required: set[str]) -> None:
        repository = self.store.execution
        run = repository.get_run(self.execution_run_id)
        chunk_sizes: dict[str, int] = {}
        for task in repository.list_tasks(self.execution_run_id, BATCHES_NODE):
            name = task.execution_payload["chunk"]["batch_name"]
            chunk_sizes[name] = chunk_sizes.get(name, 0) + 1

        def describe_task(node, task, rank):
            binding = self._binding(node.node_key)
            compatibility = binding.function_name
            batch_size = 1
            if node.node_key == BATCHES_NODE:
                compatibility = task.execution_payload["chunk"]["batch_name"]
                batch_size = chunk_sizes[compatibility]
            return TaskDispatchDescriptor(
                node_key=node.node_key,
                node_ordinal=node.ordinal,
                task_key=task.task_key,
                task_ordinal=task.ordinal,
                binding=binding,
                compatibility_key=compatibility,
                max_tasks_per_call=batch_size,
                depth=rank.depth,
                unblocking_span=rank.unblocking_span,
            )

        self._provider.repository = repository
        counts = repository.active_provider_call_counts(self.execution_run_id)
        selected = self._provider.fixed_call_candidates(
            self.execution_run_id,
            required_node_keys=required,
            describe_task=describe_task,
            available_total_slots=max(0, run.max_active_provider_calls - counts.total),
            available_gpu_slots=max(0, run.max_active_gpu_provider_calls - counts.gpu),
            now=self._now(),
        )
        if selected:
            self._ensure_output_claim()
        for candidate in selected:
            self._provider.repository = self.store.execution
            submitted = self._provider.submit_fixed_batch(
                self.execution_run_id,
                candidate,
                submission_token=candidate.candidate_key,
                kwargs=self._invocation_kwargs(
                    candidate.node_key,
                    candidate.task_keys[0],
                ),
                now=self._now(),
            )
            if submitted is None:
                return

    def _binding(self, node_key: str) -> ProviderBinding:
        function_name = {
            PREPARE_NODE: "af3score_prepare",
            BATCHES_NODE: "af3score_run",
            POSTPROCESS_NODE: "af3score_postprocess",
        }[node_key]
        uses_gpu = node_key == BATCHES_NODE
        return ProviderBinding(
            environment=self.deployment.environment,
            app_name=self.deployment.deployment_name,
            app_version=self.deployment.deployment_version,
            function_name=function_name,
            uses_gpu=uses_gpu,
            runtime_image_key="af3score-gpu" if uses_gpu else "af3score-cpu",
        )

    def _invocation_kwargs(
        self,
        node_key: str,
        task_key: str,
    ) -> dict[str, object]:
        if node_key == PREPARE_NODE:
            return {
                "run_name": self.request.run_name,
                "input_files": list(self.request.input_names),
                "input_digests": self._input_digests,
                "publication_key": self._publication_key,
                "num_jobs": self.request.max_batches,
                "prepare_workers": self.request.prepare_workers,
            }
        if node_key == POSTPROCESS_NODE:
            return {
                "run_name": self.request.run_name,
                "input_files": list(self.request.input_names),
                "input_digests": self._input_digests,
                "publication_key": self._publication_key,
            }
        task = self.store.execution.get_task(
            self.execution_run_id,
            BATCHES_NODE,
            task_key,
        )
        chunk = task.execution_payload["chunk"]
        input_ids = self._chunk_input_ids(ChunkSpec(**chunk))
        return {
            "run_name": self.request.run_name,
            "batch_name": chunk["batch_name"],
            "batch_json_dir": chunk["batch_json_dir"],
            "batch_pdb_dir": chunk["batch_pdb_dir"],
            "input_digests": {
                input_id: self._input_digests[input_id] for input_id in input_ids
            },
            "publication_key": self._publication_key,
        }

    def _ensure_output_claim(self) -> None:
        if self._claim_acquired:
            return
        acquire_output_claim(
            self.output_claims,
            claim_key=(
                "af3score-output:" + sha256(self.request.run_name.encode()).hexdigest()
            ),
            owner=str(self.execution_run_id),
            replace_owner=self.request.replace_claim_owner,
        )
        self._materialize_inputs()
        self._claim_acquired = True

    def _materialize_inputs(self) -> None:
        """Validate and publish this Run's private inputs after claim ownership."""
        source_dir = (
            self.output_root
            / ".biomodals"
            / "execution"
            / "runs"
            / self.request.staged_input_execution_run_id
            / "inputs"
        )
        self.layout.inputs_dir.mkdir(parents=True, exist_ok=True)
        for name, expected_digest in self.request.inputs:
            source = source_dir / name
            if source.is_symlink() or not source.is_file():
                raise FileNotFoundError(f"Staged AF3Score input is missing: {source}")
            destination = self.layout.inputs_dir / name
            temporary = destination.with_name(
                f".{destination.name}.{self.execution_run_id}.tmp"
            )
            digest = sha256()
            try:
                with source.open("rb") as reader, temporary.open("wb") as writer:
                    while chunk := reader.read(1024 * 1024):
                        digest.update(chunk)
                        writer.write(chunk)
                if digest.hexdigest() != expected_digest:
                    raise ValueError(f"Staged AF3Score input digest changed: {name}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)


def _result_envelope(result: object) -> dict[str, object]:
    """Encode only bounded preparation metadata or completion diagnostics."""
    if isinstance(result, TaskSpec):
        return {"kind": "prepare", "task_spec": asdict(result)}
    if isinstance(result, dict):
        value = orjson.loads(orjson.dumps(result))
        return {"kind": "postprocess", "result": value}
    if result is None:
        return {"kind": "batch"}
    return {"kind": "invalid"}


class AF3ScoreExecutionCoordinator(ExecutionCoordinatorLifecycle):
    """Bind one run-scoped writer to AF3Score publications."""

    _request_loader = staticmethod(load_execution_request)

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        output_claims: Any,
        modal_driver: Any,
        app_version: str,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only the deployment resources used by this adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            target_scientific_versions={
                "af3score": app_version,
                "alphafold3.model": DECLARED_MODEL_IDENTITY,
            },
        )
        self.output_volume = output_volume
        self.output_claims = output_claims
        self.modal_driver = modal_driver
        self.poll_interval_seconds = poll_interval_seconds

    def restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
        expected_workload_plan_fingerprint: str | None = None,
        candidate_request: AF3ScoreExecutionRequest | None = None,
    ) -> ExecutionSnapshot:
        """Create and drive a compatible Successor from conclusive state."""
        self.prepare_restart(
            predecessor_execution_run_id=predecessor_execution_run_id,
            predecessor_deployment=predecessor_deployment,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
            expected_workload_plan_fingerprint=expected_workload_plan_fingerprint,
            candidate_request=candidate_request,
        )
        return self.drive_prepared()

    def prepare_restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
        expected_workload_plan_fingerprint: str | None = None,
        candidate_request: AF3ScoreExecutionRequest | None = None,
    ) -> None:
        """Validate and persist a Successor request without driving it."""
        if candidate_request is not None and (
            max_active_provider_calls is not None
            or max_active_gpu_provider_calls is not None
        ):
            raise ValueError(
                "Candidate request and generic restart overrides are mutually exclusive"
            )
        with self._drive_lock:
            with self._writer_lock:
                self.output_volume.reload()
                with self._open_successor_source(
                    predecessor_execution_run_id,
                    predecessor_deployment=predecessor_deployment,
                    expected_workload_plan_fingerprint=(
                        expected_workload_plan_fingerprint
                    ),
                ) as (predecessor, predecessor_request, _):
                    request = candidate_request or predecessor_request
                if (
                    request.execution_plan.workload_plan_fingerprint
                    != predecessor.plan.workload_plan_fingerprint
                ):
                    raise ValueError(
                        "Restart arguments changed the Workload Plan Fingerprint"
                    )
                if candidate_request is None:
                    limit = max_active_gpu_provider_calls
                    if limit is None:
                        limit = max_active_provider_calls
                    if limit is None:
                        limit = predecessor.max_active_gpu_provider_calls
                    request = replace(request, max_batches=limit)
                register_output_claim_successor(
                    self.output_claims,
                    owner=str(self.execution_run_id),
                    predecessor=str(predecessor_execution_run_id),
                )
                request = replace(
                    request,
                    replace_claim_owner=str(predecessor_execution_run_id),
                )
                persist_execution_request(
                    self.volume_root,
                    self.execution_run_id,
                    request,
                )
                persist_execution_launch(
                    self.volume_root,
                    self.execution_run_id,
                    predecessor_execution_run_id,
                )
                self.output_volume.commit()

    def _open_runtime(
        self,
        request: AF3ScoreExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> AF3ScoreExecutionRuntime:
        runtime = self._runtime
        if runtime is not None:
            if (
                runtime.request != request
                or runtime.predecessor_execution_run_id != predecessor_execution_run_id
            ):
                raise ValueError("Active AF3Score runtime does not match request")
            return runtime
        runtime = AF3ScoreExecutionRuntime(
            request=request,
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=predecessor_execution_run_id,
            deployment=self.deployment,
            store=self._run_store(),
            modal_driver=self.modal_driver,
            output_volume=self.output_volume,
            output_claims=self.output_claims,
            output_root=self.volume_root,
            poll_interval_seconds=self.poll_interval_seconds,
        )
        self._runtime = runtime
        return runtime
