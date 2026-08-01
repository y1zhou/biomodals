"""Direct ENsiRNA adaptation of the shared execution kernel."""

from __future__ import annotations

import time
from base64 import b64decode, b64encode
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

import orjson

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    ExecutionRunNotFoundError,
    ExecutionRuntime,
    ExecutionSnapshot,
    NodeDependency,
    NodePlan,
    NodeStatus,
    ProviderBinding,
    ProviderCallStatus,
    RunStatus,
    TaskPlan,
    TaskStatus,
    drive_execution_run,
    ready_node_keys,
    result_probe_frontier,
    resume_execution_run,
)
from biomodals.execution.scheduler import (
    TaskDispatchDescriptor,
    form_fixed_batches,
    required_node_ranks,
    select_admissible_candidates,
)
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
    ExecutionRequestFile,
    ExecutionRunStore,
    ExecutionVolumeSync,
)
from biomodals.helper.output_claim import acquire_output_claim

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


def load_execution_request_from_volume(
    output_volume: Any,
    execution_run_id: UUID,
) -> EnsirnaExecutionRequest:
    """Load one request through Modal's Volume API."""
    return EnsirnaExecutionRequest.from_bytes(
        _REQUEST_FILE.load_from_volume(output_volume, execution_run_id)
    )


class EnsirnaExecutionRuntime:
    """Drive one direct ENsiRNA request through its staged DAG."""

    def __init__(
        self,
        *,
        request: EnsirnaExecutionRequest,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: ExecutionRunStore,
        modal_driver: Any,
        output_volume: Any,
        output_claims: Any,
        predecessor_execution_run_id: UUID | None = None,
        poll_interval_seconds: float = 1.0,
        now: Callable[[], int] | None = None,
    ) -> None:
        """Bind the kernel writer to ENsiRNA's content-addressed cache."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.store = store
        self.output_volume = output_volume
        self.output_claims = output_claims
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._claimed_publications: set[str] = set()
        self._volume_sync = ExecutionVolumeSync(volume=output_volume, store=store)
        self._provider = ExecutionRuntime(
            store.execution,
            modal_driver=modal_driver,
            checkpoint=self._checkpoint,
            commit_local=store.commit,
        )

    @property
    def cache_key(self) -> str:
        """Return the established content-addressed publication key."""
        app = _workload_module()
        return app._cache_key_for_fasta(
            self.request.fasta_content,
            force_generation=self.request.force_generation,
        )

    @property
    def layout(self):
        """Return the established cache layout."""
        return _workload_module()._layout_for_cache_key(self.cache_key)

    def run(
        self,
        *,
        synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> ExecutionSnapshot:
        """Create or recover the Run and drive it until it stops."""
        with synchronize():
            repository = self._initialize()
        return drive_execution_run(
            repository,
            self.execution_run_id,
            advance_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=synchronize,
        )

    def resume(
        self,
        *,
        synchronize: Callable[[], AbstractContextManager[object]] = nullcontext,
    ) -> ExecutionSnapshot:
        """Resume this Run without retrying conclusive failures."""
        with synchronize():
            repository = self._initialize()
            resume_execution_run(
                repository,
                self.execution_run_id,
                reconcile_once=self.advance_once,
                checkpoint=self._checkpoint,
                now=self._now(),
            )
        return drive_execution_run(
            self.store.execution,
            self.execution_run_id,
            advance_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=synchronize,
        )

    def cancel(self) -> ExecutionSnapshot:
        """Request cancellation while retaining uncertain call ownership."""
        self._provider.repository = self.store.execution
        self._provider.cancel_run(self.execution_run_id, now=self._now())
        return self.store.execution.snapshot(self.execution_run_id)

    def close(self) -> None:
        """Close SQLite without cancelling attached Provider Calls."""
        self.store.close()

    def advance_once(self) -> None:
        """Apply one publication, recovery, and admission cycle."""
        self._recover_publications()
        self._reconcile_nodes_and_run()
        run = self.store.execution.get_run(self.execution_run_id)
        if run.status == RunStatus.CANCEL_REQUESTED:
            self._reconcile_provider_calls(set(run.plan.node_keys))
            self._decode_completed_calls()
            self._recover_publications()
            self._reconcile_nodes_and_run()
            return
        if run.status == RunStatus.STATE_UNKNOWN:
            required = self._required_nodes()
            required_nodes = set(run.plan.node_keys if required is None else required)
            if required is not None:
                self._cancel_pruned_calls(self._prune_unrequired(required))
            self._reconcile_provider_calls(required_nodes)
            self._decode_completed_calls()
            self._recover_publications()
            self._reconcile_nodes_and_run()
            return
        if run.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return
        required = self._required_nodes()
        if required is None:
            return
        self._cancel_pruned_calls(self._prune_unrequired(required))
        self._reconcile_provider_calls(set(required))
        self._decode_completed_calls()
        self._recover_publications()
        self._reconcile_nodes_and_run()
        if self.store.execution.get_run(self.execution_run_id).status not in {
            RunStatus.PENDING,
            RunStatus.RUNNING,
        }:
            return
        self._start_ready_nodes(set(required))
        self._recover_publications()
        required = self._required_nodes()
        if required is not None:
            self._admit_remote_tasks(set(required))
        self._reconcile_nodes_and_run()

    def _initialize(self):
        self._reload_output()
        repository = self.store.execution
        plan = self.request.execution_plan
        try:
            existing = repository.get_run(self.execution_run_id)
        except LookupError:
            with self.store.transaction():
                repository.create_run(
                    execution_run_id=self.execution_run_id,
                    predecessor_execution_run_id=self.predecessor_execution_run_id,
                    plan=plan,
                    deployment=self.deployment,
                    max_active_provider_calls=self.request.prepare_workers,
                    max_active_gpu_provider_calls=1,
                    now=self._now(),
                )
            return repository
        if (
            existing.plan != plan
            or existing.predecessor_execution_run_id
            != self.predecessor_execution_run_id
            or existing.deployment != self.deployment
            or existing.max_active_provider_calls != self.request.prepare_workers
            or existing.max_active_gpu_provider_calls != 1
        ):
            raise ValueError("ENsiRNA request does not match Execution Run")
        return repository

    def _recover_publications(self) -> None:
        repository = self.store.execution
        run = repository.get_run(self.execution_run_id)
        observations: dict[str, AvailabilityStatus | None] = {}
        for node in repository.list_nodes(self.execution_run_id):
            if node.status == NodeStatus.SUCCEEDED:
                observations[node.node_key] = AvailabilityStatus.AVAILABLE
            elif node.status.is_terminal:
                observations[node.node_key] = AvailabilityStatus.MISSING
            else:
                observations[node.node_key] = node.result_observation

        while frontier := result_probe_frontier(run.plan, observations):
            observed = [
                (node_key, self._node_observation(node_key)) for node_key in frontier
            ]
            with self.store.transaction():
                for node_key, observation in observed:
                    repository.record_node_result_observation(
                        self.execution_run_id,
                        node_key,
                        observation,
                        now=self._now(),
                    )
                    observations[node_key] = observation
            if any(
                observation == AvailabilityStatus.UNKNOWN for _, observation in observed
            ):
                return

        required = self._required_nodes()
        if required is None:
            return
        task_observations = []
        for node in repository.list_nodes(self.execution_run_id):
            if (
                node.node_key in required
                and node.status == NodeStatus.RUNNING
                and node.discovery_complete
                and node.node_key not in {DOWNLOAD_MODELS_NODE, PREPARE_NODE}
            ):
                for task in repository.list_tasks(
                    self.execution_run_id,
                    node.node_key,
                ):
                    if not task.status.is_terminal:
                        task_observations.append((
                            node.node_key,
                            task.task_key,
                            self._task_observation(node.node_key, task),
                        ))
        if not task_observations:
            return
        with self.store.transaction():
            for node_key, task_key, observation in task_observations:
                repository.record_task_result_observation(
                    self.execution_run_id,
                    node_key,
                    task_key,
                    observation,
                    now=self._now(),
                )

    def _node_observation(self, node_key: str) -> AvailabilityStatus:
        app = _workload_module()
        try:
            if node_key in {DOWNLOAD_MODELS_NODE, PREPARE_NODE}:
                available = False
            elif node_key == CHUNKS_NODE:
                plan = self._try_plan_from_node(PREPARE_NODE)
                available = plan is not None and all(
                    app._chunk_artifacts_valid(chunk) for chunk in plan.chunks
                )
            elif node_key == FINALIZE_NODE:
                plan = self._try_plan_from_node(FINALIZE_NODE)
                available = plan is not None and (
                    len(app._json_records(Path(plan.json_path))) == plan.candidate_count
                )
            elif node_key == PREPROCESS_NODE:
                available = app._is_prepared(self.layout)
            elif node_key == INFERENCE_NODE:
                available = app._result_ready(self.layout, self.cache_key)
            else:
                raise ValueError(f"Unknown ENsiRNA Node {node_key!r}")
        except OSError:
            return AvailabilityStatus.UNKNOWN
        return AvailabilityStatus.AVAILABLE if available else AvailabilityStatus.MISSING

    def _task_observation(self, node_key: str, task: Any) -> AvailabilityStatus:
        if node_key == CHUNKS_NODE:
            try:
                chunk = EnsirnaPdbChunkSpec(**task.execution_payload["chunk"])
                available = _workload_module()._chunk_artifacts_valid(chunk)
            except OSError:
                return AvailabilityStatus.UNKNOWN
            return (
                AvailabilityStatus.AVAILABLE
                if available
                else AvailabilityStatus.MISSING
            )
        return self._node_observation(node_key)

    def _required_nodes(self) -> tuple[str, ...] | None:
        self._provider.repository = self.store.execution
        return self._provider.required_node_keys(self.execution_run_id)

    def _prune_unrequired(self, required: tuple[str, ...]) -> tuple[UUID, ...]:
        with self.store.transaction():
            calls = self.store.execution.prune_unrequired_nodes(
                self.execution_run_id,
                required_node_keys=set(required),
                now=self._now(),
            )
        if calls:
            self._checkpoint()
        return calls

    def _cancel_pruned_calls(self, provider_call_ids: tuple[UUID, ...]) -> None:
        for provider_call_id in provider_call_ids:
            self._provider.repository = self.store.execution
            self._provider.request_provider_call_cancellation(
                provider_call_id,
                now=self._now(),
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
        repository = self.store.execution
        with self.store.transaction():
            for call in repository.list_provider_calls(self.execution_run_id):
                if call.status != ProviderCallStatus.SUCCEEDED:
                    continue
                for task_key in call.task_keys:
                    task = repository.get_task(
                        self.execution_run_id,
                        call.node_key,
                        task_key,
                    )
                    if task.status != TaskStatus.RUNNING:
                        continue
                    observation = self._completed_task_observation(
                        call.node_key,
                        task,
                        call.result_envelope,
                    )
                    if observation == AvailabilityStatus.MISSING:
                        repository.fail_task(
                            self.execution_run_id,
                            call.node_key,
                            task_key,
                            message="ENsiRNA returned without a valid publication",
                            now=self._now(),
                        )
                    else:
                        repository.record_task_result_observation(
                            self.execution_run_id,
                            call.node_key,
                            task_key,
                            observation,
                            now=self._now(),
                        )

    def _completed_task_observation(
        self,
        node_key: str,
        task: Any,
        envelope: object,
    ) -> AvailabilityStatus:
        if not isinstance(envelope, dict):
            return AvailabilityStatus.MISSING
        if node_key == DOWNLOAD_MODELS_NODE:
            return (
                AvailabilityStatus.AVAILABLE
                if envelope.get("kind") == "none"
                else AvailabilityStatus.MISSING
            )
        if node_key in {PREPARE_NODE, FINALIZE_NODE, PREPROCESS_NODE}:
            try:
                self._plan_from_envelope(envelope)
            except (TypeError, ValueError, OSError):
                return AvailabilityStatus.MISSING
            if node_key == PREPARE_NODE:
                return AvailabilityStatus.AVAILABLE
        if node_key == INFERENCE_NODE and envelope.get("kind") != "bytes":
            return AvailabilityStatus.MISSING
        return self._task_observation(node_key, task)

    def _start_ready_nodes(self, required: set[str]) -> None:
        repository = self.store.execution
        statuses = {
            node.node_key: node.status
            for node in repository.list_nodes(self.execution_run_id)
        }
        for node_key in ready_node_keys(
            repository.get_run(self.execution_run_id).plan,
            statuses,
        ):
            if node_key not in required:
                continue
            plans = self._task_plans(node_key)
            with self.store.transaction():
                repository.start_node(
                    self.execution_run_id,
                    node_key,
                    now=self._now(),
                )
                records = repository.discover_tasks(
                    self.execution_run_id,
                    node_key,
                    plans,
                    now=self._now(),
                )
                for record in records:
                    observation = (
                        AvailabilityStatus.MISSING
                        if node_key in {DOWNLOAD_MODELS_NODE, PREPARE_NODE}
                        else self._task_observation(node_key, record)
                    )
                    repository.record_task_result_observation(
                        self.execution_run_id,
                        node_key,
                        record.task_key,
                        observation,
                        now=self._now(),
                    )

    def _task_plans(self, node_key: str) -> tuple[TaskPlan, ...]:
        if node_key == DOWNLOAD_MODELS_NODE:
            return (TaskPlan("models", {"app_version": self.request.app_version}),)
        if node_key == PREPARE_NODE:
            return (
                TaskPlan(
                    "prepare",
                    scientific_payload={
                        "fasta_sha256": sha256(self.request.fasta_content).hexdigest()
                    },
                ),
            )
        if node_key == CHUNKS_NODE:
            app = _workload_module()
            plan = self._plan_from_node(PREPARE_NODE)
            return tuple(
                TaskPlan(
                    chunk.chunk_name,
                    scientific_payload={
                        "csv_sha256": app._file_sha256(Path(chunk.csv_path))
                    },
                    execution_payload={"chunk": asdict(chunk)},
                )
                for chunk in plan.chunks
            )
        if node_key == FINALIZE_NODE:
            return (TaskPlan("finalize", {"cache_key": self.cache_key}),)
        if node_key == PREPROCESS_NODE:
            return (TaskPlan("preprocess", {"cache_key": self.cache_key}),)
        if node_key == INFERENCE_NODE:
            return (TaskPlan("inference", {"cache_key": self.cache_key}),)
        raise ValueError(f"Unknown ENsiRNA Node {node_key!r}")

    def _try_plan_from_node(self, node_key: str) -> EnsirnaPreparationPlan | None:
        try:
            return self._plan_from_node(node_key)
        except (LookupError, TypeError, ValueError, OSError):
            return None

    def _plan_from_node(self, node_key: str) -> EnsirnaPreparationPlan:
        repository = self.store.execution
        for call in repository.list_provider_calls(self.execution_run_id):
            if (
                call.node_key == node_key
                and call.status == ProviderCallStatus.SUCCEEDED
            ):
                return self._plan_from_envelope(call.result_envelope)
        if node_key == PREPROCESS_NODE:
            cached = _workload_module()._cached_preparation_plan(
                cache_key=self.cache_key,
                layout=self.layout,
            )
            if cached is not None:
                return cached
        raise LookupError(f"ENsiRNA {node_key} result is unavailable")

    def _plan_from_envelope(self, envelope: object) -> EnsirnaPreparationPlan:
        if not isinstance(envelope, dict) or envelope.get("kind") != "plan":
            raise TypeError("ENsiRNA plan envelope is invalid")
        value = envelope.get("plan")
        if not isinstance(value, dict):
            raise TypeError("ENsiRNA plan payload is invalid")
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
        parsed_chunks = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise TypeError("ENsiRNA chunk payload is invalid")
            chunk_name = chunk.get("chunk_name")
            csv_path = chunk.get("csv_path")
            chunk_json_path = chunk.get("json_path")
            pdb_dir = chunk.get("pdb_dir")
            if (
                not isinstance(chunk_name, str)
                or not isinstance(csv_path, str)
                or not isinstance(chunk_json_path, str)
                or not isinstance(pdb_dir, str)
            ):
                raise TypeError("ENsiRNA chunk payload is invalid")
            parsed_chunks.append(
                EnsirnaPdbChunkSpec(
                    chunk_name=chunk_name,
                    csv_path=csv_path,
                    json_path=chunk_json_path,
                    pdb_dir=pdb_dir,
                )
            )
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

    def _reconcile_nodes_and_run(self) -> None:
        repository = self.store.execution
        for node in repository.list_nodes(self.execution_run_id):
            if node.status == NodeStatus.RUNNING and node.discovery_complete:
                with self.store.transaction():
                    repository.reconcile_node_tasks(
                        self.execution_run_id,
                        node.node_key,
                        now=self._now(),
                    )
        with self.store.transaction():
            repository.skip_unreachable_nodes(
                self.execution_run_id,
                now=self._now(),
            )
            repository.finalize_run_from_results(
                self.execution_run_id,
                now=self._now(),
            )

    def _admit_remote_tasks(self, required: set[str]) -> None:
        repository = self.store.execution
        run = repository.get_run(self.execution_run_id)
        nodes = repository.list_nodes(self.execution_run_id)
        ranks = required_node_ranks(
            run.plan,
            required_node_keys=required,
            unfinished_node_keys={
                node.node_key for node in nodes if not node.status.is_terminal
            },
        )
        descriptors = []
        for node in nodes:
            if (
                node.node_key not in required
                or node.status != NodeStatus.RUNNING
                or not node.discovery_complete
            ):
                continue
            binding = self._binding(node.node_key)
            for task in repository.list_tasks(
                self.execution_run_id,
                node.node_key,
            ):
                if (
                    task.status != TaskStatus.PENDING
                    or task.result_observation != AvailabilityStatus.MISSING
                ):
                    continue
                rank = ranks[node.node_key]
                descriptors.append(
                    TaskDispatchDescriptor(
                        node_key=node.node_key,
                        node_ordinal=node.ordinal,
                        task_key=task.task_key,
                        task_ordinal=task.ordinal,
                        binding=binding,
                        compatibility_key=binding.function_name,
                        max_tasks_per_call=1,
                        depth=rank.depth,
                        unblocking_span=rank.unblocking_span,
                    )
                )
        self._provider.repository = repository
        descriptors = list(
            self._provider.persist_fixed_dispatch_policy(
                self.execution_run_id,
                tuple(descriptors),
                now=self._now(),
            )
        )
        repository = self.store.execution
        counts = repository.active_provider_call_counts(self.execution_run_id)
        selected = select_admissible_candidates(
            form_fixed_batches(tuple(descriptors)),
            available_total_slots=max(0, run.max_active_provider_calls - counts.total),
            available_gpu_slots=max(0, run.max_active_gpu_provider_calls - counts.gpu),
        )
        for candidate in selected:
            self._ensure_publication_claim(candidate.node_key)
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
            DOWNLOAD_MODELS_NODE: "download_ensirna_models",
            PREPARE_NODE: "ensirna_prepare_inputs",
            CHUNKS_NODE: "ensirna_prepare_pdb_chunk",
            FINALIZE_NODE: "ensirna_finalize_prepared_inputs",
            PREPROCESS_NODE: "ensirna_preprocess_dataset",
            INFERENCE_NODE: "run_ensirna_inference",
        }[node_key]
        uses_gpu = node_key in {PREPROCESS_NODE, INFERENCE_NODE}
        return ProviderBinding(
            environment=self.deployment.environment,
            app_name=self.deployment.deployment_name,
            app_version=self.deployment.deployment_version,
            function_name=function_name,
            uses_gpu=uses_gpu,
            runtime_image_key="ensirna-gpu" if uses_gpu else "ensirna-cpu",
        )

    def _invocation_kwargs(
        self,
        node_key: str,
        task_key: str,
    ) -> dict[str, object]:
        if node_key == DOWNLOAD_MODELS_NODE:
            return {"force": False}
        if node_key == PREPARE_NODE:
            return {
                "mrna_fasta_bytes": self.request.fasta_content,
                "max_prepare_jobs": self.request.prepare_workers,
                "force_generation": self.request.force_generation,
            }
        if node_key == CHUNKS_NODE:
            task = self.store.execution.get_task(
                self.execution_run_id,
                CHUNKS_NODE,
                task_key,
            )
            return {
                "chunk": EnsirnaPdbChunkSpec(**task.execution_payload["chunk"]),
                "pdb_cores": self.request.pdb_cores,
            }
        if node_key == FINALIZE_NODE:
            return {"plan": self._plan_from_node(PREPARE_NODE)}
        if node_key == PREPROCESS_NODE:
            return {
                "plan": self._plan_from_node(FINALIZE_NODE),
                "preprocess_shard_size": self.request.preprocess_shard_size,
            }
        if node_key == INFERENCE_NODE:
            return {
                "prepared_dir": self._plan_from_node(PREPROCESS_NODE).prepared_dir,
                "force": False,
            }
        raise ValueError(f"Unknown ENsiRNA Node {node_key!r}")

    def _ensure_publication_claim(self, node_key: str) -> None:
        if node_key == DOWNLOAD_MODELS_NODE:
            return
        publication = "result" if node_key == INFERENCE_NODE else "prepared"
        if publication in self._claimed_publications:
            return
        acquire_output_claim(
            self.output_claims,
            claim_key=f"ensirna-{publication}:{self.cache_key}",
            owner=str(self.execution_run_id),
            replace_owner=self.request.replace_claim_owner,
        )
        self._claimed_publications.add(publication)

    def _checkpoint(self):
        self._volume_sync.commit()
        repository = self.store.execution
        self._provider.repository = repository
        return repository

    def _reload_output(self) -> None:
        self._volume_sync.reload()
        self._provider.repository = self.store.execution


def _result_envelope(result: object) -> dict[str, object]:
    """Encode only bounded plans, digests, or diagnostics."""
    if isinstance(result, EnsirnaPreparationPlan):
        return {"kind": "plan", "plan": asdict(result)}
    if isinstance(result, bytes):
        return {
            "kind": "bytes",
            "size": len(result),
            "sha256": sha256(result).hexdigest(),
        }
    if isinstance(result, dict):
        return {"kind": "dict", "result": orjson.loads(orjson.dumps(result))}
    if result is None:
        return {"kind": "none"}
    return {"kind": "invalid"}


def _workload_module():
    """Import workload-owned validators after the Modal app finishes loading."""
    from biomodals.app.score import ensirna_app

    return ensirna_app


class EnsirnaExecutionCoordinator(ExecutionCoordinatorLifecycle):
    """Bind one run-scoped writer to ENsiRNA publications."""

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        output_volume: Any,
        output_claims: Any,
        modal_driver: Any,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Capture only the deployment resources used by this adapter."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
        )
        self.output_volume = output_volume
        self.output_claims = output_claims
        self.modal_driver = modal_driver
        self.poll_interval_seconds = poll_interval_seconds

    def run(self) -> ExecutionSnapshot:
        """Load the staged request and drive one root Run."""
        with self._drive_lock:
            with self._writer_lock:
                runtime = self._open_runtime(
                    load_execution_request(self.volume_root, self.execution_run_id)
                )
            return self._drive(runtime, resume=False)

    def cancel(self) -> ExecutionSnapshot:
        """Request cancellation and reconcile it to a terminal result."""
        with self._writer_lock:
            runtime = self._open_runtime(
                load_execution_request(self.volume_root, self.execution_run_id)
            )
            snapshot = runtime.cancel()
            self._verify_snapshot(snapshot)
        if snapshot.run.status.is_terminal:
            return snapshot
        return self._drive(runtime, resume=False)

    def resume(self) -> ExecutionSnapshot:
        """Resume this Run without retrying conclusive failures."""
        with self._drive_lock:
            with self._writer_lock:
                runtime = self._open_runtime(
                    load_execution_request(self.volume_root, self.execution_run_id)
                )
            return self._drive(runtime, resume=True)

    def restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
        expected_workload_plan_fingerprint: str | None = None,
        candidate_request: EnsirnaExecutionRequest | None = None,
    ) -> ExecutionSnapshot:
        """Create and drive a compatible Successor from conclusive state."""
        del max_active_gpu_provider_calls
        if candidate_request is not None and max_active_provider_calls is not None:
            raise ValueError(
                "Candidate request and generic restart overrides are mutually exclusive"
            )
        if predecessor_execution_run_id == self.execution_run_id:
            raise ValueError("Successor Execution Run ID must be new")
        with self._drive_lock:
            with self._writer_lock:
                self.output_volume.reload()
                predecessor_store = ExecutionRunStore(
                    self.volume_root,
                    predecessor_execution_run_id,
                )
                if not predecessor_store.ledger_path.is_file():
                    raise ExecutionRunNotFoundError(str(predecessor_execution_run_id))
                try:
                    predecessor = predecessor_store.execution.validate_successor_source(
                        predecessor_execution_run_id
                    )
                    if (
                        expected_workload_plan_fingerprint is not None
                        and predecessor.plan.workload_plan_fingerprint
                        != expected_workload_plan_fingerprint
                    ):
                        raise ValueError(
                            "Restart arguments changed the Workload Plan Fingerprint"
                        )
                    if (
                        predecessor_deployment is not None
                        and predecessor.deployment != predecessor_deployment
                    ):
                        raise ValueError(
                            "Predecessor Deployment Identity does not match Execution Run"
                        )
                    predecessor_request = load_execution_request(
                        self.volume_root,
                        predecessor_execution_run_id,
                    )
                finally:
                    predecessor_store.close()
                request = candidate_request or predecessor_request
                if (
                    request.execution_plan.workload_plan_fingerprint
                    != predecessor.plan.workload_plan_fingerprint
                ):
                    raise ValueError(
                        "Restart arguments changed the Workload Plan Fingerprint"
                    )
                if candidate_request is None:
                    request = replace(
                        request,
                        prepare_workers=(
                            predecessor.max_active_provider_calls
                            if max_active_provider_calls is None
                            else max_active_provider_calls
                        ),
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
                self.output_volume.commit()
                runtime = self._open_runtime(
                    request,
                    predecessor_execution_run_id=predecessor_execution_run_id,
                )
            return self._drive(runtime, resume=False)

    def _open_runtime(
        self,
        request: EnsirnaExecutionRequest,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> EnsirnaExecutionRuntime:
        runtime = self._runtime
        if runtime is not None:
            if (
                runtime.request != request
                or runtime.predecessor_execution_run_id != predecessor_execution_run_id
            ):
                raise ValueError("Active ENsiRNA runtime does not match request")
            return runtime
        runtime = EnsirnaExecutionRuntime(
            request=request,
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=predecessor_execution_run_id,
            deployment=self.deployment,
            store=self._run_store(),
            modal_driver=self.modal_driver,
            output_volume=self.output_volume,
            output_claims=self.output_claims,
            poll_interval_seconds=self.poll_interval_seconds,
        )
        self._runtime = runtime
        return runtime
