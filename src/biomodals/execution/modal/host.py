"""Modal Volume-backed execution host and coordinator lifecycle."""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from threading import Lock, RLock
from typing import Any, cast
from uuid import UUID

import orjson

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionNodeRecord,
    ExecutionOverview,
    ExecutionPlan,
    ExecutionRunNotFoundError,
    ExecutionRunRecord,
    ExecutionRuntime,
    ExecutionTaskRecord,
    ProviderBinding,
    ProviderCallSubmission,
    RunStatus,
    SqliteExecutionRepository,
    drive_execution_run,
    resume_execution_run,
)
from biomodals.execution.definition import ExecutionGraph
from biomodals.execution.definition_plan import execution_plan
from biomodals.execution.definition_runtime import ExecutionGraphRuntime
from biomodals.execution.scheduler import (
    NodeAdmissionRank,
    ProviderCallCandidate,
    TaskDispatchDescriptor,
)
from biomodals.execution.store import ExecutionRunStore, GraphExecutionRunStore
from biomodals.helper.artifacts import (
    VolumeHandle,
    read_bounded_file_bytes,
    read_volume_bytes,
    read_volume_file_exact,
    replace_bytes_atomic,
)
from biomodals.helper.output_claim import register_output_claim_successor

from .identity import execution_coordinator_handle


def resolve_provider_call_limits(
    *,
    default_max_containers: int,
    default_max_gpu_containers: int,
    max_containers: int | None,
    max_gpu_containers: int | None,
) -> tuple[int, int]:
    """Resolve the two CLI container ceilings into kernel call limits."""
    total = default_max_containers if max_containers is None else max_containers
    gpu = (
        min(default_max_gpu_containers, total)
        if max_gpu_containers is None
        else max_gpu_containers
    )
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total < 1
        or isinstance(gpu, bool)
        or not isinstance(gpu, int)
        or not 0 <= gpu <= total
    ):
        raise ValueError("max_gpu_containers must be between zero and max_containers")
    return total, gpu


@dataclass(frozen=True, slots=True)
class ExecutionRequestFile:
    """Store one app's bounded immutable request bytes."""

    filename: str
    max_bytes: int
    name: str

    def path(self, execution_run_id: UUID) -> PurePosixPath:
        """Return the reserved path for one Execution Run."""
        return (
            PurePosixPath(".biomodals")
            / "execution"
            / "runs"
            / str(execution_run_id)
            / self.filename
        )

    def stage(
        self,
        output_volume: Any,
        execution_run_id: UUID,
        content: bytes,
    ) -> PurePosixPath:
        """Idempotently stage bytes through the client-side Volume API."""
        self._validate(content)
        path = self.path(execution_run_id)
        existing = read_volume_bytes(
            output_volume,
            path.as_posix(),
            max_bytes=self.max_bytes,
        )
        if existing is not None:
            if existing != content:
                raise RuntimeError(f"Existing {self.name} conflicts with this run")
            return path
        with output_volume.batch_upload(force=True) as batch:
            batch.put_file(BytesIO(content), f"/{path.as_posix()}")
        return path

    def persist(
        self,
        volume_root: str | Path,
        execution_run_id: UUID,
        content: bytes,
    ) -> PurePosixPath:
        """Atomically create bytes from inside a mounted coordinator."""
        self._validate(content)
        relative = self.path(execution_run_id)
        path = Path(volume_root).joinpath(*relative.parts)
        if path.exists():
            if self.load(volume_root, execution_run_id) != content:
                raise RuntimeError(f"{self.name} is immutable")
            return relative
        replace_bytes_atomic(path, content)
        return relative

    def load(self, volume_root: str | Path, execution_run_id: UUID) -> bytes:
        """Load bytes from a coordinator-mounted Volume."""
        path = Path(volume_root).joinpath(*self.path(execution_run_id).parts)
        content = read_bounded_file_bytes(
            path,
            field_name=self.name,
            max_bytes=self.max_bytes,
        )
        self._validate(content)
        return content

    def load_from_volume(
        self,
        output_volume: Any,
        execution_run_id: UUID,
    ) -> bytes:
        """Load bytes through the client-side Volume API."""
        path = self.path(execution_run_id)
        content = read_volume_bytes(
            output_volume,
            path.as_posix(),
            max_bytes=self.max_bytes,
        )
        if content is None:
            raise FileNotFoundError(path.as_posix())
        self._validate(content)
        return content

    def _validate(self, content: bytes) -> None:
        if not isinstance(content, bytes):
            raise TypeError(f"{self.name} must be bytes")
        if not 0 < len(content) <= self.max_bytes:
            raise ValueError(f"{self.name} exceeds its byte limit")


_LAUNCH_FILE = ExecutionRequestFile(
    "launch",
    36,
    "Execution launch identity",
)


def stage_execution_launch(
    output_volume: Any,
    execution_run_id: UUID,
    predecessor_execution_run_id: UUID | None,
) -> PurePosixPath:
    """Stage immutable root/successor identity before coordinator submission."""
    return _LAUNCH_FILE.stage(
        output_volume,
        execution_run_id,
        _execution_launch_bytes(predecessor_execution_run_id),
    )


def submit_staged_execution_run(
    output_volume: Any,
    *,
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
    predecessor_execution_run_id: UUID | None,
    use_deployed_coordinator: bool,
    local_coordinator: Callable[..., Any],
    workload_name: str,
    restart_kwargs: Mapping[str, object] | None = None,
    accepted_statuses: Collection[RunStatus] = (RunStatus.SUCCEEDED,),
) -> ExecutionOverview:
    """Submit and await one staged direct App Run."""
    stage_execution_launch(
        output_volume,
        execution_run_id,
        predecessor_execution_run_id,
    )
    coordinator = execution_coordinator_handle(
        execution_run_id=execution_run_id,
        deployment=deployment,
        use_deployed_coordinator=use_deployed_coordinator,
        local_coordinator=local_coordinator,
    )
    if predecessor_execution_run_id is None:
        call = coordinator.run.spawn(development=not use_deployed_coordinator)
    else:
        kwargs = dict(restart_kwargs or ())
        kwargs["predecessor_execution_run_id"] = str(predecessor_execution_run_id)
        call = coordinator.restart_from.spawn(**kwargs)

    print(
        "Deployment Identity: "
        f"{deployment.environment}/{deployment.deployment_name}/"
        f"v{deployment.deployment_version}"
    )
    print(f"Execution Run ID: {execution_run_id}")
    print(f"Coordinator FunctionCall ID: {getattr(call, 'object_id', call)}")
    overview = call.get()
    if overview.run.status not in accepted_statuses:
        diagnostic = overview.run.status_message or (
            overview.run.status_reason.value
            if overview.run.status_reason is not None
            else overview.run.status.value
        )
        raise RuntimeError(
            f"{workload_name} Execution Run ended as "
            f"{overview.run.status.value}: {diagnostic}"
        )
    return overview


def load_execution_provider_result(
    output_volume: Any,
    *,
    execution_run_id: UUID,
    envelope: object,
    max_bytes: int = 16 * 1024 * 1024,
) -> object:
    """Read one content-bound provider result through the Volume client."""
    if not isinstance(envelope, Mapping):
        raise ValueError("Execution Result Envelope must be an object")
    envelope = cast(Mapping[str, object], envelope)
    reference = envelope.get("result_file")
    if not isinstance(reference, Mapping):
        raise ValueError("Execution Result Envelope has no result file")
    reference = cast(Mapping[str, object], reference)
    relative_path = reference.get("path")
    if not isinstance(relative_path, str):
        raise ValueError("Execution Result Envelope path is invalid")
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("Execution Result Envelope path must be contained")
    size_bytes = reference.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or not 0 < size_bytes <= max_bytes
    ):
        raise ValueError("Execution Result Envelope exceeds its byte limit")
    content = read_volume_file_exact(
        output_volume,
        (PurePosixPath("workflow-runs") / str(execution_run_id) / relative).as_posix(),
        size_bytes=size_bytes,
        content_sha256=reference.get("sha256"),
    )
    encoding = reference.get("encoding")
    if encoding not in {"bytes", "json"}:
        raise ValueError("Execution Result Envelope encoding is invalid")
    return content if encoding == "bytes" else orjson.loads(content)


def persist_execution_launch(
    volume_root: str | Path,
    execution_run_id: UUID,
    predecessor_execution_run_id: UUID | None,
) -> PurePosixPath:
    """Persist immutable launch identity from a mounted coordinator."""
    return _LAUNCH_FILE.persist(
        volume_root,
        execution_run_id,
        _execution_launch_bytes(predecessor_execution_run_id),
    )


def load_execution_launch(
    volume_root: str | Path,
    execution_run_id: UUID,
) -> UUID | None:
    """Load the predecessor identity staged for one coordinator launch."""
    return _parse_execution_launch(_LAUNCH_FILE.load(volume_root, execution_run_id))


def _parse_execution_launch(content: bytes) -> UUID | None:
    if content == b"root":
        return None
    try:
        parsed = UUID(content.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("Execution launch predecessor is invalid") from error
    if str(parsed).encode() != content:
        raise ValueError("Execution launch predecessor is not canonical")
    return parsed


def execution_lineage_root(
    output_volume: Any,
    execution_run_id: UUID,
) -> UUID:
    """Follow staged predecessor identities to the root Execution Run."""
    current = execution_run_id
    seen: set[UUID] = set()
    while current not in seen:
        seen.add(current)
        predecessor = _parse_execution_launch(
            _LAUNCH_FILE.load_from_volume(output_volume, current)
        )
        if predecessor is None:
            return current
        current = predecessor
    raise ValueError("Execution launch lineage contains a cycle")


def _execution_launch_bytes(predecessor_execution_run_id: UUID | None) -> bytes:
    return (
        b"root"
        if predecessor_execution_run_id is None
        else str(predecessor_execution_run_id).encode()
    )


class ExecutionVolumeSync:
    """Close a Run store while synchronizing its backing Volume."""

    def __init__(
        self,
        *,
        volume: VolumeHandle | None,
        store: ExecutionRunStore,
    ) -> None:
        """Bind one optional Volume to its closeable local Run store."""
        self.volume = volume
        self.store = store

    def commit(self) -> None:
        """Persist pending Volume writes when a Volume is attached."""
        if self.volume is None:
            self.store.commit()
            return
        with self.store.closed_for_storage_sync():
            self.volume.commit()

    def reload(self) -> None:
        """Refresh the Volume view when a Volume is attached."""
        if self.volume is None:
            return
        with self.store.closed_for_storage_sync():
            self.volume.reload()


class ExecutionRuntimeLifecycle:
    """Share the host lifecycle used by direct CLI App Run adapters."""

    execution_run_id: UUID
    store: ExecutionRunStore
    poll_interval_seconds: float
    _now: Callable[[], int]
    _provider: ExecutionRuntime
    _volume_sync: ExecutionVolumeSync

    def _bind_execution_runtime(
        self,
        *,
        request: Any,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        store: ExecutionRunStore,
        provider_driver: Any,
        output_volume: Any,
        predecessor_execution_run_id: UUID | None,
        poll_interval_seconds: float,
        now: Callable[[], int] | None,
        volume_io_lock: RLock | None = None,
    ) -> None:
        """Bind the common host state shared by direct App runtimes."""
        self.request = request
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.store = store
        self.output_volume = output_volume
        self.predecessor_execution_run_id = predecessor_execution_run_id
        self.poll_interval_seconds = poll_interval_seconds
        self._now = now or (lambda: int(time.time()))
        self._volume_io_lock = (
            store.volume_io_lock if volume_io_lock is None else volume_io_lock
        )
        self._volume_sync = ExecutionVolumeSync(volume=output_volume, store=store)
        self._provider = ExecutionRuntime(
            store.execution,
            provider_driver=provider_driver,
            checkpoint=self._checkpoint,
            transaction=store.transaction,
            synchronize=self._synchronize_kernel_state,
        )

    def _create_or_verify_run(
        self,
        *,
        plan: ExecutionPlan,
        max_active_provider_calls: int,
        max_active_gpu_provider_calls: int,
    ) -> SqliteExecutionRepository:
        """Create or verify this bound App Run against its immutable plan."""
        self._provider.create_or_verify_run(
            execution_run_id=self.execution_run_id,
            predecessor_execution_run_id=self.predecessor_execution_run_id,
            plan=plan,
            deployment=self.deployment,
            max_active_provider_calls=max_active_provider_calls,
            max_active_gpu_provider_calls=max_active_gpu_provider_calls,
            now=self._now(),
        )
        return self.store.execution

    def run(self) -> ExecutionOverview:
        """Create or recover the Run and drive it until it stops."""
        with self._synchronize_kernel_state():
            repository = self._initialize()
        return drive_execution_run(
            repository,
            self.execution_run_id,
            advance_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            now=self._now,
            poll_interval_seconds=self.poll_interval_seconds,
            synchronize=self._synchronize_kernel_state,
        )

    def resume(self) -> ExecutionOverview:
        """Resume this Run without retrying conclusive failures."""
        with self._synchronize_kernel_state():
            repository = self._initialize()
        resume_execution_run(
            repository,
            self.execution_run_id,
            reconcile_once=self.advance_once,
            checkpoint=self._checkpoint,
            current_repository=lambda: self.store.execution,
            synchronize=self._synchronize_kernel_state,
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
            synchronize=self._synchronize_kernel_state,
        )

    def cancel(self) -> ExecutionOverview:
        """Request cancellation while retaining uncertain call ownership."""
        with self._synchronize_kernel_state():
            repository = self.store.execution
            try:
                repository.get_run(self.execution_run_id)
            except ExecutionRunNotFoundError:
                repository = self._initialize()
        self._provider.cancel_run(self.execution_run_id, now=self._now())
        with self._synchronize_kernel_state():
            return self.store.execution.overview(self.execution_run_id)

    def close(self) -> None:
        """Close SQLite without cancelling attached Provider Calls."""
        self.store.close()

    def advance_once(self) -> None:
        """Apply one workload-owned execution cycle."""
        raise NotImplementedError

    def _initialize(self) -> SqliteExecutionRepository:
        """Create or verify the standard request-owned Execution Run."""
        return self._create_or_verify_run(
            plan=self.request.execution_plan,
            max_active_provider_calls=self.request.max_active_provider_calls,
            max_active_gpu_provider_calls=self.request.max_active_gpu_provider_calls,
        )

    def _checkpoint(self) -> SqliteExecutionRepository:
        with self._synchronize_kernel_state():
            try:
                self._volume_sync.commit()
            finally:
                repository = self.store.execution
                self._provider.repository = repository
        return repository

    def _reload_output(self) -> None:
        with self._synchronize_kernel_state():
            try:
                self._volume_sync.reload()
            finally:
                self._provider.repository = self.store.execution

    def _with_volume_io(self, operation: Callable[..., Any], *args: Any) -> Any:
        """Run only a mounted-file operation under the run-scoped Volume lock."""
        with self._volume_io_lock:
            return operation(*args)

    @contextmanager
    def _synchronize_kernel_state(self) -> Iterator[None]:
        """Order an optional output-Volume barrier before the SQLite writer."""
        volume_io_lock = getattr(self, "_volume_io_lock", None)
        if volume_io_lock is None:
            with self.store.synchronize():
                yield
            return
        with volume_io_lock, self.store.synchronize():
            yield


class StandardExecutionRuntimeLifecycle(ExecutionRuntimeLifecycle):
    """Share the standard publication, recovery, and admission cycle."""

    def advance_once(self) -> None:
        """Apply one standard workload-owned execution cycle."""
        self._provider.advance_once(
            self.execution_run_id,
            recover_publications=lambda: self._with_volume_io(
                self._recover_publications
            ),
            reconcile_provider_calls=self._reconcile_provider_calls,
            decode_completed_calls=lambda: self._with_volume_io(
                self._decode_completed_calls
            ),
            start_ready_nodes=lambda required: self._with_volume_io(
                self._start_ready_nodes,
                required,
            ),
            after_start_ready_nodes=lambda: self._with_volume_io(
                self._after_start_ready_nodes
            ),
            admit_remote_tasks=lambda required: self._with_volume_io(
                self._admit_remote_tasks,
                required,
            ),
            reconcile_results=lambda: self._with_volume_io(self._reconcile_results),
            now=self._now,
        )

    def _recover_publications(self) -> None:
        raise NotImplementedError

    def _reconcile_provider_calls(self, required: set[str]) -> None:
        raise NotImplementedError

    def _decode_completed_calls(self) -> None:
        """Decode provider envelopes when a workload does not publish by probe."""

    def _start_ready_nodes(self, required: set[str]) -> None:
        raise NotImplementedError

    def _after_start_ready_nodes(self) -> None:
        """Complete coordinator-local work after Node discovery, when present."""

    def _reconcile_results(self) -> None:
        self._provider.reconcile_nodes_and_run(
            self.execution_run_id,
            now=self._now(),
        )

    def _admit_remote_tasks(self, required: set[str]) -> None:
        """Admit fixed Provider Calls containing one Task each."""
        with self.store.synchronize():
            repository = self.store.execution
            run = repository.get_run(self.execution_run_id)
            counts = repository.active_provider_call_counts(self.execution_run_id)
        self._prepare_admission(required)
        selected = self._provider.fixed_call_candidates(
            self.execution_run_id,
            required_node_keys=required,
            describe_task=self._dispatch_descriptor,
            available_total_slots=max(0, run.max_active_provider_calls - counts.total),
            available_gpu_slots=max(0, run.max_active_gpu_provider_calls - counts.gpu),
            now=self._now(),
        )
        selected = self._prepare_candidates(selected)
        submissions = tuple(
            submission
            for candidate in selected
            if (submission := self._provider_call_submission(candidate)) is not None
        )
        self._provider.submit_provider_calls(
            self.execution_run_id,
            submissions,
            now=self._now(),
        )

    def _prepare_admission(self, required: set[str]) -> None:
        """Prepare workload-owned descriptor state without admitting work."""
        del required

    def _dispatch_descriptor(
        self,
        node: ExecutionNodeRecord,
        task: ExecutionTaskRecord,
        rank: NodeAdmissionRank,
    ) -> TaskDispatchDescriptor | None:
        """Describe standard one-Task dispatch for one ready Task."""
        binding = self._binding(node.node_key)
        return TaskDispatchDescriptor(
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

    def _prepare_candidates(
        self,
        selected: tuple[ProviderCallCandidate, ...],
    ) -> tuple[ProviderCallCandidate, ...]:
        """Claim workload publications before Provider Call submission."""
        for candidate in selected:
            self._ensure_publication_claim(
                candidate.node_key,
                candidate.task_keys[0],
            )
        return selected

    def _provider_call_submission(
        self,
        candidate: ProviderCallCandidate,
    ) -> ProviderCallSubmission | None:
        """Build the provider invocation for one admitted candidate."""
        return ProviderCallSubmission(
            candidate=candidate,
            submission_token=candidate.candidate_key,
            kwargs=self._invocation_kwargs(
                candidate.node_key,
                candidate.task_keys[0],
            ),
        )

    def _binding(self, node_key: str) -> ProviderBinding:
        raise NotImplementedError

    def _ensure_publication_claim(self, node_key: str, task_key: str) -> None:
        del node_key, task_key

    def _invocation_kwargs(
        self,
        node_key: str,
        task_key: str,
    ) -> dict[str, object]:
        del node_key, task_key
        raise NotImplementedError


class ExecutionCoordinatorLifecycle:
    """Share app-coordinator locking, status, and drive mechanics."""

    _request_loader: Callable[[str | Path, UUID], Any]
    _request_persister: Callable[[str | Path, UUID, Any], Any]
    output_volume: Any

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        target_scientific_versions: Mapping[str, str],
    ) -> None:
        """Bind the lifecycle to one Run and exact deployment."""
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.volume_root = Path(volume_root)
        self.target_scientific_versions = dict(target_scientific_versions)
        if not self.target_scientific_versions or any(
            not name or not version
            for name, version in self.target_scientific_versions.items()
        ):
            raise ValueError("Target scientific versions cannot be empty")
        self._writer_lock = RLock()
        self._volume_io_lock = RLock()
        self._drive_lock = Lock()
        self._runtime: Any | None = None

    def run(self) -> ExecutionOverview:
        """Load the staged request and drive one root Run."""
        with self._drive_lock:
            with self._volume_io_lock, self._writer_lock:
                runtime = self._open_current_runtime(recover=False)
            return self._drive(runtime, resume=False)

    def drive_prepared(self) -> ExecutionOverview:
        """Drive a prepared root or Successor Run from immutable launch state."""
        with self._drive_lock:
            with self._volume_io_lock, self._writer_lock:
                runtime = self._open_current_runtime(recover=True)
            return self._drive(runtime, resume=False)

    def cancel(self) -> ExecutionOverview:
        """Request cancellation and reconcile it to a terminal result."""
        with self._volume_io_lock, self._writer_lock:
            runtime = self._open_current_runtime(recover=True)
        cancelled = runtime.cancel()
        with self._volume_io_lock, self._writer_lock:
            self._verify_overview(cancelled)
        with self._drive_lock:
            with self._volume_io_lock, self._writer_lock:
                runtime = self._open_current_runtime(recover=True)
                overview = runtime.store.execution.overview(self.execution_run_id)
                self._verify_overview(overview)
                if overview.run.status.is_terminal:
                    return overview
            return self._drive(runtime, resume=False)

    def resume(self) -> ExecutionOverview:
        """Resume this Run without retrying conclusive failures."""
        with self._drive_lock:
            with self._volume_io_lock, self._writer_lock:
                runtime = self._open_current_runtime(recover=True)
            return self._drive(runtime, resume=True)

    def status(self) -> ExecutionOverview:
        """Read one verified overview without advancing work."""
        with self._volume_io_lock, self._writer_lock:
            runtime = self._runtime
            if runtime is not None:
                overview = runtime.store.execution.overview(self.execution_run_id)
            else:
                store = self._run_store()
                if not store.ledger_path.is_file():
                    raise ExecutionRunNotFoundError(str(self.execution_run_id))
                try:
                    overview = store.execution.overview(self.execution_run_id)
                finally:
                    store.close()
            self._verify_overview(overview)
            return overview

    def prepare_restart(
        self,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_deployment: DeploymentIdentity | None,
        max_active_provider_calls: int | None = None,
        max_active_gpu_provider_calls: int | None = None,
        expected_workload_plan_fingerprint: str | None = None,
        candidate_request: Any | None = None,
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
            with self._volume_io_lock, self._writer_lock:
                self.output_volume.reload()
                with self._open_successor_source(
                    predecessor_execution_run_id,
                    predecessor_deployment=predecessor_deployment,
                    expected_workload_plan_fingerprint=(
                        expected_workload_plan_fingerprint
                    ),
                ) as (predecessor, predecessor_request, predecessor_store):
                    request = candidate_request or replace(
                        predecessor_request,
                        max_active_provider_calls=(
                            predecessor.max_active_provider_calls
                            if max_active_provider_calls is None
                            else max_active_provider_calls
                        ),
                        max_active_gpu_provider_calls=(
                            predecessor.max_active_gpu_provider_calls
                            if max_active_gpu_provider_calls is None
                            else max_active_gpu_provider_calls
                        ),
                    )
                    request = self._prepare_successor_request(
                        request,
                        predecessor_execution_run_id=predecessor_execution_run_id,
                        predecessor_store=predecessor_store,
                    )
                self._require_successor_plan_match(predecessor, request)
                self._persist_successor_request(
                    request,
                    predecessor_execution_run_id,
                )

    def close(self) -> None:
        """Close coordinator-local state without cancelling Provider Calls."""
        with self._drive_lock:
            with self._writer_lock:
                self._close_runtime()

    def _drive(self, runtime: Any, *, resume: bool) -> ExecutionOverview:
        overview = runtime.resume() if resume else runtime.run()
        self._verify_overview(overview)
        return overview

    def _run_store(self) -> ExecutionRunStore:
        return ExecutionRunStore(
            self.volume_root,
            self.execution_run_id,
            lock=self._writer_lock,
            volume_io_lock=self._volume_io_lock,
        )

    @contextmanager
    def _open_successor_source(
        self,
        predecessor_execution_run_id: UUID,
        *,
        predecessor_deployment: DeploymentIdentity | None,
        expected_workload_plan_fingerprint: str | None = None,
    ) -> Iterator[tuple[ExecutionRunRecord, Any, ExecutionRunStore]]:
        """Open one validated predecessor and its workload request."""
        if predecessor_execution_run_id == self.execution_run_id:
            raise ValueError("Successor Execution Run ID must be new")
        store = ExecutionRunStore(
            self.volume_root,
            predecessor_execution_run_id,
            lock=self._writer_lock,
            volume_io_lock=self._volume_io_lock,
        )
        if not store.ledger_path.is_file():
            raise ExecutionRunNotFoundError(str(predecessor_execution_run_id))
        try:
            predecessor = store.execution.validate_successor_source(
                predecessor_execution_run_id
            )
            if (
                predecessor_deployment is not None
                and predecessor.deployment != predecessor_deployment
            ):
                raise ValueError(
                    "Predecessor Deployment Identity does not match Execution Run"
                )
            if (
                expected_workload_plan_fingerprint is not None
                and predecessor.plan.workload_plan_fingerprint
                != expected_workload_plan_fingerprint
            ):
                raise ValueError(
                    "Restart arguments changed the Workload Plan Fingerprint"
                )
            self._require_target_scientific_versions(predecessor.plan)
            yield (
                predecessor,
                self._request_loader(
                    self.volume_root,
                    predecessor_execution_run_id,
                ),
                store,
            )
        finally:
            store.close()

    @staticmethod
    def _require_successor_plan_match(
        predecessor: ExecutionRunRecord,
        request: Any,
    ) -> None:
        """Reject a successor request whose scientific plan changed."""
        if (
            request.execution_plan.workload_plan_fingerprint
            != predecessor.plan.workload_plan_fingerprint
        ):
            raise ValueError("Successor request changed the Workload Plan Fingerprint")

    def _require_target_scientific_versions(self, plan: Any) -> None:
        """Require every target version declared by this plan to match."""
        declared = plan.scientific_versions
        compared = self.target_scientific_versions.keys() & declared.keys()
        if not compared or any(
            declared[name] != self.target_scientific_versions[name] for name in compared
        ):
            raise ValueError("Target deployment changed declared scientific versions")

    def _persist_successor_request(
        self,
        request: Any,
        predecessor_execution_run_id: UUID,
    ) -> None:
        """Persist successor launch inputs before another container can read them."""
        self._request_persister(
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

    def _open_current_runtime(self, *, recover: bool) -> Any:
        request = self._request_loader(self.volume_root, self.execution_run_id)
        self._require_target_scientific_versions(request.execution_plan)
        return self._open_runtime(
            request,
            predecessor_execution_run_id=(
                self._existing_predecessor() if recover else None
            ),
        )

    def _open_runtime(
        self,
        request: Any,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> Any:
        runtime = self._runtime
        if runtime is not None:
            if (
                runtime.request != request
                or runtime.predecessor_execution_run_id != predecessor_execution_run_id
            ):
                raise ValueError("Active execution runtime does not match request")
            return runtime
        runtime = self._create_runtime(
            request,
            predecessor_execution_run_id=predecessor_execution_run_id,
        )
        self._runtime = runtime
        return runtime

    def _create_runtime(
        self,
        request: Any,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> Any:
        del request, predecessor_execution_run_id
        raise NotImplementedError

    def _prepare_successor_request(
        self,
        request: Any,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_store: ExecutionRunStore,
    ) -> Any:
        """Apply host-specific successor metadata before persistence."""
        del predecessor_execution_run_id, predecessor_store
        return request

    def _existing_predecessor(self) -> UUID | None:
        runtime = self._runtime
        if runtime is not None:
            return runtime.predecessor_execution_run_id
        store = self._run_store()
        if not store.ledger_path.is_file():
            return load_execution_launch(
                self.volume_root,
                self.execution_run_id,
            )
        try:
            return store.execution.get_run(
                self.execution_run_id
            ).predecessor_execution_run_id
        finally:
            store.close()

    def _verify_overview(self, overview: ExecutionOverview) -> None:
        if overview.run.execution_run_id != self.execution_run_id:
            raise ValueError("Execution Run ID does not match coordinator")
        if overview.run.deployment != self.deployment:
            raise ValueError("Deployment Identity does not match Execution Run")

    def _close_runtime(self) -> None:
        runtime = self._runtime
        if runtime is not None:
            runtime.close()
            self._runtime = None


class OutputClaimExecutionCoordinatorLifecycle(ExecutionCoordinatorLifecycle):
    """Register Modal output-claim lineage for one Successor Run."""

    output_claims: Any

    def _prepare_successor_request(
        self,
        request: Any,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_store: ExecutionRunStore,
    ) -> Any:
        del predecessor_store
        register_output_claim_successor(
            self.output_claims,
            owner=str(self.execution_run_id),
            predecessor=str(predecessor_execution_run_id),
        )
        return replace(
            request,
            replace_claim_owner=str(predecessor_execution_run_id),
        )


class ExecutionDefinitionCoordinatorLifecycle(ExecutionCoordinatorLifecycle):
    """Host one app-owned Execution Definition through the shared graph runtime."""

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        artifact_volume_name: str,
        output_volume: Any,
        provider_driver: Any,
        graph_builder: Callable[[Any, UUID | None], ExecutionGraph],
        target_scientific_versions: Mapping[str, str],
        max_parallel_nodes: int = 32,
        pull_worker_coordinator: Any | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Bind app resources while leaving graph behavior app-owned."""
        super().__init__(
            execution_run_id=execution_run_id,
            deployment=deployment,
            volume_root=volume_root,
            target_scientific_versions=target_scientific_versions,
        )
        self.artifact_volume_name = artifact_volume_name
        self.output_volume = output_volume
        self.provider_driver = provider_driver
        self.graph_builder = graph_builder
        self.max_parallel_nodes = max_parallel_nodes
        self.pull_worker_coordinator = pull_worker_coordinator
        self.poll_interval_seconds = poll_interval_seconds

    def _drive(
        self,
        runtime: ExecutionGraphRuntime,
        *,
        resume: bool,
    ) -> ExecutionOverview:
        if resume:
            runtime.resume()
        else:
            runtime.run()
        overview = runtime.store.execution.overview(self.execution_run_id)
        self._verify_overview(overview)
        return overview

    def _run_store(self) -> GraphExecutionRunStore:
        return GraphExecutionRunStore(
            self.volume_root,
            self.execution_run_id,
            lock=self._writer_lock,
            volume_io_lock=self._volume_io_lock,
        )

    def _create_runtime(
        self,
        request: Any,
        *,
        predecessor_execution_run_id: UUID | None = None,
    ) -> ExecutionGraphRuntime:
        graph = self.graph_builder(request, predecessor_execution_run_id)
        workload_run_key = request.execution_plan.workload_run_key
        if not workload_run_key:
            raise ValueError("App Execution Plan must define a Workload Run Key")
        if (
            execution_plan(
                graph.validate(),
                workload_run_key=workload_run_key,
            )
            != request.execution_plan
        ):
            raise ValueError("App Execution Definition changed its Execution Plan")
        store = self._run_store()
        return ExecutionGraphRuntime(
            graph=graph,
            execution_run_id=self.execution_run_id,
            deployment=self.deployment,
            volume_root=self.volume_root,
            artifact_volume_name=self.artifact_volume_name,
            workload_run_key=workload_run_key,
            request=request,
            predecessor_execution_run_id=predecessor_execution_run_id,
            provider_driver=self.provider_driver,
            storage_sync=ExecutionVolumeSync(
                volume=self.output_volume,
                store=store,
            ),
            max_parallel_nodes=self.max_parallel_nodes,
            max_active_provider_calls=request.max_active_provider_calls,
            max_active_gpu_provider_calls=(request.max_active_gpu_provider_calls),
            pull_worker_coordinator=self.pull_worker_coordinator,
            store=store,
            volume_io_lock=self._volume_io_lock,
            poll_interval_seconds=self.poll_interval_seconds,
        )


class OutputClaimExecutionDefinitionCoordinatorLifecycle(
    ExecutionDefinitionCoordinatorLifecycle
):
    """Host an Execution Definition with successor output-claim lineage."""

    def __init__(self, *, output_claims: Any, **kwargs: Any) -> None:
        """Bind the shared claim store beside the graph host resources."""
        super().__init__(**kwargs)
        self.output_claims = output_claims

    def _prepare_successor_request(
        self,
        request: Any,
        *,
        predecessor_execution_run_id: UUID,
        predecessor_store: ExecutionRunStore,
    ) -> Any:
        del predecessor_store
        register_output_claim_successor(
            self.output_claims,
            owner=str(self.execution_run_id),
            predecessor=str(predecessor_execution_run_id),
        )
        return replace(
            request,
            replace_claim_owner=str(predecessor_execution_run_id),
        )
