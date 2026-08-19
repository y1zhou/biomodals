"""Physical state shared by remotely coordinated execution adapters."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from threading import Lock, RLock
from typing import Any, Protocol, cast
from uuid import UUID

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionOverview,
    ExecutionPlan,
    ExecutionRunNotFoundError,
    ExecutionRunRecord,
    ExecutionRuntime,
    ProviderBinding,
    ProviderCallSubmission,
    SqliteExecutionRepository,
    drive_execution_run,
    resume_execution_run,
)
from biomodals.execution.scheduler import TaskDispatchDescriptor
from biomodals.helper.artifacts import VolumeHandle, read_volume_bytes

LEDGER_FILENAME = "ledger.sqlite3"


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
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        return relative

    def load(self, volume_root: str | Path, execution_run_id: UUID) -> bytes:
        """Load bytes from a coordinator-mounted Volume."""
        path = Path(volume_root).joinpath(*self.path(execution_run_id).parts)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Expected regular file: {path}")
        if path.stat().st_size > self.max_bytes:
            raise ValueError(f"{self.name} exceeds its byte limit")
        content = path.read_bytes()
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


class ExecutionRunStore:
    """Own one Run's kernel connection and reserved state path."""

    def __init__(
        self,
        volume_root: str | Path,
        execution_run_id: UUID,
        *,
        lock: Any | None = None,
        volume_io_lock: Any | None = None,
    ) -> None:
        """Bind storage only to the host Volume root and opaque Run ID."""
        self.volume_root = Path(volume_root)
        self.execution_run_id = execution_run_id
        self._connection: sqlite3.Connection | None = None
        self._execution: SqliteExecutionRepository | None = None
        self._lock = RLock() if lock is None else lock
        self.volume_io_lock = RLock() if volume_io_lock is None else volume_io_lock
        self._volume_sync_active = False

    @property
    def state_root(self) -> Path:
        """Return the reserved directory containing only execution state."""
        return (
            self.volume_root
            / ".biomodals"
            / "execution"
            / "runs"
            / str(self.execution_run_id)
        )

    @property
    def ledger_path(self) -> Path:
        """Return the per-Run App Run Ledger path."""
        return self.state_root / LEDGER_FILENAME

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the active caller-owned SQLite connection."""
        with self._lock:
            return self._connect()

    @property
    def execution(self) -> SqliteExecutionRepository:
        """Return the shared execution repository on the active connection."""
        with self._lock:
            self._connect()
            if self._execution is None:
                raise RuntimeError("Execution repository was not initialized")
            return self._execution

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit or roll back one caller-owned execution transaction."""
        with self._lock:
            connection = self._connect()
            if connection.in_transaction:
                raise RuntimeError("Nested execution transactions are unsupported")
            try:
                yield
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @contextmanager
    def synchronize(self) -> Iterator[None]:
        """Serialize one compound repository and Volume state boundary."""
        with self._lock:
            yield

    @contextmanager
    def closed_for_volume_sync(self) -> Iterator[None]:
        """Commit and close SQLite while the host synchronizes its Volume."""
        with self._lock:
            connection = self._connection
            if connection is not None:
                connection.commit()
            self._close()
            self._volume_sync_active = True
            try:
                yield
            finally:
                self._volume_sync_active = False

    def close(self) -> None:
        """Close the active connection without inventing an implicit commit."""
        with self._lock:
            self._close()

    def commit(self) -> None:
        """Commit coordinator-local SQLite changes without syncing its Volume."""
        with self._lock:
            self._connect().commit()

    def _connect(self) -> sqlite3.Connection:
        if self._volume_sync_active:
            raise RuntimeError("Run store is closed for Volume synchronization")
        if self._connection is not None:
            return self._connection

        self.state_root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.ledger_path,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            execution = SqliteExecutionRepository(connection)
            execution.initialize_schema()
            self._initialize_additional_schema(connection)
            connection.commit()
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self._execution = execution
        return connection

    def _initialize_additional_schema(self, connection: sqlite3.Connection) -> None:
        """Allow an adapter-owned store to share this transaction boundary."""

    def _close(self) -> None:
        connection = self._connection
        self._connection = None
        self._execution = None
        if connection is not None:
            connection.close()


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
        with self.store.closed_for_volume_sync():
            self.volume.commit()

    def reload(self) -> None:
        """Refresh the Volume view when a Volume is attached."""
        if self.volume is None:
            return
        with self.store.closed_for_volume_sync():
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
        raise NotImplementedError

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


class _SingleTaskAdmissionHooks(Protocol):
    """Workload hooks for the standard one-Task-per-call admission path."""

    def _binding(self, node_key: str) -> ProviderBinding: ...

    def _ensure_publication_claim(self, node_key: str) -> None: ...

    def _invocation_kwargs(
        self,
        node_key: str,
        task_key: str,
    ) -> dict[str, object]: ...


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
            admit_remote_tasks=lambda required: self._with_volume_io(
                self._admit_remote_tasks,
                required,
            ),
            now=self._now,
        )

    def _recover_publications(self) -> None:
        raise NotImplementedError

    def _reconcile_provider_calls(self, required: set[str]) -> None:
        raise NotImplementedError

    def _decode_completed_calls(self) -> None:
        raise NotImplementedError

    def _start_ready_nodes(self, required: set[str]) -> None:
        raise NotImplementedError

    def _admit_remote_tasks(self, required: set[str]) -> None:
        """Admit fixed Provider Calls containing one Task each."""
        hooks = cast(_SingleTaskAdmissionHooks, self)
        with self.store.synchronize():
            repository = self.store.execution
            run = repository.get_run(self.execution_run_id)
            counts = repository.active_provider_call_counts(self.execution_run_id)
        selected = self._provider.fixed_call_candidates(
            self.execution_run_id,
            required_node_keys=required,
            describe_task=lambda node, task, rank: TaskDispatchDescriptor(
                node_key=node.node_key,
                node_ordinal=node.ordinal,
                task_key=task.task_key,
                task_ordinal=task.ordinal,
                binding=hooks._binding(node.node_key),
                compatibility_key=hooks._binding(node.node_key).function_name,
                max_tasks_per_call=1,
                depth=rank.depth,
                unblocking_span=rank.unblocking_span,
            ),
            available_total_slots=max(0, run.max_active_provider_calls - counts.total),
            available_gpu_slots=max(0, run.max_active_gpu_provider_calls - counts.gpu),
            now=self._now(),
        )
        for candidate in selected:
            hooks._ensure_publication_claim(candidate.node_key)
        self._provider.submit_provider_calls(
            self.execution_run_id,
            tuple(
                ProviderCallSubmission(
                    candidate=candidate,
                    submission_token=candidate.candidate_key,
                    kwargs=hooks._invocation_kwargs(
                        candidate.node_key,
                        candidate.task_keys[0],
                    ),
                )
                for candidate in selected
            ),
            now=self._now(),
        )


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
