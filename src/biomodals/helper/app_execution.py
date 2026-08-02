"""Physical state shared by remotely coordinated execution adapters."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from threading import Lock, RLock
from typing import Any, Protocol
from uuid import UUID

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionRunNotFoundError,
    ExecutionRunRecord,
    ExecutionRuntime,
    ExecutionSnapshot,
    SqliteExecutionRepository,
    drive_execution_run,
    resume_execution_run,
)

LEDGER_FILENAME = "ledger.sqlite3"


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
        existing = self._read_volume(output_volume, path)
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
        content = self._read_volume(output_volume, path)
        if content is None:
            raise FileNotFoundError(path.as_posix())
        self._validate(content)
        return content

    def _read_volume(
        self,
        output_volume: Any,
        path: PurePosixPath,
    ) -> bytes | None:
        content = bytearray()
        try:
            for chunk in output_volume.read_file(path.as_posix()):
                if not isinstance(chunk, bytes):
                    raise TypeError(f"Volume returned non-bytes for {path}")
                if len(content) + len(chunk) > self.max_bytes:
                    raise ValueError(f"Volume file exceeds its byte limit: {path}")
                content.extend(chunk)
        except FileNotFoundError:
            return None
        return bytes(content)

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
    content = _LAUNCH_FILE.load(volume_root, execution_run_id)
    if content == b"root":
        return None
    try:
        predecessor = content.decode("ascii")
        parsed = UUID(predecessor)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("Execution launch predecessor is invalid") from error
    if str(parsed) != predecessor:
        raise ValueError("Execution launch predecessor is not canonical")
    return parsed


def _execution_launch_bytes(predecessor_execution_run_id: UUID | None) -> bytes:
    return (
        b"root"
        if predecessor_execution_run_id is None
        else str(predecessor_execution_run_id).encode()
    )


class ExecutionRunStore:
    """Own one Run's kernel connection and reserved state path."""

    def __init__(self, volume_root: str | Path, execution_run_id: UUID) -> None:
        """Bind storage only to the host Volume root and opaque Run ID."""
        self.volume_root = Path(volume_root)
        self.execution_run_id = execution_run_id
        self._connection: sqlite3.Connection | None = None
        self._execution: SqliteExecutionRepository | None = None
        self._lock = RLock()
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


class ExecutionVolume(Protocol):
    """Minimal Modal Volume boundary used by execution hosts."""

    def commit(self) -> object:
        """Persist pending writes."""

    def reload(self) -> object:
        """Refresh writes made by other containers."""


class ExecutionVolumeSync:
    """Close a Run store while synchronizing its backing Volume."""

    def __init__(
        self,
        *,
        volume: ExecutionVolume | None,
        store: ExecutionRunStore,
    ) -> None:
        """Bind one optional Volume to its closeable local Run store."""
        self.volume = volume
        self.store = store
        self._lock = RLock()

    def commit(self) -> None:
        """Persist pending Volume writes when a Volume is attached."""
        with self._lock:
            with self.store.closed_for_volume_sync():
                if self.volume is not None:
                    self.volume.commit()

    def reload(self) -> None:
        """Refresh the Volume view when a Volume is attached."""
        if self.volume is None:
            return
        with self._lock:
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
        repository = self.store.execution
        try:
            repository.get_run(self.execution_run_id)
        except ExecutionRunNotFoundError:
            repository = self._initialize()
        self._provider.repository = repository
        self._provider.cancel_run(self.execution_run_id, now=self._now())
        return self.store.execution.snapshot(self.execution_run_id)

    def close(self) -> None:
        """Close SQLite without cancelling attached Provider Calls."""
        self.store.close()

    def advance_once(self) -> None:
        """Apply one workload-owned execution cycle."""
        raise NotImplementedError

    def _initialize(self) -> SqliteExecutionRepository:
        raise NotImplementedError

    def _checkpoint(self) -> SqliteExecutionRepository:
        self._volume_sync.commit()
        repository = self.store.execution
        self._provider.repository = repository
        return repository

    def _reload_output(self) -> None:
        self._volume_sync.reload()
        self._provider.repository = self.store.execution


class ExecutionCoordinatorLifecycle:
    """Share app-coordinator locking, status, and drive mechanics."""

    _request_loader: Callable[[str | Path, UUID], Any]

    def __init__(
        self,
        *,
        execution_run_id: UUID,
        deployment: DeploymentIdentity,
        volume_root: str | Path,
        target_scientific_versions: Mapping[str, str] | None = None,
    ) -> None:
        """Bind the lifecycle to one Run and exact deployment."""
        self.execution_run_id = execution_run_id
        self.deployment = deployment
        self.volume_root = Path(volume_root)
        self.target_scientific_versions = dict(target_scientific_versions or {})
        self._writer_lock = RLock()
        self._drive_lock = Lock()
        self._runtime: Any | None = None

    def run(self) -> ExecutionSnapshot:
        """Load the staged request and drive one root Run."""
        with self._drive_lock:
            with self._writer_lock:
                runtime = self._open_current_runtime(recover=False)
            return self._drive(runtime, resume=False)

    def drive_prepared(self) -> ExecutionSnapshot:
        """Drive a prepared root or Successor Run from immutable launch state."""
        with self._drive_lock:
            with self._writer_lock:
                runtime = self._open_current_runtime(recover=True)
            return self._drive(runtime, resume=False)

    def cancel(self) -> ExecutionSnapshot:
        """Request cancellation and reconcile it to a terminal result."""
        with self._writer_lock:
            runtime = self._open_current_runtime(recover=True)
            self._verify_snapshot(runtime.cancel())
        with self._drive_lock:
            with self._writer_lock:
                runtime = self._open_current_runtime(recover=True)
                snapshot = runtime.store.execution.snapshot(self.execution_run_id)
                self._verify_snapshot(snapshot)
                if snapshot.run.status.is_terminal:
                    self._close_runtime()
                    return snapshot
            return self._drive(runtime, resume=False)

    def resume(self) -> ExecutionSnapshot:
        """Resume this Run without retrying conclusive failures."""
        with self._drive_lock:
            with self._writer_lock:
                runtime = self._open_current_runtime(recover=True)
            return self._drive(runtime, resume=True)

    def status(self) -> ExecutionSnapshot:
        """Read one verified snapshot without advancing work."""
        with self._writer_lock:
            runtime = self._runtime
            if runtime is not None:
                snapshot = runtime.store.execution.snapshot(self.execution_run_id)
            else:
                store = self._run_store()
                if not store.ledger_path.is_file():
                    raise ExecutionRunNotFoundError(str(self.execution_run_id))
                try:
                    snapshot = store.execution.snapshot(self.execution_run_id)
                finally:
                    store.close()
            self._verify_snapshot(snapshot)
            return snapshot

    def close(self) -> None:
        """Close coordinator-local state without cancelling Provider Calls."""
        with self._drive_lock:
            with self._writer_lock:
                self._close_runtime()

    def synchronize(self) -> AbstractContextManager[object]:
        """Return the single-writer boundary used between drive cycles."""
        return self._writer_lock

    def _drive(self, runtime: Any, *, resume: bool) -> ExecutionSnapshot:
        try:
            snapshot = (
                runtime.resume(synchronize=self.synchronize)
                if resume
                else runtime.run(synchronize=self.synchronize)
            )
            self._verify_snapshot(snapshot)
            return snapshot
        finally:
            with self._writer_lock:
                self._close_runtime()

    def _run_store(self) -> ExecutionRunStore:
        return ExecutionRunStore(self.volume_root, self.execution_run_id)

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
            if any(
                predecessor.plan.scientific_versions.get(name) != version
                for name, version in self.target_scientific_versions.items()
            ):
                raise ValueError(
                    "Target deployment changed declared scientific versions"
                )
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

    def _open_current_runtime(self, *, recover: bool) -> Any:
        request = self._request_loader(self.volume_root, self.execution_run_id)
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

    def _verify_snapshot(self, snapshot: ExecutionSnapshot) -> None:
        if snapshot.run.execution_run_id != self.execution_run_id:
            raise ValueError("Execution Run ID does not match coordinator")
        if snapshot.run.deployment != self.deployment:
            raise ValueError("Deployment Identity does not match Execution Run")

    def _close_runtime(self) -> None:
        runtime = self._runtime
        if runtime is not None:
            runtime.close()
            self._runtime = None
