"""Tests for durable Direct CLI App Run storage."""

# ruff: noqa: D101,D102,D107

from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock, Thread
from time import sleep
from types import SimpleNamespace
from uuid import UUID

import pytest

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionPlan,
    NodePlan,
    RunStatus,
)
from biomodals.helper.app_execution import (
    ExecutionCoordinatorLifecycle,
    ExecutionRequestFile,
    ExecutionRunStore,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
REQUEST_FILE = ExecutionRequestFile(
    "example-request.json",
    32,
    "Example execution request",
)


class FakeVolume:
    """Small chunked Volume double for request staging tests."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def read_file(self, path: str):
        selected = self.root / path.lstrip("/")
        if not selected.is_file():
            raise FileNotFoundError(path)
        content = selected.read_bytes()
        midpoint = len(content) // 2
        yield content[:midpoint]
        yield content[midpoint:]

    @contextmanager
    def batch_upload(self, *, force: bool):
        assert force
        root = self.root

        class Batch:
            def put_file(self, source, destination: str) -> None:
                path = root / destination.lstrip("/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source.read())

        yield Batch()


def test_app_execution_store_uses_the_reserved_run_namespace(
    tmp_path: Path,
) -> None:
    """App execution state is isolated from scientific output paths."""
    store = ExecutionRunStore(tmp_path, RUN_ID)

    assert store.state_root == (
        tmp_path / ".biomodals" / "execution" / "runs" / str(RUN_ID)
    )
    assert store.ledger_path == store.state_root / "ledger.sqlite3"
    assert not store.ledger_path.exists()


def test_app_execution_store_persists_the_shared_repository(tmp_path: Path) -> None:
    """Closing and reopening retains one kernel Execution Run."""
    store = ExecutionRunStore(tmp_path, RUN_ID)
    plan = ExecutionPlan("example", (NodePlan("run"),))
    with store.transaction():
        store.execution.create_run(
            execution_run_id=RUN_ID,
            plan=plan,
            deployment=DeploymentIdentity("main", "Example", 3),
            max_active_provider_calls=2,
            max_active_gpu_provider_calls=1,
            now=10,
        )
    store.close()

    reopened = ExecutionRunStore(tmp_path, RUN_ID)
    assert reopened.execution.get_run(RUN_ID).plan == plan
    reopened.close()


def test_app_execution_store_closes_sqlite_during_volume_sync(
    tmp_path: Path,
) -> None:
    """A mounted SQLite file is never open while its Volume is synchronized."""
    store = ExecutionRunStore(tmp_path, RUN_ID)
    original = store.connection

    with store.closed_for_volume_sync():
        with pytest.raises(RuntimeError, match="closed for Volume synchronization"):
            _ = store.connection

    assert store.connection is not original
    store.close()


def test_request_bytes_stage_and_load_idempotently(tmp_path: Path) -> None:
    """Thin clients and mounted coordinators share one immutable byte boundary."""
    path = REQUEST_FILE.path(RUN_ID)
    volume = FakeVolume(tmp_path)

    assert REQUEST_FILE.stage(volume, RUN_ID, b'{"ok":true}') == path
    assert REQUEST_FILE.stage(volume, RUN_ID, b'{"ok":true}') == path
    assert REQUEST_FILE.load(tmp_path, RUN_ID) == b'{"ok":true}'
    assert REQUEST_FILE.load_from_volume(volume, RUN_ID) == b'{"ok":true}'


def test_request_bytes_reject_conflicts_and_oversized_volume_files(
    tmp_path: Path,
) -> None:
    """Existing request bytes remain immutable and bounded while reading."""
    request_file = ExecutionRequestFile("request.json", 16, "Example request")
    path = request_file.path(RUN_ID)
    volume = FakeVolume(tmp_path)
    selected = tmp_path.joinpath(*path.parts)
    selected.parent.mkdir(parents=True)
    selected.write_bytes(b"existing")

    with pytest.raises(RuntimeError, match="conflicts with this run"):
        request_file.stage(volume, RUN_ID, b"diff")
    with pytest.raises(ValueError, match="byte limit"):
        ExecutionRequestFile("request.json", 4, "Example request").load_from_volume(
            volume,
            RUN_ID,
        )


def test_request_bytes_persist_atomically_and_remain_immutable(tmp_path: Path) -> None:
    """Successor coordinators may create, but never replace, request state."""
    path = REQUEST_FILE.path(RUN_ID)

    assert REQUEST_FILE.persist(tmp_path, RUN_ID, b"request") == path
    assert REQUEST_FILE.persist(tmp_path, RUN_ID, b"request") == path
    with pytest.raises(RuntimeError, match="immutable"):
        REQUEST_FILE.persist(tmp_path, RUN_ID, b"changed")


def test_app_coordinator_cancel_does_not_start_a_second_driver(
    tmp_path: Path,
) -> None:
    """Cancellation interleaves with polling but never owns a second drive loop."""

    class FakeRuntime:
        def __init__(self) -> None:
            self.started = Event()
            self.cancel_requested = Event()
            self.status = RunStatus.RUNNING
            self.active_drivers = 0
            self.max_active_drivers = 0
            self.closed_while_driving = False
            self._lock = Lock()

        def snapshot(self):
            return SimpleNamespace(
                run=SimpleNamespace(
                    execution_run_id=RUN_ID,
                    deployment=DeploymentIdentity("main", "Example", 3),
                    status=self.status,
                )
            )

        def run(self, *, synchronize):
            with self._lock:
                self.active_drivers += 1
                self.max_active_drivers = max(
                    self.max_active_drivers,
                    self.active_drivers,
                )
            self.started.set()
            assert self.cancel_requested.wait(timeout=1)
            sleep(0.05)
            with synchronize():
                self.status = RunStatus.CANCELLED
            with self._lock:
                self.active_drivers -= 1
            return self.snapshot()

        def cancel(self):
            self.status = RunStatus.CANCEL_REQUESTED
            self.cancel_requested.set()
            return self.snapshot()

        def close(self) -> None:
            with self._lock:
                self.closed_while_driving |= self.active_drivers > 0

    runtime = FakeRuntime()

    class Coordinator(ExecutionCoordinatorLifecycle):
        def _open_current_runtime(self, *, recover: bool):
            del recover
            self._runtime = runtime
            return runtime

    coordinator = Coordinator(
        execution_run_id=RUN_ID,
        deployment=DeploymentIdentity("main", "Example", 3),
        volume_root=tmp_path,
    )
    errors: list[BaseException] = []

    def call(operation) -> None:
        try:
            operation()
        except BaseException as error:  # pragma: no cover - assertion aid
            errors.append(error)

    run_thread = Thread(target=call, args=(coordinator.run,))
    run_thread.start()
    assert runtime.started.wait(timeout=1)
    cancel_thread = Thread(target=call, args=(coordinator.cancel,))
    cancel_thread.start()
    run_thread.join(timeout=2)
    cancel_thread.join(timeout=2)

    assert not run_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert errors == []
    assert runtime.max_active_drivers == 1
    assert not runtime.closed_while_driving
