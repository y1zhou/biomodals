"""Caller-driven execution runtime fault-boundary tests."""

# ruff: noqa: D101, D102, D103, D107, S106

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, RLock, Thread

import pytest

from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ExecutionPlan,
    NodePlan,
    NodeStatus,
    ProviderCallStatus,
    RunStatus,
    SqliteExecutionRepository,
)
from biomodals.execution.coordinator import drive_execution_run
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
    ModalDefiniteSubmissionError,
    ModalDeploymentUnavailableError,
    ModalSubmissionOutcomeUnknownError,
)
from biomodals.execution.runtime import ExecutionRuntime, ProviderCallSubmission
from biomodals.execution.scheduler import (
    NodeAdmissionRank,
    ProviderCallCandidate,
    TaskDispatchDescriptor,
)
from biomodals.helper.app_execution import ExecutionRunStore, ExecutionVolumeSync

from .provider_call_helpers import (
    GPU_BINDING,
    RUN_ID,
    create_repository,
    persist_fixed_policy,
    persist_pull_policy,
)


@dataclass
class FakeResolvedFunction:
    name: str


class FakeModalDriver:
    def __init__(self) -> None:
        self.resolve_count = 0
        self.spawn_count = 0
        self.observe_count = 0
        self.cancelled: list[str] = []
        self.spawn_kwargs: list[dict[str, object]] = []
        self.resolve_error: Exception | None = None
        self.spawn_error: Exception | None = None
        self.observation = ModalCallObservation(ModalCallObservationKind.RUNNING)

    def resolve(self, binding):
        self.resolve_count += 1
        if self.resolve_error is not None:
            raise self.resolve_error
        return FakeResolvedFunction(binding.function_name)

    def spawn(self, function, *, args, kwargs):
        self.spawn_count += 1
        self.spawn_kwargs.append(dict(kwargs))
        if self.spawn_error is not None:
            raise self.spawn_error
        return f"fc-{function.name}-{self.spawn_count}"

    def observe(self, provider_call_handle_id):
        self.observe_count += 1
        return self.observation

    def cancel(self, provider_call_handle_id):
        self.cancelled.append(provider_call_handle_id)


def _candidate(task_index: int = 0) -> ProviderCallCandidate:
    return ProviderCallCandidate(
        candidate_key=f"inference:{task_index}",
        node_key="inference",
        node_ordinal=0,
        task_keys=(f"seed-{task_index}",),
        task_ordinal=task_index,
        binding=GPU_BINDING,
        compatibility_key="af3",
        depth=0,
        unblocking_span=0,
        max_tasks_per_call=1,
    )


def _transaction(connection: sqlite3.Connection):
    @contextmanager
    def transaction():
        try:
            yield
        except BaseException:
            connection.rollback()
            raise
        else:
            connection.commit()

    return transaction


def test_runtime_creates_or_verifies_one_run_identity() -> None:
    connection = sqlite3.connect(":memory:")
    repository = SqliteExecutionRepository(connection)
    repository.initialize_schema()
    runtime = ExecutionRuntime(
        repository,
        modal_driver=FakeModalDriver(),
        checkpoint=connection.commit,
        transaction=_transaction(connection),
    )
    plan = ExecutionPlan("demo", (NodePlan("compute"),))
    deployment = DeploymentIdentity("main", "Demo", 7)

    created = runtime.create_or_verify_run(
        execution_run_id=RUN_ID,
        predecessor_execution_run_id=None,
        plan=plan,
        deployment=deployment,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=2,
        now=10,
    )
    recovered = runtime.create_or_verify_run(
        execution_run_id=RUN_ID,
        predecessor_execution_run_id=None,
        plan=plan,
        deployment=deployment,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=2,
        now=11,
    )

    assert recovered == created
    with pytest.raises(ValueError, match="initialization does not match"):
        runtime.create_or_verify_run(
            execution_run_id=RUN_ID,
            predecessor_execution_run_id=None,
            plan=plan,
            deployment=deployment,
            max_active_provider_calls=5,
            max_active_gpu_provider_calls=2,
            now=12,
        )


def test_runtime_owns_result_frontier_recovery() -> None:
    connection = sqlite3.connect(":memory:")
    repository = create_repository(connection=connection, task_count=1)
    runtime = ExecutionRuntime(
        repository,
        modal_driver=FakeModalDriver(),
        checkpoint=connection.commit,
        transaction=_transaction(connection),
    )

    required = runtime.recover_publications(
        RUN_ID,
        observe_node=lambda _node_key: AvailabilityStatus.AVAILABLE,
        observe_task=lambda _node_key, _task: None,
        now=110,
    )

    assert required == ()
    assert repository.get_node(RUN_ID, "inference").status == NodeStatus.SUCCEEDED


def test_runtime_builds_and_limits_fixed_call_candidates() -> None:
    connection = sqlite3.connect(":memory:")
    repository = create_repository(
        connection=connection,
        task_count=3,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
    )
    runtime = ExecutionRuntime(
        repository,
        modal_driver=FakeModalDriver(),
        checkpoint=connection.commit,
        transaction=_transaction(connection),
    )

    def descriptor(node, task, rank: NodeAdmissionRank):
        return TaskDispatchDescriptor(
            node_key=node.node_key,
            node_ordinal=node.ordinal,
            task_key=task.task_key,
            task_ordinal=task.ordinal,
            binding=GPU_BINDING,
            compatibility_key="af3",
            max_tasks_per_call=1,
            depth=rank.depth,
            unblocking_span=rank.unblocking_span,
        )

    candidates = runtime.fixed_call_candidates(
        RUN_ID,
        required_node_keys={"inference"},
        describe_task=descriptor,
        available_total_slots=2,
        available_gpu_slots=1,
        now=110,
    )

    assert [candidate.task_keys for candidate in candidates] == [("seed-0",)]


def test_preclaim_checkpoint_precedes_spawn_and_replay_never_spawns_twice() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()
    checkpoints: list[int] = []
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: checkpoints.append(driver.spawn_count),
    )

    first = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        kwargs={"seed": 0},
        now=110,
    )
    duplicate = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        kwargs={"seed": 0},
        now=111,
    )

    assert first is not None
    assert first.status == ProviderCallStatus.ATTACHED
    assert duplicate == first
    assert checkpoints == [0, 1]
    assert driver.spawn_count == 1


def test_admission_set_batches_resolution_and_volume_checkpoints() -> None:
    repository = create_repository(
        task_count=2,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=2,
    )
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1"),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()
    checkpoints: list[int] = []
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: checkpoints.append(driver.spawn_count),
    )

    calls = runtime.submit_provider_calls(
        RUN_ID,
        tuple(
            ProviderCallSubmission(
                candidate=_candidate(index),
                submission_token=f"batch-{index}",
                kwargs={"seed": index},
            )
            for index in range(2)
        ),
        now=110,
    )

    assert all(call is not None for call in calls)
    assert driver.resolve_count == 1
    assert driver.spawn_count == 2
    assert checkpoints == [0, 2]


def test_modal_operations_never_hold_the_repository_writer() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    writer_active = False
    operations: list[str] = []

    @contextmanager
    def synchronize():
        nonlocal writer_active
        assert not writer_active
        writer_active = True
        try:
            yield
        finally:
            writer_active = False

    class LockCheckingDriver(FakeModalDriver):
        def resolve(self, binding):
            assert not writer_active
            operations.append("resolve")
            return super().resolve(binding)

        def spawn(self, function, *, args, kwargs):
            assert not writer_active
            operations.append("spawn")
            return super().spawn(function, args=args, kwargs=kwargs)

        def observe(self, provider_call_handle_id):
            assert not writer_active
            operations.append("observe")
            return super().observe(provider_call_handle_id)

        def cancel(self, provider_call_handle_id):
            assert not writer_active
            operations.append("cancel")
            return super().cancel(provider_call_handle_id)

    driver = LockCheckingDriver()
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: (
            pytest.fail("checkpoint escaped the writer") if not writer_active else None
        ),
        synchronize=synchronize,
    )
    runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        now=110,
    )
    runtime.cancel_run(RUN_ID, now=111)
    runtime.reconcile_provider_calls(
        RUN_ID,
        required_node_keys={"inference"},
        encode_result=lambda result: result,
        now=112,
    )

    assert operations == ["resolve", "spawn", "cancel", "observe"]


def test_cancellation_during_spawn_cancels_the_attached_call() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    runtime: ExecutionRuntime

    class CancellingDriver(FakeModalDriver):
        def spawn(self, function, *, args, kwargs):
            handle_id = super().spawn(function, args=args, kwargs=kwargs)
            runtime.cancel_run(RUN_ID, now=111)
            return handle_id

    driver = CancellingDriver()
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    )

    call = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        now=110,
    )

    assert call is not None
    assert call.status == ProviderCallStatus.ATTACHED
    assert repository.get_run(RUN_ID).status.value == "cancel_requested"
    assert driver.cancelled == [call.provider_call_handle_id]


def test_terminal_call_set_uses_one_volume_checkpoint() -> None:
    repository = create_repository(
        task_count=2,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=2,
    )
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1"),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()
    checkpoints: list[str] = []
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: checkpoints.append("checkpoint"),
    )
    runtime.submit_provider_calls(
        RUN_ID,
        tuple(
            ProviderCallSubmission(
                candidate=_candidate(index),
                submission_token=f"batch-{index}",
            )
            for index in range(2)
        ),
        now=110,
    )
    checkpoints.clear()
    driver.observation = ModalCallObservation(
        ModalCallObservationKind.SUCCEEDED,
        result={"path": "/outputs/result"},
    )

    reconciled = runtime.reconcile_provider_calls(
        RUN_ID,
        required_node_keys={"inference"},
        encode_result=lambda result: result,
        now=120,
    )

    assert all(
        updated.status == ProviderCallStatus.SUCCEEDED for _, updated in reconciled
    )
    assert driver.observe_count == 2
    assert checkpoints == ["checkpoint"]


def test_checkpoint_may_replace_a_volume_backed_repository(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.sqlite3"
    connection = sqlite3.connect(ledger_path)
    repository = create_repository(connection=connection, task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    connection.commit()
    driver = FakeModalDriver()
    active_connection = connection

    def checkpoint():
        nonlocal active_connection
        active_connection.commit()
        active_connection.close()
        active_connection = sqlite3.connect(ledger_path)
        return type(repository)(active_connection)

    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=checkpoint,
    )

    call = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        now=110,
    )

    assert call is not None
    assert call.status == ProviderCallStatus.ATTACHED
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    assert runtime.repository is not repository


def test_resolution_failure_happens_before_durable_preclaim() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )

    class BrokenResolver(FakeModalDriver):
        def resolve(self, binding):
            raise RuntimeError("deployment unavailable")

    runtime = ExecutionRuntime(
        repository,
        modal_driver=BrokenResolver(),
        checkpoint=lambda: None,
    )

    with pytest.raises(RuntimeError, match="deployment unavailable"):
        runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            kwargs={"seed": 0},
            now=110,
        )

    assert repository.list_provider_calls(RUN_ID) == ()


def test_publication_recovery_reopens_repository_after_concurrent_checkpoint(
    tmp_path,
) -> None:
    """A Volume barrier during observation must not leave a stale SQLite handle."""

    class Volume:
        def commit(self) -> None:
            pass

        def reload(self) -> None:
            pass

    store = ExecutionRunStore(tmp_path, RUN_ID)
    create_repository(connection=store.connection, task_count=1)
    volume_sync = ExecutionVolumeSync(volume=Volume(), store=store)

    def checkpoint():
        volume_sync.commit()
        return store.execution

    runtime = ExecutionRuntime(
        store.execution,
        modal_driver=FakeModalDriver(),
        checkpoint=checkpoint,
        transaction=store.transaction,
        synchronize=store.synchronize,
    )

    def observe_node(_node_key):
        runtime.checkpoint()
        return AvailabilityStatus.AVAILABLE

    required = runtime.recover_publications(
        RUN_ID,
        observe_node=observe_node,
        observe_task=lambda _node_key, _task: None,
        now=110,
    )

    assert required == ()
    assert store.execution.get_node(RUN_ID, "inference").status == NodeStatus.SUCCEEDED
    store.close()


def test_unavailable_exact_deployment_fails_without_a_preclaim() -> None:
    """Conclusive version loss becomes the accepted terminal Run reason."""
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )

    class UnavailableResolver(FakeModalDriver):
        def resolve(self, binding):
            raise ModalDeploymentUnavailableError("version 23 is unavailable")

    runtime = ExecutionRuntime(
        repository,
        modal_driver=UnavailableResolver(),
        checkpoint=lambda: None,
    )

    assert (
        runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            now=110,
        )
        is None
    )

    run = repository.get_run(RUN_ID)
    assert run.status.value == "failed"
    assert run.status_reason is not None
    assert run.status_reason.value == "deployment_unavailable"
    assert repository.list_provider_calls(RUN_ID) == ()


def test_unavailable_deployment_first_drains_attached_calls() -> None:
    """Known child ownership remains observable before the Run fails closed."""
    repository = create_repository(task_count=2)
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1"),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    )
    first = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(0),
        submission_token="first",
        now=110,
    )
    assert first is not None

    driver.resolve_error = ModalDeploymentUnavailableError("version 23 is unavailable")
    assert (
        runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(1),
            submission_token="second",
            now=120,
        )
        is None
    )
    assert repository.get_run(RUN_ID).status.value == "running"
    assert len(repository.list_provider_calls(RUN_ID)) == 1

    repository.fail_provider_call(
        first.provider_call_id,
        message="first call finished unsuccessfully",
        now=130,
    )
    assert (
        runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(1),
            submission_token="second",
            now=140,
        )
        is None
    )
    reason = repository.get_run(RUN_ID).status_reason
    assert reason is not None
    assert reason.value == "deployment_unavailable"


def test_failed_preclaim_checkpoint_never_reaches_spawn_and_recovers_unknown() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()

    def fail_checkpoint():
        raise RuntimeError("checkpoint failed")

    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=fail_checkpoint,
    )

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            kwargs={"seed": 0},
            now=110,
        )

    assert driver.spawn_count == 0
    recovered = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    ).reconcile_provider_call(
        repository.list_provider_calls(RUN_ID)[0].provider_call_id,
        encode_result=lambda value: value,
        now=120,
    )
    assert recovered.status == ProviderCallStatus.OUTCOME_UNKNOWN
    assert driver.spawn_count == 0


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (
            ModalDefiniteSubmissionError("rejected"),
            ProviderCallStatus.FAILED,
        ),
        (
            ModalSubmissionOutcomeUnknownError("response lost"),
            ProviderCallStatus.OUTCOME_UNKNOWN,
        ),
        (
            RuntimeError("unexpected transport failure"),
            ProviderCallStatus.OUTCOME_UNKNOWN,
        ),
    ],
)
def test_spawn_failure_is_durably_classified_without_reauthorization(
    error: Exception,
    expected_status: ProviderCallStatus,
) -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()
    driver.spawn_error = error
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    )

    submitted = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        kwargs={"seed": 0},
        now=110,
    )

    call = repository.list_provider_calls(RUN_ID)[0]
    assert submitted == call
    assert call.status == expected_status
    replay = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        kwargs={"seed": 0},
        now=120,
    )
    assert replay == call
    assert driver.spawn_count == 1


def test_definite_submission_rejection_finishes_run_without_suspension() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()
    driver.spawn_error = ModalDefiniteSubmissionError("rejected")
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    )

    def advance_once() -> None:
        runtime.submit_provider_calls(
            RUN_ID,
            (
                ProviderCallSubmission(
                    candidate=_candidate(),
                    submission_token="batch",
                ),
            ),
            now=110,
        )
        repository.reconcile_node_tasks(RUN_ID, "inference", now=111)
        repository.finalize_run_from_results(RUN_ID, now=112)

    snapshot = drive_execution_run(
        repository,
        RUN_ID,
        advance_once=advance_once,
        checkpoint=lambda: None,
        sleep=lambda _: None,
        poll_interval_seconds=0,
    )

    assert snapshot.run.status == RunStatus.FAILED


def test_cancellation_stops_the_rest_of_a_multi_call_admission_set() -> None:
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    repository = create_repository(
        connection=connection,
        task_count=3,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=3,
    )
    persist_fixed_policy(
        repository,
        ("seed-0", "seed-1", "seed-2"),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    first_spawn_started = Event()
    release_first_spawn = Event()
    writer = RLock()

    class BlockingDriver(FakeModalDriver):
        def spawn(self, function, *, args, kwargs):
            handle_id = super().spawn(function, args=args, kwargs=kwargs)
            if self.spawn_count == 1:
                first_spawn_started.set()
                assert release_first_spawn.wait(timeout=5)
            return handle_id

    driver = BlockingDriver()
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=connection.commit,
        transaction=_transaction(connection),
        synchronize=lambda: writer,
    )
    submissions = tuple(
        ProviderCallSubmission(
            candidate=_candidate(index),
            submission_token=f"batch-{index}",
        )
        for index in range(3)
    )
    failure: list[BaseException] = []

    def submit() -> None:
        try:
            runtime.submit_provider_calls(RUN_ID, submissions, now=110)
        except BaseException as error:  # pragma: no cover - assertion aid
            failure.append(error)

    thread = Thread(target=submit)
    thread.start()
    assert first_spawn_started.wait(timeout=5)
    runtime.cancel_run(RUN_ID, now=120)
    release_first_spawn.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert failure == []
    assert driver.spawn_count == 1
    assert driver.cancelled == [f"fc-{GPU_BINDING.function_name}-1"]
    assert [call.status for call in repository.list_provider_calls(RUN_ID)] == [
        ProviderCallStatus.ATTACHED,
        ProviderCallStatus.CANCELLED,
        ProviderCallStatus.CANCELLED,
    ]


def test_unattached_returned_call_is_cancelled_and_checkpointed_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()
    checkpoints: list[ProviderCallStatus] = []
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: checkpoints.append(
            repository.list_provider_calls(RUN_ID)[0].status
        ),
    )

    def fail_attachment(*args, **kwargs):
        raise RuntimeError("attachment failed")

    monkeypatch.setattr(repository, "attach_provider_call", fail_attachment)

    with pytest.raises(RuntimeError, match="attachment failed"):
        runtime.submit_fixed_batch(
            RUN_ID,
            _candidate(),
            submission_token="batch",
            now=110,
        )

    call = repository.list_provider_calls(RUN_ID)[0]
    assert call.status == ProviderCallStatus.OUTCOME_UNKNOWN
    assert driver.cancelled == [f"fc-{GPU_BINDING.function_name}-1"]
    assert checkpoints == [
        ProviderCallStatus.SUBMITTING,
        ProviderCallStatus.OUTCOME_UNKNOWN,
    ]


def test_recovery_collects_attached_call_once_then_replays_durable_envelope() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    )
    call = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        kwargs={"seed": 0},
        now=110,
    )
    assert call is not None
    driver.observation = ModalCallObservation(
        ModalCallObservationKind.SUCCEEDED,
        result={"path": "/outputs/seed-0"},
    )

    completed = runtime.reconcile_provider_call(
        call.provider_call_id,
        encode_result=lambda result: {"tasks": {"seed-0": result}},
        now=120,
    )
    replay = runtime.reconcile_provider_call(
        call.provider_call_id,
        encode_result=lambda result: {"tasks": {"seed-0": result}},
        now=121,
    )

    assert completed.status == ProviderCallStatus.SUCCEEDED
    assert replay == completed
    assert driver.observe_count == 1


def test_running_poll_does_not_cross_host_checkpoint() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()
    checkpoints: list[str] = []
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: checkpoints.append("checkpoint"),
    )
    call = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch",
        now=110,
    )
    assert call is not None
    checkpoints.clear()

    running = runtime.reconcile_provider_call(
        call.provider_call_id,
        encode_result=lambda result: result,
        now=120,
    )

    assert running.status == ProviderCallStatus.RUNNING
    assert checkpoints == []


def test_pull_worker_submission_receives_its_durable_call_identity() -> None:
    repository = create_repository(task_count=2)
    persist_pull_policy(
        repository,
        binding=GPU_BINDING,
        compatibility_key="af3",
        claim_capacity=2,
    )
    driver = FakeModalDriver()
    checkpoints: list[str] = []
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: checkpoints.append("checkpoint"),
    )

    call = runtime.submit_pull_worker(
        RUN_ID,
        node_key="inference",
        submission_token="worker-0",
        binding=GPU_BINDING,
        compatibility_key="af3",
        claim_capacity=2,
        kwargs={"coordinator": "run-pool"},
        now=110,
    )

    assert call is not None
    assert call.status == ProviderCallStatus.ATTACHED
    assert driver.spawn_kwargs == [
        {
            "coordinator": "run-pool",
            "provider_call_id": str(call.provider_call_id),
        }
    ]
    assert checkpoints == ["checkpoint", "checkpoint"]


def test_fixed_submission_can_receive_its_durable_call_identity() -> None:
    repository = create_repository(task_count=1)
    persist_fixed_policy(
        repository,
        ("seed-0",),
        binding=GPU_BINDING,
        compatibility_key="af3",
    )
    driver = FakeModalDriver()
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: None,
    )

    call = runtime.submit_fixed_batch(
        RUN_ID,
        _candidate(),
        submission_token="batch-with-owner",
        kwargs={"seed": 0},
        provider_call_id_kwarg="claim_owner",
        now=110,
    )

    assert call is not None
    assert driver.spawn_kwargs == [
        {
            "seed": 0,
            "claim_owner": str(call.provider_call_id),
        }
    ]


def test_pull_claim_and_completion_cross_checkpoint_before_return() -> None:
    repository = create_repository(task_count=1)
    persist_pull_policy(
        repository,
        binding=GPU_BINDING,
        compatibility_key="af3",
        claim_capacity=1,
    )
    driver = FakeModalDriver()
    checkpoints: list[str] = []
    runtime = ExecutionRuntime(
        repository,
        modal_driver=driver,
        checkpoint=lambda: checkpoints.append("checkpoint"),
    )
    call = runtime.submit_pull_worker(
        RUN_ID,
        node_key="inference",
        submission_token="worker-0",
        binding=GPU_BINDING,
        compatibility_key="af3",
        claim_capacity=1,
        now=110,
    )
    assert call is not None

    claim = runtime.claim_pull_tasks(
        call.provider_call_id,
        request_id="claim-0",
        capacity=1,
        now=111,
    )
    completed = runtime.record_pull_task_completion(
        call.provider_call_id,
        "seed-0",
        request_id="complete-0",
        observation=AvailabilityStatus.AVAILABLE,
        now=112,
    )

    assert [assignment.task_key for assignment in claim.assignments] == ["seed-0"]
    assert completed.status.value == "succeeded"
    assert checkpoints == ["checkpoint"] * 4
