"""Tests for direct Rosetta SQLite pull-worker scheduling."""

# ruff: noqa: D101,D102,D103,D107

from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, cast
from uuid import UUID

from biomodals.app.bioinfo.rosetta.execution_contracts import (
    RosettaTaskSpec,
    execute_rosetta_task,
    task_publication_path,
)
from biomodals.app.bioinfo.rosetta.execution_request import (
    RosettaExecutionRequest,
)
from biomodals.app.bioinfo.rosetta.execution_runtime import (
    RosettaExecutionRuntime,
)
from biomodals.execution import (
    DeploymentIdentity,
    ProviderCallStatus,
    RunStatus,
    TaskStatus,
)
from biomodals.execution.modal import (
    ExecutionRunStore,
    ProviderCallObservation,
    ProviderCallObservationKind,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "Rosetta", 7)


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


class RecordingDriver:
    def __init__(self) -> None:
        self.spawns: list[dict[str, object]] = []
        self.succeeded = False
        self.failed = False
        self.state_unknown = False
        self.cancelled: set[str] = set()

    def resolve(self, binding):
        return binding

    def spawn(self, function, *, args, kwargs):
        handle = f"fc-{len(self.spawns) + 1}"
        self.spawns.append({
            "function": function,
            "args": args,
            "kwargs": kwargs,
            "handle": handle,
        })
        return handle

    def observe(self, provider_call_handle_id: str):
        if provider_call_handle_id in self.cancelled:
            return ProviderCallObservation(ProviderCallObservationKind.CANCELLED)
        if self.state_unknown:
            return ProviderCallObservation(ProviderCallObservationKind.STATE_UNKNOWN)
        if self.failed:
            return ProviderCallObservation(
                ProviderCallObservationKind.FAILED,
                message="completion RPC connection lost",
            )
        if self.succeeded:
            return ProviderCallObservation(
                ProviderCallObservationKind.SUCCEEDED,
                result={"claimed_tasks": 1, "claim_requests": 2},
            )
        return ProviderCallObservation(ProviderCallObservationKind.RUNNING)

    def cancel(self, provider_call_handle_id: str) -> None:
        self.cancelled.add(provider_call_handle_id)


def _request() -> RosettaExecutionRequest:
    tasks = tuple(
        RosettaTaskSpec(
            task_key=str(index),
            index=index,
            binary="relax",
            pdb=f"inputs/{index}/input.pdb",
            rosetta_script=None,
            flags_file=None,
            output_dir=f"outputs/{index}",
            worker_log=f"logs/{index}.log",
            expected_files=(),
            input_sha256=sha256(f"ATOM {index}\n".encode()).hexdigest(),
        )
        for index in range(1, 4)
    )
    return RosettaExecutionRequest(
        run_name="example",
        run_id="workload",
        tasks=tasks,
        app_version="2025.51",
        max_active_provider_calls=2,
        claim_capacity=2,
        max_parallel_per_worker=2,
    )


def _runtime(
    tmp_path: Path,
    driver: RecordingDriver,
    *,
    volume_io_lock: Any | None = None,
) -> RosettaExecutionRuntime:
    return RosettaExecutionRuntime(
        request=_request(),
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        provider_driver=driver,
        output_volume=FakeVolume(),
        output_root=tmp_path,
        pull_worker_coordinator="coordinator",
        poll_interval_seconds=0,
        now=lambda: 10,
        volume_io_lock=volume_io_lock,
    )


def test_volume_barriers_own_the_run_scoped_lock(tmp_path: Path) -> None:
    volume_io_lock = RLock()
    runtime = _runtime(
        tmp_path,
        RecordingDriver(),
        volume_io_lock=volume_io_lock,
    )
    output = cast(FakeVolume, runtime.output_volume)
    real_commit = output.commit
    real_reload = output.reload

    def commit() -> None:
        assert volume_io_lock._is_owned()
        real_commit()

    def reload() -> None:
        assert volume_io_lock._is_owned()
        real_reload()

    output.commit = commit
    output.reload = reload

    runtime._initialize()
    runtime.advance_once()
    runtime.refresh_publications()
    runtime.close()


def _publish_assignment(runtime, assignment) -> dict[str, object]:
    task = RosettaTaskSpec.from_dict(assignment.execution_payload)
    input_path = runtime.run_root / task.pdb
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(f"ATOM {task.index}\n".encode())

    def run_command(command, *, output_mode, log_file):
        del command, output_mode
        Path(log_file).write_text("log\n", encoding="utf-8")
        output = runtime.run_root / task.output_dir / "result.pdb"
        output.write_text("ATOM\n", encoding="utf-8")

    return execute_rosetta_task(
        run_root=runtime.run_root,
        task=task,
        task_fingerprint=assignment.task_fingerprint,
        run_command=run_command,
    )


def test_initialization_reuses_the_host_volume_view(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, RecordingDriver())

    runtime._initialize()

    output = cast(FakeVolume, runtime.output_volume)
    assert output.reloads == 0
    assert output.commits == 0
    runtime.attach()
    runtime.attach()
    assert output.reloads == 0
    runtime.refresh_publications()
    assert output.reloads == 1
    runtime.close()


def test_running_provider_poll_does_not_synchronize_the_output_volume(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, RecordingDriver())
    runtime._initialize()
    runtime.advance_once()
    output = cast(FakeVolume, runtime.output_volume)
    commits = output.commits
    reloads = output.reloads

    runtime.advance_once()

    assert output.commits == commits
    assert output.reloads == reloads
    runtime.close()


def test_workers_claim_disjoint_microbatches_and_complete_each_task(
    tmp_path: Path,
) -> None:
    driver = RecordingDriver()
    runtime = _runtime(tmp_path, driver)
    runtime._initialize()
    runtime.advance_once()

    calls = runtime.store.execution.list_provider_calls(RUN_ID)
    assert len(calls) == 2
    for spawn in driver.spawns:
        kwargs = cast(dict[str, Any], spawn["kwargs"])
        assert kwargs["coordinator"] == "coordinator"
    claims = [
        runtime.claim_pull_tasks(
            call.provider_call_id,
            request_id=f"claim-{ordinal}",
            capacity=2,
        )
        for ordinal, call in enumerate(calls)
    ]
    claimed_keys = []
    for ordinal, (call, claim) in enumerate(zip(calls, claims, strict=True)):
        completions = []
        for assignment in claim.assignments:
            claimed_keys.append(assignment.task_key)
            result = _publish_assignment(runtime, assignment)
            completions.append((
                assignment.task_key,
                f"complete-{assignment.task_key}",
                result,
            ))
        output = cast(FakeVolume, runtime.output_volume)
        commits = output.commits
        next_claim = runtime.complete_pull_tasks_and_claim(
            call.provider_call_id,
            tuple(completions),
            request_id=f"claim-{ordinal}-next",
            capacity=2,
        )
        assert output.commits == commits + 1
        assert next_claim.assignments == ()

    assert claimed_keys == ["1", "2", "3"]
    driver.succeeded = True
    runtime.advance_once()
    assert runtime.store.execution.get_run(RUN_ID).status == RunStatus.SUCCEEDED
    assert {
        task.status
        for task in runtime.store.execution.list_tasks(
            RUN_ID,
            "rosetta-tasks",
        )
    } == {TaskStatus.SUCCEEDED}
    runtime.close()


def test_fused_completion_replay_uses_its_durable_observations(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path, RecordingDriver())
    runtime._initialize()
    runtime.advance_once()
    [call, _other_call] = runtime.store.execution.list_provider_calls(RUN_ID)
    claim = runtime.claim_pull_tasks(
        call.provider_call_id,
        request_id="claim",
        capacity=2,
    )
    completions = tuple(
        (
            assignment.task_key,
            f"complete-{assignment.task_key}",
            _publish_assignment(runtime, assignment),
        )
        for assignment in claim.assignments
    )
    next_claim = runtime.complete_pull_tasks_and_claim(
        call.provider_call_id,
        completions,
        request_id="claim-next",
        capacity=2,
    )
    task_publication_path(
        runtime.run_root,
        claim.assignments[0].task_key,
    ).unlink()

    replay = runtime.complete_pull_tasks_and_claim(
        call.provider_call_id,
        completions,
        request_id="claim-next",
        capacity=2,
    )

    assert replay == next_claim
    runtime.close()


def test_terminal_worker_recovers_committed_outputs_after_lost_callback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    driver = RecordingDriver()
    runtime = _runtime(tmp_path, driver)
    runtime._initialize()
    runtime.advance_once()
    calls = runtime.store.execution.list_provider_calls(RUN_ID)
    for ordinal, call in enumerate(calls):
        claim = runtime.claim_pull_tasks(
            call.provider_call_id,
            request_id=f"claim-{ordinal}",
            capacity=2,
        )
        for assignment in claim.assignments:
            _publish_assignment(runtime, assignment)

    output = cast(FakeVolume, runtime.output_volume)
    from biomodals.app.bioinfo.rosetta import execution_runtime as runtime_module

    real_validate = runtime_module.validate_task_publication

    def validate_after_reload(*args, **kwargs):
        return output.reloads > 0 and real_validate(*args, **kwargs)

    monkeypatch.setattr(
        runtime_module,
        "validate_task_publication",
        validate_after_reload,
    )
    driver.failed = True

    runtime.advance_once()

    assert output.reloads == 1
    assert {
        task.status
        for task in runtime.store.execution.list_tasks(RUN_ID, "rosetta-tasks")
    } == {TaskStatus.SUCCEEDED}
    assert runtime.store.execution.get_run(RUN_ID).status == RunStatus.SUCCEEDED
    runtime.close()


def test_one_worker_failure_is_recorded_without_losing_sibling_success(
    tmp_path: Path,
) -> None:
    driver = RecordingDriver()
    runtime = _runtime(tmp_path, driver)
    runtime._initialize()
    runtime.advance_once()
    calls = runtime.store.execution.list_provider_calls(RUN_ID)

    for ordinal, call in enumerate(calls):
        claim = runtime.claim_pull_tasks(
            call.provider_call_id,
            request_id=f"claim-{ordinal}",
            capacity=2,
        )
        batch_index = 0
        while claim.assignments:
            claim = runtime.complete_pull_tasks_and_claim(
                call.provider_call_id,
                tuple(
                    (
                        assignment.task_key,
                        f"complete-{assignment.task_key}",
                        (
                            {
                                "status": "failed",
                                "task_key": assignment.task_key,
                                "error": "Rosetta failed",
                            }
                            if assignment.task_key == "2"
                            else _publish_assignment(runtime, assignment)
                        ),
                    )
                    for assignment in claim.assignments
                ),
                request_id=f"claim-{ordinal}-next-{batch_index}",
                capacity=2,
            )
            batch_index += 1

    driver.succeeded = True
    runtime.advance_once()
    tasks = runtime.store.execution.list_tasks(RUN_ID, "rosetta-tasks")
    assert [task.status for task in tasks] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.SUCCEEDED,
    ]
    assert runtime.store.execution.get_run(RUN_ID).status == RunStatus.FAILED
    runtime.close()


def test_cancel_requested_run_reconciles_worker_cancellation(
    tmp_path: Path,
) -> None:
    driver = RecordingDriver()
    runtime = _runtime(tmp_path, driver)
    runtime._initialize()
    runtime.advance_once()

    requested = runtime.cancel()
    assert requested.run.status == RunStatus.CANCEL_REQUESTED

    runtime.advance_once()

    snapshot = runtime.store.execution.snapshot(RUN_ID)
    assert snapshot.run.status == RunStatus.CANCELLED
    assert {call.status for call in snapshot.provider_calls} == {
        ProviderCallStatus.CANCELLED
    }
    runtime.close()


def test_unknown_run_prunes_workers_after_task_publications_appear(
    tmp_path: Path,
) -> None:
    driver = RecordingDriver()
    runtime = _runtime(tmp_path, driver)
    runtime._initialize()
    runtime.advance_once()
    calls = runtime.store.execution.list_provider_calls(RUN_ID)
    for ordinal, call in enumerate(calls):
        claim = runtime.claim_pull_tasks(
            call.provider_call_id,
            request_id=f"claim-{ordinal}",
            capacity=2,
        )
        for assignment in claim.assignments:
            _publish_assignment(runtime, assignment)
    for call in calls:
        with runtime.store.transaction():
            runtime.store.execution.mark_provider_call_state_unknown(
                call.provider_call_id,
                message="Modal state lookup was inconclusive",
                now=11,
            )
    driver.state_unknown = True

    overview = runtime.resume()
    snapshot = runtime.store.execution.snapshot(RUN_ID)

    assert overview.run.status == RunStatus.SUCCEEDED
    assert driver.cancelled == {str(spawn["handle"]) for spawn in driver.spawns}
    assert {call.status for call in snapshot.provider_calls} == {
        ProviderCallStatus.CANCELLED
    }
    assert {task.status for task in snapshot.tasks} == {TaskStatus.SUCCEEDED}
    runtime.close()
