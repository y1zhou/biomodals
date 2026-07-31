"""Tests for direct Rosetta SQLite pull-worker scheduling."""

# ruff: noqa: D101,D102,D103,D107

from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from biomodals.app.bioinfo.rosetta.execution_contracts import (
    RosettaTaskSpec,
    execute_rosetta_task,
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
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.helper.app_execution import AppExecutionRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "Rosetta", 7)


class FakeVolume:
    def commit(self) -> None:
        pass

    def reload(self) -> None:
        pass


class RecordingDriver:
    def __init__(self) -> None:
        self.spawns: list[dict[str, object]] = []
        self.succeeded = False
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
            return ModalCallObservation(ModalCallObservationKind.CANCELLED)
        if self.state_unknown:
            return ModalCallObservation(ModalCallObservationKind.STATE_UNKNOWN)
        if self.succeeded:
            return ModalCallObservation(
                ModalCallObservationKind.SUCCEEDED,
                result={"claimed_tasks": 1, "claim_requests": 2},
            )
        return ModalCallObservation(ModalCallObservationKind.RUNNING)

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
) -> RosettaExecutionRuntime:
    return RosettaExecutionRuntime(
        request=_request(),
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=AppExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=FakeVolume(),
        output_root=tmp_path,
        pull_worker_coordinator="coordinator",
        poll_interval_seconds=0,
        now=lambda: 10,
    )


def _publish_assignment(runtime, assignment) -> dict[str, object]:
    task = RosettaTaskSpec.from_dict(assignment.execution_payload)

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
    claimed_keys = []
    for ordinal, call in enumerate(calls):
        claim = runtime.claim_pull_tasks(
            call.provider_call_id,
            request_id=f"claim-{ordinal}",
            capacity=2,
        )
        for assignment in claim.assignments:
            claimed_keys.append(assignment.task_key)
            result = _publish_assignment(runtime, assignment)
            runtime.complete_pull_task(
                call.provider_call_id,
                assignment.task_key,
                request_id=f"complete-{assignment.task_key}",
                result=result,
            )

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
        for assignment in claim.assignments:
            result = (
                {
                    "status": "failed",
                    "task_key": assignment.task_key,
                    "error": "Rosetta failed",
                }
                if assignment.task_key == "2"
                else _publish_assignment(runtime, assignment)
            )
            runtime.complete_pull_task(
                call.provider_call_id,
                assignment.task_key,
                request_id=f"complete-{assignment.task_key}",
                result=result,
            )

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
        runtime.store.execution.mark_provider_call_state_unknown(
            call.provider_call_id,
            message="Modal state lookup was inconclusive",
            now=11,
        )
    driver.state_unknown = True

    snapshot = runtime.resume()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert driver.cancelled == {str(spawn["handle"]) for spawn in driver.spawns}
    assert {call.status for call in snapshot.provider_calls} == {
        ProviderCallStatus.CANCELLED
    }
    assert {task.status for task in snapshot.tasks} == {TaskStatus.SUCCEEDED}
    runtime.close()
