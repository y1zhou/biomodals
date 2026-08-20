"""Tests for Rosetta's shared-graph pull-worker adapter."""

# ruff: noqa: D101,D102,D103,D107

from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from biomodals.app.bioinfo.rosetta.execution_contracts import (
    RosettaTaskSpec,
    execute_rosetta_task,
    task_publication_path,
)
from biomodals.app.bioinfo.rosetta.execution_request import RosettaExecutionRequest
from biomodals.app.bioinfo.rosetta.execution_runtime import rosetta_execution_graph
from biomodals.execution import (
    DeploymentIdentity,
    GraphExecutionRunStore,
    ProviderCallStatus,
    RunStatus,
    TaskStatus,
)
from biomodals.execution.definition_plan import execution_plan
from biomodals.execution.definition_runtime import ExecutionGraphRuntime
from biomodals.execution.modal import (
    ExecutionVolumeSync,
    ProviderCallObservation,
    ProviderCallObservationKind,
)
from biomodals.schema import AppRunResult, AppRunStatus

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
    return RosettaExecutionRequest(
        run_name="example",
        run_id="workload",
        tasks=tuple(
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
        ),
        app_version="2025.51",
        max_active_provider_calls=2,
        claim_capacity=2,
        max_parallel_per_worker=2,
    )


def _runtime(
    tmp_path: Path,
    driver: RecordingDriver,
) -> tuple[ExecutionGraphRuntime, FakeVolume]:
    request = _request()
    volume = FakeVolume()
    store = GraphExecutionRunStore(tmp_path, RUN_ID)
    runtime = ExecutionGraphRuntime(
        graph=rosetta_execution_graph(request, output_root=tmp_path),
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        volume_root=tmp_path,
        artifact_volume_name="Rosetta-outputs",
        workload_run_key=request.workload_run_key,
        request=request,
        provider_driver=driver,
        storage_sync=ExecutionVolumeSync(volume=volume, store=store),
        max_active_provider_calls=request.max_active_provider_calls,
        max_active_gpu_provider_calls=0,
        pull_worker_coordinator="coordinator",
        store=store,
        poll_interval_seconds=0,
        now=iter(range(10, 1000)).__next__,
    )
    return runtime, volume


def _run_root(runtime: ExecutionGraphRuntime) -> Path:
    request = cast(RosettaExecutionRequest, runtime.request)
    return runtime.volume_root / request.workload_run_key


def _publish_assignment(
    runtime: ExecutionGraphRuntime,
    assignment: Any,
) -> AppRunResult:
    task = RosettaTaskSpec.from_dict(assignment.execution_payload)
    run_root = _run_root(runtime)
    input_path = run_root / task.pdb
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(f"ATOM {task.index}\n".encode())

    def run_command(command, *, output_mode, log_file):
        del command, output_mode
        Path(log_file).write_text("log\n", encoding="utf-8")
        output = run_root / task.output_dir / "result.pdb"
        output.write_text("ATOM\n", encoding="utf-8")

    execute_rosetta_task(
        run_root=run_root,
        task=task,
        task_fingerprint=assignment.task_fingerprint,
        run_command=run_command,
    )
    return AppRunResult(status=AppRunStatus.SUCCEEDED)


def test_graph_preserves_the_staged_execution_plan(tmp_path: Path) -> None:
    request = _request()

    plan = execution_plan(
        rosetta_execution_graph(request, output_root=tmp_path).validate(),
        workload_run_key=request.workload_run_key,
    )

    assert plan == request.execution_plan


def test_workers_claim_disjoint_microbatches_and_complete_each_task(
    tmp_path: Path,
) -> None:
    driver = RecordingDriver()
    runtime, _volume = _runtime(tmp_path, driver)
    runtime.attach()
    runtime.advance_once()

    calls = runtime.store.execution.list_provider_calls(RUN_ID)
    assert len(calls) == 2
    assert all(
        cast(dict[str, Any], spawn["kwargs"])["coordinator"] == "coordinator"
        for spawn in driver.spawns
    )
    claimed_keys = []
    for ordinal, call in enumerate(calls):
        claim = runtime.claim_pull_tasks(
            call.provider_call_id,
            request_id=f"claim-{ordinal}",
            capacity=2,
        )
        batch_index = 0
        while claim.assignments:
            claimed_keys.extend(assignment.task_key for assignment in claim.assignments)
            claim = runtime.complete_pull_tasks_and_claim(
                call.provider_call_id,
                tuple(
                    (
                        assignment.task_key,
                        f"complete-{assignment.task_key}",
                        _publish_assignment(runtime, assignment),
                    )
                    for assignment in claim.assignments
                ),
                request_id=f"claim-{ordinal}-next-{batch_index}",
                capacity=2,
            )
            batch_index += 1

    assert claimed_keys == ["1", "2", "3"]
    driver.succeeded = True
    runtime.advance_once()
    assert runtime.store.execution.get_run(RUN_ID).status == RunStatus.SUCCEEDED
    assert {
        task.status
        for task in runtime.store.execution.list_tasks(RUN_ID, "rosetta-tasks")
    } == {TaskStatus.SUCCEEDED}
    runtime.close()


def test_fused_completion_replay_uses_its_durable_observation(
    tmp_path: Path,
) -> None:
    runtime, _volume = _runtime(tmp_path, RecordingDriver())
    runtime.attach()
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
    task_publication_path(_run_root(runtime), claim.assignments[0].task_key).unlink()

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
    runtime, volume = _runtime(tmp_path, driver)
    runtime.attach()
    runtime.advance_once()
    for ordinal, call in enumerate(runtime.store.execution.list_provider_calls(RUN_ID)):
        claim = runtime.claim_pull_tasks(
            call.provider_call_id,
            request_id=f"claim-{ordinal}",
            capacity=2,
        )
        for assignment in claim.assignments:
            _publish_assignment(runtime, assignment)

    from biomodals.app.bioinfo.rosetta import execution_runtime as runtime_module

    real_validate = runtime_module.validate_task_publication

    def validate_after_reload(*args, **kwargs):
        return volume.reloads > 0 and real_validate(*args, **kwargs)

    monkeypatch.setattr(
        runtime_module,
        "validate_task_publication",
        validate_after_reload,
    )
    driver.failed = True

    runtime.advance_once()

    assert volume.reloads == 1
    assert {
        task.status
        for task in runtime.store.execution.list_tasks(RUN_ID, "rosetta-tasks")
    } == {TaskStatus.SUCCEEDED}
    assert runtime.store.execution.get_run(RUN_ID).status == RunStatus.SUCCEEDED
    runtime.close()


def test_one_worker_failure_preserves_sibling_success(tmp_path: Path) -> None:
    driver = RecordingDriver()
    runtime, _volume = _runtime(tmp_path, driver)
    runtime.attach()
    runtime.advance_once()

    for ordinal, call in enumerate(runtime.store.execution.list_provider_calls(RUN_ID)):
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
                            AppRunResult(
                                status=AppRunStatus.FAILED,
                                warnings=["Rosetta failed"],
                            )
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
    assert [
        task.status
        for task in runtime.store.execution.list_tasks(RUN_ID, "rosetta-tasks")
    ] == [TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.SUCCEEDED]
    assert runtime.store.execution.get_run(RUN_ID).status == RunStatus.FAILED
    runtime.close()


def test_resume_recovers_publications_owned_by_unknown_workers(tmp_path: Path) -> None:
    driver = RecordingDriver()
    runtime, volume = _runtime(tmp_path, driver)
    runtime.attach()
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
                message="Provider state lookup was inconclusive",
                now=11,
            )
    driver.state_unknown = True

    result = runtime.resume()
    snapshot = runtime.store.execution.snapshot(RUN_ID)

    assert result.status == AppRunStatus.SUCCEEDED
    assert volume.reloads == 1
    assert driver.cancelled == {str(spawn["handle"]) for spawn in driver.spawns}
    assert {call.status for call in snapshot.provider_calls} == {
        ProviderCallStatus.CANCELLED
    }
    assert {task.status for task in snapshot.tasks} == {TaskStatus.SUCCEEDED}
    runtime.close()
