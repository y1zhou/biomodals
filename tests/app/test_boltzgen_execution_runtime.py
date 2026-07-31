"""Tests for BoltzGen's direct-fan-out execution adapter."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from biomodals.app.design.boltzgen.execution_contracts import (
    boltzgen_run_root,
    write_boltzgen_task_publication,
    write_collection_publication,
)
from biomodals.app.design.boltzgen.execution_request import (
    DESIGN_RUNS_NODE,
    BoltzGenExecutionRequest,
    prepare_execution_request,
)
from biomodals.app.design.boltzgen.execution_runtime import (
    BoltzGenExecutionRuntime,
)
from biomodals.execution import (
    DeploymentIdentity,
    ProviderCallStatus,
    RunStatus,
    TaskPlan,
)
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.helper.app_execution import ExecutionRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
DEPLOYMENT = DeploymentIdentity("main", "BoltzGen", 7)


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


class RecordingCallDriver:
    def __init__(self) -> None:
        self.spawns: list[dict[str, object]] = []
        self.cancelled: set[str] = set()
        self.state_unknown = False
        self.succeeded = False
        self.on_success: Any = None

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
            if self.on_success is not None:
                self.on_success()
            return ModalCallObservation(
                ModalCallObservationKind.SUCCEEDED,
                result={"path": "result"},
            )
        return ModalCallObservation(ModalCallObservationKind.RUNNING)

    def cancel(self, provider_call_handle_id: str) -> None:
        self.cancelled.add(provider_call_handle_id)


def _request(
    *,
    run_ids: tuple[str, ...] = ("run-a", "run-b"),
    protocol: str = "nanobody-anything",
) -> BoltzGenExecutionRequest:
    return prepare_execution_request(
        run_name="example",
        run_ids=run_ids,
        yaml_content=b"name: example\n",
        additional_files={},
        protocol=protocol,
        num_designs=10,
        budget=5,
        steps=None,
        extra_args=None,
        filter_results=False,
        filter_rmsd_threshold=2.5,
        app_version="0.3.2",
        repo_commit_hash="abc123",
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=2,
    )


def _runtime(
    tmp_path: Path,
    *,
    request: BoltzGenExecutionRequest | None = None,
    driver: object | None = None,
) -> BoltzGenExecutionRuntime:
    return BoltzGenExecutionRuntime(
        request=request or _request(),
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver or RecordingCallDriver(),
        output_volume=FakeVolume(),
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 10,
    )


def _task_fingerprint(request: BoltzGenExecutionRequest, run_id: str) -> str:
    return TaskPlan(
        task_key=run_id,
        scientific_payload={"run_id": run_id},
    ).fingerprint(
        workload_plan_fingerprint=(request.execution_plan.workload_plan_fingerprint),
        node_key=DESIGN_RUNS_NODE,
    )


def _publish_run(
    tmp_path: Path,
    request: BoltzGenExecutionRequest,
    run_id: str,
) -> None:
    final = boltzgen_run_root(tmp_path, "example", run_id) / "final_ranked_designs"
    final.mkdir(parents=True, exist_ok=True)
    (final / "results_overview.pdf").write_bytes(b"pdf")
    write_boltzgen_task_publication(
        final.parent,
        task_fingerprint=_task_fingerprint(request, run_id),
    )


def test_initialization_does_not_checkpoint_the_output_volume(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    runtime._initialize()

    output = cast(FakeVolume, runtime.output_volume)
    assert output.reloads == 1
    assert output.commits == 0
    runtime.close()


def test_running_provider_poll_does_not_synchronize_the_output_volume(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    runtime._initialize()
    runtime.advance_once()
    output = cast(FakeVolume, runtime.output_volume)
    commits = output.commits
    reloads = output.reloads

    runtime.advance_once()

    assert output.commits == commits
    assert output.reloads == reloads
    runtime.close()


def test_provider_publication_is_reloaded_after_success_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(run_ids=("run-a",))
    driver = RecordingCallDriver()
    runtime = _runtime(tmp_path, request=request, driver=driver)
    runtime._initialize()
    runtime.advance_once()
    publication_ready = False

    def make_publication_ready() -> None:
        nonlocal publication_ready
        publication_ready = True

    original_reload = runtime._reload_output

    def reload_output() -> None:
        original_reload()
        if publication_ready:
            _publish_run(tmp_path, request, "run-a")

    driver.on_success = make_publication_ready
    driver.succeeded = True
    monkeypatch.setattr(runtime, "_reload_output", reload_output)

    runtime.advance_once()

    task = runtime.store.execution.get_task(
        RUN_ID,
        DESIGN_RUNS_NODE,
        "run-a",
    )
    assert task.status.value == "succeeded"
    runtime.close()


def test_missing_runs_are_admitted_once_as_independent_gpu_calls(
    tmp_path: Path,
) -> None:
    driver = RecordingCallDriver()
    runtime = _runtime(tmp_path, driver=driver)
    runtime._initialize()

    runtime.advance_once()
    runtime.advance_once()

    calls = runtime.store.execution.list_provider_calls(RUN_ID)
    assert len(driver.spawns) == 2
    assert [call.task_keys for call in calls] == [("run-a",), ("run-b",)]
    assert {call.status for call in calls} == {ProviderCallStatus.RUNNING}
    for call, spawn in zip(calls, driver.spawns, strict=True):
        function = cast(Any, spawn["function"])
        kwargs = cast(dict[str, object], spawn["kwargs"])
        assert function.function_name == "run_boltzgen_task"
        assert kwargs["claim_owner"] == str(call.provider_call_id)
        assert (
            kwargs["task_fingerprint"]
            == runtime.store.execution.get_task(
                RUN_ID,
                DESIGN_RUNS_NODE,
                call.task_keys[0],
            ).fingerprint
        )
    runtime.close()


def test_cached_run_is_reused_and_only_missing_run_is_submitted(
    tmp_path: Path,
) -> None:
    request = _request()
    _publish_run(tmp_path, request, "run-a")
    driver = RecordingCallDriver()
    runtime = _runtime(tmp_path, request=request, driver=driver)
    runtime._initialize()

    runtime.advance_once()

    assert len(driver.spawns) == 1
    kwargs = cast(dict[str, object], driver.spawns[0]["kwargs"])
    assert cast(str, kwargs["out_dir"]).endswith("/outputs/run-b")
    runtime.close()


def test_task_publication_from_other_science_is_not_reused(tmp_path: Path) -> None:
    original = _request()
    _publish_run(tmp_path, original, "run-a")
    changed = _request(protocol="protein-anything")
    driver = RecordingCallDriver()
    runtime = _runtime(tmp_path, request=changed, driver=driver)
    runtime._initialize()

    runtime.advance_once()

    assert len(driver.spawns) == 2
    runtime.close()


def test_collection_waits_for_all_design_publications(tmp_path: Path) -> None:
    request = _request()
    _publish_run(tmp_path, request, "run-a")
    _publish_run(tmp_path, request, "run-b")
    driver = RecordingCallDriver()
    runtime = _runtime(tmp_path, request=request, driver=driver)
    runtime._initialize()

    runtime.advance_once()
    runtime.advance_once()
    assert len(driver.spawns) == 1
    function = cast(Any, driver.spawns[0]["function"])
    assert function.function_name == "collect_boltzgen_data"
    kwargs = cast(dict[str, object], driver.spawns[0]["kwargs"])
    assert kwargs["task_fingerprints"] == {
        run_id: _task_fingerprint(request, run_id) for run_id in request.run_ids
    }
    runtime.advance_once()
    assert len(driver.spawns) == 1
    runtime.close()


def test_terminal_collection_publication_prunes_all_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _request()
    write_collection_publication(
        tmp_path,
        request.collection_publication_path,
        {
            "run_name": request.run_name,
            "run_ids": list(request.run_ids),
            "filtered": False,
        },
    )
    driver = RecordingCallDriver()
    runtime = _runtime(tmp_path, request=request, driver=driver)
    runtime._initialize()
    original_observation = runtime._node_observation

    def observe_only_terminal(node_key: str):
        if node_key == DESIGN_RUNS_NODE:
            raise AssertionError("cached terminal result must prune ancestor probes")
        return original_observation(node_key)

    monkeypatch.setattr(runtime, "_node_observation", observe_only_terminal)

    runtime.advance_once()

    snapshot = runtime.store.execution.snapshot(RUN_ID)
    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert snapshot.provider_calls == ()
    assert driver.spawns == []
    runtime.close()


def test_unknown_run_prunes_calls_after_terminal_publication_appears(
    tmp_path: Path,
) -> None:
    request = _request()
    driver = RecordingCallDriver()
    runtime = _runtime(tmp_path, request=request, driver=driver)
    runtime._initialize()
    runtime.advance_once()
    calls = runtime.store.execution.list_provider_calls(RUN_ID)
    for call in calls:
        runtime.store.execution.mark_provider_call_state_unknown(
            call.provider_call_id,
            message="Modal state lookup was inconclusive",
            now=11,
        )
    driver.state_unknown = True
    write_collection_publication(
        tmp_path,
        request.collection_publication_path,
        {
            "run_name": request.run_name,
            "run_ids": list(request.run_ids),
            "filtered": False,
        },
    )

    snapshot = runtime.resume()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert driver.cancelled == {str(spawn["handle"]) for spawn in driver.spawns}
    assert len(driver.spawns) == len(calls)
    assert {call.status for call in snapshot.provider_calls} == {
        ProviderCallStatus.CANCELLED
    }
    runtime.close()


def test_cancel_requested_run_reconciles_provider_cancellation(
    tmp_path: Path,
) -> None:
    driver = RecordingCallDriver()
    runtime = _runtime(tmp_path, driver=driver)
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
