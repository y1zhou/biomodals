"""Tests for BoltzGen's direct-fan-out execution adapter."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from typing import Any, cast
from uuid import UUID

from biomodals.app.design.boltzgen.execution_contracts import (
    boltzgen_run_root,
    write_collection_publication,
)
from biomodals.app.design.boltzgen.execution_request import (
    BoltzGenExecutionRequest,
    prepare_execution_request,
)
from biomodals.app.design.boltzgen.execution_runtime import (
    BoltzGenExecutionRuntime,
)
from biomodals.execution import DeploymentIdentity, ProviderCallStatus, RunStatus
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.helper.app_execution import AppExecutionRunStore

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
        del provider_call_handle_id
        return ModalCallObservation(ModalCallObservationKind.RUNNING)

    def cancel(self, provider_call_handle_id: str) -> None:
        raise AssertionError(provider_call_handle_id)


def _request(
    *,
    run_ids: tuple[str, ...] = ("run-a", "run-b"),
) -> BoltzGenExecutionRequest:
    return prepare_execution_request(
        run_name="example",
        run_ids=run_ids,
        yaml_content=b"name: example\n",
        additional_files={},
        protocol="nanobody-anything",
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
        store=AppExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver or RecordingCallDriver(),
        output_volume=FakeVolume(),
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 10,
    )


def _publish_run(tmp_path: Path, run_id: str) -> None:
    final = boltzgen_run_root(tmp_path, "example", run_id) / "final_ranked_designs"
    final.mkdir(parents=True, exist_ok=True)
    (final / "results_overview.pdf").write_bytes(b"pdf")


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
    runtime.close()


def test_cached_run_is_reused_and_only_missing_run_is_submitted(
    tmp_path: Path,
) -> None:
    _publish_run(tmp_path, "run-a")
    driver = RecordingCallDriver()
    runtime = _runtime(tmp_path, driver=driver)
    runtime._initialize()

    runtime.advance_once()

    assert len(driver.spawns) == 1
    kwargs = cast(dict[str, object], driver.spawns[0]["kwargs"])
    assert cast(str, kwargs["out_dir"]).endswith("/outputs/run-b")
    runtime.close()


def test_collection_waits_for_all_design_publications(tmp_path: Path) -> None:
    _publish_run(tmp_path, "run-a")
    _publish_run(tmp_path, "run-b")
    driver = RecordingCallDriver()
    runtime = _runtime(tmp_path, driver=driver)
    runtime._initialize()

    runtime.advance_once()
    assert len(driver.spawns) == 1
    function = cast(Any, driver.spawns[0]["function"])
    assert function.function_name == "collect_boltzgen_data"
    runtime.advance_once()
    assert len(driver.spawns) == 1
    runtime.close()


def test_terminal_collection_publication_prunes_all_calls(tmp_path: Path) -> None:
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

    runtime.advance_once()

    snapshot = runtime.store.execution.snapshot(RUN_ID)
    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert snapshot.provider_calls == ()
    assert driver.spawns == []
    runtime.close()
