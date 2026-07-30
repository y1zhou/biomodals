"""Tests for Rosetta's deployment-local execution coordinator."""

# ruff: noqa: D101,D102,D103,D107

from hashlib import sha256
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

import biomodals.app.bioinfo.rosetta.execution_coordinator as coordinator_module
from biomodals.app.bioinfo.rosetta.execution_contracts import RosettaTaskSpec
from biomodals.app.bioinfo.rosetta.execution_coordinator import (
    RosettaExecutionCoordinator,
)
from biomodals.app.bioinfo.rosetta.execution_request import (
    RosettaExecutionRequest,
    load_execution_request,
    persist_execution_request,
)
from biomodals.execution import DeploymentIdentity
from biomodals.helper.app_execution import AppExecutionRunStore

PREDECESSOR_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SUCCESSOR_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
DEPLOYMENT = DeploymentIdentity("main", "Rosetta", 7)
SUCCESSOR_DEPLOYMENT = DeploymentIdentity("main", "Rosetta", 8)


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


class FakeRuntime:
    created: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.request = cast(RosettaExecutionRequest, kwargs["request"])
        self.execution_run_id = cast(UUID, kwargs["execution_run_id"])
        self.predecessor_execution_run_id = cast(
            UUID | None,
            kwargs["predecessor_execution_run_id"],
        )
        self.deployment = cast(DeploymentIdentity, kwargs["deployment"])
        self.store = cast(AppExecutionRunStore, kwargs["store"])
        self.created.append(kwargs)

    def run(self, *, synchronize):
        del synchronize
        return self._snapshot()

    def resume(self, *, synchronize):
        del synchronize
        return self._snapshot()

    def cancel(self):
        with self.store.transaction():
            self.store.execution.request_run_cancellation(
                self.execution_run_id,
                now=20,
            )
            self.store.execution.finalize_run_from_results(
                self.execution_run_id,
                now=21,
            )
        return self.store.execution.snapshot(self.execution_run_id)

    def close(self) -> None:
        self.store.close()

    def _snapshot(self):
        try:
            return self.store.execution.snapshot(self.execution_run_id)
        except LookupError:
            with self.store.transaction():
                self.store.execution.create_run(
                    execution_run_id=self.execution_run_id,
                    predecessor_execution_run_id=(self.predecessor_execution_run_id),
                    plan=self.request.execution_plan,
                    deployment=self.deployment,
                    max_active_provider_calls=(self.request.max_active_provider_calls),
                    max_active_gpu_provider_calls=0,
                    now=10,
                )
            return self.store.execution.snapshot(self.execution_run_id)


def _request() -> RosettaExecutionRequest:
    return RosettaExecutionRequest(
        run_name="example",
        run_id="workload-run",
        tasks=(
            RosettaTaskSpec(
                task_key="1",
                index=1,
                binary="relax",
                pdb="inputs/1/input.pdb",
                rosetta_script=None,
                flags_file=None,
                output_dir="outputs/1",
                worker_log="logs/1.log",
                expected_files=(),
                input_sha256=sha256(b"ATOM\n").hexdigest(),
            ),
        ),
        app_version="2025.51",
        max_active_provider_calls=2,
        claim_capacity=4,
        max_parallel_per_worker=4,
    )


def _coordinator(
    tmp_path: Path,
    volume: FakeVolume,
    *,
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
) -> RosettaExecutionCoordinator:
    return RosettaExecutionCoordinator(
        execution_run_id=execution_run_id,
        deployment=deployment,
        volume_root=tmp_path,
        output_volume=volume,
        modal_driver=object(),
        pull_worker_coordinator=object(),
        poll_interval_seconds=0,
    )


def _terminal_predecessor(
    tmp_path: Path,
    request: RosettaExecutionRequest,
) -> None:
    persist_execution_request(tmp_path, PREDECESSOR_ID, request)
    store = AppExecutionRunStore(tmp_path, PREDECESSOR_ID)
    with store.transaction():
        store.execution.create_run(
            execution_run_id=PREDECESSOR_ID,
            plan=request.execution_plan,
            deployment=DEPLOYMENT,
            max_active_provider_calls=request.max_active_provider_calls,
            max_active_gpu_provider_calls=0,
            now=1,
        )
        store.execution.request_run_cancellation(PREDECESSOR_ID, now=2)
        store.execution.finalize_run_from_results(PREDECESSOR_ID, now=3)
    store.close()


def test_root_run_uses_staged_request_and_remote_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRuntime.created.clear()
    monkeypatch.setattr(
        coordinator_module,
        "RosettaExecutionRuntime",
        FakeRuntime,
    )
    request = _request()
    persist_execution_request(tmp_path, PREDECESSOR_ID, request)
    volume = FakeVolume()
    coordinator = _coordinator(
        tmp_path,
        volume,
        execution_run_id=PREDECESSOR_ID,
        deployment=DEPLOYMENT,
    )

    snapshot = coordinator.run()

    assert snapshot.run.execution_run_id == PREDECESSOR_ID
    assert snapshot.run.predecessor_execution_run_id is None
    assert snapshot.run.plan == request.execution_plan
    assert len(FakeRuntime.created) == 1
    assert volume.commits == 1
    assert coordinator.status() == snapshot


def test_restart_links_successor_and_only_changes_worker_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRuntime.created.clear()
    monkeypatch.setattr(
        coordinator_module,
        "RosettaExecutionRuntime",
        FakeRuntime,
    )
    request = _request()
    _terminal_predecessor(tmp_path, request)
    coordinator = _coordinator(
        tmp_path,
        FakeVolume(),
        execution_run_id=SUCCESSOR_ID,
        deployment=SUCCESSOR_DEPLOYMENT,
    )

    snapshot = coordinator.restart(
        predecessor_execution_run_id=PREDECESSOR_ID,
        predecessor_deployment=DEPLOYMENT,
        max_active_provider_calls=3,
        claim_capacity=2,
        max_parallel_per_worker=2,
        expected_workload_plan_fingerprint=(
            request.execution_plan.workload_plan_fingerprint
        ),
    )

    successor_request = load_execution_request(tmp_path, SUCCESSOR_ID)
    assert snapshot.run.predecessor_execution_run_id == PREDECESSOR_ID
    assert snapshot.run.plan == request.execution_plan
    assert snapshot.run.max_active_provider_calls == 3
    assert successor_request.claim_capacity == 2
    assert successor_request.max_parallel_per_worker == 2
    assert (
        successor_request.execution_plan.workload_plan_fingerprint
        == request.execution_plan.workload_plan_fingerprint
    )


def test_launch_time_restart_rejects_changed_science(tmp_path: Path) -> None:
    request = _request()
    _terminal_predecessor(tmp_path, request)
    coordinator = _coordinator(
        tmp_path,
        FakeVolume(),
        execution_run_id=SUCCESSOR_ID,
        deployment=SUCCESSOR_DEPLOYMENT,
    )

    with pytest.raises(ValueError, match="changed the Workload Plan Fingerprint"):
        coordinator.restart(
            predecessor_execution_run_id=PREDECESSOR_ID,
            predecessor_deployment=DEPLOYMENT,
            expected_workload_plan_fingerprint="different",
        )
