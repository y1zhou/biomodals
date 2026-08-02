"""Tests for BoltzGen's deployment-local coordinator adapter."""

# ruff: noqa: D101,D102,D103,D107,S106

from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

import biomodals.app.design.boltzgen.execution_coordinator as coordinator_module
from biomodals.app.design.boltzgen.execution_coordinator import (
    BoltzGenExecutionCoordinator,
)
from biomodals.app.design.boltzgen.execution_request import (
    BoltzGenExecutionRequest,
    load_execution_request,
    persist_execution_request,
    prepare_execution_request,
)
from biomodals.execution import (
    AvailabilityStatus,
    DeploymentIdentity,
    ProviderBinding,
    TaskPlan,
)
from biomodals.execution.scheduler import TaskDispatchDescriptor
from biomodals.helper.app_execution import ExecutionRunStore

PREDECESSOR_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SUCCESSOR_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
DEPLOYMENT = DeploymentIdentity("main", "BoltzGen", 7)
SUCCESSOR_DEPLOYMENT = DeploymentIdentity("main", "BoltzGen", 8)


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
        self.request = cast(BoltzGenExecutionRequest, kwargs["request"])
        self.execution_run_id = cast(UUID, kwargs["execution_run_id"])
        self.predecessor_execution_run_id = cast(
            UUID | None,
            kwargs["predecessor_execution_run_id"],
        )
        self.deployment = cast(DeploymentIdentity, kwargs["deployment"])
        self.store = cast(ExecutionRunStore, kwargs["store"])
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
                    max_active_gpu_provider_calls=(
                        self.request.max_active_gpu_provider_calls
                    ),
                    now=10,
                )
            return self.store.execution.snapshot(self.execution_run_id)


def _request() -> BoltzGenExecutionRequest:
    return prepare_execution_request(
        run_name="example",
        run_ids=("run-a",),
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
        max_active_provider_calls=1,
        max_active_gpu_provider_calls=1,
    )


def _coordinator(
    tmp_path: Path,
    volume: FakeVolume,
    *,
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
    app_version: str = "0.3.2",
    repo_commit_hash: str = "abc123",
) -> BoltzGenExecutionCoordinator:
    return BoltzGenExecutionCoordinator(
        execution_run_id=execution_run_id,
        deployment=deployment,
        volume_root=tmp_path,
        output_volume=volume,
        modal_driver=object(),
        app_version=app_version,
        repo_commit_hash=repo_commit_hash,
        poll_interval_seconds=0,
    )


def _terminal_predecessor(
    tmp_path: Path,
    request: BoltzGenExecutionRequest,
) -> str:
    persist_execution_request(tmp_path, PREDECESSOR_ID, request)
    store = ExecutionRunStore(tmp_path, PREDECESSOR_ID)
    with store.transaction():
        store.execution.create_run(
            execution_run_id=PREDECESSOR_ID,
            plan=request.execution_plan,
            deployment=DEPLOYMENT,
            max_active_provider_calls=request.max_active_provider_calls,
            max_active_gpu_provider_calls=(request.max_active_gpu_provider_calls),
            now=1,
        )
        store.execution.start_node(
            PREDECESSOR_ID,
            "design-runs",
            now=2,
        )
        store.execution.discover_tasks(
            PREDECESSOR_ID,
            "design-runs",
            (
                TaskPlan(
                    task_key="run-a",
                    scientific_payload={"run_id": "run-a"},
                    execution_payload={"run_id": "run-a"},
                ),
            ),
            now=3,
        )
        store.execution.record_task_result_observation(
            PREDECESSOR_ID,
            "design-runs",
            "run-a",
            AvailabilityStatus.MISSING,
            now=4,
        )
        binding = ProviderBinding(
            environment="main",
            app_name="BoltzGen",
            app_version=7,
            function_name="run_boltzgen_task",
            uses_gpu=True,
        )
        store.execution.persist_fixed_dispatch_policy(
            PREDECESSOR_ID,
            (
                TaskDispatchDescriptor(
                    node_key="design-runs",
                    node_ordinal=0,
                    task_key="run-a",
                    task_ordinal=0,
                    binding=binding,
                    compatibility_key="run_boltzgen_task",
                    max_tasks_per_call=1,
                    depth=0,
                    unblocking_span=1,
                ),
            ),
            now=4,
        )
        preclaim = store.execution.preclaim_fixed_batch(
            PREDECESSOR_ID,
            "design-runs",
            ("run-a",),
            submission_token="run-a",
            binding=binding,
            compatibility_key="run_boltzgen_task",
            now=5,
        )
        assert preclaim is not None
        store.execution.fail_provider_call(
            preclaim.call.provider_call_id,
            message="worker failed",
            now=6,
        )
        store.execution.reconcile_node_tasks(
            PREDECESSOR_ID,
            "design-runs",
            now=7,
        )
        store.execution.skip_unreachable_nodes(PREDECESSOR_ID, now=8)
        store.execution.finalize_run_from_results(PREDECESSOR_ID, now=9)
    store.close()
    return str(preclaim.call.provider_call_id)


def test_close_does_not_checkpoint_unchanged_state(tmp_path: Path) -> None:
    volume = FakeVolume()
    coordinator = _coordinator(
        tmp_path,
        volume,
        execution_run_id=PREDECESSOR_ID,
        deployment=DEPLOYMENT,
    )

    coordinator.close()

    assert volume.commits == 0


def test_restart_links_successor_and_preserves_scientific_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRuntime.created.clear()
    monkeypatch.setattr(
        coordinator_module,
        "BoltzGenExecutionRuntime",
        FakeRuntime,
    )
    request = _request()
    old_provider_call_id = _terminal_predecessor(tmp_path, request)
    coordinator = _coordinator(
        tmp_path,
        FakeVolume(),
        execution_run_id=SUCCESSOR_ID,
        deployment=SUCCESSOR_DEPLOYMENT,
    )

    snapshot = coordinator.restart(
        predecessor_execution_run_id=PREDECESSOR_ID,
        predecessor_deployment=DEPLOYMENT,
        max_active_provider_calls=2,
        max_active_gpu_provider_calls=1,
        expected_workload_plan_fingerprint=(
            request.execution_plan.workload_plan_fingerprint
        ),
    )

    assert snapshot.run.predecessor_execution_run_id == PREDECESSOR_ID
    assert snapshot.run.plan == request.execution_plan
    assert snapshot.run.max_active_provider_calls == 2
    successor_request = load_execution_request(tmp_path, SUCCESSOR_ID)
    assert (
        successor_request.execution_plan.workload_plan_fingerprint
        == request.execution_plan.workload_plan_fingerprint
    )
    assert successor_request.replace_claim_owners == (("run-a", old_provider_call_id),)


def test_launch_time_restart_rejects_changed_science(
    tmp_path: Path,
) -> None:
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


def test_restart_rejects_target_deployment_version_drift(tmp_path: Path) -> None:
    request = _request()
    _terminal_predecessor(tmp_path, request)
    coordinator = _coordinator(
        tmp_path,
        FakeVolume(),
        execution_run_id=SUCCESSOR_ID,
        deployment=SUCCESSOR_DEPLOYMENT,
        app_version="0.4.0",
    )

    with pytest.raises(ValueError, match="declared scientific versions"):
        coordinator.prepare_restart(
            predecessor_execution_run_id=PREDECESSOR_ID,
            predecessor_deployment=DEPLOYMENT,
        )

    assert not ExecutionRunStore(tmp_path, SUCCESSOR_ID).ledger_path.exists()
