"""Tests for AlphaFold3's deployment-local coordinator adapter."""

# ruff: noqa: D101,D102,D107

from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

import biomodals.app.fold.alphafold3.execution_coordinator as coordinator_module
from biomodals.app.fold.alphafold3.execution_coordinator import (
    AlphaFold3ExecutionCoordinator,
    _restart_request,
)
from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
    persist_execution_request,
)
from biomodals.execution import DeploymentIdentity, RunStatus
from biomodals.helper.app_execution import AppExecutionRunStore

PREDECESSOR_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SUCCESSOR_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
DEPLOYMENT = DeploymentIdentity("main", "AlphaFold3", 7)
SUCCESSOR_DEPLOYMENT = DeploymentIdentity("main", "AlphaFold3", 8)


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
        self.request = cast(AlphaFold3ExecutionRequest, kwargs["request"])
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
                    max_active_gpu_provider_calls=self.request.max_num_gpus,
                    now=10,
                )
            return self.store.execution.snapshot(self.execution_run_id)


def _request() -> AlphaFold3ExecutionRequest:
    return AlphaFold3ExecutionRequest.prepare(
        AF3Config(
            name="example",
            modelSeeds=[1],
            sequences=[
                AF3SequenceEntry(
                    protein=AF3Protein(id="A", sequence="ACDE"),
                )
            ],
        ),
        search_msa=False,
        search_protein_templates=False,
        max_parallel_search_workers=2,
        max_num_gpus=1,
        recycle=10,
        sample=1,
    )


def _coordinator(
    tmp_path: Path,
    volume: FakeVolume,
    *,
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
) -> AlphaFold3ExecutionCoordinator:
    return AlphaFold3ExecutionCoordinator(
        execution_run_id=execution_run_id,
        deployment=deployment,
        volume_root=tmp_path,
        output_volume=volume,
        modal_driver=object(),
        search_runtime=cast(Any, object()),
        template_runtime=cast(Any, object()),
        inference_runtime=cast(Any, object()),
        poll_interval_seconds=0,
    )


def test_root_run_loads_staged_request_and_binds_remote_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deployment adapter trusts staged bytes, not method arguments."""
    FakeRuntime.created.clear()
    monkeypatch.setattr(
        coordinator_module,
        "AlphaFold3ExecutionRuntime",
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


def test_restart_links_a_new_ledger_and_only_changes_operational_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conclusive predecessor yields a compatible, separately stored Run."""
    FakeRuntime.created.clear()
    monkeypatch.setattr(
        coordinator_module,
        "AlphaFold3ExecutionRuntime",
        FakeRuntime,
    )
    request = _request()
    persist_execution_request(tmp_path, PREDECESSOR_ID, request)
    predecessor_store = AppExecutionRunStore(tmp_path, PREDECESSOR_ID)
    with predecessor_store.transaction():
        predecessor_store.execution.create_run(
            execution_run_id=PREDECESSOR_ID,
            plan=request.execution_plan,
            deployment=DEPLOYMENT,
            max_active_provider_calls=request.max_active_provider_calls,
            max_active_gpu_provider_calls=request.max_num_gpus,
            now=1,
        )
        predecessor_store.execution.start_node(
            PREDECESSOR_ID,
            "stage-request-input",
            now=2,
        )
        predecessor_store.execution.discover_tasks(
            PREDECESSOR_ID,
            "stage-request-input",
            (),
            now=3,
        )
        predecessor_store.execution.skip_unreachable_nodes(
            PREDECESSOR_ID,
            now=4,
        )
        predecessor_store.execution.finalize_run_from_results(
            PREDECESSOR_ID,
            now=5,
        )
    assert (
        predecessor_store.execution.get_run(PREDECESSOR_ID).status == RunStatus.FAILED
    )
    predecessor_store.close()
    volume = FakeVolume()
    coordinator = _coordinator(
        tmp_path,
        volume,
        execution_run_id=SUCCESSOR_ID,
        deployment=SUCCESSOR_DEPLOYMENT,
    )

    snapshot = coordinator.restart(
        predecessor_execution_run_id=PREDECESSOR_ID,
        predecessor_deployment=DEPLOYMENT,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=2,
    )

    assert snapshot.run.predecessor_execution_run_id == PREDECESSOR_ID
    assert snapshot.run.plan == request.execution_plan
    assert snapshot.run.max_active_provider_calls == 3
    assert snapshot.run.max_active_gpu_provider_calls == 2
    assert (
        cast(
            AlphaFold3ExecutionRequest, FakeRuntime.created[0]["request"]
        ).execution_plan
        == request.execution_plan
    )


def test_restart_rejects_a_gpu_limit_above_the_total_limit() -> None:
    """Operational overrides retain the kernel's GPU-subset invariant."""
    request = _request()

    with pytest.raises(ValueError, match="cannot exceed"):
        _restart_request(
            request,
            predecessor_max_active_provider_calls=2,
            predecessor_max_active_gpu_provider_calls=1,
            max_active_provider_calls=1,
            max_active_gpu_provider_calls=2,
        )
