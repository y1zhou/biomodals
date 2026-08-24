"""Tests for AlphaFold3's deployment-local graph coordinator."""

# ruff: noqa: D101,D102,D103,D107

from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from uniaf3.schema.alphafold3 import AF3Config, AF3Protein, AF3SequenceEntry

import biomodals.execution.modal.host as host_module
from biomodals.app.fold.alphafold3.environment import EnvironmentRuntime
from biomodals.app.fold.alphafold3.execution_coordinator import (
    AlphaFold3ExecutionCoordinator,
)
from biomodals.app.fold.alphafold3.execution_request import (
    AlphaFold3ExecutionRequest,
    load_execution_request,
    persist_execution_request,
)
from biomodals.app.fold.alphafold3.msa_search import SearchRuntime
from biomodals.app.fold.alphafold3.seed_predictions import InferenceRuntime
from biomodals.app.fold.alphafold3.template_search import TemplateRuntime
from biomodals.execution import DeploymentIdentity, GraphExecutionRunStore
from biomodals.schema import AppRunResult, AppRunStatus

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


class FakeClaims:
    def put(self, key, value, *, skip_if_exists=False):
        del key, value, skip_if_exists
        return True

    def get(self, key, default=None):
        del key
        return default


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
        self.store = cast(GraphExecutionRunStore, kwargs["store"])
        self.created.append(kwargs)

    def run(self) -> AppRunResult:
        self._ensure_run()
        return AppRunResult(status=AppRunStatus.SUCCEEDED)

    def resume(self) -> AppRunResult:
        return self.run()

    def close(self) -> None:
        self.store.close()

    def _ensure_run(self) -> None:
        try:
            self.store.execution.get_run(self.execution_run_id)
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


def _request() -> AlphaFold3ExecutionRequest:
    return AlphaFold3ExecutionRequest.prepare(
        AF3Config(
            name="example",
            modelSeeds=[1, 2],
            sequences=[AF3SequenceEntry(protein=AF3Protein(id="A", sequence="ACDE"))],
        ),
        search_msa=False,
        search_protein_templates=False,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=2,
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
    claims = cast(Any, FakeClaims())
    return AlphaFold3ExecutionCoordinator(
        execution_run_id=execution_run_id,
        deployment=deployment,
        volume_root=tmp_path,
        output_volume=volume,
        output_volume_name="AlphaFold3-outputs",
        provider_driver=object(),
        search_runtime=SearchRuntime(
            sharded_volume=cast(Any, FakeVolume()),
            cache_volume=cast(Any, FakeVolume()),
            claims=claims,
            container_id="coordinator",
            maximum_age_seconds=100,
            wait_timeout_seconds=100,
            sharded_root=tmp_path / "sharded",
            cache_root=tmp_path / "cache",
        ),
        template_runtime=TemplateRuntime(
            source_volume=cast(Any, FakeVolume()),
            cache_volume=cast(Any, FakeVolume()),
            claims=claims,
            container_id="coordinator",
            maximum_age_seconds=100,
            wait_timeout_seconds=100,
            source_root=tmp_path / "source",
            cache_root=tmp_path / "cache",
        ),
        inference_runtime=InferenceRuntime(
            output_root=tmp_path,
            volume=cast(Any, volume),
            claims=claims,
            container_id="coordinator",
            maximum_age_seconds=100,
            summary_maximum_age_seconds=100,
            wait_timeout_seconds=100,
        ),
        environment_runtime=EnvironmentRuntime(
            model_volume=cast(Any, FakeVolume()),
            source_volume=cast(Any, FakeVolume()),
            sharded_volume=cast(Any, FakeVolume()),
            claims=claims,
            container_id="coordinator",
            model_root=tmp_path / "models",
            source_root=tmp_path / "source",
            sharded_root=tmp_path / "sharded",
        ),
        poll_interval_seconds=0,
    )


def _terminal_predecessor(
    tmp_path: Path,
    request: AlphaFold3ExecutionRequest,
) -> None:
    persist_execution_request(tmp_path, PREDECESSOR_ID, request)
    store = GraphExecutionRunStore(tmp_path, PREDECESSOR_ID)
    with store.transaction():
        store.execution.create_run(
            execution_run_id=PREDECESSOR_ID,
            plan=request.execution_plan,
            deployment=DEPLOYMENT,
            max_active_provider_calls=request.max_active_provider_calls,
            max_active_gpu_provider_calls=request.max_active_gpu_provider_calls,
            now=1,
        )
        store.execution.request_run_cancellation(PREDECESSOR_ID, now=2)
        store.execution.finalize_run_from_results(PREDECESSOR_ID, now=3)
    store.close()


def test_root_run_uses_staged_request_and_graph_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRuntime.created.clear()
    monkeypatch.setattr(host_module, "ExecutionGraphRuntime", FakeRuntime)
    request = _request()
    persist_execution_request(tmp_path, PREDECESSOR_ID, request)
    coordinator = _coordinator(
        tmp_path,
        FakeVolume(),
        execution_run_id=PREDECESSOR_ID,
        deployment=DEPLOYMENT,
    )

    snapshot = coordinator.run()

    assert snapshot.run.execution_run_id == PREDECESSOR_ID
    assert snapshot.run.plan == request.execution_plan
    assert len(FakeRuntime.created) == 1


def test_restart_uses_shared_operational_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRuntime.created.clear()
    monkeypatch.setattr(host_module, "ExecutionGraphRuntime", FakeRuntime)
    request = _request()
    candidate = replace(
        request,
        max_active_provider_calls=6,
        max_active_gpu_provider_calls=3,
    )
    _terminal_predecessor(tmp_path, request)
    coordinator = _coordinator(
        tmp_path,
        FakeVolume(),
        execution_run_id=SUCCESSOR_ID,
        deployment=SUCCESSOR_DEPLOYMENT,
    )

    coordinator.prepare_restart(
        predecessor_execution_run_id=PREDECESSOR_ID,
        predecessor_deployment=DEPLOYMENT,
        candidate_request=candidate,
    )
    snapshot = coordinator.drive_prepared()

    stored = load_execution_request(tmp_path, SUCCESSOR_ID)
    assert snapshot.run.predecessor_execution_run_id == PREDECESSOR_ID
    assert snapshot.run.plan == request.execution_plan
    assert stored.max_active_provider_calls == 6
    assert stored.max_active_gpu_provider_calls == 3


def test_restart_rejects_changed_science(tmp_path: Path) -> None:
    request = _request()
    changed = AlphaFold3ExecutionRequest.prepare(
        request.config.model_copy(update={"modelSeeds": [9]}),
        search_msa=request.search_msa,
        search_protein_templates=request.search_protein_templates,
        max_active_provider_calls=4,
        max_active_gpu_provider_calls=2,
        recycle=request.recycle,
        sample=request.sample,
    )
    _terminal_predecessor(tmp_path, request)
    coordinator = _coordinator(
        tmp_path,
        FakeVolume(),
        execution_run_id=SUCCESSOR_ID,
        deployment=SUCCESSOR_DEPLOYMENT,
    )

    with pytest.raises(ValueError, match="changed the Workload Plan Fingerprint"):
        coordinator.prepare_restart(
            predecessor_execution_run_id=PREDECESSOR_ID,
            predecessor_deployment=DEPLOYMENT,
            candidate_request=changed,
        )
