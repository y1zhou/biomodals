"""Direct GROMACS execution-adapter tests."""

# ruff: noqa: D101,D102,D103,D107

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

import orjson
import pytest

from biomodals.app.bioinfo.gromacs_execution_runtime import (
    GromacsExecutionRequest,
    GromacsExecutionRuntime,
)
from biomodals.execution import DeploymentIdentity, RunStatus
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.helper.app_execution import ExecutionRunStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SECOND_RUN_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
THIRD_RUN_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
DEPLOYMENT = DeploymentIdentity("main", "Gromacs", 7)


class FakeVolume:
    def commit(self) -> None:
        pass

    def reload(self) -> None:
        pass


class FakeClaims:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.lock = Lock()

    def get(self, key: str, default=None):
        with self.lock:
            return self.values.get(key, default)

    def put(self, key: str, value: str, *, skip_if_exists: bool = False) -> bool:
        with self.lock:
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True


class CompletingDriver:
    def __init__(self, root: Path, run_name: str) -> None:
        self.root = root / run_name
        self.run_name = run_name
        self.calls: dict[str, tuple[Any, dict[str, object]]] = {}
        self.spawns: list[tuple[str, dict[str, object]]] = []

    def resolve(self, binding):
        return binding

    def spawn(self, function, *, args, kwargs):
        handle = f"fc-{len(self.calls) + 1}"
        copied = dict(kwargs)
        self.calls[handle] = (function, copied)
        self.spawns.append((function.function_name, copied))
        return handle

    def observe(self, provider_call_handle_id: str):
        function, kwargs = self.calls[provider_call_handle_id]
        self._publish(function.function_name, kwargs)
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=str(self.root),
        )

    def cancel(self, provider_call_handle_id: str) -> None:
        pass

    def _publish(self, function_name: str, kwargs: dict[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if function_name.startswith("prepare_tpr_"):
            (self.root / f"{self.run_name}.pdb").write_bytes(b"pdb")
            (self.root / f"nvt_{self.run_name}.tpr").write_bytes(b"tpr")
            (self.root / f"nvt_{self.run_name}.xtc").write_bytes(b"xtc")
            (self.root / f"npt_{self.run_name}.tpr").write_bytes(b"tpr")
            (self.root / f"npt_{self.run_name}.xtc").write_bytes(b"xtc")
            (self.root / f"production_{self.run_name}.tpr").write_bytes(b"tpr")
            (self.root / "production.mdp").write_bytes(b"mdp")
            return
        if function_name.startswith("production_run_"):
            (self.root / f"production_{self.run_name}.xtc").write_bytes(b"xtc")
            return
        prefix = str(kwargs["traj_prefix"])
        for metric in ("rmsd", "rg", "rmsf"):
            for suffix in ("csv", "png"):
                (self.root / f"{metric}_{prefix}{self.run_name}.{suffix}").write_bytes(
                    b"result"
                )
        if kwargs.get("save_processed_traj"):
            (self.root / f"{prefix}{self.run_name}_nopbc.xtc").write_bytes(b"xtc")


class IncompletePreparationDriver(CompletingDriver):
    def _publish(self, function_name: str, kwargs: dict[str, object]) -> None:
        if function_name.startswith("prepare_tpr_"):
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / f"production_{self.run_name}.tpr").write_bytes(b"tpr")
            (self.root / "production.mdp").write_bytes(b"mdp")
            return
        super()._publish(function_name, kwargs)


def _request() -> GromacsExecutionRequest:
    return GromacsExecutionRequest(
        run_name="example",
        pdb_content=b"ATOM\n",
        simulation_time_ns=5,
        run_pdbfixer=True,
        cpu_only=False,
        num_threads=8,
        use_openmp_threads=False,
        ld_seed=-1,
        gen_seed=-1,
        genion_seed=0,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=1,
    )


def _runtime(
    tmp_path: Path,
    request: GromacsExecutionRequest,
    claims: FakeClaims,
    execution_run_id: UUID,
    predecessor_execution_run_id: UUID | None,
) -> GromacsExecutionRuntime:
    return GromacsExecutionRuntime(
        request=request,
        execution_run_id=execution_run_id,
        predecessor_execution_run_id=predecessor_execution_run_id,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, execution_run_id),
        modal_driver=CompletingDriver(tmp_path, request.run_name),
        output_volume=FakeVolume(),
        output_claims=claims,
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 10,
    )


def test_gromacs_execution_request_round_trips_scientific_and_operational_data(
    tmp_path: Path,
) -> None:
    request = _request()

    decoded = GromacsExecutionRequest.from_bytes(request.to_bytes())

    assert decoded == request
    assert decoded.execution_plan.workload_run_key == "example"
    assert decoded.run_root(tmp_path) == tmp_path / "example"
    assert decoded.execution_plan.scientific_versions == {
        "gromacs": request.gromacs_version,
        "biomodals.gromacs.execution_plan": request.execution_plan_version,
    }


def test_gromacs_random_seeds_are_part_of_scientific_identity() -> None:
    request = _request()
    fingerprint = request.execution_plan.workload_plan_fingerprint

    assert (
        replace(request, ld_seed=17).execution_plan.workload_plan_fingerprint
        != fingerprint
    )
    assert (
        replace(request, gen_seed=23).execution_plan.workload_plan_fingerprint
        != fingerprint
    )
    assert (
        replace(request, genion_seed=29).execution_plan.workload_plan_fingerprint
        != fingerprint
    )
    assert (
        replace(
            request,
            gromacs_version="different-gromacs",
        ).execution_plan.workload_plan_fingerprint
        != fingerprint
    )
    assert (
        replace(
            request,
            execution_plan_version="different-plan",
        ).execution_plan.workload_plan_fingerprint
        != fingerprint
    )


def test_direct_runtime_drives_the_shared_parallel_graph(tmp_path: Path) -> None:
    request = _request()
    driver = CompletingDriver(tmp_path, request.run_name)
    runtime = GromacsExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=FakeVolume(),
        output_claims=FakeClaims(),
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    snapshot = runtime.run()

    assert snapshot.run.status == RunStatus.SUCCEEDED
    assert [name for name, _ in driver.spawns] == [
        "prepare_tpr_gpu",
        "production_run_gpu",
        "collect_traj_stats",
        "collect_traj_stats",
        "collect_traj_stats",
    ]
    assert [
        kwargs.get("traj_prefix")
        for name, kwargs in driver.spawns
        if name == "collect_traj_stats"
    ] == ["nvt_", "npt_", "production_"]
    runtime.close()


def test_same_run_name_rejects_outputs_from_changed_science(
    tmp_path: Path,
) -> None:
    first_request = _request()
    claims = FakeClaims()
    first_driver = CompletingDriver(tmp_path, first_request.run_name)
    first = GromacsExecutionRuntime(
        request=first_request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=first_driver,
        output_volume=FakeVolume(),
        output_claims=claims,
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 10,
    )
    assert first.run().run.status == RunStatus.SUCCEEDED
    first.close()

    changed_request = replace(first_request, pdb_content=b"ATOM changed\n")
    changed_driver = CompletingDriver(tmp_path, changed_request.run_name)
    changed = GromacsExecutionRuntime(
        request=changed_request,
        execution_run_id=SECOND_RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, SECOND_RUN_ID),
        modal_driver=changed_driver,
        output_volume=FakeVolume(),
        output_claims=claims,
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 20,
    )

    try:
        with pytest.raises(ValueError, match="different scientific inputs"):
            changed.run()
        assert changed_driver.spawns == []
    finally:
        changed.close()


def test_same_science_reuses_published_run_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    claims = FakeClaims()
    first = GromacsExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=CompletingDriver(tmp_path, request.run_name),
        output_volume=FakeVolume(),
        output_claims=claims,
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 10,
    )
    assert first.run().run.status == RunStatus.SUCCEEDED
    first.close()
    driver = CompletingDriver(tmp_path, request.run_name)
    resumed = GromacsExecutionRuntime(
        request=request,
        execution_run_id=THIRD_RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, THIRD_RUN_ID),
        modal_driver=driver,
        output_volume=FakeVolume(),
        output_claims=claims,
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 20,
    )
    observed: list[str] = []
    observe = resumed._node_publication_ready

    def record_observation(node_key: str) -> bool:
        observed.append(node_key)
        return observe(node_key)

    monkeypatch.setattr(resumed, "_node_publication_ready", record_observation)

    try:
        assert resumed.run().run.status == RunStatus.SUCCEEDED
        assert driver.spawns == []
        assert observed == list(request.execution_plan.terminal_node_keys)
    finally:
        resumed.close()


def test_prepare_publication_requires_downstream_inputs(tmp_path: Path) -> None:
    request = _request()
    driver = IncompletePreparationDriver(tmp_path, request.run_name)
    runtime = GromacsExecutionRuntime(
        request=request,
        execution_run_id=RUN_ID,
        deployment=DEPLOYMENT,
        store=ExecutionRunStore(tmp_path, RUN_ID),
        modal_driver=driver,
        output_volume=FakeVolume(),
        output_claims=FakeClaims(),
        output_root=tmp_path,
        poll_interval_seconds=0,
        now=lambda: 10,
    )

    try:
        assert runtime.run().run.status == RunStatus.FAILED
        assert [name for name, _kwargs in driver.spawns] == ["prepare_tpr_gpu"]
    finally:
        runtime.close()


def test_concurrent_same_name_roots_elect_one_output_owner(tmp_path: Path) -> None:
    request = _request()
    claims = FakeClaims()
    runtimes = tuple(
        GromacsExecutionRuntime(
            request=request,
            execution_run_id=execution_run_id,
            deployment=DEPLOYMENT,
            store=ExecutionRunStore(tmp_path, execution_run_id),
            modal_driver=CompletingDriver(tmp_path, request.run_name),
            output_volume=FakeVolume(),
            output_claims=claims,
            output_root=tmp_path,
            poll_interval_seconds=0,
            now=lambda: 10,
        )
        for execution_run_id in (RUN_ID, SECOND_RUN_ID)
    )

    def ensure(runtime: GromacsExecutionRuntime) -> bool | str:
        try:
            return runtime._ensure_run_identity()
        except (RuntimeError, ValueError) as error:
            return str(error)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = tuple(executor.map(ensure, runtimes))
        assert outcomes.count(True) == 1
        assert (
            sum(
                "already claimed" in str(outcome)
                or "unclaimed existing outputs" in str(outcome)
                for outcome in outcomes
            )
            == 1
        )
    finally:
        for runtime in runtimes:
            runtime.close()


def test_successors_transfer_incomplete_output_ownership(tmp_path: Path) -> None:
    request = _request()
    claims = FakeClaims()

    generations = (
        _runtime(tmp_path, request, claims, RUN_ID, None),
        _runtime(tmp_path, request, claims, SECOND_RUN_ID, RUN_ID),
        _runtime(tmp_path, request, claims, THIRD_RUN_ID, SECOND_RUN_ID),
    )
    try:
        assert [item._ensure_run_identity() for item in generations] == [True] * 3
        marker = orjson.loads(generations[-1]._run_identity_path().read_bytes())
        assert marker["owner_execution_run_id"] == str(THIRD_RUN_ID)
    finally:
        for item in generations:
            item.close()


def test_sibling_successor_cannot_replace_active_owner(tmp_path: Path) -> None:
    request = _request()
    claims = FakeClaims()
    owner = _runtime(tmp_path, request, claims, RUN_ID, None)
    first_successor = _runtime(tmp_path, request, claims, SECOND_RUN_ID, RUN_ID)
    sibling = _runtime(tmp_path, request, claims, THIRD_RUN_ID, RUN_ID)
    try:
        assert owner._ensure_run_identity()
        assert first_successor._ensure_run_identity()
        with pytest.raises(RuntimeError, match="already claimed"):
            sibling._ensure_run_identity()
        marker = orjson.loads(first_successor._run_identity_path().read_bytes())
        assert marker["owner_execution_run_id"] == str(SECOND_RUN_ID)
    finally:
        owner.close()
        first_successor.close()
        sibling.close()


def test_cache_reading_successor_preserves_repair_lineage(tmp_path: Path) -> None:
    request = _request()
    claims = FakeClaims()
    owner = _runtime(tmp_path, request, claims, RUN_ID, None)
    cache_reader = _runtime(tmp_path, request, claims, SECOND_RUN_ID, RUN_ID)
    repair = _runtime(tmp_path, request, claims, THIRD_RUN_ID, SECOND_RUN_ID)
    try:
        assert owner.run().run.status == RunStatus.SUCCEEDED
        assert not cache_reader._ensure_run_identity()
        terminal_node = request.execution_plan.terminal_node_keys[0]
        owner._node_publication_path(terminal_node).unlink()

        assert repair._ensure_run_identity()
        marker = orjson.loads(repair._run_identity_path().read_bytes())
        assert marker["owner_execution_run_id"] == str(THIRD_RUN_ID)
    finally:
        owner.close()
        cache_reader.close()
        repair.close()
