"""Direct GROMACS execution-definition tests."""

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
    GromacsExecutionCoordinator,
    GromacsExecutionRequest,
    GromacsPublications,
    gromacs_execution_graph,
    persist_execution_request,
)
from biomodals.execution import DeploymentIdentity, RunStatus
from biomodals.execution.definition_plan import execution_plan
from biomodals.execution.modal import (
    ProviderCallObservation,
    ProviderCallObservationKind,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SECOND_RUN_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
THIRD_RUN_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
DEPLOYMENT = DeploymentIdentity("main", "Gromacs", 7)
OUTPUT_VOLUME_NAME = "Gromacs-outputs"


class FakeVolume:
    def __init__(self) -> None:
        self.commits = 0
        self.reloads = 0

    def commit(self) -> None:
        self.commits += 1

    def reload(self) -> None:
        self.reloads += 1


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
        return ProviderCallObservation(
            ProviderCallObservationKind.SUCCEEDED,
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
            (self.root / f"production_{self.run_name}.edr").write_bytes(b"edr")
            return
        prefix = str(kwargs["traj_prefix"])
        for metric in ("rmsd", "rg", "rmsf"):
            for suffix in ("csv", "png"):
                (self.root / f"{metric}_{prefix}{self.run_name}.{suffix}").write_bytes(
                    b"result"
                )
        if kwargs.get("save_processed_traj"):
            (self.root / f"{prefix}{self.run_name}_nopbc.xtc").write_bytes(b"xtc")
            (self.root / f"{prefix}{self.run_name}_nopbc_centered.pdb").write_bytes(
                b"ATOM\n"
            )


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
        ld_seed=11,
        gen_seed=12,
        genion_seed=13,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=1,
    )


def _publications(
    tmp_path: Path,
    request: GromacsExecutionRequest,
    claims: FakeClaims,
    execution_run_id: UUID,
    predecessor_execution_run_id: UUID | None = None,
) -> GromacsPublications:
    return GromacsPublications(
        request=request,
        execution_run_id=execution_run_id,
        predecessor_execution_run_id=predecessor_execution_run_id,
        output_root=tmp_path,
        output_claims=claims,
        output_volume_name=OUTPUT_VOLUME_NAME,
    )


def _coordinator(
    tmp_path: Path,
    request: GromacsExecutionRequest,
    claims: FakeClaims,
    execution_run_id: UUID,
    *,
    driver: CompletingDriver | None = None,
) -> GromacsExecutionCoordinator:
    persist_execution_request(tmp_path, execution_run_id, request)
    return GromacsExecutionCoordinator(
        execution_run_id=execution_run_id,
        deployment=DEPLOYMENT,
        volume_root=tmp_path,
        output_volume=FakeVolume(),
        output_volume_name=OUTPUT_VOLUME_NAME,
        provider_driver=driver or CompletingDriver(tmp_path, request.run_name),
        output_claims=claims,
        poll_interval_seconds=0,
    )


def _restart(
    coordinator: GromacsExecutionCoordinator,
    predecessor_execution_run_id: UUID,
):
    coordinator.prepare_restart(
        predecessor_execution_run_id=predecessor_execution_run_id,
        predecessor_deployment=DEPLOYMENT,
    )
    return coordinator.drive_prepared()


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


def test_gromacs_definition_preserves_exact_execution_plan(tmp_path: Path) -> None:
    request = _request()
    graph = gromacs_execution_graph(
        request,
        _publications(tmp_path, request, FakeClaims(), RUN_ID),
    )

    assert (
        execution_plan(
            graph.validate(),
            workload_run_key=request.run_name,
        )
        == request.execution_plan
    )


def test_gromacs_random_seeds_are_part_of_scientific_identity() -> None:
    request = _request()
    fingerprint = request.execution_plan.workload_plan_fingerprint

    assert request.ld_seed != -1
    assert request.gen_seed != -1
    assert request.genion_seed != 0
    assert GromacsExecutionRequest.from_bytes(request.to_bytes()) == request
    assert replace(request, ld_seed=17).execution_plan.workload_plan_fingerprint != (
        fingerprint
    )
    assert replace(request, gen_seed=23).execution_plan.workload_plan_fingerprint != (
        fingerprint
    )
    assert replace(
        request, genion_seed=29
    ).execution_plan.workload_plan_fingerprint != (fingerprint)
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


def test_gromacs_request_rejects_unmaterialized_random_sentinels() -> None:
    with pytest.raises(ValueError, match="random sentinels"):
        replace(_request(), ld_seed=-1)
    with pytest.raises(ValueError, match="random sentinels"):
        replace(_request(), gen_seed=-1)
    with pytest.raises(ValueError, match="random sentinels"):
        replace(_request(), genion_seed=0)


def test_gromacs_request_allows_zero_gpu_admission_for_cached_results() -> None:
    assert replace(_request(), max_active_gpu_provider_calls=0).cpu_only is False


def test_direct_coordinator_drives_the_shared_parallel_graph(tmp_path: Path) -> None:
    request = _request()
    driver = CompletingDriver(tmp_path, request.run_name)
    coordinator = _coordinator(tmp_path, request, FakeClaims(), RUN_ID, driver=driver)

    snapshot = coordinator.run()

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
    coordinator.close()


def test_same_run_name_rejects_outputs_from_changed_science(tmp_path: Path) -> None:
    request = _request()
    claims = FakeClaims()
    first = _coordinator(tmp_path, request, claims, RUN_ID)
    assert first.run().run.status == RunStatus.SUCCEEDED
    first.close()

    changed_request = replace(request, pdb_content=b"ATOM changed\n")
    driver = CompletingDriver(tmp_path, changed_request.run_name)
    changed = _coordinator(
        tmp_path,
        changed_request,
        claims,
        SECOND_RUN_ID,
        driver=driver,
    )
    try:
        with pytest.raises(ValueError, match="different scientific inputs"):
            changed.run()
        assert driver.spawns == []
    finally:
        changed.close()


def test_same_science_reuses_published_run_name(tmp_path: Path) -> None:
    request = _request()
    claims = FakeClaims()
    first = _coordinator(tmp_path, request, claims, RUN_ID)
    assert first.run().run.status == RunStatus.SUCCEEDED
    first.close()
    driver = CompletingDriver(tmp_path, request.run_name)
    reader = _coordinator(
        tmp_path,
        request,
        claims,
        THIRD_RUN_ID,
        driver=driver,
    )

    try:
        assert reader.run().run.status == RunStatus.SUCCEEDED
        assert driver.spawns == []
    finally:
        reader.close()


def test_prepare_publication_requires_downstream_inputs(tmp_path: Path) -> None:
    request = _request()
    driver = IncompletePreparationDriver(tmp_path, request.run_name)
    coordinator = _coordinator(
        tmp_path,
        request,
        FakeClaims(),
        RUN_ID,
        driver=driver,
    )

    try:
        assert coordinator.run().run.status == RunStatus.FAILED
        assert [name for name, _kwargs in driver.spawns] == ["prepare_tpr_gpu"]
    finally:
        coordinator.close()


def test_terminal_publication_covers_required_user_outputs(tmp_path: Path) -> None:
    request = _request()
    publications = _publications(tmp_path, request, FakeClaims(), RUN_ID)
    root = request.run_root(tmp_path)

    paths = {
        path.relative_to(root).as_posix()
        for path in publications.node_paths(
            request.execution_plan.terminal_node_keys[0]
        )
    }

    assert {
        "production.mdp",
        f"production_{request.run_name}.edr",
        f"production_{request.run_name}.tpr",
        f"production_{request.run_name}_nopbc.xtc",
        f"production_{request.run_name}_nopbc_centered.pdb",
    } <= paths


def test_production_invalidation_removes_the_checkpoint_output_set(
    tmp_path: Path,
) -> None:
    request = _request()
    publications = _publications(tmp_path, request, FakeClaims(), RUN_ID)
    root = request.run_root(tmp_path)
    root.mkdir()
    prefix = f"production_{request.run_name}"
    output_names = (
        f"{prefix}.xtc",
        f"{prefix}.edr",
        f"{prefix}.cpt",
        f"{prefix}_prev.cpt",
        f"{prefix}.log",
        f"{prefix}.gro",
        f"{prefix}.trr",
        f"{prefix}.tng",
    )
    for name in output_names:
        (root / name).write_bytes(b"partial")
    topology = root / f"{prefix}.tpr"
    topology.write_bytes(b"tpr")

    publications.invalidate("production_run_gpu")

    assert not any((root / name).exists() for name in output_names)
    assert topology.read_bytes() == b"tpr"


@pytest.mark.parametrize(
    ("missing_name", "repair_function"),
    [
        ("production.mdp", "prepare_tpr_gpu"),
        ("production_example.tpr", "prepare_tpr_gpu"),
        (
            "production_example_nopbc_centered.pdb",
            "collect_traj_stats",
        ),
    ],
)
def test_successor_repairs_missing_terminal_output(
    tmp_path: Path,
    missing_name: str,
    repair_function: str,
) -> None:
    request = _request()
    claims = FakeClaims()
    owner = _coordinator(tmp_path, request, claims, RUN_ID)
    assert owner.run().run.status == RunStatus.SUCCEEDED
    owner.close()
    (request.run_root(tmp_path) / missing_name).unlink()
    driver = CompletingDriver(tmp_path, request.run_name)
    successor = _coordinator(
        tmp_path,
        request,
        claims,
        SECOND_RUN_ID,
        driver=driver,
    )
    try:
        assert _restart(successor, RUN_ID).run.status == RunStatus.SUCCEEDED
        assert repair_function in {name for name, _kwargs in driver.spawns}
    finally:
        successor.close()


def test_successor_replaces_digest_invalid_preparation_output(
    tmp_path: Path,
) -> None:
    request = _request()
    claims = FakeClaims()
    owner = _coordinator(tmp_path, request, claims, RUN_ID)
    assert owner.run().run.status == RunStatus.SUCCEEDED
    owner.close()
    production_mdp = request.run_root(tmp_path) / "production.mdp"
    production_mdp.write_bytes(b"bad")
    driver = CompletingDriver(tmp_path, request.run_name)
    successor = _coordinator(
        tmp_path,
        request,
        claims,
        SECOND_RUN_ID,
        driver=driver,
    )
    try:
        assert _restart(successor, RUN_ID).run.status == RunStatus.SUCCEEDED
        assert "prepare_tpr_gpu" in {name for name, _kwargs in driver.spawns}
        assert production_mdp.read_bytes() == b"mdp"
    finally:
        successor.close()


def test_concurrent_same_name_roots_elect_one_output_owner(tmp_path: Path) -> None:
    request = _request()
    claims = FakeClaims()
    publications = tuple(
        _publications(tmp_path, request, claims, execution_run_id)
        for execution_run_id in (RUN_ID, SECOND_RUN_ID)
    )

    def ensure(item: GromacsPublications) -> bool | str:
        try:
            return item.ensure_run_identity()
        except (RuntimeError, ValueError) as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(ensure, publications))
    assert outcomes.count(True) == 1
    assert (
        sum(
            "already claimed" in str(outcome)
            or "unclaimed existing outputs" in str(outcome)
            for outcome in outcomes
        )
        == 1
    )


def test_successors_transfer_incomplete_output_ownership(tmp_path: Path) -> None:
    request = _request()
    claims = FakeClaims()
    generations = (
        _publications(tmp_path, request, claims, RUN_ID),
        _publications(tmp_path, request, claims, SECOND_RUN_ID, RUN_ID),
        _publications(tmp_path, request, claims, THIRD_RUN_ID, SECOND_RUN_ID),
    )

    assert [item.ensure_run_identity() for item in generations] == [True] * 3
    marker = orjson.loads(generations[-1].run_identity_path().read_bytes())
    assert marker["owner_execution_run_id"] == str(THIRD_RUN_ID)


def test_sibling_successor_cannot_replace_active_owner(tmp_path: Path) -> None:
    request = _request()
    claims = FakeClaims()
    owner = _publications(tmp_path, request, claims, RUN_ID)
    first_successor = _publications(
        tmp_path,
        request,
        claims,
        SECOND_RUN_ID,
        RUN_ID,
    )
    sibling = _publications(tmp_path, request, claims, THIRD_RUN_ID, RUN_ID)

    assert owner.ensure_run_identity()
    assert first_successor.ensure_run_identity()
    with pytest.raises(RuntimeError, match="already claimed"):
        sibling.ensure_run_identity()
    marker = orjson.loads(first_successor.run_identity_path().read_bytes())
    assert marker["owner_execution_run_id"] == str(SECOND_RUN_ID)


def test_cache_reading_successor_preserves_repair_lineage(tmp_path: Path) -> None:
    request = _request()
    claims = FakeClaims()
    owner = _coordinator(tmp_path, request, claims, RUN_ID)
    assert owner.run().run.status == RunStatus.SUCCEEDED
    owner.close()
    cache_reader = _publications(tmp_path, request, claims, SECOND_RUN_ID, RUN_ID)
    repair = _publications(
        tmp_path,
        request,
        claims,
        THIRD_RUN_ID,
        SECOND_RUN_ID,
    )

    assert not cache_reader.ensure_run_identity()
    terminal_node = request.execution_plan.terminal_node_keys[0]
    cache_reader.publication_path(terminal_node).unlink()

    assert repair.ensure_run_identity()
    marker = orjson.loads(repair.run_identity_path().read_bytes())
    assert marker["owner_execution_run_id"] == str(THIRD_RUN_ID)
