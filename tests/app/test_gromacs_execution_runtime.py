"""Direct GROMACS execution-adapter tests."""

# ruff: noqa: D101,D102,D103,D107

from pathlib import Path
from typing import Any
from uuid import UUID

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
DEPLOYMENT = DeploymentIdentity("main", "Gromacs", 7)


class FakeVolume:
    def commit(self) -> None:
        pass

    def reload(self) -> None:
        pass


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
            (self.root / f"production_{self.run_name}.tpr").touch()
            (self.root / "production.mdp").touch()
            return
        if function_name.startswith("production_run_"):
            (self.root / f"production_{self.run_name}.xtc").touch()
            return
        prefix = str(kwargs["traj_prefix"])
        for metric in ("rmsd", "rg", "rmsf"):
            for suffix in ("csv", "png"):
                (self.root / f"{metric}_{prefix}{self.run_name}.{suffix}").touch()
        if kwargs.get("save_processed_traj"):
            (self.root / f"{prefix}{self.run_name}_nopbc.xtc").touch()


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


def test_gromacs_execution_request_round_trips_scientific_and_operational_data(
    tmp_path: Path,
) -> None:
    request = _request()

    decoded = GromacsExecutionRequest.from_bytes(request.to_bytes())

    assert decoded == request
    assert decoded.execution_plan.workload_run_key == "example"
    assert decoded.run_root(tmp_path) == tmp_path / "example"


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
