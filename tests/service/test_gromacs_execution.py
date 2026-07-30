"""GROMACS service execution-kernel integration tests."""

# ruff: noqa: D101, D102, D103, D107, S105, S106

import asyncio
from pathlib import Path
from uuid import UUID

from biomodals.execution import (
    DeploymentIdentity,
    ProviderCallStatus,
    RunStatus,
)
from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.service.auth import AuthService
from biomodals.service.gromacs.archive import GROMACS_ARCHIVE_SCHEMA_VERSION
from biomodals.service.gromacs.contracts import GromacsJobOptions
from biomodals.service.gromacs.execution import GromacsExecutionCoordinator
from biomodals.service.gromacs.plan import execution_plan
from biomodals.service.gromacs.results import FinalArchive
from biomodals.service.runtime_config import (
    DatabaseOverridableSetting,
    JobAdmissionConfiguration,
)
from biomodals.service.store import JobState, ServiceStore

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
JOB_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")


class FakeGromacsExecutionAdapter:
    def __init__(self) -> None:
        self.spawn_waves: list[list[str]] = []
        self._current_wave: list[str] = []
        self._calls: dict[str, str] = {}
        self.publish_count = 0

    def begin_wave(self) -> None:
        self._current_wave = []
        self.spawn_waves.append(self._current_wave)

    async def resolve(self, binding):
        return binding

    async def spawn(self, function, *, args, kwargs):
        call_id = f"fc-{len(self._calls)}"
        self._calls[call_id] = function.function_name
        self._current_wave.append(function.function_name)
        return call_id

    async def observe(self, provider_call_handle_id):
        function_name = self._calls[provider_call_handle_id]
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=f"/outputs/{function_name}",
        )

    async def cancel(self, provider_call_handle_id):
        return None

    async def publish_archive(self, job, *, completed_at):
        self.publish_count += 1
        return FinalArchive(
            state=JobState.SUCCEEDED,
            volume_name="Gromacs-outputs",
            path=f"api-results/{job.run_name}/result.zip",
            filename=f"{job.run_name}.zip",
            size_bytes=100,
            sha256="a" * 64,
            warnings_json="[]",
        )


def _store(tmp_path: Path) -> tuple[ServiceStore, UUID]:
    store = ServiceStore(tmp_path / "service.sqlite3")
    store.initialize()
    auth = AuthService(store, frontend_url="https://biomodals.internal")
    link = auth.create_user("admin@example.com", display_name="Admin", is_admin=True)
    token = link.url.partition("#token=")[2]
    user_id = auth.set_password(
        token,
        "correct horse battery staple",
    ).principal.user_id
    return store, user_id


def _admit(store: ServiceStore, user_id: UUID) -> None:
    options = GromacsJobOptions(simulation_time_ns=5, cpu_only=False)
    plan = execution_plan(
        cpu_only=False,
        workload_run_key="simulation-1",
        pdb_sha256="b" * 64,
        simulation_time_ns=5,
        run_pdbfixer=False,
    )
    store.admit_job(
        owner_user_id=user_id,
        display_name="Simulation 1",
        idempotency_key="one",
        request_hash="c" * 64,
        parameters_json=options.model_dump_json(),
        artifact_request_sha256="d" * 64,
        configuration=JobAdmissionConfiguration(
            workload="gromacs",
            modal_environment=DatabaseOverridableSetting("production", False),
            modal_app_name=DatabaseOverridableSetting("Gromacs", False),
            modal_app_version=DatabaseOverridableSetting(23, False),
            workload_active_job_limit=DatabaseOverridableSetting(10, False),
            global_active_job_limit=DatabaseOverridableSetting(20, False),
        ),
        now=100,
        new_job_id=JOB_ID,
        execution_plan=plan,
        execution_run_id=RUN_ID,
        max_active_provider_calls=3,
        max_active_gpu_provider_calls=1,
        input_content=b"ATOM\n",
    )


def test_gromacs_kernel_advances_parallel_function_waves_and_local_result(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store, user_id = _store(tmp_path)
        _admit(store, user_id)
        adapter = FakeGromacsExecutionAdapter()
        coordinator = GromacsExecutionCoordinator(
            store,
            adapter,
            now=iter(range(110, 200)).__next__,
        )

        for _ in range(4):
            adapter.begin_wave()
            await coordinator.advance(JOB_ID)
            if len(adapter.spawn_waves) == 1:
                running = store.get_job_by_id(JOB_ID)
                assert running is not None
                assert running.state == JobState.RUNNING
                assert [operation.operation for operation in running.operations] == [
                    "prepare_tpr_gpu"
                ]
                assert running.operations[0].modal_call_id == "fc-0"

        assert adapter.spawn_waves == [
            ["prepare_tpr_gpu"],
            [
                "production_run_gpu",
                "collect_traj_stats",
                "collect_traj_stats",
            ],
            ["collect_traj_stats"],
            [],
        ]
        with store.execution_repository() as repository:
            snapshot = repository.snapshot(RUN_ID)
        assert snapshot.run.status == RunStatus.SUCCEEDED
        assert all(
            call.status == ProviderCallStatus.SUCCEEDED
            for call in snapshot.provider_calls
        )
        assert snapshot.run.deployment == DeploymentIdentity(
            "production",
            "Gromacs",
            23,
        )
        assert adapter.publish_count == 1
        assert store.load_job_input(JOB_ID) is None
        job = store.get_job_by_id(JOB_ID)
        assert job is not None
        assert job.state == JobState.SUCCEEDED
        assert [operation.operation for operation in job.operations] == [
            "prepare_tpr_gpu",
            "collect_traj_stats:nvt_",
            "collect_traj_stats:npt_",
            "production_run_gpu",
            "collect_traj_stats:production_",
            "prepare_result",
        ]
        assert job.result_archive_schema_version == GROMACS_ARCHIVE_SCHEMA_VERSION

    asyncio.run(scenario())
