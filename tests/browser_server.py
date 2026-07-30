"""Real HTTP test service with deterministic fake GROMACS compute."""

# This module is started only by the cross-repository Playwright gate.

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import time
import zipfile
from pathlib import Path

import orjson

from biomodals.execution.modal import (
    ModalCallObservation,
    ModalCallObservationKind,
)
from biomodals.service.api import create_app
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthService
from biomodals.service.config import ServiceSettings
from biomodals.service.gromacs import (
    GromacsExecutionCoordinator,
    create_registration,
)
from biomodals.service.gromacs.modal import FinalArchive
from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import (
    JobOperationRecord,
    JobOperationState,
    JobRecord,
    JobState,
    ServiceStore,
)

ORIGIN = os.environ["BIOMODALS_BROWSER_ORIGIN"]
STAGE_SECONDS = 1.0
EQUILIBRATION_ANALYSIS_SECONDS = 4.0


def _result_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("input.pdb", "ATOM\nEND\n")
        archive.writestr("outputs/trajectory_nopbc.xtc", b"trajectory")
        archive.writestr("metadata/manifest.json", "{}\n")
    return buffer.getvalue()


class _FakeGromacsAdapter:
    """Fake only the established Modal adapter seam, never an HTTP route."""

    def __init__(self, stats_path: Path) -> None:
        self.stats_path = stats_path
        self.password_link = ""
        self.secondary_password_link = ""
        self.submit_calls = 0
        self.preflight_versions: list[int] = []
        self.submit_versions: list[int] = []
        self.log_fetches = 0
        self.calls: dict[str, tuple[float, str]] = {}
        self.cancelled: set[str] = set()
        self.archive = _result_archive()
        self._write_stats()

    def _write_stats(self) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.stats_path.with_suffix(".tmp")
        temporary.write_bytes(
            orjson.dumps(
                {
                    "password_link": self.password_link,
                    "secondary_password_link": self.secondary_password_link,
                    "preflight_versions": self.preflight_versions,
                    "submit_calls": self.submit_calls,
                    "submit_versions": self.submit_versions,
                    "provider_calls": len(self.calls),
                    "cancel_calls": len(self.cancelled),
                    "log_fetches": self.log_fetches,
                },
                option=orjson.OPT_SORT_KEYS,
            )
        )
        temporary.replace(self.stats_path)

    async def preflight(
        self,
        _app_name: str,
        _environment_name: str,
        app_version: int,
    ) -> None:
        self.preflight_versions.append(app_version)
        self._write_stats()

    async def resolve(self, binding):
        return binding

    async def spawn(self, function, *, args, kwargs) -> str:
        del args
        operation = function.function_name
        if operation == "collect_traj_stats":
            operation = f"collect_traj_stats:{kwargs['traj_prefix']}"
        call_id = f"fake-{len(self.calls) + 1}"
        self.calls[call_id] = (time.monotonic(), operation)
        if operation.startswith("prepare_tpr_"):
            self.submit_calls += 1
            self.submit_versions.append(function.app_version)
        self._write_stats()
        return call_id

    async def observe(
        self,
        provider_call_handle_id: str,
    ) -> ModalCallObservation:
        if provider_call_handle_id in self.cancelled:
            return ModalCallObservation(ModalCallObservationKind.CANCELLED)
        started, submitted_operation = self.calls[provider_call_handle_id]
        duration = (
            EQUILIBRATION_ANALYSIS_SECONDS
            if submitted_operation
            in {"collect_traj_stats:nvt_", "collect_traj_stats:npt_"}
            else STAGE_SECONDS
        )
        if time.monotonic() - started >= duration:
            return ModalCallObservation(
                ModalCallObservationKind.SUCCEEDED,
                result=f"/outputs/{submitted_operation}",
            )
        return ModalCallObservation(ModalCallObservationKind.RUNNING)

    async def cancel(self, provider_call_handle_id: str) -> None:
        self.cancelled.add(provider_call_handle_id)
        self._write_stats()

    async def publish_archive(
        self,
        job: JobRecord,
        *,
        completed_at: int,
    ) -> FinalArchive:
        del completed_at
        digest = hashlib.sha256(self.archive).hexdigest()
        return FinalArchive(
            state=JobState.SUCCEEDED,
            volume_name="browser-test-results",
            path=f"results/{job.job_id}.zip",
            filename="result.zip",
            size_bytes=len(self.archive),
            sha256=digest,
            warnings_json="[]",
        )

    async def recover_archive(self, job: JobRecord) -> FinalArchive:
        return await self.publish_archive(job, completed_at=0)

    async def cleanup_intermediates(self, job: JobRecord) -> None:
        del job
        return None

    async def read_artifact(self, _job: JobRecord):
        yield self.archive

    async def rebuild_artifact(self, _job: JobRecord):
        yield self.archive

    async def open_operation_logs(
        self,
        _job: JobRecord,
        operation: JobOperationRecord,
        _selection,
    ):
        self.log_fetches += 1
        self._write_stats()

        async def chunks():
            await asyncio.sleep(0.25)
            yield (
                b"2026-07-22 14:05:33+08:00 \x1b[31mBrowser test remote log\x1b[0m\n"
            )
            yield (b"2026-07-22 14:05:34+08:00 " + b"A" * 4_096 + b"\n")
            if operation.state in {
                JobOperationState.RUNNING,
                JobOperationState.STATE_UNKNOWN,
            }:
                await asyncio.Event().wait()

        return chunks()


def _create_browser_app():
    root = Path(os.environ["BIOMODALS_BROWSER_ROOT"])
    settings = ServiceSettings.from_environment({
        "MODAL_TOKEN_ID": "browser-test-token-id",
        "MODAL_TOKEN_SECRET": "browser-test-token-secret",
        "BIOMODALS_STATE_DIR": str(root / "state"),
        "BIOMODALS_CACHE_DIR": str(root / "cache"),
        "BIOMODALS_PUBLIC_URL": ORIGIN,
        "BIOMODALS_SECURE_COOKIES": "false",
        "BIOMODALS_GROMACS_APP_VERSION": "7",
        "BIOMODALS_RECONCILE_SECONDS": "0.1",
    })
    store = ServiceStore(settings.database_path)
    store.initialize()
    auth = AuthService(store, frontend_url=ORIGIN)
    adapter = _FakeGromacsAdapter(root / "stats.json")
    link = auth.create_user(
        "browser-admin@example.com",
        display_name="Browser Administrator",
        is_admin=True,
    )
    adapter.password_link = link.url
    secondary_link = auth.create_user(
        "browser-user@example.com",
        display_name="Browser Regular User",
    )
    adapter.secondary_password_link = secondary_link.url
    adapter._write_stats()
    lifecycle_locks = JobLifecycleLocks()
    reconciler = GromacsExecutionCoordinator(
        store,
        adapter,
        lifecycle_locks=lifecycle_locks,
    )
    cache = ArtifactCache(settings.cache_dir / "results")
    workloads = [
        create_registration(
            adapter,
            reconciler=reconciler,
            lifecycle_locks=lifecycle_locks,
            open_operation_logs=adapter.open_operation_logs,
            preflight=adapter.preflight,
            read_artifact=adapter.read_artifact,
            rebuild_artifact=adapter.rebuild_artifact,
        )
    ]
    configuration = RuntimeConfiguration(
        store,
        settings,
        workload_definitions=[workload.definition for workload in workloads],
    )
    return create_app(
        store=store,
        auth=auth,
        configuration=configuration,
        workloads=workloads,
        allowed_origin=ORIGIN,
        secure_cookies=False,
        cache=cache,
        reconcile_interval_seconds=settings.reconcile_interval_seconds,
    )


app = _create_browser_app()
