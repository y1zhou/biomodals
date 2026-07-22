"""Real HTTP test service with deterministic fake GROMACS compute."""

# This module is started only by the cross-repository Playwright gate.

from __future__ import annotations

import hashlib
import io
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import orjson

from biomodals.service.api import create_app
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthService
from biomodals.service.config import ServiceSettings
from biomodals.service.gromacs import (
    GromacsJobOptions,
    GromacsReconciler,
    create_registration,
)
from biomodals.service.gromacs.modal import FinalArchive, PollOutcome
from biomodals.service.jobs import JobLifecycleLocks
from biomodals.service.runtime_config import (
    ModalConfigurationSnapshot,
    RuntimeConfiguration,
)
from biomodals.service.store import (
    JobOperationRecord,
    JobRecord,
    JobState,
    ServiceStore,
)

ORIGIN = os.environ["BIOMODALS_BROWSER_ORIGIN"]
STAGE_SECONDS = 1.0
EQUILIBRATION_ANALYSIS_SECONDS = 4.0


@dataclass(frozen=True, slots=True)
class _SubmittedCall:
    modal_call_id: str
    run_name: str
    operation: str


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
        self.submit_calls = 0
        self.preflight_versions: list[int] = []
        self.submit_versions: list[int] = []
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
                    "preflight_versions": self.preflight_versions,
                    "submit_calls": self.submit_calls,
                    "submit_versions": self.submit_versions,
                    "provider_calls": len(self.calls),
                    "cancel_calls": len(self.cancelled),
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

    def _call(self, run_name: str, operation: str) -> _SubmittedCall:
        call_id = f"fake-{len(self.calls) + 1}"
        self.calls[call_id] = (time.monotonic(), operation)
        self._write_stats()
        return _SubmittedCall(call_id, run_name, operation)

    async def submit(
        self,
        _pdb_content: bytes,
        options: GromacsJobOptions,
        *,
        run_name: str,
        modal_configuration: ModalConfigurationSnapshot,
    ) -> _SubmittedCall:
        self.submit_calls += 1
        self.submit_versions.append(modal_configuration.app_version)
        self._write_stats()
        operation = "prepare_tpr_cpu" if options.cpu_only else "prepare_tpr_gpu"
        return self._call(run_name, operation)

    async def submit_operation(
        self,
        job: JobRecord,
        operation: str,
    ) -> _SubmittedCall:
        if job.run_name is None:
            raise ValueError("Fake Job has no run name")
        return self._call(job.run_name, operation)

    async def poll(
        self,
        modal_call_id: str,
        *,
        operation: str | None = None,
    ) -> PollOutcome:
        if modal_call_id in self.cancelled:
            return PollOutcome("cancelled")
        started, submitted_operation = self.calls[modal_call_id]
        effective_operation = operation or submitted_operation
        duration = (
            EQUILIBRATION_ANALYSIS_SECONDS
            if effective_operation
            in {"collect_traj_stats:nvt_", "collect_traj_stats:npt_"}
            else STAGE_SECONDS
        )
        if time.monotonic() - started >= duration:
            return PollOutcome("completed")
        return PollOutcome("running")

    async def cancel(self, modal_call_id: str) -> None:
        self.cancelled.add(modal_call_id)
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

    async def read_artifact(self, _job: JobRecord):
        yield self.archive

    async def rebuild_artifact(self, _job: JobRecord):
        yield self.archive

    async def open_operation_logs(
        self,
        _job: JobRecord,
        _operation: JobOperationRecord,
    ):
        async def chunks():
            yield b"Browser test remote log\n"

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
    configuration = RuntimeConfiguration(store, settings)
    auth = AuthService(store, frontend_url=ORIGIN)
    adapter = _FakeGromacsAdapter(root / "stats.json")
    link = auth.create_user(
        "browser-admin@example.com",
        display_name="Browser Administrator",
        is_admin=True,
    )
    adapter.password_link = link.url
    adapter._write_stats()
    lifecycle_locks = JobLifecycleLocks()
    reconciler = GromacsReconciler(
        store,
        cast(Any, adapter),
        lifecycle_locks=lifecycle_locks,
    )
    cache = ArtifactCache(settings.cache_dir / "results")
    return create_app(
        store=store,
        auth=auth,
        configuration=configuration,
        workloads=[
            create_registration(
                cast(Any, adapter),
                reconciler=reconciler,
                lifecycle_locks=lifecycle_locks,
                open_operation_logs=adapter.open_operation_logs,
                preflight=adapter.preflight,
                read_artifact=adapter.read_artifact,
                rebuild_artifact=adapter.rebuild_artifact,
            )
        ],
        allowed_origin=ORIGIN,
        secure_cookies=False,
        cache=cache,
        reconcile_interval_seconds=settings.reconcile_interval_seconds,
    )


app = _create_browser_app()
