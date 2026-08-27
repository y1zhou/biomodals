"""Real HTTP test service with deterministic fake remote coordination."""

# This module is started only by the cross-repository Playwright gate.

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import time
import zipfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid5

import orjson

from biomodals.app.bioinfo.gromacs_execution_runtime import GromacsExecutionRequest
from biomodals.execution import (
    ActiveProviderCallCounts,
    DeploymentIdentity,
    ExecutionNodeRecord,
    ExecutionOverview,
    ExecutionRunRecord,
    NodeStatus,
    NodeTaskStatusCounts,
    ProviderCallDiagnostic,
    ProviderCallPage,
    ProviderCallStatus,
    RunStatus,
)
from biomodals.execution.model import ProviderCallOverview
from biomodals.service.alphafold3.router import create_router as af3_router
from biomodals.service.alphafold3.validation import ValidatedInputStore
from biomodals.service.api import create_app
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.auth import AuthService
from biomodals.service.billing import BillingSummary, CostGroup
from biomodals.service.config import ServiceSettings
from biomodals.service.gromacs.router import create_router as gromacs_router
from biomodals.service.pending import PendingRequestStore
from biomodals.service.remote_execution import ExecutionLocator
from biomodals.service.runtime_config import RuntimeConfiguration
from biomodals.service.store import JobRecord, ServiceStore
from biomodals.service.tool_runtime import (
    JobLifecycle,
    PreparedResult,
    ToolRegistration,
)
from biomodals.service.tools import ALPHAFOLD3_TOOL, GROMACS_TOOL, TOOLS

ORIGIN = os.environ["BIOMODALS_BROWSER_ORIGIN"]
_CALL_NAMESPACE = UUID("156d600f-2a56-4ce7-8d0c-886bfa35698a")


def _result_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("input.pdb", "ATOM\nEND\n")
        archive.writestr("outputs/trajectory_nopbc.xtc", b"trajectory")
        archive.writestr("metadata/manifest.json", "{}\n")
    return buffer.getvalue()


class _FakeRemote:
    """Mimic a deployed coordinator while retaining the real service boundary."""

    _functions = {
        "prepare_tpr_gpu": "prepare_tpr_gpu",
        "collect_traj_stats:nvt_": "collect_traj_stats",
        "collect_traj_stats:npt_": "collect_traj_stats",
        "production_run_gpu": "production_run_gpu",
        "collect_traj_stats:production_": "collect_traj_stats",
    }

    def __init__(self, stats_path: Path) -> None:
        self.stats_path = stats_path
        self.password_link = ""
        self.secondary_password_link = ""
        self.preflight_versions: list[int] = []
        self.submit_versions: list[int] = []
        self.started: dict[UUID, tuple[float, int, object]] = {}
        self.cancelled: set[UUID] = set()
        self.log_fetches = 0
        self._write_stats()

    def bind_plan(self, job_id: UUID, plan: object) -> None:
        self.started[job_id] = (0, 0, plan)

    def _write_stats(self) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.stats_path.with_suffix(".tmp")
        active_calls = sum(len(self._calls(run_id)) for run_id in self.started)
        temporary.write_bytes(
            orjson.dumps(
                {
                    "password_link": self.password_link,
                    "secondary_password_link": self.secondary_password_link,
                    "preflight_versions": self.preflight_versions,
                    "submit_calls": len(self.submit_versions),
                    "submit_versions": self.submit_versions,
                    "provider_calls": active_calls,
                    "cancel_calls": len(self.cancelled),
                    "log_fetches": self.log_fetches,
                },
                option=orjson.OPT_SORT_KEYS,
            )
        )
        temporary.replace(self.stats_path)

    async def preflight(self, deployment: DeploymentIdentity) -> None:
        self.preflight_versions.append(deployment.deployment_version)
        self._write_stats()

    async def launch(self, locator: ExecutionLocator) -> str:
        _, _, plan = self.started[locator.execution_run_id]
        self.started[locator.execution_run_id] = (
            time.monotonic(),
            int(time.time()),
            plan,
        )
        self.submit_versions.append(locator.deployment.deployment_version)
        self._write_stats()
        return f"fake-root-{locator.execution_run_id}"

    async def poll_root(
        self, _locator: ExecutionLocator, function_call_id: str
    ) -> ExecutionOverview | None:
        run_id = UUID(function_call_id.removeprefix("fake-root-"))
        return self._overview(run_id)

    async def status(self, locator: ExecutionLocator) -> ExecutionOverview:
        return self._overview(locator.execution_run_id)

    async def cancel(self, locator: ExecutionLocator) -> ExecutionOverview:
        self.cancelled.add(locator.execution_run_id)
        self._write_stats()
        return self._overview(locator.execution_run_id)

    async def provider_calls(
        self,
        locator: ExecutionLocator,
        *,
        node_key: str | None = None,
        cursor: UUID | None = None,
        limit: int = 50,
        newest_first: bool = False,
    ) -> ProviderCallPage:
        calls = [
            call
            for call in self._calls(locator.execution_run_id)
            if node_key is None or call.node_key == node_key
        ]
        if newest_first:
            calls.reverse()
        if cursor is not None:
            calls = calls[
                next(
                    (
                        index + 1
                        for index, call in enumerate(calls)
                        if call.provider_call_id == cursor
                    ),
                    len(calls),
                ) :
            ]
        selected = calls[:limit]
        return ProviderCallPage(
            tuple(selected),
            selected[-1].provider_call_id if len(calls) > limit else None,
        )

    async def provider_call(
        self,
        locator: ExecutionLocator,
        provider_call_id: UUID,
        *,
        node_key: str | None = None,
    ) -> ProviderCallDiagnostic | None:
        return next(
            (
                call
                for call in self._calls(locator.execution_run_id)
                if call.provider_call_id == provider_call_id
                and (node_key is None or call.node_key == node_key)
            ),
            None,
        )

    async def log_entries(self, _call_id: str, *, live: bool, **_window):
        self.log_fetches += 1
        self._write_stats()
        yield SimpleNamespace(
            timestamp=datetime(2026, 7, 22, 6, 5, 33, tzinfo=UTC),
            message="\x1b[31mBrowser test remote log\x1b[0m\n",
            source="stdout",
        )
        yield SimpleNamespace(
            timestamp=datetime(2026, 7, 22, 6, 5, 34, tzinfo=UTC),
            message="A" * 4_096 + "\n",
            source="stdout",
        )
        if live:
            await asyncio.Event().wait()

    def _elapsed(self, run_id: UUID) -> float:
        started, _, _ = self.started[run_id]
        return max(0, time.monotonic() - started) if started else 0

    def _node_status(self, run_id: UUID, node_key: str) -> NodeStatus:
        if run_id in self.cancelled:
            return NodeStatus.CANCELLED
        elapsed = self._elapsed(run_id)
        windows = {
            "prepare_tpr_gpu": (0, 1),
            "collect_traj_stats:nvt_": (1, 5),
            "collect_traj_stats:npt_": (1, 5),
            "production_run_gpu": (1, 2.5),
            "collect_traj_stats:production_": (2.5, 5.5),
            "prepare_result": (5.5, 6),
        }
        start, end = windows[node_key]
        if elapsed < start:
            return NodeStatus.PENDING
        if elapsed < end:
            return NodeStatus.RUNNING
        return NodeStatus.SUCCEEDED

    def _calls(self, run_id: UUID) -> list[ProviderCallDiagnostic]:
        _, started_at, _ = self.started[run_id]
        calls = []
        for node_key, function_name in self._functions.items():
            status = self._node_status(run_id, node_key)
            if status == NodeStatus.PENDING:
                continue
            provider_status = (
                ProviderCallStatus.SUCCEEDED
                if status == NodeStatus.SUCCEEDED
                else ProviderCallStatus.RUNNING
            )
            provider_id = uuid5(_CALL_NAMESPACE, f"{run_id}:{node_key}")
            calls.append(
                ProviderCallDiagnostic(
                    provider_call_id=provider_id,
                    node_key=node_key,
                    function_name=function_name,
                    status=provider_status,
                    provider_call_handle_id=f"fake-{provider_id}",
                    created_at=started_at,
                    started_at=started_at,
                    completed_at=started_at + 1
                    if provider_status.is_terminal
                    else None,
                )
            )
        return calls

    def _overview(self, run_id: UUID) -> ExecutionOverview:
        _, started_at, plan = self.started[run_id]
        cancelled = run_id in self.cancelled
        status = (
            RunStatus.CANCELLED
            if cancelled
            else (
                RunStatus.SUCCEEDED if self._elapsed(run_id) >= 6 else RunStatus.RUNNING
            )
        )
        deployment = DeploymentIdentity("main", "Gromacs", 7)
        nodes = tuple(
            ExecutionNodeRecord(
                execution_run_id=run_id,
                node_key=node.node_key,
                ordinal=ordinal,
                dependencies=node.dependencies,
                aggregation_policy=node.aggregation_policy,
                allow_empty_result=node.allow_empty_result,
                status=self._node_status(run_id, node.node_key),
                status_reason=None,
                discovery_complete=True,
                result_observation=None,
                result_observed_at=None,
                result_provenance=None,
                error_message=None,
                created_at=started_at,
                updated_at=int(time.time()),
                started_at=started_at,
                completed_at=int(time.time()) if status.is_terminal else None,
            )
            for ordinal, node in enumerate(plan.nodes)
        )
        calls = self._calls(run_id)
        return ExecutionOverview(
            run=ExecutionRunRecord(
                execution_run_id=run_id,
                predecessor_execution_run_id=None,
                plan=plan,
                deployment=deployment,
                status=status,
                status_reason=None,
                status_message=None,
                max_active_provider_calls=3,
                max_active_gpu_provider_calls=1,
                created_at=started_at,
                updated_at=int(time.time()),
                started_at=started_at,
                completed_at=int(time.time()) if status.is_terminal else None,
            ),
            nodes=nodes,
            representative_provider_calls=tuple(
                ProviderCallOverview(
                    call.node_key,
                    str(call.provider_call_id),
                    call.status,
                    call.provider_call_handle_id,
                    None,
                )
                for call in calls
            ),
            active_provider_calls=ActiveProviderCallCounts(
                total=sum(not call.status.is_terminal for call in calls),
                gpu=sum(
                    call.function_name in {"prepare_tpr_gpu", "production_run_gpu"}
                    and not call.status.is_terminal
                    for call in calls
                ),
            ),
            node_task_status_counts=tuple(
                NodeTaskStatusCounts(node.node_key, **{node.status.value: 1})
                for node in nodes
            ),
        )


class _FakeAdapter:
    def __init__(
        self, remote: _FakeRemote, pending: PendingRequestStore, archive: bytes
    ) -> None:
        self.remote = remote
        self.pending = pending
        self.archive = archive

    async def stage(self, job: JobRecord) -> None:
        content = self.pending.get(job.job_id)
        if content is None:
            raise FileNotFoundError("Browser test request is unavailable")
        self.remote.bind_plan(
            job.job_id, GromacsExecutionRequest.from_bytes(content).execution_plan
        )

    async def discard_pending(self, job: JobRecord) -> None:
        self.pending.delete(job.job_id)

    async def prepare_result(
        self, job: JobRecord, cache: ArtifactCache, *, completed_at: int
    ) -> PreparedResult:
        del completed_at
        digest = hashlib.sha256(self.archive).hexdigest()

        async def chunks():
            yield self.archive

        lease = await cache.store(
            str(job.job_id),
            size_bytes=len(self.archive),
            sha256=digest,
            chunks=chunks(),
        )
        lease.close()
        return PreparedResult(
            "result.zip",
            "application/zip",
            len(self.archive),
            digest,
            "gromacs-result/1",
        )


class _UnusedAdapter:
    async def stage(self, _job):
        raise AssertionError("AlphaFold3 is not submitted by this browser fixture")

    async def discard_pending(self, _job):
        return None

    async def prepare_result(self, _job, _cache, *, completed_at):
        raise AssertionError("AlphaFold3 is not submitted by this browser fixture")


class _FakeBilling:
    async def report(self, **_kwargs) -> BillingSummary:
        return BillingSummary(
            total=Decimal("1.25"),
            environments=(CostGroup("main", Decimal("1.25")),),
            tools=(CostGroup("gromacs", Decimal("1.00")),),
            other=Decimal("0.25"),
        )


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
    pending = PendingRequestStore(settings.state_dir)
    pending.initialize()
    validations = ValidatedInputStore(settings.state_dir)
    validations.initialize()
    remote = _FakeRemote(root / "stats.json")
    link = auth.create_user(
        "browser-admin@example.com", display_name="Browser Administrator", is_admin=True
    )
    remote.password_link = link.url
    secondary_link = auth.create_user(
        "browser-user@example.com", display_name="Browser Regular User"
    )
    remote.secondary_password_link = secondary_link.url
    remote._write_stats()
    cache = ArtifactCache(settings.cache_dir / "results")
    registrations = (
        ToolRegistration(
            GROMACS_TOOL, _FakeAdapter(remote, pending, _result_archive())
        ),
        ToolRegistration(ALPHAFOLD3_TOOL, _UnusedAdapter()),
    )
    configuration = RuntimeConfiguration(store, settings, tool_definitions=TOOLS)
    lifecycle = JobLifecycle(store, remote, registrations, cache)
    app = create_app(
        store=store,
        auth=auth,
        configuration=configuration,
        registrations=registrations,
        tool_routers=(
            gromacs_router(
                store=store,
                configuration=configuration,
                pending=pending,
                remote=remote,
                lifecycle=lifecycle,
            ),
            af3_router(
                store=store,
                configuration=configuration,
                validations=validations,
                adapter=registrations[1].adapter,
                remote=remote,
                lifecycle=lifecycle,
            ),
        ),
        remote=remote,
        lifecycle=lifecycle,
        cache=cache,
        allowed_origin=ORIGIN,
        secure_cookies=False,
        reconcile_interval_seconds=settings.reconcile_interval_seconds,
    )
    app.state.pending_requests = pending
    app.state.validated_inputs = validations
    app.state.billing = _FakeBilling()
    return app


app = _create_browser_app()
