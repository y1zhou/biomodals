"""Real API composition for credential-mode browser E2E verification."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from pwdlib import PasswordHash
from service.antibody_fixture import reference_csv

from biomodals.execution.model import (
    ActiveProviderCallCounts,
    ExecutionOverview,
    ExecutionPlan,
    ExecutionRunRecord,
    RunStatus,
)
from biomodals.service.antibody_sequence_analysis.reference import TherapeuticReference
from biomodals.service.api import create_deployed_app
from biomodals.service.remote_execution import (
    RemoteDeploymentUnavailableError,
    RemoteExecutionClient,
)

ROOT = Path(os.environ["BIOMODALS_BROWSER_ROOT"])
ORIGIN = os.environ["BIOMODALS_BROWSER_ORIGIN"]
MODE = os.environ.get("BIOMODALS_BROWSER_CREDENTIALS", "absent")
PASSWORD = "-".join(("local", "tools", "e2e", "passphrase"))
CALLS: list[str] = []

# Never inherit the developer's provider credentials or deployment settings.
for key in tuple(os.environ):
    if key.startswith(("BIOMODALS_", "MODAL_")):
        del os.environ[key]
os.environ.update({
    "BIOMODALS_STATE_DIR": str(ROOT / "state"),
    "BIOMODALS_CACHE_DIR": str(ROOT / "cache"),
    "BIOMODALS_PUBLIC_URL": ORIGIN,
    "BIOMODALS_SECURE_COOKIES": "false",
    "BIOMODALS_DEFAULT_ADMIN_EMAIL": "local-admin@example.com",
    "BIOMODALS_DEFAULT_ADMIN_PASSWORD_HASH": PasswordHash.recommended().hash(PASSWORD),
    "BIOMODALS_RECONCILE_SECONDS": "0.1",
})
os.environ.update(
    {
        "absent": {},
        "id-only": {"MODAL_TOKEN_ID": "fixture-id"},
        "secret-only": {"MODAL_TOKEN_SECRET": "fixture-secret"},
        "whitespace": {"MODAL_TOKEN_ID": "  ", "MODAL_TOKEN_SECRET": "  "},
        "complete": {
            "MODAL_TOKEN_ID": "fixture-id",
            "MODAL_TOKEN_SECRET": "fixture-secret",
        },
        "rejected": {
            "MODAL_TOKEN_ID": "fixture-id",
            "MODAL_TOKEN_SECRET": "fixture-secret",
        },
    }[MODE]
)


def block_network(event, args):
    """Trap outbound network access, including SDK paths outside the adapter."""
    if event == "socket.connect":
        CALLS.append("unexpected_network")
        raise RuntimeError("Outbound networking is forbidden in this E2E fixture")


sys.addaudithook(block_network)


async def preflight(self, deployment):
    """Model only provider resolution; never alter the real app assembly."""
    CALLS.append("preflight")
    if MODE != "complete":
        raise RemoteDeploymentUnavailableError("Fixture rejected provider preflight")


async def poll_root(self, locator, function_call_id):
    """Keep the retained fixture Job running when reconciliation is enabled."""
    CALLS.append("poll_root")
    if MODE != "complete":
        raise AssertionError("Local mode attempted provider reconciliation")
    return None


async def remote_status(self, locator):
    """Return a running provider observation when the browser reads old Jobs."""
    CALLS.append("status")
    if MODE != "complete":
        raise AssertionError("Local mode attempted provider status")
    now = int(time.time())
    return ExecutionOverview(
        run=ExecutionRunRecord(
            execution_run_id=locator.execution_run_id,
            predecessor_execution_run_id=None,
            plan=ExecutionPlan(workload_name="gromacs", nodes=()),
            deployment=locator.deployment,
            status=RunStatus.RUNNING,
            status_reason=None,
            status_message=None,
            max_active_provider_calls=1,
            max_active_gpu_provider_calls=1,
            created_at=now,
            updated_at=now,
            started_at=now,
            completed_at=None,
        ),
        nodes=(),
        representative_provider_calls=(),
        active_provider_calls=ActiveProviderCallCounts(total=1, gpu=0),
    )


RemoteExecutionClient.preflight = preflight
RemoteExecutionClient.poll_root = poll_root
RemoteExecutionClient.status = remote_status
app = create_deployed_app()
store = app.state.store
owner = store.get_user_by_email("local-admin@example.com")
assert owner is not None
admission = store.admit_job(
    owner_user_id=owner.user_id,
    tool="gromacs",
    display_name="Retained remote Job",
    idempotency_key="local-mode-existing-job",
    request_digest="a" * 64,
    modal_environment="production",
    modal_app_name="fixture-gromacs",
    modal_app_version=1,
    tool_active_job_limit=10,
    global_active_job_limit=10,
    max_active_provider_calls=1,
    max_active_gpu_provider_calls=1,
    now=int(time.time()) - 120,
)
JOB_ID = admission.job.job_id
if admission.created:
    store.record_launch(
        JOB_ID, function_call_id="fc-fixture", now=int(time.time()) - 120
    )
app.state.antibody_analysis.reference = TherapeuticReference(
    ROOT / "reference.json", download=reference_csv
)


@app.get("/__test/status", include_in_schema=False)
async def fixture_status():
    """Expose test evidence without adding instrumentation to production."""
    task = getattr(app.state, "reconciler_task", None)
    return {
        "provider_calls": CALLS,
        "reconciler_running": task is not None and not task.done(),
        "job": asdict(store.get_job_by_id(JOB_ID)),
    }
