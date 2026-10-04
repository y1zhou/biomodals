"""Real ASGI response lifetimes must release owned files and log capacity."""

import asyncio
import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import orjson
import pytest
from starlette.requests import ClientDisconnect

from biomodals.execution import ProviderCallDiagnostic, ProviderCallStatus
from biomodals.service.artifacts import ArtifactCache
from biomodals.service.job_logs_api import create_job_logs_router
from biomodals.service.jobs_api import _response
from biomodals.service.tools import HUMANIZATION_TOOL


async def deliver(response, version, disconnect):
    """Exercise the actual ASGI send/disconnect boundary."""
    disconnected = asyncio.Event()
    messages = []

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message["type"])
        if message["type"] == disconnect:
            if version == "2.4":
                raise OSError("Client closed connection")
            disconnected.set()
            await asyncio.Event().wait()

    try:
        await response(
            {"type": "http", "asgi": {"spec_version": version}}, receive, send
        )
    except ClientDisconnect:
        assert disconnect is not None
    return messages


@pytest.mark.parametrize("version", ["2.3", "2.4"])
@pytest.mark.parametrize(
    "disconnect", ["http.response.start", "http.response.body", None]
)
def test_response_releases_archive_and_log_capacity(tmp_path, version, disconnect):
    """Repeated disconnects leave archives removable and log admission usable."""

    async def scenario():
        job_id, user_id, target = uuid4(), uuid4(), uuid4()
        content = b"verified archive bytes"
        cache = ArtifactCache(tmp_path)
        (tmp_path / f"{job_id}.result").write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        job = SimpleNamespace(
            job_id=job_id,
            tool="humanization",
            deleted_at=None,
            modal_environment="main",
            modal_app_name="HumanizationWorkflow",
            modal_app_version=1,
        )
        call = ProviderCallDiagnostic(
            provider_call_id=target,
            node_key="generate_sapiens",
            function_name="function",
            status=ProviderCallStatus.RUNNING,
            provider_call_handle_id="fc-example",
            created_at=1,
            started_at=2,
            completed_at=None,
        )

        class Remote:
            async def provider_call(self, *args):
                return call

            async def log_entries(self, *args, **kwargs):
                yield SimpleNamespace(
                    timestamp=datetime.now(UTC), message="model output", source="stdout"
                )

        state = SimpleNamespace(
            remote_execution=Remote(),
            store=SimpleNamespace(get_job=lambda *args: job),
            registrations={
                "humanization": SimpleNamespace(definition=HUMANIZATION_TOOL)
            },
            configuration=SimpleNamespace(
                tool=lambda key: SimpleNamespace(
                    job_logs_visible_to_owner=SimpleNamespace(value=True)
                )
            ),
        )
        request = SimpleNamespace(app=SimpleNamespace(state=state))
        session = SimpleNamespace(
            principal=SimpleNamespace(user_id=user_id, is_admin=False)
        )
        router = create_job_logs_router()
        logs = next(
            route.endpoint for route in router.routes if route.path.endswith("/logs")
        )
        transcripts = []
        try:
            for _ in range(6):
                lease = cache.acquire(
                    str(job_id), size_bytes=len(content), sha256=digest
                )
                response = _response(
                    lease,
                    filename="result.zip",
                    media_type="application/zip",
                    size_bytes=len(content),
                    sha256=digest,
                    range_header=None,
                )
                transcripts.append(await deliver(response, version, disconnect))
                with pytest.raises(ValueError, match="closed"):
                    lease.read()
                response = await logs(request, job_id, session, target, None, None)
                transcripts.append(await deliver(response, version, disconnect))
            assert cache.remove_job_files(str(job_id))
            assert not (tmp_path / f"{job_id}.result").exists()
            (tmp_path / "response-lifecycle.json").write_bytes(
                orjson.dumps(transcripts)
            )
        finally:
            await cache.shutdown()

    asyncio.run(scenario())
