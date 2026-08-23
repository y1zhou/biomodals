"""Minimal remote execution trust and SDK streaming contracts."""

# ruff: noqa: D101,D102,D103,D107

from types import SimpleNamespace
from uuid import UUID

import pytest

from biomodals.execution import DeploymentIdentity
from biomodals.service.remote_execution import (
    ExecutionLocator,
    RemoteExecutionClient,
    RemoteExecutionIdentityMismatchError,
)

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
OTHER_RUN_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
DEPLOYMENT = DeploymentIdentity("main", "AlphaFold3", 7)
LOCATOR = ExecutionLocator(RUN_ID, DEPLOYMENT)


@pytest.fixture(autouse=True)
def direct_to_thread(monkeypatch):
    async def invoke(function, *arguments, **keywords):
        return function(*arguments, **keywords)

    monkeypatch.setattr(
        "biomodals.service.remote_execution.asyncio.to_thread",
        invoke,
    )


class RemoteMethod:
    def __init__(self, result):
        self.result = result
        self.arguments = None

    def remote(self, *arguments):
        self.arguments = arguments
        return self.result


def _overview(run_id=RUN_ID, deployment=DEPLOYMENT):
    return SimpleNamespace(
        run=SimpleNamespace(execution_run_id=run_id, deployment=deployment)
    )


@pytest.mark.anyio
async def test_every_remote_overview_must_match_the_pinned_locator(
    monkeypatch,
) -> None:
    wrong = _overview(run_id=OTHER_RUN_ID)
    coordinator = SimpleNamespace(
        status=RemoteMethod(wrong),
        cancel=RemoteMethod(wrong),
    )
    monkeypatch.setattr(
        RemoteExecutionClient,
        "_coordinator",
        staticmethod(lambda _locator: coordinator),
    )
    monkeypatch.setattr(
        "biomodals.service.remote_execution.modal.FunctionCall",
        SimpleNamespace(
            from_id=lambda _call_id: SimpleNamespace(get=lambda **_kw: wrong)
        ),
    )
    client = RemoteExecutionClient()

    with pytest.raises(RemoteExecutionIdentityMismatchError):
        await client.poll_root(LOCATOR, "fc-root")
    with pytest.raises(RemoteExecutionIdentityMismatchError):
        await client.status(LOCATOR)
    with pytest.raises(RemoteExecutionIdentityMismatchError):
        await client.cancel(LOCATOR)


@pytest.mark.anyio
async def test_provider_call_uses_one_direct_coordinator_lookup(monkeypatch) -> None:
    call_id = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
    expected = SimpleNamespace(node_key="inference")
    method = RemoteMethod(expected)
    monkeypatch.setattr(
        RemoteExecutionClient,
        "_coordinator",
        staticmethod(lambda _locator: SimpleNamespace(provider_call=method)),
    )

    selected = await RemoteExecutionClient().provider_call(LOCATOR, call_id)

    assert selected is expected
    assert method.arguments == (str(call_id),)


@pytest.mark.anyio
async def test_live_logs_use_one_unbounded_async_sdk_stream(monkeypatch) -> None:
    class Stream:
        arguments = None

        def aio(self, **arguments):
            self.arguments = arguments

            async def entries():
                yield SimpleNamespace(message="one")
                yield SimpleNamespace(message="two")

            return entries()

    stream = Stream()
    logs = SimpleNamespace(stream=stream)
    monkeypatch.setattr(
        "biomodals.service.remote_execution.modal.FunctionCall",
        SimpleNamespace(from_id=lambda _call_id: SimpleNamespace(logs=logs)),
    )

    entries = [
        entry.message
        async for entry in RemoteExecutionClient().log_entries(
            "fc-worker",
            live=True,
        )
    ]

    assert entries == ["one", "two"]
    assert stream.arguments == {"timeout": None}
