"""Minimal remote execution trust and SDK streaming contracts."""

# ruff: noqa: D101,D102,D103,D107

from types import SimpleNamespace
from uuid import UUID

import modal
import pytest

from biomodals.execution import (
    DeploymentIdentity,
    ExecutionOverview,
    ProviderCallStatus,
)
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
    return ExecutionOverview(
        run=SimpleNamespace(execution_run_id=run_id, deployment=deployment),
        nodes=(),
        representative_provider_calls=(),
        active_provider_calls=SimpleNamespace(),
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
@pytest.mark.parametrize("remote_result", [17, {"status": "succeeded"}])
async def test_root_result_must_be_an_execution_overview(
    monkeypatch,
    remote_result,
) -> None:
    monkeypatch.setattr(
        "biomodals.service.remote_execution.modal.FunctionCall",
        SimpleNamespace(
            from_id=lambda _call_id: SimpleNamespace(
                get=lambda **_arguments: remote_result
            )
        ),
    )

    with pytest.raises(
        RemoteExecutionIdentityMismatchError,
        match="not an execution overview",
    ):
        await RemoteExecutionClient().poll_root(LOCATOR, "fc-root")


@pytest.mark.anyio
async def test_missing_root_call_is_an_unknown_remote_identity(monkeypatch) -> None:
    def missing(**_arguments):
        raise modal.exception.NotFoundError("expired")

    monkeypatch.setattr(
        "biomodals.service.remote_execution.modal.FunctionCall",
        SimpleNamespace(
            from_id=lambda _call_id: SimpleNamespace(get=missing),
        ),
    )

    with pytest.raises(
        RemoteExecutionIdentityMismatchError,
        match="root Function Call is unavailable",
    ):
        await RemoteExecutionClient().poll_root(LOCATOR, "fc-expired")


@pytest.mark.anyio
async def test_builtin_root_poll_timeout_means_still_running(monkeypatch) -> None:
    def active(**_arguments):
        raise TimeoutError

    monkeypatch.setattr(
        "biomodals.service.remote_execution.modal.FunctionCall",
        SimpleNamespace(
            from_id=lambda _call_id: SimpleNamespace(get=active),
        ),
    )

    assert await RemoteExecutionClient().poll_root(LOCATOR, "fc-active") is None


@pytest.mark.anyio
async def test_unassigned_active_provider_calls_are_presented_as_queued(
    monkeypatch,
) -> None:
    provider_call = SimpleNamespace(
        provider_call_handle_id="fc-provider",
        status=ProviderCallStatus.RUNNING,
    )
    overview = SimpleNamespace(representative_provider_calls=(provider_call,))
    monkeypatch.setattr(
        "biomodals.service.remote_execution.modal.FunctionCall",
        SimpleNamespace(
            from_id=lambda _call_id: SimpleNamespace(
                get_call_graph=lambda: [
                    SimpleNamespace(
                        function_call_id="fc-provider",
                        task_id="",
                        status=modal.types.InputStatus.PENDING,
                    )
                ]
            )
        ),
    )

    queued = await RemoteExecutionClient().queued_provider_call_handles(overview)

    assert queued == frozenset({"fc-provider"})


@pytest.mark.anyio
async def test_assigned_or_uninspectable_provider_calls_default_to_running(
    monkeypatch,
) -> None:
    calls = (
        SimpleNamespace(
            provider_call_handle_id="fc-assigned",
            status=ProviderCallStatus.RUNNING,
        ),
        SimpleNamespace(
            provider_call_handle_id="fc-unavailable",
            status=ProviderCallStatus.RUNNING,
        ),
    )

    def from_id(call_id):
        if call_id == "fc-unavailable":
            return SimpleNamespace(
                get_call_graph=lambda: (_ for _ in ()).throw(RuntimeError("no graph"))
            )
        return SimpleNamespace(
            get_call_graph=lambda: [
                SimpleNamespace(
                    function_call_id=call_id,
                    task_id="ta-assigned",
                    status=modal.types.InputStatus.PENDING,
                )
            ]
        )

    monkeypatch.setattr(
        "biomodals.service.remote_execution.modal.FunctionCall",
        SimpleNamespace(from_id=from_id),
    )

    assert (
        await RemoteExecutionClient().queued_provider_call_handles(
            SimpleNamespace(representative_provider_calls=calls)
        )
        == frozenset()
    )


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


@pytest.mark.anyio
async def test_closing_live_logs_closes_the_modal_stream(monkeypatch) -> None:
    class Source:
        def __init__(self):
            self.closed = False
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.index += 1
            return SimpleNamespace(message=f"entry-{self.index}")

        async def aclose(self):
            self.closed = True

    source = Source()
    logs = SimpleNamespace(
        stream=SimpleNamespace(aio=lambda **_arguments: source),
    )
    monkeypatch.setattr(
        "biomodals.service.remote_execution.modal.FunctionCall",
        SimpleNamespace(from_id=lambda _call_id: SimpleNamespace(logs=logs)),
    )
    entries = RemoteExecutionClient().log_entries("fc-worker", live=True)

    assert (await anext(entries)).message == "entry-1"
    await entries.aclose()

    assert source.closed is True
