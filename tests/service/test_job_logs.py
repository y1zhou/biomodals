"""Bounded Modal log request contracts."""

# ruff: noqa: D103

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import HTTPException

from biomodals.execution import (
    DeploymentIdentity,
    ProviderCallDiagnostic,
    ProviderCallPage,
    ProviderCallStatus,
)
from biomodals.service.job_logs_api import (
    _redact_provider_call_id,
    _stage_calls,
    _validate_window_parameters,
)
from biomodals.service.remote_execution import ExecutionLocator
from biomodals.service.tools import HUMANIZATION_TOOL


def _call(status: ProviderCallStatus) -> ProviderCallDiagnostic:
    return ProviderCallDiagnostic(
        provider_call_id=uuid4(),
        node_key="node",
        function_name="function",
        status=status,
        provider_call_handle_id="fc-test",
        created_at=1,
        started_at=2,
        completed_at=3 if status.is_terminal else None,
    )


def test_live_log_stream_needs_no_historical_window() -> None:
    assert _validate_window_parameters(since=None, until=None)


@pytest.mark.parametrize(
    ("since", "until"),
    [
        (None, datetime.now(UTC)),
        (datetime.now(UTC), None),
        (datetime.now(), datetime.now()),
        (datetime.now(UTC), datetime.now(UTC) - timedelta(seconds=1)),
        (datetime.now(UTC), datetime.now(UTC) + timedelta(hours=2)),
    ],
)
def test_historical_log_window_is_complete_aware_ordered_and_bounded(
    since: datetime | None,
    until: datetime | None,
) -> None:
    with pytest.raises(HTTPException) as captured:
        _validate_window_parameters(since=since, until=until)
    assert captured.value.status_code == 422


def test_historical_window_can_read_an_active_call() -> None:
    since = datetime.now(UTC)
    assert not _validate_window_parameters(
        since=since,
        until=since + timedelta(minutes=10),
    )


@pytest.mark.anyio
async def test_private_provider_call_id_is_redacted_across_chunks() -> None:
    async def chunks():
        yield b'{"message":"call fc-'
        yield b'test finished"}\n'

    content = b"".join([
        chunk async for chunk in _redact_provider_call_id(chunks(), "fc-test")
    ])

    assert content == b'{"message":"call [function-call-id-redacted] finished"}\n'


@pytest.mark.anyio
async def test_stage_targets_query_only_mapped_nodes_newest_first() -> None:
    calls = {
        "first": _call(ProviderCallStatus.SUCCEEDED),
        "second": _call(ProviderCallStatus.RUNNING),
    }

    class Remote:
        requested: list[tuple[str | None, bool]] = []

        async def provider_calls(
            self,
            _locator,
            *,
            node_key=None,
            limit=50,
            newest_first=False,
        ):
            self.requested.append((node_key, newest_first))
            return ProviderCallPage((calls[node_key],), None)

    remote = Remote()
    selected = await _stage_calls(
        remote,  # type: ignore[arg-type]
        ExecutionLocator(
            uuid4(),
            DeploymentIdentity("main", "Tool", 1),
        ),
        node_keys=("first", "second"),
        limit=10,
    )

    assert remote.requested == [("first", True), ("second", True)]
    assert set(selected) == set(calls.values())


@pytest.mark.anyio
@pytest.mark.parametrize("method", ["sapiens", "humatch", "pabnativ2", "hudiff_ab"])
async def test_humanization_log_target_selects_only_its_method(method):
    stage = next(
        stage
        for stage in HUMANIZATION_TOOL.stages
        if stage.code == f"generate_{method}"
    )
    requested = []

    class Remote:
        async def provider_calls(self, locator, *, node_key, limit, newest_first):
            requested.append(node_key)
            return ProviderCallPage((), None)

    await _stage_calls(
        Remote(),
        ExecutionLocator(
            uuid4(), DeploymentIdentity("main", "HumanizationWorkflow", 1)
        ),
        node_keys=stage.node_keys,
        limit=10,
    )
    assert requested == [f"generate_{method}"]
