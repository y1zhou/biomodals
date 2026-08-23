"""Bounded Modal log request contracts."""

# ruff: noqa: D103

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import HTTPException

from biomodals.execution import ProviderCallDiagnostic, ProviderCallStatus
from biomodals.service.job_logs_api import _validate_window


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
    assert _validate_window(
        _call(ProviderCallStatus.RUNNING),
        since=None,
        until=None,
    )


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
        _validate_window(
            _call(ProviderCallStatus.SUCCEEDED),
            since=since,
            until=until,
        )
    assert captured.value.status_code == 422


def test_historical_window_can_read_an_active_call() -> None:
    since = datetime.now(UTC)
    assert not _validate_window(
        _call(ProviderCallStatus.RUNNING),
        since=since,
        until=since + timedelta(minutes=10),
    )
