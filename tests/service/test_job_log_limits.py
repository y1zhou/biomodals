"""Job-log stream admission tests."""

import asyncio
from uuid import uuid4

import pytest

from biomodals.service.http_contract import CodedAPIError
from biomodals.service.job_logs_api import _LogStreams


def test_live_log_streams_enforce_and_release_limits() -> None:
    """A closed live stream immediately frees its admission permit."""
    streams = _LogStreams(mode="live", global_limit=2, user_limit=1, job_limit=2)
    user_id = uuid4()
    other_user_id = uuid4()
    job_id = uuid4()

    async def exercise() -> None:
        await streams.acquire(user_id, job_id)
        with pytest.raises(CodedAPIError) as error:
            await streams.acquire(user_id, uuid4())
        assert error.value.status_code == 429
        assert error.value.code == "log_stream_limit"

        await streams.acquire(other_user_id, job_id)
        with pytest.raises(CodedAPIError):
            await streams.acquire(uuid4(), uuid4())

        await streams.release(user_id, job_id)
        await streams.acquire(user_id, uuid4())

    asyncio.run(exercise())


def test_live_log_streams_enforce_job_limit() -> None:
    """One heavily watched Job cannot consume every live stream permit."""
    streams = _LogStreams(mode="live", global_limit=3, user_limit=3, job_limit=1)
    job_id = uuid4()

    async def exercise() -> None:
        await streams.acquire(uuid4(), job_id)
        with pytest.raises(CodedAPIError) as error:
            await streams.acquire(uuid4(), job_id)
        assert error.value.code == "log_stream_limit"

    asyncio.run(exercise())


def test_historical_and_live_streams_have_independent_limits() -> None:
    """Historical fetch fan-out cannot consume live-stream capacity."""
    live = _LogStreams(mode="live", global_limit=1, user_limit=1, job_limit=1)
    historical = _LogStreams(
        mode="historical", global_limit=1, user_limit=1, job_limit=1
    )
    user_id = uuid4()
    job_id = uuid4()

    async def exercise() -> None:
        await historical.acquire(user_id, job_id)
        with pytest.raises(CodedAPIError) as error:
            await historical.acquire(uuid4(), uuid4())
        assert "historical" in error.value.detail

        await live.acquire(user_id, job_id)
        await historical.release(user_id, job_id)
        await live.release(user_id, job_id)

    asyncio.run(exercise())
