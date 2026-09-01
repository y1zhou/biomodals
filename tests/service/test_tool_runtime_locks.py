"""Lifecycle lock ownership tests."""

from __future__ import annotations

import asyncio
import gc
from typing import Any, cast
from uuid import uuid4

import pytest

from biomodals.service.tool_runtime import JobLifecycle


class _MissingStore:
    def get_job_by_id(self, _job_id):
        return None


class _ResolvingStore(_MissingStore):
    def __init__(self) -> None:
        self.called = False

    def resolve_state_unknown(self, *_args, **_kwargs):
        self.called = True
        return object()


def test_job_lock_is_released_after_lifecycle_pass() -> None:
    """Completed lifecycle passes must not retain one lock per Job forever."""
    lifecycle = JobLifecycle(
        cast(Any, _MissingStore()),
        cast(Any, object()),
        (),
        cast(Any, object()),
    )

    with pytest.raises(LookupError):
        asyncio.run(lifecycle.advance(uuid4()))

    gc.collect()
    assert not lifecycle._locks


@pytest.mark.anyio
async def test_admin_resolution_uses_the_job_lifecycle_lock() -> None:
    """Administrative decisions must not race remote lifecycle side effects."""
    store = _ResolvingStore()
    lifecycle = JobLifecycle(
        cast(Any, store),
        cast(Any, object()),
        (),
        cast(Any, object()),
    )
    job_id = uuid4()
    lock = lifecycle._locks.setdefault(job_id, asyncio.Lock())
    await lock.acquire()
    resolution = asyncio.create_task(
        lifecycle.resolve_state_unknown(
            job_id,
            resolution="resume",
            function_call_id="fc-root",
        )
    )

    await asyncio.sleep(0)
    assert not store.called
    lock.release()
    await resolution
    assert store.called
