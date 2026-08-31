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
