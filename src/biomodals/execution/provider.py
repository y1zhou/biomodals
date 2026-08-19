"""Provider-neutral call lifecycle contracts."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from biomodals.execution.model import ProviderBinding

if sys.version_info >= (3, 11):  # noqa: UP036 - mounted in Python 3.10 apps
    from enum import StrEnum
else:
    from backports.strenum import StrEnum  # noqa: UP035,I001


class ProviderDefiniteSubmissionError(RuntimeError):
    """Raised when a provider conclusively rejects work before it starts."""


class ProviderSubmissionOutcomeUnknownError(RuntimeError):
    """Raised when work may have started without returning a durable handle."""


class ProviderDeploymentUnavailableError(RuntimeError):
    """Raised when an exact provider deployment cannot be resolved."""


class ProviderCallObservationKind(StrEnum):
    """One provider-neutral observation of retained remote work."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STATE_UNKNOWN = "state_unknown"


@dataclass(frozen=True)
class ProviderCallObservation:
    """One nonblocking observation of a concrete Provider Call."""

    kind: ProviderCallObservationKind
    result: Any = None
    message: str | None = None


class ProviderDriver(Protocol):
    """Synchronous provider operations required by the execution runtime."""

    def resolve(self, binding: ProviderBinding) -> Any:
        """Resolve one exact deployed operation."""
        ...

    def spawn(
        self,
        operation: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str:
        """Submit one operation and return its durable provider handle."""
        ...

    def observe(self, provider_call_handle_id: str) -> ProviderCallObservation:
        """Observe one retained Provider Call."""
        ...

    def cancel(self, provider_call_handle_id: str) -> None:
        """Request Provider Call cancellation."""
        ...


class AsyncProviderDriver(Protocol):
    """Asynchronous provider operations required by the execution runtime."""

    async def resolve(self, binding: ProviderBinding) -> Any: ...

    async def spawn(
        self,
        operation: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str: ...

    async def observe(
        self,
        provider_call_handle_id: str,
    ) -> ProviderCallObservation: ...

    async def cancel(self, provider_call_handle_id: str) -> None: ...
