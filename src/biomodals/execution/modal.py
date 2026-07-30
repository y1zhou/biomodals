"""Small internal boundary around the Modal call lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import modal

from biomodals.execution.model import ProviderBinding


class ModalDefiniteSubmissionError(RuntimeError):
    """Raised when Modal conclusively rejects a spawn before work starts."""


class ModalSubmissionOutcomeUnknownError(RuntimeError):
    """Raised when a spawn may have started without returning a call ID."""


class ModalCallObservationKind(StrEnum):
    """Provider-neutral observations produced by the Modal boundary."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STATE_UNKNOWN = "state_unknown"


@dataclass(frozen=True)
class ModalCallObservation:
    """One nonblocking observation of a concrete Modal Function Call."""

    kind: ModalCallObservationKind
    result: Any = None
    message: str | None = None


_DEFINITE_SUBMISSION_ERRORS = (
    modal.exception.AuthError,
    modal.exception.InvalidError,
    modal.exception.NotFoundError,
    modal.exception.PermissionDeniedError,
    modal.exception.RequestSizeError,
    modal.exception.SerializationError,
    TypeError,
    ValueError,
)
_INCONCLUSIVE_SERVICE_ERRORS = (
    modal.exception.AuthError,
    modal.exception.ClientClosed,
    modal.exception.ConnectionError,
    modal.exception.DataLossError,
    modal.exception.InternalError,
    modal.exception.NotFoundError,
    modal.exception.PermissionDeniedError,
    modal.exception.ResourceExhaustedError,
    modal.exception.ServiceError,
)
_CONCLUSIVE_EXECUTION_ERRORS = (
    modal.exception.ExecutionError,
    modal.exception.FunctionTimeoutError,
    modal.exception.RemoteError,
    modal.exception.UserCodeException,
)


class ModalCallDriver:
    """Resolve, invoke, recover, observe, and cancel deployed Modal functions."""

    def __init__(
        self,
        *,
        function_resolver: Callable[..., Any] = modal.Function.from_name,
        call_resolver: Callable[[str], Any] = modal.FunctionCall.from_id,
    ) -> None:
        """Inject only the two external Modal handle constructors used in tests."""
        self._function_resolver = function_resolver
        self._call_resolver = call_resolver

    def resolve(self, binding: ProviderBinding) -> Any:
        """Hydrate one exact deployed function before durable preclaim."""
        function = self._function_resolver(
            binding.app_name,
            binding.function_name,
            environment_name=binding.environment,
            version=binding.app_version,
        )
        function.hydrate()
        return function

    def spawn(
        self,
        function: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str:
        """Spawn once and return the durable Function Call ID."""
        try:
            call = function.spawn(*args, **dict(kwargs))
            return str(call.object_id)
        except _DEFINITE_SUBMISSION_ERRORS as error:
            raise ModalDefiniteSubmissionError(str(error)) from error
        except Exception as error:
            raise ModalSubmissionOutcomeUnknownError(
                "Modal did not return a durable Function Call ID"
            ) from error

    def observe(self, provider_call_handle_id: str) -> ModalCallObservation:
        """Poll one retained call result without clearing it."""
        call = self._call_resolver(provider_call_handle_id)
        try:
            result = call.get(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            return ModalCallObservation(ModalCallObservationKind.RUNNING)
        except modal.exception.InputCancellation as error:
            return ModalCallObservation(
                ModalCallObservationKind.CANCELLED,
                message=str(error),
            )
        except (
            modal.exception.OutputExpiredError,
            *_INCONCLUSIVE_SERVICE_ERRORS,
        ) as error:
            return ModalCallObservation(
                ModalCallObservationKind.STATE_UNKNOWN,
                message=str(error),
            )
        except _CONCLUSIVE_EXECUTION_ERRORS as error:
            return ModalCallObservation(
                ModalCallObservationKind.FAILED,
                message=str(error),
            )
        except Exception as error:
            return ModalCallObservation(
                ModalCallObservationKind.STATE_UNKNOWN,
                message=str(error),
            )
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=result,
        )

    def cancel(self, provider_call_handle_id: str) -> None:
        """Request cancellation of one attached Function Call."""
        self._call_resolver(provider_call_handle_id).cancel()


class AsyncModalCallDriver:
    """Async Modal SDK boundary for API-hosted execution coordination."""

    def __init__(
        self,
        *,
        function_resolver: Callable[..., Any] = modal.Function.from_name,
        call_resolver: Callable[[str], Any] = modal.FunctionCall.from_id,
    ) -> None:
        """Inject external Modal handle constructors for deterministic tests."""
        self._function_resolver = function_resolver
        self._call_resolver = call_resolver

    async def resolve(self, binding: ProviderBinding) -> Any:
        """Hydrate one exact deployed function before durable preclaim."""
        function = self._function_resolver(
            binding.app_name,
            binding.function_name,
            environment_name=binding.environment,
            version=binding.app_version,
        )
        await function.hydrate.aio()
        return function

    async def spawn(
        self,
        function: Any,
        *,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> str:
        """Spawn once and return the durable Function Call ID."""
        try:
            call = await function.spawn.aio(*args, **dict(kwargs))
            return str(call.object_id)
        except _DEFINITE_SUBMISSION_ERRORS as error:
            raise ModalDefiniteSubmissionError(str(error)) from error
        except Exception as error:
            raise ModalSubmissionOutcomeUnknownError(
                "Modal did not return a durable Function Call ID"
            ) from error

    async def observe(
        self,
        provider_call_handle_id: str,
    ) -> ModalCallObservation:
        """Poll one retained call result without clearing it."""
        call = self._call_resolver(provider_call_handle_id)
        try:
            result = await call.get.aio(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            return ModalCallObservation(ModalCallObservationKind.RUNNING)
        except modal.exception.InputCancellation as error:
            return ModalCallObservation(
                ModalCallObservationKind.CANCELLED,
                message=str(error),
            )
        except (
            modal.exception.OutputExpiredError,
            *_INCONCLUSIVE_SERVICE_ERRORS,
        ) as error:
            return ModalCallObservation(
                ModalCallObservationKind.STATE_UNKNOWN,
                message=str(error),
            )
        except _CONCLUSIVE_EXECUTION_ERRORS as error:
            return ModalCallObservation(
                ModalCallObservationKind.FAILED,
                message=str(error),
            )
        except Exception as error:
            return ModalCallObservation(
                ModalCallObservationKind.STATE_UNKNOWN,
                message=str(error),
            )
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=result,
        )

    async def cancel(self, provider_call_handle_id: str) -> None:
        """Request cancellation of one attached Function Call."""
        call = self._call_resolver(provider_call_handle_id)
        await call.cancel.aio()
