"""Small internal boundary around the Modal call lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

import modal

from biomodals.execution.model import DeploymentIdentity, ProviderBinding


class ModalDefiniteSubmissionError(RuntimeError):
    """Raised when Modal conclusively rejects a spawn before work starts."""


class ModalSubmissionOutcomeUnknownError(RuntimeError):
    """Raised when a spawn may have started without returning a call ID."""


class ModalDeploymentUnavailableError(RuntimeError):
    """Raised when an exact deployed function version cannot be hydrated."""


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
_DEPLOYMENT_UNAVAILABLE_ERRORS = (
    modal.exception.InvalidError,
    modal.exception.NotFoundError,
)


def _observation_from_error(error: Exception) -> ModalCallObservation:
    """Classify one sync or async retained-call error."""
    if isinstance(error, (TimeoutError, modal.exception.TimeoutError)):
        kind = ModalCallObservationKind.RUNNING
    elif isinstance(error, modal.exception.InputCancellation):
        kind = ModalCallObservationKind.CANCELLED
    elif isinstance(
        error,
        (modal.exception.OutputExpiredError, *_INCONCLUSIVE_SERVICE_ERRORS),
    ):
        kind = ModalCallObservationKind.STATE_UNKNOWN
    elif isinstance(error, _CONCLUSIVE_EXECUTION_ERRORS):
        kind = ModalCallObservationKind.FAILED
    else:
        kind = ModalCallObservationKind.STATE_UNKNOWN
    return ModalCallObservation(kind, message=str(error))


def deployed_execution_coordinator(
    *,
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
    class_resolver: Callable[..., Any] = modal.Cls.from_name,
) -> Any:
    """Resolve and parameterize the standard exact-version coordinator class."""
    coordinator_class = class_resolver(
        deployment.deployment_name,
        "ExecutionCoordinator",
        environment_name=deployment.environment,
        version=deployment.deployment_version,
    )
    return coordinator_class(
        execution_run_id=str(execution_run_id),
        deployment_environment=deployment.environment,
        deployment_name=deployment.deployment_name,
        deployment_version=deployment.deployment_version,
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
        try:
            function = self._function_resolver(
                binding.app_name,
                binding.function_name,
                environment_name=binding.environment,
                version=binding.app_version,
            )
            function.hydrate()
        except _DEPLOYMENT_UNAVAILABLE_ERRORS as error:
            raise ModalDeploymentUnavailableError(
                "Exact Modal deployment is unavailable: "
                f"{binding.environment}/{binding.app_name}/"
                f"v{binding.app_version}/{binding.function_name}"
            ) from error
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
        except Exception as error:
            return _observation_from_error(error)
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=result,
        )

    def cancel(self, provider_call_handle_id: str) -> None:
        """Request cancellation of one attached Function Call."""
        self._call_resolver(provider_call_handle_id).cancel()


def development_modal_call_driver(
    function_handles: Mapping[str, Any],
    *,
    workload_name: str,
) -> ModalCallDriver:
    """Build a driver that resolves only current-source development handles."""
    handles = dict(function_handles)

    def resolve(
        _app_name: str,
        function_name: str,
        **_kwargs: object,
    ) -> Any:
        try:
            return handles[function_name]
        except KeyError as error:
            raise ValueError(
                f"No {workload_name} development function {function_name!r}"
            ) from error

    return ModalCallDriver(function_resolver=resolve)


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
        try:
            function = self._function_resolver(
                binding.app_name,
                binding.function_name,
                environment_name=binding.environment,
                version=binding.app_version,
            )
            await function.hydrate.aio()
        except _DEPLOYMENT_UNAVAILABLE_ERRORS as error:
            raise ModalDeploymentUnavailableError(
                "Exact Modal deployment is unavailable: "
                f"{binding.environment}/{binding.app_name}/"
                f"v{binding.app_version}/{binding.function_name}"
            ) from error
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
        except Exception as error:
            return _observation_from_error(error)
        return ModalCallObservation(
            ModalCallObservationKind.SUCCEEDED,
            result=result,
        )

    async def cancel(self, provider_call_handle_id: str) -> None:
        """Request cancellation of one attached Function Call."""
        call = self._call_resolver(provider_call_handle_id)
        await call.cancel.aio()
