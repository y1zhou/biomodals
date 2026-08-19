"""Modal SDK adapter for Provider Call operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import modal

from biomodals.execution.model import DeploymentIdentity, ProviderBinding
from biomodals.execution.provider import (
    ProviderCallObservation,
    ProviderCallObservationKind,
    ProviderDefiniteSubmissionError,
    ProviderDeploymentUnavailableError,
    ProviderSubmissionOutcomeUnknownError,
)

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


def _observation_from_error(error: Exception) -> ProviderCallObservation:
    if isinstance(error, (TimeoutError, modal.exception.TimeoutError)):
        kind = ProviderCallObservationKind.RUNNING
    elif isinstance(error, modal.exception.InputCancellation):
        kind = ProviderCallObservationKind.CANCELLED
    elif isinstance(
        error,
        (modal.exception.OutputExpiredError, *_INCONCLUSIVE_SERVICE_ERRORS),
    ):
        kind = ProviderCallObservationKind.STATE_UNKNOWN
    elif isinstance(error, _CONCLUSIVE_EXECUTION_ERRORS):
        kind = ProviderCallObservationKind.FAILED
    else:
        kind = ProviderCallObservationKind.STATE_UNKNOWN
    return ProviderCallObservation(kind, message=str(error))


class ModalCallDriver:
    """Adapt synchronous Modal Function Calls to the provider interface."""

    def __init__(
        self,
        *,
        function_resolver: Callable[..., Any] = modal.Function.from_name,
        call_resolver: Callable[[str], Any] = modal.FunctionCall.from_id,
    ) -> None:
        self._function_resolver = function_resolver
        self._call_resolver = call_resolver

    def resolve(self, binding: ProviderBinding) -> Any:
        try:
            function = self._function_resolver(
                binding.app_name,
                binding.function_name,
                environment_name=binding.environment,
                version=binding.app_version,
            )
            function.hydrate()
        except _DEPLOYMENT_UNAVAILABLE_ERRORS as error:
            raise ProviderDeploymentUnavailableError(
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
        try:
            call = function.spawn(*args, **dict(kwargs))
            return str(call.object_id)
        except _DEFINITE_SUBMISSION_ERRORS as error:
            raise ProviderDefiniteSubmissionError(str(error)) from error
        except Exception as error:
            raise ProviderSubmissionOutcomeUnknownError(
                "Modal did not return a durable Function Call ID"
            ) from error

    def observe(self, provider_call_handle_id: str) -> ProviderCallObservation:
        call = self._call_resolver(provider_call_handle_id)
        try:
            result = call.get(timeout=0)
        except Exception as error:
            return _observation_from_error(error)
        return ProviderCallObservation(
            ProviderCallObservationKind.SUCCEEDED,
            result=result,
        )

    def cancel(self, provider_call_handle_id: str) -> None:
        self._call_resolver(provider_call_handle_id).cancel()


def deployed_function_handle(
    deployment: DeploymentIdentity,
    function_name: str,
) -> Any:
    """Resolve one exact deployed function for a Modal client."""
    return ModalCallDriver().resolve(
        ProviderBinding(
            environment=deployment.environment,
            app_name=deployment.deployment_name,
            app_version=deployment.deployment_version,
            function_name=function_name,
            uses_gpu=False,
            runtime_image_key="local-entrypoint",
        )
    )


def development_modal_call_driver(
    function_handles: Mapping[str, Any],
    *,
    workload_name: str,
) -> ModalCallDriver:
    """Build a driver that resolves current-source development handles."""
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
    """Adapt asynchronous Modal calls to the provider interface."""

    def __init__(
        self,
        *,
        function_resolver: Callable[..., Any] = modal.Function.from_name,
        call_resolver: Callable[[str], Any] = modal.FunctionCall.from_id,
    ) -> None:
        self._function_resolver = function_resolver
        self._call_resolver = call_resolver

    async def resolve(self, binding: ProviderBinding) -> Any:
        try:
            function = self._function_resolver(
                binding.app_name,
                binding.function_name,
                environment_name=binding.environment,
                version=binding.app_version,
            )
            await function.hydrate.aio()
        except _DEPLOYMENT_UNAVAILABLE_ERRORS as error:
            raise ProviderDeploymentUnavailableError(
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
        try:
            call = await function.spawn.aio(*args, **dict(kwargs))
            return str(call.object_id)
        except _DEFINITE_SUBMISSION_ERRORS as error:
            raise ProviderDefiniteSubmissionError(str(error)) from error
        except Exception as error:
            raise ProviderSubmissionOutcomeUnknownError(
                "Modal did not return a durable Function Call ID"
            ) from error

    async def observe(
        self,
        provider_call_handle_id: str,
    ) -> ProviderCallObservation:
        call = self._call_resolver(provider_call_handle_id)
        try:
            result = await call.get.aio(timeout=0)
        except Exception as error:
            return _observation_from_error(error)
        return ProviderCallObservation(
            ProviderCallObservationKind.SUCCEEDED,
            result=result,
        )

    async def cancel(self, provider_call_handle_id: str) -> None:
        call = self._call_resolver(provider_call_handle_id)
        await call.cancel.aio()
