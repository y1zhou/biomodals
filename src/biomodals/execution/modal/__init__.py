"""Modal provider integration for the execution kernel."""

from biomodals.execution.modal.driver import (
    AsyncModalCallDriver,
    ModalCallDriver,
    deployed_function_handle,
    development_modal_call_driver,
)
from biomodals.execution.modal.host import (
    deployed_execution_coordinator,
    execution_coordinator_adapter,
    execution_coordinator_handle,
    execution_coordinator_identity,
    initialize_execution_coordinator_host,
)
from biomodals.execution.provider import (
    ProviderCallObservation,
    ProviderCallObservationKind,
    ProviderDefiniteSubmissionError,
    ProviderDeploymentUnavailableError,
    ProviderSubmissionOutcomeUnknownError,
)

__all__ = [
    "AsyncModalCallDriver",
    "ModalCallDriver",
    "ProviderCallObservation",
    "ProviderCallObservationKind",
    "ProviderDefiniteSubmissionError",
    "ProviderDeploymentUnavailableError",
    "ProviderSubmissionOutcomeUnknownError",
    "deployed_execution_coordinator",
    "deployed_function_handle",
    "development_modal_call_driver",
    "execution_coordinator_adapter",
    "execution_coordinator_handle",
    "execution_coordinator_identity",
    "initialize_execution_coordinator_host",
]
