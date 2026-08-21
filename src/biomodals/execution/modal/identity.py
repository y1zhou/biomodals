"""Modal coordinator identity helpers."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Any, TypeVar
from uuid import UUID

import modal

from biomodals.execution.model import DeploymentIdentity

_T = TypeVar("_T")


def deployed_execution_coordinator(
    *,
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
    class_resolver: Callable[..., Any] = modal.Cls.from_name,
) -> Any:
    """Resolve and parameterize the exact-version Modal coordinator class."""
    coordinator_class = class_resolver(
        deployment.deployment_name,
        "ExecutionCoordinator",
        environment_name=deployment.environment,
        version=deployment.deployment_version,
    )
    return coordinator_class(
        **_coordinator_parameters(execution_run_id, deployment, development=False)
    )


def execution_coordinator_handle(
    *,
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
    use_deployed_coordinator: bool,
    local_coordinator: Callable[..., Any],
    class_resolver: Callable[..., Any] = modal.Cls.from_name,
) -> Any:
    """Bind one run to its deployed or current-source coordinator class."""
    if use_deployed_coordinator:
        return deployed_execution_coordinator(
            execution_run_id=execution_run_id,
            deployment=deployment,
            class_resolver=class_resolver,
        )
    return local_coordinator(
        **_coordinator_parameters(execution_run_id, deployment, development=True)
    )


def execution_coordinator_identity(
    parameters: Any,
) -> tuple[UUID, DeploymentIdentity]:
    """Read the standard execution identity from Modal class parameters."""
    execution_run_id = UUID(parameters.execution_run_id)
    if str(execution_run_id) != parameters.execution_run_id:
        raise ValueError("Execution Run ID must use canonical UUID text")
    return execution_run_id, DeploymentIdentity(
        environment=parameters.deployment_environment,
        deployment_name=parameters.deployment_name,
        deployment_version=parameters.deployment_version,
    )


def initialize_execution_coordinator_host(host: Any) -> None:
    """Initialize one concurrent Modal coordinator container."""
    host._coordinator_adapter = None
    host._development = bool(getattr(host, "development", False))
    host._coordinator_adapter_lock = Lock()


def execution_coordinator_adapter(  # noqa: UP047 - mounted in Python 3.10/3.11 apps
    host: Any,
    *,
    development: bool | None,
    factory: Callable[[bool], _T],
) -> _T:
    """Return the one mode-pinned adapter owned by a coordinator container."""
    with host._coordinator_adapter_lock:
        adapter = host._coordinator_adapter
        selected_mode = host._development
        if adapter is not None:
            if development is not None and selected_mode != development:
                raise ValueError("Coordinator execution mode cannot change in place")
            return adapter
        if development is not None and selected_mode != development:
            raise ValueError("Coordinator execution mode does not match its identity")
        adapter = factory(selected_mode)
        host._coordinator_adapter = adapter
        host._development = selected_mode
        return adapter


def _coordinator_parameters(
    execution_run_id: UUID,
    deployment: DeploymentIdentity,
    *,
    development: bool,
) -> dict[str, object]:
    return {
        "execution_run_id": str(execution_run_id),
        "deployment_environment": deployment.environment,
        "deployment_name": deployment.deployment_name,
        "deployment_version": deployment.deployment_version,
        "development": development,
    }
