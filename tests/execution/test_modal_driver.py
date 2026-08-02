"""Modal SDK boundary tests without live remote calls."""

# ruff: noqa: D101, D102, D103, D107

import asyncio
from dataclasses import replace
from threading import Thread
from time import sleep
from types import SimpleNamespace
from uuid import UUID

import modal
import pytest

from biomodals.execution import DeploymentIdentity
from biomodals.execution.modal import (
    AsyncModalCallDriver,
    ModalCallDriver,
    ModalCallObservationKind,
    ModalDeploymentUnavailableError,
    ModalSubmissionOutcomeUnknownError,
    deployed_execution_coordinator,
    development_modal_call_driver,
    execution_coordinator_adapter,
    execution_coordinator_handle,
    execution_coordinator_identity,
    initialize_execution_coordinator_host,
)

from .provider_call_helpers import GPU_BINDING


class FakeFunction:
    def __init__(self) -> None:
        self.hydrated = False

    def hydrate(self):
        self.hydrated = True
        return self

    def spawn(self, *args, **kwargs):
        assert args == ("input",)
        assert kwargs == {"seed": 1}
        return type("Call", (), {"object_id": "fc-123"})()


def test_deployed_coordinator_uses_exact_class_version_and_run_parameters() -> None:
    resolved: list[tuple[tuple[object, ...], dict[str, object]]] = []
    parameters: dict[str, object] = {}

    class Coordinator:
        def __init__(self, **kwargs: object) -> None:
            parameters.update(kwargs)

    def resolve(*args: object, **kwargs: object) -> type[Coordinator]:
        resolved.append((args, kwargs))
        return Coordinator

    handle = deployed_execution_coordinator(
        execution_run_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        deployment=DeploymentIdentity("production", "ShortMDWorkflow", 7),
        class_resolver=resolve,
    )

    assert isinstance(handle, Coordinator)
    assert resolved == [
        (
            ("ShortMDWorkflow", "ExecutionCoordinator"),
            {"environment_name": "production", "version": 7},
        )
    ]
    assert parameters == {
        "execution_run_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "deployment_environment": "production",
        "deployment_name": "ShortMDWorkflow",
        "deployment_version": 7,
    }


def test_coordinator_handle_uses_current_source_class_in_development() -> None:
    parameters: dict[str, object] = {}

    class Coordinator:
        def __init__(self, **kwargs: object) -> None:
            parameters.update(kwargs)

    handle = execution_coordinator_handle(
        execution_run_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        deployment=DeploymentIdentity("development", "Gromacs", 3),
        use_deployed_coordinator=False,
        local_coordinator=Coordinator,
        class_resolver=lambda *args, **kwargs: pytest.fail(
            "development binding must not resolve a deployed class"
        ),
    )

    assert isinstance(handle, Coordinator)
    assert parameters == {
        "execution_run_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        "deployment_environment": "development",
        "deployment_name": "Gromacs",
        "deployment_version": 3,
    }


def test_coordinator_identity_reads_standard_modal_parameters() -> None:
    execution_run_id, deployment = execution_coordinator_identity(
        SimpleNamespace(
            execution_run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            deployment_environment="production",
            deployment_name="AlphaFold3",
            deployment_version=11,
        )
    )

    assert execution_run_id == UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    assert deployment == DeploymentIdentity("production", "AlphaFold3", 11)


def test_concurrent_coordinator_inputs_share_one_adapter() -> None:
    host = SimpleNamespace()
    initialize_execution_coordinator_host(host)
    created: list[object] = []
    results: list[object] = []

    def factory(_development: bool) -> object:
        adapter = object()
        created.append(adapter)
        sleep(0.02)
        return adapter

    def resolve() -> None:
        results.append(
            execution_coordinator_adapter(
                host,
                development=False,
                factory=factory,
            )
        )

    threads = [Thread(target=resolve) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert all(not thread.is_alive() for thread in threads)
    assert len(created) == 1
    assert results == created * 8


def test_driver_resolves_exact_version_and_spawns_detached_call() -> None:
    resolved: list[tuple] = []
    function = FakeFunction()

    def resolve(*args, **kwargs):
        resolved.append((args, kwargs))
        return function

    driver = ModalCallDriver(function_resolver=resolve)

    handle = driver.resolve(GPU_BINDING)
    call_id = driver.spawn(handle, args=("input",), kwargs={"seed": 1})

    assert resolved == [
        (
            ("biomodals-alphafold3", "run_inference"),
            {"environment_name": "production", "version": 23},
        )
    ]
    assert function.hydrated
    assert call_id == "fc-123"


def test_development_driver_resolves_only_declared_local_handles() -> None:
    function = FakeFunction()
    driver = development_modal_call_driver(
        {"run_inference": function},
        workload_name="Example",
    )

    assert driver.resolve(GPU_BINDING) is function
    with pytest.raises(ValueError, match="No Example development function 'missing'"):
        driver.resolve(replace(GPU_BINDING, function_name="missing"))


def test_driver_classifies_ambiguous_spawn_failure() -> None:
    class FailingFunction(FakeFunction):
        def spawn(self, *args, **kwargs):
            raise TimeoutError("response lost")

    function = FailingFunction()
    driver = ModalCallDriver(function_resolver=lambda *args, **kwargs: function)

    with pytest.raises(ModalSubmissionOutcomeUnknownError):
        driver.spawn(
            driver.resolve(GPU_BINDING),
            args=(),
            kwargs={},
        )


def test_driver_classifies_an_unavailable_exact_version() -> None:
    """A missing retained deployment is distinct from provider uncertainty."""

    class MissingFunction(FakeFunction):
        def hydrate(self):
            raise modal.exception.NotFoundError("version expired")

    driver = ModalCallDriver(
        function_resolver=lambda *args, **kwargs: MissingFunction()
    )

    with pytest.raises(ModalDeploymentUnavailableError, match="v23"):
        driver.resolve(GPU_BINDING)


def test_driver_observes_timeout_as_running_and_retained_result_as_success() -> None:
    class Call:
        def __init__(self):
            self.ready = False

        def get(self, timeout=0):
            assert timeout == 0
            if not self.ready:
                raise TimeoutError
            return {"done": True}

    call = Call()
    driver = ModalCallDriver(call_resolver=lambda call_id: call)

    assert driver.observe("fc-123").kind == ModalCallObservationKind.RUNNING
    call.ready = True
    observation = driver.observe("fc-123")
    assert observation.kind == ModalCallObservationKind.SUCCEEDED
    assert observation.result == {"done": True}


def test_async_driver_uses_exact_deployment_and_retained_call_handle() -> None:
    async def scenario() -> None:
        resolved: list[tuple] = []

        class AwaitableMethod:
            def __init__(self, function):
                self.aio = function

        class Function:
            def __init__(self):
                self.hydrate = AwaitableMethod(self._hydrate)
                self.spawn = AwaitableMethod(self._spawn)
                self.hydrated = False

            async def _hydrate(self):
                self.hydrated = True

            async def _spawn(self, **kwargs):
                assert kwargs == {"seed": 1}
                return type("Call", (), {"object_id": "fc-async"})()

        class Call:
            def __init__(self):
                self.get = AwaitableMethod(self._get)

            async def _get(self, timeout=0):
                assert timeout == 0
                return {"done": True}

        function = Function()

        def resolve(*args, **kwargs):
            resolved.append((args, kwargs))
            return function

        driver = AsyncModalCallDriver(
            function_resolver=resolve,
            call_resolver=lambda call_id: Call(),
        )
        handle = await driver.resolve(GPU_BINDING)
        call_id = await driver.spawn(handle, args=(), kwargs={"seed": 1})
        observation = await driver.observe(call_id)

        assert resolved == [
            (
                ("biomodals-alphafold3", "run_inference"),
                {"environment_name": "production", "version": 23},
            )
        ]
        assert function.hydrated
        assert call_id == "fc-async"
        assert observation.kind == ModalCallObservationKind.SUCCEEDED
        assert observation.result == {"done": True}

    asyncio.run(scenario())
