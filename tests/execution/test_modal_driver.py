"""Modal SDK boundary tests without live remote calls."""

# ruff: noqa: D101, D102, D103, D107

import pytest

from biomodals.execution.modal import (
    ModalCallDriver,
    ModalCallObservationKind,
    ModalSubmissionOutcomeUnknownError,
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
