"""Contract tests for the Modal GROMACS job backend."""

# ruff: noqa: D101,D102,D103,D107

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import modal
import pytest

from biomodals.schema import (
    AppOutput,
    AppRunResult,
    AppRunStatus,
    ArtifactKind,
    VolumePath,
)
from biomodals.service.gromacs_api import (
    GromacsJobOptions,
    JobNotFoundError,
    JobState,
)
from biomodals.service.modal_gromacs import (
    JobRegistry,
    ModalGromacsBackend,
    ModalJobRecord,
    create_deployed_app,
)


class FakeAsyncMethod:
    def __init__(self, function: Callable[..., Any]) -> None:
        self.function = function

    async def aio(self, *args, **kwargs):
        result = self.function(*args, **kwargs)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeRegistry:
    def __init__(self) -> None:
        self.items: dict[str, str] = {}
        self.get = FakeAsyncMethod(
            lambda key, default=None: self.items.get(key, default)
        )
        self.put = FakeAsyncMethod(self._put)

    def _put(self, key: str, value: str) -> bool:
        self.items[key] = value
        return True


class FakeCall:
    object_id = "fc-123"


class FakeSpawn:
    def __init__(self) -> None:
        self.kwargs = {}

    async def aio(self, **kwargs):
        self.kwargs = kwargs
        return FakeCall()


class FakeFunction:
    def __init__(self) -> None:
        self.spawn = FakeSpawn()


class FakeGet:
    def __init__(self, result) -> None:
        self.result = result

    async def aio(self, *, timeout: int):
        assert timeout == 0
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class FakeResolvedCall:
    def __init__(
        self,
        result,
        *,
        graph_status: modal.call_graph.InputStatus = modal.call_graph.InputStatus.PENDING,
        cancel_result=None,
    ) -> None:
        self.get = FakeGet(result)
        self.cancelled = False
        self.get_call_graph = FakeAsyncMethod(
            lambda: [SimpleNamespace(status=graph_status)]
        )

        def cancel():
            self.cancelled = True
            return cancel_result

        self.cancel = FakeAsyncMethod(cancel)


def registered_backend(
    result,
    *,
    call: FakeResolvedCall | None = None,
) -> tuple[ModalGromacsBackend, FakeRegistry, FakeResolvedCall]:
    registry = FakeRegistry()
    registry.items["job-123"] = ModalJobRecord(
        modal_call_id="fc-123",
        run_name="api-123",
    ).model_dump_json()
    resolved_call = call or FakeResolvedCall(result)
    backend = ModalGromacsBackend(
        cast(modal.Function, FakeFunction()),
        cast(JobRegistry, registry),
        call_resolver=cast(
            Callable[[str], modal.FunctionCall],
            lambda raw_call_id: (
                resolved_call
                if raw_call_id == "fc-123"
                else (_ for _ in ()).throw(AssertionError(raw_call_id))
            ),
        ),
    )
    return backend, registry, resolved_call


def test_submit_registers_opaque_job_id_and_run_name() -> None:
    function = FakeFunction()
    registry = FakeRegistry()
    backend = ModalGromacsBackend(
        cast(modal.Function, function),
        cast(JobRegistry, registry),
    )

    result = asyncio.run(
        backend.submit(
            b"PDB bytes validated by the HTTP boundary",
            GromacsJobOptions(simulation_time_ns=3, cpu_only=True),
        )
    )

    assert result.job_id.startswith("job-")
    assert result.job_id != "fc-123"
    assert result.status == JobState.PENDING
    assert result.run_name is not None
    record = ModalJobRecord.model_validate_json(registry.items[result.job_id])
    assert record.modal_call_id == "fc-123"
    assert record.run_name == result.run_name
    assert function.spawn.kwargs["run_name"] == result.run_name
    assert function.spawn.kwargs["simulation_time_ns"] == 3
    assert function.spawn.kwargs["run_pdbfixer"] is False
    assert function.spawn.kwargs["cpu_only"] is True


def test_inspect_maps_app_result_to_portable_manifest() -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="gromacs_run",
                kind=ArtifactKind.DIRECTORY,
                storage=VolumePath(volume_name="Gromacs-outputs", path="api-123"),
                metadata={
                    "run_name": "api-123",
                    "files": [
                        {
                            "path": "production_api-123.xtc",
                            "role": "trajectory",
                        }
                    ],
                },
            )
        ],
    )
    backend, _registry, _call = registered_backend(result)

    snapshot = asyncio.run(backend.inspect("job-123"))

    assert snapshot.job_id == "job-123"
    assert snapshot.status == JobState.SUCCEEDED
    assert snapshot.run_name == "api-123"
    assert snapshot.result is not None
    assert snapshot.result.run_name == "api-123"
    assert snapshot.result.artifacts[0].files[0].path == "production_api-123.xtc"


def test_inspect_rejects_result_for_a_different_run() -> None:
    result = AppRunResult(
        status=AppRunStatus.SUCCEEDED,
        outputs=[
            AppOutput(
                name="gromacs_run",
                kind=ArtifactKind.DIRECTORY,
                storage=VolumePath(volume_name="Gromacs-outputs", path="api-other"),
                metadata={"run_name": "api-other"},
            )
        ],
    )
    backend, _registry, _call = registered_backend(result)

    snapshot = asyncio.run(backend.inspect("job-123"))

    assert snapshot.status == JobState.FAILED
    assert snapshot.result is None


def test_inspect_returns_pending_without_waiting() -> None:
    backend, _registry, _call = registered_backend(TimeoutError())

    snapshot = asyncio.run(backend.inspect("job-123"))

    assert snapshot.status == JobState.PENDING
    assert snapshot.run_name == "api-123"
    assert snapshot.result is None


def test_remote_timeout_is_not_misclassified_as_pending() -> None:
    call = FakeResolvedCall(
        TimeoutError("remote timeout"),
        graph_status=modal.call_graph.InputStatus.FAILURE,
    )
    backend, _registry, _call = registered_backend(TimeoutError(), call=call)

    snapshot = asyncio.run(backend.inspect("job-123"))

    assert snapshot.status == JobState.FAILED
    assert snapshot.detail == "GROMACS job failed; inspect Modal logs"


def test_inspect_distinguishes_expired_output_and_keeps_run_name() -> None:
    backend, _registry, _call = registered_backend(
        modal.exception.OutputExpiredError("expired")
    )

    snapshot = asyncio.run(backend.inspect("job-123"))

    assert snapshot.status == JobState.EXPIRED
    assert snapshot.run_name == "api-123"
    assert snapshot.result is None


def test_inspect_hides_remote_failure_details() -> None:
    backend, _registry, _call = registered_backend(RuntimeError("secret gmx output"))

    snapshot = asyncio.run(backend.inspect("job-123"))

    assert snapshot.status == JobState.FAILED
    assert snapshot.detail == "GROMACS job failed; inspect Modal logs"
    assert "secret" not in snapshot.detail


def test_cancel_uses_registered_raw_call_id() -> None:
    backend, _registry, call = registered_backend(TimeoutError())

    result = asyncio.run(backend.cancel("job-123"))

    assert call.cancelled is True
    assert result.job_id == "job-123"
    assert result.status == JobState.CANCELLED
    assert result.run_name == "api-123"


def test_invalid_call_during_cancel_returns_portable_not_found() -> None:
    call = FakeResolvedCall(
        TimeoutError(),
        cancel_result=modal.exception.NotFoundError("missing"),
    )
    backend, _registry, _call = registered_backend(TimeoutError(), call=call)

    with pytest.raises(JobNotFoundError) as exc_info:
        asyncio.run(backend.cancel("job-123"))

    assert exc_info.value.job_id == "job-123"


def test_unknown_or_raw_modal_call_id_is_rejected_before_resolution() -> None:
    backend, _registry, _call = registered_backend(TimeoutError())

    for job_id in ("job-missing", "fc-123"):
        with pytest.raises(JobNotFoundError) as exc_info:
            asyncio.run(backend.inspect(job_id))
        assert exc_info.value.job_id == job_id


def test_deployed_app_factory_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("BIOMODALS_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="BIOMODALS_API_KEY"):
        create_deployed_app()


def test_deployed_app_factory_looks_up_configured_modal_objects(
    monkeypatch,
) -> None:
    function = FakeFunction()
    registry = FakeRegistry()
    function_lookups = []
    registry_lookups = []

    def function_from_name(app_name: str, function_name: str):
        function_lookups.append((app_name, function_name))
        return function

    def dict_from_name(name: str, *, create_if_missing: bool):
        registry_lookups.append((name, create_if_missing))
        return registry

    monkeypatch.setenv("BIOMODALS_API_KEY", "secret")
    monkeypatch.delenv("BIOMODALS_GROMACS_APP", raising=False)
    monkeypatch.setenv("BIOMODALS_GROMACS_JOB_DICT", "MyJobs")
    monkeypatch.setattr(modal.Function, "from_name", function_from_name)
    monkeypatch.setattr(modal.Dict, "from_name", dict_from_name)

    web_app = create_deployed_app()

    assert web_app.title == "Biomodals GROMACS API"
    assert function_lookups == [("GromacsAPI", "run_gromacs_job")]
    assert registry_lookups == [("MyJobs", True)]
