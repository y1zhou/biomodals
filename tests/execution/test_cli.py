"""CLI tests for explicit remote Execution Run lifecycle controls."""

# ruff: noqa: D101, D102, D103, D107

from types import SimpleNamespace
from uuid import UUID

import pytest
from typer.testing import CliRunner

from biomodals.cli import app

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
LOCATION_FLAGS = [
    "--environment",
    "production",
    "--deployment-name",
    "ShortMDWorkflow",
    "--deployment-version",
    "7",
    "--execution-run-id",
    str(RUN_ID),
]
runner = CliRunner()


class FakeRemoteMethod:
    def __init__(self, result: object) -> None:
        self.result = result
        self.remote_calls = 0
        self.spawn_calls = 0

    def remote(self) -> object:
        self.remote_calls += 1
        return self.result

    def spawn(self) -> object:
        self.spawn_calls += 1
        return self.result


class FakeCoordinator:
    def __init__(self) -> None:
        self.snapshot = object()
        self.status = FakeRemoteMethod(self.snapshot)
        self.cancel = FakeRemoteMethod(self.snapshot)
        self.resume = FakeRemoteMethod(SimpleNamespace(object_id="fc-resume"))


def _patch_coordinator(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FakeCoordinator, dict[str, object]]:
    coordinator = FakeCoordinator()
    calls: dict[str, object] = {}

    def resolve(**kwargs: object) -> FakeCoordinator:
        calls.update(kwargs)
        return coordinator

    monkeypatch.setattr("biomodals.cli.deployed_execution_coordinator", resolve)
    return coordinator, calls


def test_run_status_uses_explicit_location_and_prints_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, calls = _patch_coordinator(monkeypatch)
    printed: list[object] = []
    monkeypatch.setattr(
        "biomodals.cli._print_execution_snapshot",
        printed.append,
    )

    result = runner.invoke(app, ["run", "status", *LOCATION_FLAGS])

    assert result.exit_code == 0
    assert coordinator.status.remote_calls == 1
    assert printed == [coordinator.snapshot]
    assert calls["execution_run_id"] == RUN_ID
    deployment = calls["deployment"]
    assert deployment.environment == "production"
    assert deployment.deployment_name == "ShortMDWorkflow"
    assert deployment.deployment_version == 7


def test_run_cancel_calls_the_same_coordinator_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _ = _patch_coordinator(monkeypatch)
    monkeypatch.setattr(
        "biomodals.cli._print_execution_snapshot",
        lambda _snapshot: None,
    )

    result = runner.invoke(app, ["run", "cancel", *LOCATION_FLAGS])

    assert result.exit_code == 0
    assert coordinator.cancel.remote_calls == 1


def test_run_resume_spawns_a_detached_coordinator_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _ = _patch_coordinator(monkeypatch)

    result = runner.invoke(app, ["run", "resume", *LOCATION_FLAGS])

    assert result.exit_code == 0
    assert coordinator.resume.spawn_calls == 1
    assert str(RUN_ID) in result.output
    assert "production/ShortMDWorkflow/v7" in result.output
    assert "fc-resume" in result.output


def test_top_level_run_is_reserved_for_execution_lifecycle() -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "status" in result.output
    assert "cancel" in result.output
    assert "resume" in result.output
