"""CLI tests for explicit remote Execution Run lifecycle controls."""

# ruff: noqa: D101, D102, D103, D107

from types import SimpleNamespace
from uuid import UUID

import pytest
from typer.testing import CliRunner

from biomodals.cli import app
from biomodals.execution import DeploymentIdentity, RunStatus, RunStatusReason

RUN_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
SUCCESSOR_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
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
TARGET_FLAGS = [
    "--target-environment",
    "production",
    "--target-deployment-name",
    "ShortMDWorkflow",
    "--target-deployment-version",
    "8",
]
runner = CliRunner()


class FakeRemoteMethod:
    def __init__(
        self,
        result: object,
        *,
        name: str = "method",
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.name = name
        self.events = [] if events is None else events
        self.remote_calls = 0
        self.spawn_calls = 0
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def remote(self, *args: object, **kwargs: object) -> object:
        self.events.append(f"{self.name}.remote")
        self.remote_calls += 1
        self.calls.append((args, kwargs))
        return self.result

    def spawn(self, *args: object, **kwargs: object) -> object:
        self.events.append(f"{self.name}.spawn")
        self.spawn_calls += 1
        self.calls.append((args, kwargs))
        return self.result


class FakeCoordinator:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.snapshot = object()
        self.status = FakeRemoteMethod(self.snapshot)
        self.cancel = FakeRemoteMethod(self.snapshot)
        self.resume = FakeRemoteMethod(SimpleNamespace(object_id="fc-resume"))
        self.prepare_restart = FakeRemoteMethod(
            None,
            name="prepare_restart",
            events=self.events,
        )
        self.drive_prepared = FakeRemoteMethod(
            SimpleNamespace(object_id="fc-restart"),
            name="drive_prepared",
            events=self.events,
        )


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


def test_run_status_uses_explicit_location_and_prints_overview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, calls = _patch_coordinator(monkeypatch)
    printed: list[object] = []
    monkeypatch.setattr(
        "biomodals.cli._print_execution_overview",
        printed.append,
    )

    result = runner.invoke(app, ["run", "status", *LOCATION_FLAGS])

    assert result.exit_code == 0
    assert coordinator.status.remote_calls == 1
    assert printed == [coordinator.snapshot]
    assert calls["execution_run_id"] == RUN_ID
    deployment = calls["deployment"]
    assert isinstance(deployment, DeploymentIdentity)
    assert deployment.environment == "production"
    assert deployment.deployment_name == "ShortMDWorkflow"
    assert deployment.deployment_version == 7


def test_run_status_distinguishes_durable_slots_from_live_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _ = _patch_coordinator(monkeypatch)
    coordinator.snapshot = SimpleNamespace(
        run=SimpleNamespace(
            execution_run_id=RUN_ID,
            deployment=DeploymentIdentity(
                environment="production",
                deployment_name="ShortMDWorkflow",
                deployment_version=7,
            ),
            status=RunStatus.STATE_UNKNOWN,
            status_reason=RunStatusReason.PROVIDER_OUTCOME_UNKNOWN,
            status_message="Modal call state was inconclusive",
        ),
        active_provider_calls=SimpleNamespace(total=12, gpu=12),
    )
    coordinator.status.result = coordinator.snapshot

    result = runner.invoke(app, ["run", "status", *LOCATION_FLAGS])

    assert result.exit_code == 0
    assert "Occupied Provider Call Slots: 12 total, 12 GPU" in result.output
    assert "durable ownership records" in result.output
    assert "not confirmed live Modal containers" in result.output
    assert "Automatic scheduling is stopped" in result.output


def test_run_cancel_calls_the_same_coordinator_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _ = _patch_coordinator(monkeypatch)
    monkeypatch.setattr(
        "biomodals.cli._print_execution_overview",
        lambda _overview: None,
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


def test_run_restart_creates_a_new_explicit_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, calls = _patch_coordinator(monkeypatch)
    monkeypatch.setattr("biomodals.cli.uuid4", lambda: SUCCESSOR_ID)

    result = runner.invoke(
        app,
        [
            "run",
            "restart",
            *LOCATION_FLAGS,
            *TARGET_FLAGS,
            "--max-active-provider-calls",
            "12",
            "--max-active-gpu-provider-calls",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert coordinator.prepare_restart.remote_calls == 1
    assert coordinator.drive_prepared.spawn_calls == 1
    assert coordinator.events == ["prepare_restart.remote", "drive_prepared.spawn"]
    assert calls["execution_run_id"] == SUCCESSOR_ID
    deployment = calls["deployment"]
    assert isinstance(deployment, DeploymentIdentity)
    assert deployment.environment == "production"
    assert deployment.deployment_name == "ShortMDWorkflow"
    assert deployment.deployment_version == 8
    assert coordinator.prepare_restart.calls == [
        (
            (),
            {
                "predecessor_execution_run_id": str(RUN_ID),
                "predecessor_deployment_environment": "production",
                "predecessor_deployment_name": "ShortMDWorkflow",
                "predecessor_deployment_version": 7,
                "max_active_provider_calls": 12,
                "max_active_gpu_provider_calls": 3,
            },
        )
    ]
    assert str(SUCCESSOR_ID) in result.output
    assert "production/ShortMDWorkflow/v8" in result.output
    assert "fc-restart" in result.output


def test_run_restart_does_not_drive_when_preparation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _ = _patch_coordinator(monkeypatch)
    monkeypatch.setattr("biomodals.cli.uuid4", lambda: SUCCESSOR_ID)

    def fail_preparation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("invalid predecessor")

    monkeypatch.setattr(coordinator.prepare_restart, "remote", fail_preparation)

    result = runner.invoke(
        app,
        ["run", "restart", *LOCATION_FLAGS, *TARGET_FLAGS],
    )

    assert result.exit_code == 1
    assert coordinator.drive_prepared.spawn_calls == 0
    assert "invalid predecessor" in result.output
    assert str(SUCCESSOR_ID) in result.output
    assert "submission outcome is unknown" in result.output


def test_run_restart_reports_unknown_outcome_when_drive_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _ = _patch_coordinator(monkeypatch)
    monkeypatch.setattr("biomodals.cli.uuid4", lambda: SUCCESSOR_ID)

    def fail_drive(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(coordinator.drive_prepared, "spawn", fail_drive)

    result = runner.invoke(
        app,
        ["run", "restart", *LOCATION_FLAGS, *TARGET_FLAGS],
    )

    assert result.exit_code == 1
    assert "submission outcome is unknown" in result.output
    assert str(SUCCESSOR_ID) in result.output


def test_run_restart_reports_successor_when_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator, _ = _patch_coordinator(monkeypatch)
    monkeypatch.setattr("biomodals.cli.uuid4", lambda: SUCCESSOR_ID)

    def interrupt(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(coordinator.prepare_restart, "remote", interrupt)

    result = runner.invoke(
        app,
        ["run", "restart", *LOCATION_FLAGS, *TARGET_FLAGS],
    )

    assert result.exit_code == 130
    assert "submission outcome is unknown" in result.output
    assert str(SUCCESSOR_ID) in result.output


def test_top_level_run_is_reserved_for_execution_lifecycle() -> None:
    result = runner.invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "status" in result.output
    assert "cancel" in result.output
    assert "resume" in result.output
    assert "restart" in result.output


def test_run_resume_help_describes_both_resumable_states() -> None:
    result = runner.invoke(app, ["run", "resume", "--help"])

    assert result.exit_code == 0
    assert "suspended or state-unknown Run" in result.output
