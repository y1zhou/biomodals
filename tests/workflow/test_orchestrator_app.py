"""Tests for the mocked workflow orchestrator boundary."""

# ruff: noqa: D103

from pathlib import Path

from biomodals.schema import AppRunResult, AppRunStatus
from biomodals.workflow import orchestrator_app


def test_orchestrator_helper_uses_runtime_from_definition(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeRuntime:
        @classmethod
        def from_definition(
            cls,
            *,
            workflow_name: str,
            workflow_definition: dict[str, object],
            volume_root: Path,
        ):
            calls["workflow_name"] = workflow_name
            calls["workflow_definition"] = workflow_definition
            calls["volume_root"] = volume_root
            return cls()

        def run(self, *, run_id: str, force: bool = False) -> AppRunResult:
            calls["run_id"] = run_id
            calls["force"] = force
            return AppRunResult(status=AppRunStatus.SUCCEEDED)

    monkeypatch.setattr(orchestrator_app, "WorkflowRuntime", FakeRuntime)
    monkeypatch.setattr(
        orchestrator_app.CONF,
        "output_volume_mountpoint",
        "/workflow-outputs",
    )

    result = orchestrator_app._run_workflow_orchestrator(
        workflow_name="demo",
        run_id="run-1",
        workflow_definition={"nodes": []},
        force=True,
    )

    assert result.status == AppRunStatus.SUCCEEDED
    assert calls == {
        "workflow_name": "demo",
        "workflow_definition": {"nodes": []},
        "volume_root": Path("/workflow-outputs"),
        "run_id": "run-1",
        "force": True,
    }
