"""Tests for workflow-aware CLI catalog loading."""

# ruff: noqa: D103

from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from biomodals.cli import _load_entry, app
from biomodals.helper.catalog import AppFunction

runner = CliRunner()


def test_cli_loads_workflow_namespace_names() -> None:
    workflow = _load_entry("workflow", "ppiflow")

    assert workflow.module == "biomodals.workflow.ppiflow_workflow"
    assert workflow.category == "workflow"


def test_workflow_list_command_shows_workflow_names_without_legacy_prefix() -> None:
    result = runner.invoke(app, ["workflow", "list", "--short"])

    assert result.exit_code == 0
    assert "ppiflow" in result.output
    assert "workflow-ppiflow" not in result.output
    assert "orchestrator" not in result.output


def test_app_list_command_is_namespaced() -> None:
    result = runner.invoke(app, ["app", "list", "--short"])

    assert result.exit_code == 0
    assert "rosetta" in result.output


def test_top_level_list_remains_app_compatibility_alias() -> None:
    result = runner.invoke(app, ["list", "--short"])

    assert result.exit_code == 0
    assert "rosetta" in result.output


def test_app_deploy_command_is_namespaced() -> None:
    result = runner.invoke(app, ["app", "deploy", "--help"])

    assert result.exit_code == 0
    assert "Name or path of the app to deploy" in result.output


def test_top_level_deploy_remains_app_compatibility_alias() -> None:
    result = runner.invoke(app, ["deploy", "--help"])

    assert result.exit_code == 0
    assert "Name or path of the app to deploy" in result.output


def test_workflow_run_rejects_files_outside_workflow_package(tmp_path: Path) -> None:
    ad_hoc_workflow = tmp_path / "ad_hoc_workflow.py"
    ad_hoc_workflow.write_text('"""Not a packaged Biomodals workflow."""\n')

    result = runner.invoke(app, ["workflow", "run", str(ad_hoc_workflow)])

    assert result.exit_code == 1
    assert "Workflow paths must be under" in result.output
    assert "biomodals.workflow" in result.output


@dataclass
class _FakeWorkflow:
    name: str = "ambiguous"
    module: str = "biomodals.workflow.ambiguous_workflow"
    path: Path = Path("src/biomodals/workflow/ambiguous_workflow.py")
    _entrypoint: str | None = None

    def __post_init__(self) -> None:
        self._local_entrypoint_idx = [0, 1]
        self.functions = [
            AppFunction("first", "local_entrypoint", None, []),
            AppFunction("second", "local_entrypoint", None, []),
        ]

    def __getitem__(self, name: str | int) -> AppFunction:
        if isinstance(name, str):
            for function in self.functions:
                if function.name == name:
                    return function
            raise KeyError(name)
        return self.functions[name]


@dataclass
class _SingleEntrypointWorkflow:
    name: str = "shortmd"
    module: str = "biomodals.workflow.shortmd_workflow"
    path: Path = Path("src/biomodals/workflow/shortmd_workflow.py")
    _entrypoint: str | None = "submit_shortmd_workflow"

    def __post_init__(self) -> None:
        self._local_entrypoint_idx = []
        self.functions = []


@dataclass
class _SingleEntrypointApp:
    name: str = "rosetta"
    path: Path = Path("src/biomodals/app/bioinfo/rosetta_app.py")
    _entrypoint: str | None = "main"


@dataclass
class _NoEntrypointApp:
    name: str = "rosetta"
    path: Path = Path("src/biomodals/app/bioinfo/rosetta_app.py")
    _entrypoint: str | None = None


def test_workflow_run_requires_entrypoint_for_multiple_local_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("biomodals.cli._load_entry", lambda *_args: _FakeWorkflow())

    result = runner.invoke(app, ["workflow", "run", "ambiguous"])

    assert result.exit_code == 1
    assert "contains multiple local entrypoints" in result.output
    assert "::first" in result.output
    assert "::second" in result.output


def test_workflow_run_dry_run_forwards_entrypoint_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    def fake_run_command(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _SingleEntrypointWorkflow(),
    )
    monkeypatch.setattr("biomodals.cli.run_command", fake_run_command)

    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "shortmd",
            "--dry-run",
            "--",
            "/inputs",
            "--replicates",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert calls["command"][1:3] == ["-m", "modal"]
    assert (
        "biomodals.workflow.shortmd_workflow::submit_shortmd_workflow"
        in calls["command"]
    )
    module_idx = calls["command"].index(
        "biomodals.workflow.shortmd_workflow::submit_shortmd_workflow"
    )
    assert calls["command"][module_idx + 1 :] == [
        "--dry-run",
        "/inputs",
        "--replicates",
        "1",
    ]
    assert calls["kwargs"]["output_mode"] == "inherit"


def test_app_run_uses_inherited_output_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    def fake_run_command(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs

    monkeypatch.setattr(
        "biomodals.cli._load_entry", lambda *_args: _SingleEntrypointApp()
    )
    monkeypatch.setattr("biomodals.cli.run_command", fake_run_command)

    result = runner.invoke(app, ["app", "run", "rosetta", "--", "--example"])

    assert result.exit_code == 0
    assert calls["command"][-1] == "--example"
    assert calls["kwargs"]["output_mode"] == "inherit"


def test_app_run_without_entrypoint_renders_help_without_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {}

    def fake_run_command(*_args, **_kwargs):
        calls["run_command"] = True

    def fake_show_entry_help(*args, **kwargs):
        calls["show_entry_help"] = (args, kwargs)

    monkeypatch.setattr("biomodals.cli._load_entry", lambda *_args: _NoEntrypointApp())
    monkeypatch.setattr("biomodals.cli.run_command", fake_run_command)
    monkeypatch.setattr("biomodals.cli._show_entry_help", fake_show_entry_help)

    result = runner.invoke(app, ["app", "run", "rosetta"])

    assert result.exit_code == 0
    assert "run_command" not in calls
    assert calls["show_entry_help"] == (
        ("app", "src/biomodals/app/bioinfo/rosetta_app.py"),
        {"verbose": False},
    )
