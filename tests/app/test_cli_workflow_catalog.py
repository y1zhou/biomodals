"""Tests for workflow-aware CLI catalog loading."""

# ruff: noqa: D103

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from biomodals.cli import _load_entry, app
from biomodals.helper.catalog import AppFunction
from biomodals.helper.styling import strip_ansi

runner = CliRunner()


def test_cli_loads_workflow_namespace_names() -> None:
    workflow = _load_entry("workflow", "ppiflow")

    assert workflow.module == "biomodals.workflow.ppiflow_workflow"
    assert workflow.category == "workflow"


def test_cli_loads_hyphenated_workflow_aliases() -> None:
    workflow = _load_entry("workflow", "rfd-ligandmpnn")

    assert workflow.name == "rfd_ligandmpnn"
    assert workflow.module == "biomodals.workflow.rfd_ligandmpnn_workflow"


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


@pytest.mark.parametrize("command", ["list", "ls", "l", "help", "h", "deploy", "d"])
def test_top_level_app_compatibility_aliases_are_removed(command: str) -> None:
    result = runner.invoke(app, [command])

    assert result.exit_code == 2
    assert f"No such command '{command}'" in strip_ansi(result.output)


def test_app_deploy_command_is_namespaced() -> None:
    result = runner.invoke(app, ["app", "deploy", "--help"])
    output = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "Name or path of the app to deploy" in output
    assert "--env" in output
    assert "--strategy" in output
    assert "rolling" in output
    assert "recreate" in output


def test_workflow_deploy_uses_importable_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "biomodals.cli.run_command",
        lambda command, **_kwargs: commands.append(command),
    )

    result = runner.invoke(
        app,
        [
            "workflow",
            "deploy",
            "rfd_ligandmpnn",
            "--name",
            "RFDLigandMPNN-staging",
            "--tag",
            "candidate",
            "--env",
            "staging",
            "--strategy",
            "recreate",
        ],
    )

    assert result.exit_code == 0
    assert commands == [
        [
            sys.executable,
            "-m",
            "modal",
            "deploy",
            "--name",
            "RFDLigandMPNN-staging",
            "--tag",
            "candidate",
            "--env",
            "staging",
            "--strategy",
            "recreate",
            "-m",
            "biomodals.workflow.rfd_ligandmpnn_workflow",
        ]
    ]


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


@dataclass
class _CoordinatedApp:
    name: str = "alphafold3"
    module: str = "biomodals.app.fold.alphafold3_app"
    path: Path = Path("src/biomodals/app/fold/alphafold3_app.py")
    _entrypoint: str | None = "submit_alphafold3_task"


def test_workflow_run_requires_entrypoint_for_multiple_local_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("biomodals.cli._load_entry", lambda *_args: _FakeWorkflow())

    result = runner.invoke(app, ["workflow", "run", "ambiguous"])

    assert result.exit_code == 1
    assert "contains multiple local entrypoints" in result.output
    assert "::first" in result.output
    assert "::second" in result.output


def test_workflow_dry_run_invokes_the_entrypoint_locally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_invoke_local_entrypoint(**kwargs):
        calls.update(kwargs)

    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _SingleEntrypointWorkflow(),
    )
    monkeypatch.setattr(
        "biomodals.cli.invoke_local_entrypoint",
        fake_invoke_local_entrypoint,
    )

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
    assert calls["module_name"] == "biomodals.workflow.shortmd_workflow"
    assert calls["entrypoint_name"] == "submit_shortmd_workflow"
    assert calls["flags"] == [
        "/inputs",
        "--replicates",
        "1",
    ]
    assert calls["overrides"] == {
        "dry_run": True,
        "use_deployed_coordinator": False,
        "deployment_environment": "development",
        "deployment_name": None,
        "deployment_version": 1,
        "restart_from": None,
    }
    assert calls["environment_name"] is None


def test_workflow_run_resolves_and_forwards_an_exact_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    entrypoint_calls: list[dict[str, Any]] = []

    def fake_run_command(command, **kwargs):
        calls.append((command, kwargs))
        if "history" in command:
            return ['[{"version":"v7"},{"version":"v4"}]']
        return []

    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _SingleEntrypointWorkflow(),
    )
    monkeypatch.setattr(
        "biomodals.cli._deployment_name",
        lambda _workflow: "ShortMDWorkflow",
    )
    monkeypatch.setattr("biomodals.cli.run_command", fake_run_command)
    monkeypatch.setattr(
        "biomodals.cli.invoke_local_entrypoint",
        lambda **kwargs: entrypoint_calls.append(kwargs),
    )

    result = runner.invoke(
        app,
        ["workflow", "run", "shortmd", "--", "/inputs"],
    )

    assert result.exit_code == 0
    history_command, history_kwargs = calls[0]
    assert history_command[-5:] == [
        "history",
        "ShortMDWorkflow",
        "--env",
        "main",
        "--json",
    ]
    assert history_kwargs["output_mode"] == "capture"
    assert len(calls) == 1
    assert entrypoint_calls == [
        {
            "module_name": "biomodals.workflow.shortmd_workflow",
            "entrypoint_name": "submit_shortmd_workflow",
            "flags": ["/inputs"],
            "overrides": {
                "dry_run": False,
                "use_deployed_coordinator": True,
                "deployment_environment": "main",
                "deployment_name": "ShortMDWorkflow",
                "deployment_version": 7,
                "restart_from": None,
            },
            "program_name": (
                "biomodals workflow run shortmd::submit_shortmd_workflow --"
            ),
            "environment_name": "main",
        }
    ]


def test_workflow_restart_forwards_only_the_explicit_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch-time workflow restart remains an explicit successor operation."""
    commands: list[list[str]] = []
    entrypoint_calls: list[dict[str, Any]] = []
    predecessor = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    def fake_run_command(command, **_kwargs):
        commands.append(command)
        if "history" in command:
            return ['[{"version":"v7"}]']
        return []

    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _SingleEntrypointWorkflow(),
    )
    monkeypatch.setattr(
        "biomodals.cli._deployment_name",
        lambda _workflow: "ShortMDWorkflow",
    )
    monkeypatch.setattr("biomodals.cli.run_command", fake_run_command)
    monkeypatch.setattr(
        "biomodals.cli.invoke_local_entrypoint",
        lambda **kwargs: entrypoint_calls.append(kwargs),
    )

    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "shortmd",
            "--restart-from",
            predecessor,
            "--",
            "/inputs",
        ],
    )

    assert result.exit_code == 0
    assert len(commands) == 1
    assert entrypoint_calls[0]["overrides"]["restart_from"] == predecessor


@pytest.mark.parametrize(
    "mode_flags",
    [
        ("--development",),
        ("--dry-run",),
        ("--mode", "shell"),
    ],
)
def test_workflow_restart_rejects_launches_without_durable_lookup(
    monkeypatch: pytest.MonkeyPatch,
    mode_flags: tuple[str, ...],
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _SingleEntrypointWorkflow(),
    )
    monkeypatch.setattr(
        "biomodals.cli.run_command",
        lambda command, **_kwargs: commands.append(command),
    )

    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "shortmd",
            *mode_flags,
            "--restart-from",
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "--",
            "/inputs",
        ],
    )

    assert result.exit_code == 1
    assert "--restart-from" in strip_ansi(result.output)
    assert commands == []


def test_workflow_run_validates_an_explicit_version_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    entrypoint_calls: list[dict[str, Any]] = []

    def fake_run_command(command, **_kwargs):
        commands.append(command)
        if "history" in command:
            return ['[{"version":"v7"},{"version":"v4"}]']
        return []

    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _SingleEntrypointWorkflow(),
    )
    monkeypatch.setattr("biomodals.cli.run_command", fake_run_command)
    monkeypatch.setattr(
        "biomodals.cli.invoke_local_entrypoint",
        lambda **kwargs: entrypoint_calls.append(kwargs),
    )

    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "shortmd",
            "--environment",
            "production",
            "--deployment-name",
            "shortmd-prod",
            "--version",
            "4",
            "--",
            "/inputs",
        ],
    )

    assert result.exit_code == 0
    assert commands[0][-4:] == [
        "shortmd-prod",
        "--env",
        "production",
        "--json",
    ]
    assert len(commands) == 1
    assert entrypoint_calls[0]["overrides"]["deployment_version"] == 4


def test_workflow_run_development_mode_skips_deployment_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run_command(command, **_kwargs):
        commands.append(command)
        return []

    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _SingleEntrypointWorkflow(),
    )
    monkeypatch.setattr("biomodals.cli.run_command", fake_run_command)

    result = runner.invoke(
        app,
        ["workflow", "run", "shortmd", "--development", "--", "/inputs"],
    )

    assert result.exit_code == 0
    assert len(commands) == 1
    assert "history" not in commands[0]
    assert "--use-deployed-coordinator" not in commands[0]


def test_workflow_run_fails_closed_for_an_unavailable_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _SingleEntrypointWorkflow(),
    )
    monkeypatch.setattr(
        "biomodals.cli.run_command",
        lambda *_args, **_kwargs: ['[{"version":"v7"}]'],
    )

    result = runner.invoke(
        app,
        ["workflow", "run", "shortmd", "--version", "4", "--", "/inputs"],
    )

    assert result.exit_code == 1
    output = strip_ansi(result.output)
    assert "version 4 is" in output
    assert "not available" in output


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

    result = runner.invoke(
        app,
        ["app", "run", "rosetta", "--development", "--", "--example"],
    )

    assert result.exit_code == 0
    assert calls["command"][-1] == "--example"
    assert calls["kwargs"]["output_mode"] == "inherit"


def test_uncoordinated_app_requires_explicit_development_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _SingleEntrypointApp(),
    )

    result = runner.invoke(app, ["app", "run", "rosetta", "--", "--example"])

    assert result.exit_code == 1
    assert "does not expose a deployment coordinator" in result.output
    assert "--development" in result.output


def test_coordinated_app_run_resolves_and_forwards_an_exact_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coordinator-aware app entrypoints use the same pinned lookup as workflows."""
    calls: list[tuple[list[str], dict[str, object]]] = []
    entrypoint_calls: list[dict[str, Any]] = []

    def fake_run_command(command, **kwargs):
        calls.append((command, kwargs))
        if "history" in command:
            return ['[{"version":"v9"},{"version":"v7"}]']
        return []

    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _CoordinatedApp(),
    )
    monkeypatch.setattr(
        "biomodals.cli._deployment_name",
        lambda _app: "AlphaFold3",
    )
    monkeypatch.setattr("biomodals.cli.run_command", fake_run_command)
    monkeypatch.setattr(
        "biomodals.cli.invoke_local_entrypoint",
        lambda **kwargs: entrypoint_calls.append(kwargs),
    )

    result = runner.invoke(
        app,
        [
            "app",
            "run",
            "alphafold3::submit_alphafold3_task",
            "--environment",
            "production",
            "--version",
            "7",
            "--",
            "--input-json",
            "input.json",
        ],
    )

    assert result.exit_code == 0
    assert calls[0][0][-4:] == [
        "AlphaFold3",
        "--env",
        "production",
        "--json",
    ]
    assert len(calls) == 1
    assert entrypoint_calls == [
        {
            "module_name": "biomodals.app.fold.alphafold3_app",
            "entrypoint_name": "submit_alphafold3_task",
            "flags": ["--input-json", "input.json"],
            "overrides": {
                "use_deployed_coordinator": True,
                "deployment_environment": "production",
                "deployment_name": "AlphaFold3",
                "deployment_version": 7,
                "restart_from": None,
            },
            "program_name": ("biomodals app run alphafold3::submit_alphafold3_task --"),
            "environment_name": "production",
        }
    ]


def test_coordinated_app_restart_forwards_only_the_explicit_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch-time restart remains a thin convenience over Successor Run creation."""
    commands: list[list[str]] = []
    entrypoint_calls: list[dict[str, Any]] = []
    predecessor = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"

    def fake_run_command(command, **_kwargs):
        commands.append(command)
        if "history" in command:
            return ['[{"version":"v9"}]']
        return []

    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _CoordinatedApp(),
    )
    monkeypatch.setattr(
        "biomodals.cli._deployment_name",
        lambda _app: "AlphaFold3",
    )
    monkeypatch.setattr("biomodals.cli.run_command", fake_run_command)
    monkeypatch.setattr(
        "biomodals.cli.invoke_local_entrypoint",
        lambda **kwargs: entrypoint_calls.append(kwargs),
    )

    result = runner.invoke(
        app,
        [
            "app",
            "run",
            "alphafold3::submit_alphafold3_task",
            "--restart-from",
            predecessor,
            "--",
            "--input-json",
            "input.json",
        ],
    )

    assert result.exit_code == 0
    assert len(commands) == 1
    assert entrypoint_calls[0]["overrides"]["restart_from"] == predecessor


def test_coordinated_app_development_mode_skips_deployment_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Development mode keeps the source-backed no-resume behavior explicit."""
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "biomodals.cli._load_entry",
        lambda *_args: _CoordinatedApp(),
    )
    monkeypatch.setattr(
        "biomodals.cli.run_command",
        lambda command, **_kwargs: commands.append(command),
    )

    result = runner.invoke(
        app,
        [
            "app",
            "run",
            "alphafold3::submit_alphafold3_task",
            "--development",
            "--",
            "--input-json",
            "input.json",
        ],
    )

    assert result.exit_code == 0
    assert len(commands) == 1
    assert "history" not in commands[0]
    assert "--use-deployed-coordinator" not in commands[0]


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
