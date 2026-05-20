"""Tests for workflow-aware CLI catalog loading."""

# ruff: noqa: D103

from typer.testing import CliRunner

from biomodals.cli import _catalog_for_list_type, _load_catalog_entry, app

runner = CliRunner()


def test_cli_workflow_catalog_uses_natural_workflow_names() -> None:
    workflows = _catalog_for_list_type("workflow", use_absolute_paths=True)

    assert "ppiflow" in workflows
    assert "workflow-ppiflow" not in workflows
    assert workflows["ppiflow"].name == "ppiflow_workflow.py"


def test_cli_loads_workflow_namespace_names() -> None:
    workflow = _load_catalog_entry("workflow", "ppiflow")

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
