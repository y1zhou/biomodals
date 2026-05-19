"""Tests for workflow-aware CLI catalog loading."""

# ruff: noqa: D103

from biomodals.cli import _catalog_for_list_type, _load_app


def test_cli_workflow_catalog_uses_workflow_suffix() -> None:
    workflows = _catalog_for_list_type("workflow", use_absolute_paths=True)

    assert "workflow-ppiflow" in workflows
    assert workflows["workflow-ppiflow"].name == "ppiflow_workflow.py"


def test_cli_load_app_resolves_workflow_names() -> None:
    app = _load_app("workflow-ppiflow")

    assert app.module == "biomodals.workflow.ppiflow_workflow"
    assert app.category == "workflow"
