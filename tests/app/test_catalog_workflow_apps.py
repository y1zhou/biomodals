"""Tests for Biomodals app catalog workflow discovery."""

# ruff: noqa: D103

from biomodals.app.catalog import BiomodalsApp, get_all_apps


def test_catalog_discovers_workflow_orchestrator() -> None:
    apps = get_all_apps(use_absolute_paths=True)

    assert "workflow-orchestrator" in apps
    assert apps["workflow-orchestrator"].name == "orchestrator_app.py"


def test_workflow_orchestrator_resolves_to_workflow_module() -> None:
    app = BiomodalsApp("workflow-orchestrator")

    assert app.module == "biomodals.workflow.orchestrator_app"
    assert app.category == "workflow"
