"""Tests for Biomodals catalog workflow discovery."""

# ruff: noqa: D103

from biomodals.app.catalog import WORKFLOW_HOME, BiomodalsApp, get_all_apps


def test_default_catalog_does_not_collect_workflows() -> None:
    apps = get_all_apps(use_absolute_paths=True)

    assert "workflow-ppiflow" not in apps
    assert "workflow-orchestrator" not in apps


def test_catalog_discovers_workflows_with_suffix() -> None:
    workflows = get_all_apps(
        use_absolute_paths=True,
        app_home=WORKFLOW_HOME,
        suffix="workflow",
    )

    assert "workflow-ppiflow" in workflows
    assert workflows["workflow-ppiflow"].name == "ppiflow_workflow.py"


def test_workflow_file_resolves_to_workflow_module() -> None:
    workflows = get_all_apps(
        use_absolute_paths=True,
        app_home=WORKFLOW_HOME,
        suffix="workflow",
    )
    app = BiomodalsApp("workflow-ppiflow", all_apps=workflows)

    assert app.module == "biomodals.workflow.ppiflow_workflow"
    assert app.category == "workflow"
