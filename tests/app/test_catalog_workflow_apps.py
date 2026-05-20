"""Tests for Biomodals catalog workflow discovery."""

# ruff: noqa: D103

from biomodals.app.catalog import (
    BiomodalsApp,
    get_all_apps,
    get_app_catalog,
    get_workflow_catalog,
)


def test_default_catalog_does_not_collect_workflows() -> None:
    apps = get_all_apps(use_absolute_paths=True)

    assert "workflow-ppiflow" not in apps
    assert apps["ppiflow"].name == "ppiflow_app.py"
    assert "ppiflow_workflow" not in apps
    assert "workflow-orchestrator" not in apps


def test_app_catalog_does_not_collect_workflow_scripts() -> None:
    apps = get_app_catalog(use_absolute_paths=True)

    assert "workflow-ppiflow" not in apps
    assert "ppiflow_workflow" not in apps


def test_workflow_catalog_discovers_natural_workflow_names() -> None:
    workflows = get_workflow_catalog(use_absolute_paths=True)

    assert "ppiflow" in workflows
    assert "workflow-ppiflow" not in workflows
    assert "orchestrator" not in workflows
    assert workflows["ppiflow"].name == "ppiflow_workflow.py"


def test_workflow_catalog_can_include_legacy_workflow_aliases() -> None:
    workflows = get_workflow_catalog(
        use_absolute_paths=True,
        include_compat_aliases=True,
    )

    assert workflows["ppiflow"] == workflows["workflow-ppiflow"]


def test_workflow_file_resolves_to_workflow_module_with_natural_name() -> None:
    workflows = get_workflow_catalog(use_absolute_paths=True)
    app = BiomodalsApp("ppiflow", all_apps=workflows)

    assert app.module == "biomodals.workflow.ppiflow_workflow"
    assert app.category == "workflow"
