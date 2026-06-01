"""Tests for Biomodals catalog workflow discovery."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace

import modal
import pytest

from biomodals.helper import catalog
from biomodals.helper.catalog import BiomodalsApp, get_catalog, include_dependency_apps


def test_default_catalog_does_not_collect_workflows() -> None:
    apps = get_catalog("app", use_absolute_paths=True)

    assert "workflow-ppiflow" not in apps
    assert apps["ppiflow"].name == "ppiflow_app.py"
    assert "ppiflow_workflow" not in apps
    assert "workflow-orchestrator" not in apps


def test_app_catalog_does_not_collect_workflow_scripts() -> None:
    apps = get_catalog("app", use_absolute_paths=True)

    assert "workflow-ppiflow" not in apps
    assert "ppiflow_workflow" not in apps


def test_workflow_catalog_discovers_natural_workflow_names() -> None:
    workflows = get_catalog("workflow", use_absolute_paths=True)

    assert "ppiflow" in workflows
    assert "shortmd" in workflows
    assert "workflow-ppiflow" not in workflows
    assert "orchestrator" not in workflows
    assert workflows["ppiflow"].name == "ppiflow_workflow.py"
    assert workflows["shortmd"].name == "shortmd_workflow.py"


def test_workflow_file_resolves_to_workflow_module_with_natural_name() -> None:
    workflows = get_catalog("workflow", use_absolute_paths=True)
    app = BiomodalsApp("ppiflow", all_apps=workflows)

    assert app.module == "biomodals.workflow.ppiflow_workflow"
    assert app.category == "workflow"


def test_include_dependency_apps_resolves_catalog_app_and_includes_modal_app(
    monkeypatch,
) -> None:
    workflow_app = modal.App("workflow")
    dependency_app = modal.App("dependency")

    @dependency_app.function(name="dependency_function", serialized=True)
    def dependency_function() -> None:
        return None

    class FakeBiomodalsApp:
        def __init__(self, app_name_or_path: str, all_apps: dict[str, Path]) -> None:
            assert app_name_or_path == "dependency"
            assert all_apps == {"dependency": Path("/apps/dependency_app.py")}
            self.module = "fake.dependency_app"

    monkeypatch.setattr(
        catalog,
        "get_catalog",
        lambda catalog_type, *, use_absolute_paths=False, cwd=None: {
            "dependency": Path("/apps/dependency_app.py")
        },
    )
    monkeypatch.setattr(catalog, "BiomodalsApp", FakeBiomodalsApp)
    monkeypatch.setattr(
        catalog.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(app=dependency_app),
    )

    assert include_dependency_apps(workflow_app, ("dependency",)) is workflow_app
    assert "dependency_function" in workflow_app._local_state.functions


def test_include_dependency_apps_rejects_duplicate_modal_tags(monkeypatch) -> None:
    workflow_app = modal.App("workflow")
    dependency_app = modal.App("dependency")

    @workflow_app.function(name="duplicate_function", serialized=True)
    def workflow_duplicate_function() -> None:
        return None

    @dependency_app.function(name="duplicate_function", serialized=True)
    def dependency_duplicate_function() -> None:
        return None

    class FakeBiomodalsApp:
        def __init__(self, app_name_or_path: str, all_apps: dict[str, Path]) -> None:
            self.module = "fake.dependency_app"

    monkeypatch.setattr(
        catalog,
        "get_catalog",
        lambda catalog_type, *, use_absolute_paths=False, cwd=None: {
            "dependency": Path("/apps/dependency_app.py")
        },
    )
    monkeypatch.setattr(catalog, "BiomodalsApp", FakeBiomodalsApp)
    monkeypatch.setattr(
        catalog.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(app=dependency_app),
    )

    with pytest.raises(ValueError, match="duplicate_function"):
        include_dependency_apps(workflow_app, ("dependency",))
