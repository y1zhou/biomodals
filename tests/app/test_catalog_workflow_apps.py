"""Tests for Biomodals catalog workflow discovery."""

# ruff: noqa: D103

from pathlib import Path
from types import SimpleNamespace

import modal
import pytest

from biomodals.helper import catalog
from biomodals.helper.catalog import get_catalog, include_dependency_apps


def test_app_catalog_discovers_single_file_and_package_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    category = tmp_path / "design"
    package = category / "package"
    package.mkdir(parents=True)
    single_file = category / "single_app.py"
    package_app = package / "app.py"
    single_file.write_text('"""Single-file app."""\n')
    package_app.write_text('"""Package app."""\n')
    monkeypatch.setattr(catalog, "APP_HOME", tmp_path)

    apps = get_catalog("app", use_absolute_paths=True)

    assert apps == {
        "package": package_app.resolve(),
        "single": single_file.resolve(),
    }


def test_app_catalog_rejects_duplicate_names_across_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    category = tmp_path / "design"
    package = category / "duplicate"
    package.mkdir(parents=True)
    (category / "duplicate_app.py").write_text('"""Single-file app."""\n')
    (package / "app.py").write_text('"""Package app."""\n')
    monkeypatch.setattr(catalog, "APP_HOME", tmp_path)

    with pytest.raises(ValueError, match="Duplicate app name 'duplicate'"):
        get_catalog("app", use_absolute_paths=True)


def test_workflow_catalog_supports_both_layouts_and_rejects_duplicates(
    tmp_path: Path, monkeypatch
) -> None:
    package = tmp_path / "complex"
    package.mkdir()
    (package / "workflow.py").write_text('"""Package workflow."""\n')
    (package / "helper.py").write_text('"""Not a workflow."""\n')
    (tmp_path / "simple_workflow.py").write_text('"""Simple workflow."""\n')
    monkeypatch.setattr(catalog, "WORKFLOW_HOME", tmp_path)
    assert get_catalog("workflow", cwd=tmp_path) == {
        "complex": Path("complex/workflow.py"),
        "simple": Path("simple_workflow.py"),
    }
    assert get_catalog("workflow", use_absolute_paths=True)["complex"] == (
        package / "workflow.py"
    )
    assert catalog._catalog_entry_name(package / "workflow.py") == "complex"
    (tmp_path / "complex_workflow.py").write_text('"""Duplicate."""\n')
    with pytest.raises(ValueError, match="Duplicate workflow name 'complex'"):
        get_catalog("workflow")


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


def test_include_dependency_apps_excludes_the_child_coordinator(
    monkeypatch,
) -> None:
    """A workflow includes child functions but owns the only execution ledger."""
    workflow_app = modal.App("workflow")
    dependency_app = modal.App("dependency")

    @dependency_app.function(name="worker", serialized=True)
    def worker() -> None:
        return None

    @dependency_app.cls(serialized=True)
    class ExecutionCoordinator:
        @modal.method()
        def run(self) -> None:
            return None

    class FakeBiomodalsApp:
        def __init__(self, app_name_or_path: str, all_apps: dict[str, Path]) -> None:
            self.module = "fake.dependency_app"

    monkeypatch.setattr(
        catalog,
        "get_catalog",
        lambda *args, **kwargs: {"dependency": Path("/apps/dependency_app.py")},
    )
    monkeypatch.setattr(catalog, "BiomodalsApp", FakeBiomodalsApp)
    monkeypatch.setattr(
        catalog.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(app=dependency_app),
    )

    include_dependency_apps(workflow_app, ("dependency",))

    assert "worker" in workflow_app._local_state.functions
    assert not any(
        tag == "ExecutionCoordinator" or tag.startswith("ExecutionCoordinator.")
        for tag in (
            *workflow_app._local_state.functions,
            *workflow_app._local_state.classes,
        )
    )
